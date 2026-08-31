"""Insurance capability layer used by the MCP server.

This module intentionally contains NO agent routing and NO MCP client/server code.
It exposes deterministic/domain capabilities that MCP tools can call:
    - fraud risk scoring + SHAP explainability
    - policy hybrid retrieval + reranking + grounded answer generation

Architecture:
    Specialist Agent -> MCP Tool -> this capability layer -> ML/RAG components
"""

from __future__ import annotations

import logging
import os
import re
import threading
import warnings
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import weaviate
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import logging as transformers_logging

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
class Spinner:
    """Minimal CLI progress indicator retained for local/server logs."""

    def __init__(self, message: str = "Processing...") -> None:
        self.message = message
        self._stop = False
        self._thread: threading.Thread | None = None

    def __enter__(self):
        # Do not animate in environments that are likely to be non-interactive.
        if os.getenv("DISABLE_SPINNERS", "0") == "1":
            return self
        try:
            import sys
            import time

            self._sys = sys
            self._time = time
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        except Exception:
            self._thread = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=0.2)
        try:
            self._sys.stdout.write("\r\033[K")
            self._sys.stdout.flush()
        except Exception:
            pass

    def _spin(self):
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not self._stop:
            try:
                self._sys.stdout.write(f"\r{chars[i % len(chars)]} {self.message}")
                self._sys.stdout.flush()
                self._time.sleep(0.08)
                i += 1
            except Exception:
                break


class InsuranceRiskEngine:
    """Fraud model, feature engineering, historical baselines and SHAP."""

    def __init__(self, artifact_path: str | None = None, dataset_path: str | None = None):
        artifact_path = artifact_path or os.getenv(
            "FRAUD_MODEL_PATH", "output/fraud_detection_model.pkl"
        )
        dataset_path = dataset_path or os.getenv(
            "FRAUD_DATASET_PATH", "Health Insurance Fraud Claims.xlsx"
        )

        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"Trained model missing at '{artifact_path}'. Run train_fraud_model.py first."
            )

        with Spinner("Loading fraud model artifacts..."):
            artifacts = joblib.load(artifact_path)
            self.model = artifacts["model"]
            self.preprocessor = artifacts["preprocessor"]
            self.optimal_threshold = float(artifacts["optimal_threshold"])
            self.explainer = artifacts.get("explainer")
            self.feature_names = artifacts.get("feature_names", [])
            self.required_features = artifacts["required_features"]

        with Spinner("Loading historical claim baselines..."):
            self._load_dataset_statistics(dataset_path)

    def _load_dataset_statistics(self, dataset_path: str) -> None:
        self.patient_avg_map: Dict[str, float] = {}
        self.procedure_avg_map: Dict[str, float] = {}
        self.provider_avg_map: Dict[str, float] = {}
        self.specialty_avg_map: Dict[str, float] = {}
        self.specialty_to_code_map: Dict[str, Tuple[str, float]] = {}
        self.global_mean = 5014.20

        df = None
        if os.path.exists(dataset_path):
            try:
                if dataset_path.lower().endswith(".csv"):
                    df = pd.read_csv(dataset_path)
                else:
                    df = pd.read_excel(dataset_path)
            except Exception as exc:
                logger.warning("Could not load baseline dataset: %s", exc)

        processed_csv = "output/processed_claim_features.csv"
        if df is None and os.path.exists(processed_csv):
            try:
                df = pd.read_csv(processed_csv)
            except Exception as exc:
                logger.warning("Could not load processed baseline dataset: %s", exc)

        if df is not None:
            amt_col = "ClaimAmount" if "ClaimAmount" in df.columns else "claim_amount"
            if amt_col in df.columns:
                self.global_mean = float(pd.to_numeric(df[amt_col], errors="coerce").mean())

                patient_col = "PatientID" if "PatientID" in df.columns else "patient_id"
                if patient_col in df.columns:
                    self.patient_avg_map = (
                        df.groupby(patient_col)[amt_col].mean().dropna().to_dict()
                    )

                proc_col = "ProcedureCode" if "ProcedureCode" in df.columns else "procedure_code"
                if proc_col in df.columns:
                    self.procedure_avg_map = (
                        df.groupby(proc_col)[amt_col].mean().dropna().to_dict()
                    )

                prov_col = "ProviderID" if "ProviderID" in df.columns else (
                    "provider_hospital" if "provider_hospital" in df.columns else None
                )
                if prov_col:
                    self.provider_avg_map = (
                        df.groupby(prov_col)[amt_col].mean().dropna().to_dict()
                    )

                spec_col = "ProviderSpecialty" if "ProviderSpecialty" in df.columns else (
                    "provider_specialty" if "provider_specialty" in df.columns else None
                )
                if spec_col:
                    self.specialty_avg_map = (
                        df.groupby(spec_col)[amt_col].mean().dropna().to_dict()
                    )
                    if proc_col in df.columns:
                        for spec, group in df.groupby(spec_col):
                            if group[proc_col].empty:
                                continue
                            top_code = group[proc_col].mode().iloc[0]
                            code_avg = float(
                                self.procedure_avg_map.get(
                                    top_code, group[amt_col].mean()
                                )
                            )
                            self.specialty_to_code_map[str(spec).lower()] = (
                                str(top_code),
                                code_avg,
                            )

        self.procedure_codes_list = [str(x) for x in self.procedure_avg_map.keys()]
        self.default_procedure = (
            self.procedure_codes_list[0] if self.procedure_codes_list else "AA395"
        )

    def resolve_procedure_code(self, user_input: str) -> Tuple[str, str]:
        if not user_input or not user_input.strip():
            return self.default_procedure, "General Practice"

        query = user_input.strip().lower()

        for code in self.procedure_codes_list:
            if code.lower() in query:
                return code, "General Practice"

        tokens = [token for token in re.split(r"\W+", query) if len(token) >= 3]
        for spec_name, (top_code, _spec_avg) in self.specialty_to_code_map.items():
            for token in tokens:
                if token in spec_name or spec_name.startswith(token):
                    return top_code, spec_name.title()

        return self.default_procedure, "General Practice"

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _feature_pipeline(self, raw_claim: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        amount = self._safe_float(raw_claim.get("ClaimAmount"), 0.0)
        income = self._safe_float(raw_claim.get("PatientIncome"), 35000.0)
        age = self._safe_float(raw_claim.get("PatientAge"), 35.0)

        patient_id = str(raw_claim.get("PatientID", ""))
        procedure_code = str(
            raw_claim.get("ProcedureCode", self.default_procedure)
        )
        provider_id = str(raw_claim.get("ProviderID", ""))
        provider_specialty = str(
            raw_claim.get("ProviderSpecialty", "General Practice")
        )
        claim_type = str(raw_claim.get("ClaimType", "Outpatient"))

        baseline_peer_avg = float(
            self.procedure_avg_map.get(procedure_code, self.global_mean)
        )
        baseline_patient_avg = float(
            self.patient_avg_map.get(patient_id, baseline_peer_avg)
        )
        baseline_provider_avg = float(
            self.provider_avg_map.get(provider_id, baseline_peer_avg)
        )

        peer_deviation_pct = (
            ((amount - baseline_peer_avg) / baseline_peer_avg) * 100.0
            if baseline_peer_avg > 0
            else 0.0
        )
        patient_ratio = (
            amount / baseline_patient_avg if baseline_patient_avg > 0 else 1.0
        )
        provider_ratio = (
            amount / baseline_provider_avg if baseline_provider_avg > 0 else 1.0
        )

        dispersion_metrics = {
            "procedure_code": procedure_code,
            "provider_specialty": provider_specialty,
            "peer_avg": baseline_peer_avg,
            "peer_deviation_pct": peer_deviation_pct,
            "patient_avg": baseline_patient_avg,
            "patient_ratio": patient_ratio,
            "provider_ratio": provider_ratio,
        }

        processed = {
            "claim_amount": amount,
            "claim_type": claim_type,
            "provider_hospital": provider_id or "PRV-201",
            "geography": str(raw_claim.get("ProviderLocation", "Urban")),
            "diagnosis_code": str(raw_claim.get("DiagnosisCode", "D001")),
            "procedure_code": procedure_code,
            "claim_status": str(raw_claim.get("ClaimStatus", "Submitted")),
            "submission_method": str(raw_claim.get("ClaimSubmissionMethod", "Electronic")),
            "patient_age": age,
            "patient_income": income,
            "patient_gender": str(raw_claim.get("PatientGender", "Male")),
            "marital_status": str(raw_claim.get("PatientMaritalStatus", "Single")),
            "employment_status": str(raw_claim.get("PatientEmploymentStatus", "Employed")),
            "provider_specialty": provider_specialty,
            "customer_tenure": self._safe_float(raw_claim.get("customer_tenure"), 365.0),
            "patient_previous_claim_count": self._safe_float(
                raw_claim.get("patient_previous_claim_count"), 1.0
            ),
            "avg_historical_claim_amount": baseline_patient_avg,
            "num_claims_last_12m": self._safe_float(raw_claim.get("num_claims_last_12m"), 1.0),
            "previously_rejected_claims": self._safe_float(
                raw_claim.get("previously_rejected_claims"), 0.0
            ),
            "provider_claim_frequency": self._safe_float(
                raw_claim.get("provider_claim_frequency"), 10.0
            ),
            "provider_historical_avg_claim": baseline_provider_avg,
            "provider_historical_fraud_rate": self._safe_float(
                raw_claim.get("provider_historical_fraud_rate"), 0.02
            ),
            "deviation_from_peer_claims": (
                (amount - baseline_peer_avg) / baseline_peer_avg
                if baseline_peer_avg > 0
                else 0.0
            ),
            "claim_amount_vs_patient_average": patient_ratio,
            "claim_amount_vs_provider_average": provider_ratio,
            "log_claim_amount": np.log1p(max(0.0, amount)),
        }

        df = pd.DataFrame([processed])
        for col in self.required_features:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.required_features], dispersion_metrics

    def score_claim(self, raw_claim: Dict[str, Any]) -> Dict[str, Any]:
        with Spinner("Risk engine: scoring claim and calculating SHAP..."):
            df_features, dispersion_metrics = self._feature_pipeline(raw_claim)
            X_trans = self.preprocessor.transform(df_features)
            fraud_probability = float(self.model.predict_proba(X_trans)[0][1])
            prediction = int(fraud_probability >= self.optimal_threshold)

            explanation = "SHAP explainer unavailable."
            if self.explainer is not None:
                try:
                    shap_values = self.explainer.shap_values(X_trans)
                    if isinstance(shap_values, list):
                        sv = np.asarray(shap_values[1])[0]
                    else:
                        arr = np.asarray(shap_values)
                        sv = arr[0] if arr.ndim > 1 else arr

                    impacts = sorted(
                        zip(self.feature_names, sv),
                        key=lambda pair: abs(float(pair[1])),
                        reverse=True,
                    )
                    lines = ["Key Model Feature Impact (SHAP Analysis):"]
                    for feature, impact in impacts[:5]:
                        impact = float(impact)
                        direction = "pushed RISK UP" if impact > 0 else "pulled RISK DOWN"
                        lines.append(f"- {feature}: {direction} by {abs(impact):.3f}")
                    explanation = "\n".join(lines)
                except Exception as exc:
                    explanation = f"Risk explanation notice: {exc}"

            return {
                "risk_level": "HIGH_RISK" if prediction else "LOW_RISK",
                "risk_score": round(fraud_probability, 6),
                "decision_cutoff": round(self.optimal_threshold, 6),
                "triage_status": (
                    "HIGH RISK (SUSPICIOUS)" if prediction else "LOW RISK (NORMAL)"
                ),
                "risk_explanation": explanation,
                "dispersion_metrics": dispersion_metrics,
                "claim": raw_claim,
            }


class PolicyRAGEngine:
    """Hybrid Weaviate retrieval, cross-encoder reranking and grounded generation."""

    def __init__(self, use_local_llm: bool = True):
        self.use_local_llm = use_local_llm
        with Spinner("Loading policy embedding model..."):
            self.embedder = SentenceTransformer(
                os.getenv(
                    "POLICY_EMBED_MODEL",
                    "sentence-transformers/all-MiniLM-L6-v2",
                )
            )
        with Spinner("Loading policy reranker model..."):
            self.reranker = CrossEncoder(
                os.getenv(
                    "POLICY_RERANK_MODEL",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                )
            )

        self.llm_generator = None
        if self.use_local_llm:
            from transformers import pipeline

            with Spinner("Loading grounded policy generator..."):
                self.llm_generator = pipeline(
                    "text-generation",
                    model=os.getenv(
                        "POLICY_GENERATOR_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
                    ),
                    device_map="auto",
                )
                if hasattr(self.llm_generator.model, "config"):
                    self.llm_generator.model.config.max_length = None

    def retrieve_policy(self, question: str, claim_type: str = "Outpatient") -> List[Dict[str, Any]]:
        query_vector = self.embedder.encode(question).tolist()
        candidate_docs: List[Dict[str, Any]] = []

        try:
            with weaviate.connect_to_local() as client:
                kb = client.collections.use("InsuranceKnowledge")
                meta_filter = None
                if claim_type in {"Inpatient", "Outpatient"}:
                    try:
                        from weaviate.classes.query import Filter, MetadataQuery

                        meta_filter = Filter.by_property("category").like(
                            f"*{claim_type}*"
                        )
                    except Exception:
                        meta_filter = None
                else:
                    from weaviate.classes.query import MetadataQuery

                from weaviate.classes.query import MetadataQuery

                response = kb.query.hybrid(
                    query=question,
                    vector=query_vector,
                    alpha=0.5,
                    filters=meta_filter,
                    limit=10,
                    return_metadata=MetadataQuery(score=True),
                )

                if not response.objects and meta_filter is not None:
                    response = kb.query.hybrid(
                        query=question,
                        vector=query_vector,
                        alpha=0.5,
                        limit=10,
                        return_metadata=MetadataQuery(score=True),
                    )

                for obj in response.objects:
                    raw_content = str(obj.properties.get("content", "")).strip()
                    snippet = " ".join(raw_content.split())
                    if len(snippet) > 220:
                        snippet = snippet[:220] + "..."
                    candidate_docs.append(
                        {
                            "source_file": obj.properties.get("source_file", "Unknown PDF"),
                            "category": obj.properties.get("category", "Policy Document"),
                            "full_text": raw_content,
                            "snippet": snippet,
                            "hybrid_score": float(obj.metadata.score) if obj.metadata and obj.metadata.score is not None else 0.0,
                        }
                    )
        except Exception as exc:
            logger.exception("Weaviate policy retrieval failed")
            return [{
                "source_file": "RAG_ERROR",
                "category": "System",
                "full_text": f"Weaviate retrieval failed: {exc}",
                "snippet": f"Weaviate retrieval failed: {exc}",
                "hybrid_score": 0.0,
                "rerank_score": 0.0,
            }]

        if not candidate_docs:
            return []

        pairs = [[question, doc["full_text"]] for doc in candidate_docs]
        cross_scores = self.reranker.predict(pairs)
        for doc, raw_score in zip(candidate_docs, cross_scores):
            score = float(raw_score)
            doc["rerank_score"] = float(1.0 / (1.0 + np.exp(-score))) * 100.0

        candidate_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidate_docs[:2]

    def answer_policy(
        self,
        question: str,
        claim_type: str = "Outpatient",
        procedure_code: str = "AA395",
    ) -> Dict[str, Any]:
        docs = self.retrieve_policy(question, claim_type=claim_type)
        usable_docs = [d for d in docs if d.get("category") != "System"]

        if not usable_docs:
            return {
                "question": question,
                "answer": "No matching policy context was found.",
                "retrieved_docs": [],
            }

        context = "\n\n".join(
            f"[{doc['source_file']}]\n{doc['full_text']}" for doc in usable_docs
        )

        answer = f"Retrieved policy context:\n{context}"
        if self.use_local_llm and self.llm_generator is not None:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict health insurance policy officer. "
                        "Answer using ONLY the provided policy excerpts. "
                        "Do not invent exclusions, limits, or requirements. "
                        "When the excerpts do not answer a detail, explicitly say that the "
                        "available policy excerpts do not specify it."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Claim type: {claim_type}\n"
                        f"Procedure code: {procedure_code}\n"
                        f"Policy excerpts:\n{context}\n\n"
                        f"Question: {question}\n\nAnswer:"
                    ),
                },
            ]
            generated = self.llm_generator(
                messages,
                max_new_tokens=256,
                temperature=0.1,
                clean_up_tokenization_spaces=False,
            )
            generated_text = generated[0].get("generated_text", "")
            if isinstance(generated_text, list) and generated_text:
                generated_text = generated_text[-1].get("content", "")
            answer = str(generated_text).strip() or answer

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": usable_docs,
        }


_risk_engine: InsuranceRiskEngine | None = None
_policy_engine: PolicyRAGEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_risk_engine() -> InsuranceRiskEngine:
    global _risk_engine
    if _risk_engine is None:
        with _ENGINE_LOCK:
            if _risk_engine is None:
                _risk_engine = InsuranceRiskEngine()
    return _risk_engine


def get_policy_engine() -> PolicyRAGEngine:
    global _policy_engine
    if _policy_engine is None:
        with _ENGINE_LOCK:
            if _policy_engine is None:
                _policy_engine = PolicyRAGEngine(
                    use_local_llm=os.getenv("USE_LOCAL_POLICY_LLM", "1") == "1"
                )
    return _policy_engine


# Backward-compatible helper functions for local scripts/tests.
def score_claim_capability(raw_claim: Dict[str, Any]) -> Dict[str, Any]:
    return get_risk_engine().score_claim(raw_claim)


def lookup_policy_capability(
    question: str,
    claim_type: str = "Outpatient",
    procedure_code: str = "AA395",
) -> Dict[str, Any]:
    return get_policy_engine().answer_policy(question, claim_type, procedure_code)


if __name__ == "__main__":
    print("Capability layer loaded. Use insurance_mcp_server.py to expose MCP tools.")
