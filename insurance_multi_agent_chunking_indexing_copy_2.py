import os
import sys
import time
import warnings
import threading
import joblib
import json
import logging
import re
import numpy as np
import pandas as pd
import weaviate
from weaviate.classes.query import MetadataQuery, Filter
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import TypedDict, Dict, Any, List, Literal
from langgraph.graph import StateGraph, START, END
from transformers import pipeline, logging as transformers_logging

# Suppress non-critical warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)


# =====================================================================
# CLI SPINNER FOR BACKGROUND TASKS
# =====================================================================
class Spinner:
    """A lightweight animated CLI spinner for long-running background tasks."""
    def __init__(self, message="Processing..."):
        self.message = message
        self.stop_running = False
        self.thread = None

    def __enter__(self):
        self.stop_running = False
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_running = True
        if self.thread:
            self.thread.join()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _spin(self):
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not self.stop_running:
            sys.stdout.write(f"\r{chars[i % len(chars)]} {self.message}")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1


# =====================================================================
# 1. SHARED GRAPH STATE
# =====================================================================
class ClaimState(TypedDict):
    raw_claim: Dict[str, Any]
    user_query: str
    target_route: str
    
    # Feature Ratios & Dispersion
    dispersion_metrics: Dict[str, Any]
    
    # Agent Outputs
    ml_probability: float
    ml_prediction: int
    triage_status: str
    risk_explanation: str
    retrieved_docs: List[Dict[str, Any]]
    rag_generated_answer: str
    needs_clarification: bool
    final_report: str


# =====================================================================
# 2. MULTI-AGENT SYSTEM
# =====================================================================
class InsuranceAgentSystem:
    def __init__(self, use_local_llm: bool = True):
        # 1. Load Model Artifacts
        with Spinner("Loading ML Model Artifacts from output/fraud_detection_model.pkl..."):
            artifact_path = "output/fraud_detection_model.pkl"
            if not os.path.exists(artifact_path):
                raise FileNotFoundError(f"Trained model missing at '{artifact_path}'. Run your training script first.")
                
            artifacts = joblib.load(artifact_path)
            self.model = artifacts["model"]
            self.preprocessor = artifacts["preprocessor"]
            self.optimal_threshold = artifacts["optimal_threshold"]
            self.explainer = artifacts["explainer"]
            self.feature_names = artifacts["feature_names"]
            self.required_features = artifacts["required_features"]

        # 2. Load Dataset Baseline Averages
        with Spinner("Calculating Dataset Baselines from dataset..."):
            self._load_dataset_statistics()

        # 3. Load Bi-Encoder Embedding Model
        with Spinner("Loading Bi-Encoder Embedding Model ('all-MiniLM-L6-v2')..."):
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # 4. Load Cross-Encoder Re-Ranker Model
        with Spinner("Loading Cross-Encoder Re-Ranker ('cross-encoder/ms-marco-MiniLM-L-6-v2')..."):
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 5. Load Generator LLM
        self.use_local_llm = use_local_llm
        if self.use_local_llm:
            with Spinner("Loading Local Generator LLM ('Qwen/Qwen2.5-1.5B-Instruct')..."):
                self.llm_generator = pipeline(
                    "text-generation",
                    model="Qwen/Qwen2.5-1.5B-Instruct",
                    device_map="auto"
                )
                if hasattr(self.llm_generator.model, "config"):
                    self.llm_generator.model.config.max_length = None

        self.graph = self._build_graph()
        print(f"[✓] Model Loaded | Decision Cutoff Threshold: {self.optimal_threshold:.2%}")
        print(f"[✓] Loaded Dataset Statistics: {len(self.procedure_avg_map)} Unique Procedures, {len(self.specialty_avg_map)} Specialties, {len(self.patient_avg_map)} Patients, {len(self.provider_avg_map)} Providers (Global Mean: ${self.global_mean:,.2f}).")
        print("[✓] Multi-Stage Production RAG & Multi-Agent System Ready.\n")

    def _load_dataset_statistics(self):
        """Reads raw dataset Excel or processed CSV to build exact group averages."""
        self.patient_avg_map = {}
        self.procedure_avg_map = {}
        self.provider_avg_map = {}
        self.specialty_avg_map = {}
        self.specialty_to_code_map = {}
        self.global_mean = 5014.20

        raw_excel = "Health Insurance Fraud Claims.xlsx"
        processed_csv = "output/processed_claim_features.csv"

        df = None
        if os.path.exists(raw_excel):
            try:
                df = pd.read_excel(raw_excel)
            except Exception:
                df = None

        if df is None and os.path.exists(processed_csv):
            try:
                df = pd.read_csv(processed_csv)
            except Exception:
                df = None

        if df is not None:
            amt_col = "ClaimAmount" if "ClaimAmount" in df.columns else "claim_amount"
            if amt_col in df.columns:
                self.global_mean = float(df[amt_col].mean())
                
                patient_col = "PatientID" if "PatientID" in df.columns else "patient_id"
                if patient_col in df.columns:
                    self.patient_avg_map = df.groupby(patient_col)[amt_col].mean().to_dict()

                proc_col = "ProcedureCode" if "ProcedureCode" in df.columns else "procedure_code"
                if proc_col in df.columns:
                    self.procedure_avg_map = df.groupby(proc_col)[amt_col].mean().to_dict()

                prov_col = "ProviderID" if "ProviderID" in df.columns else ("provider_hospital" if "provider_hospital" in df.columns else None)
                if prov_col and prov_col in df.columns:
                    self.provider_avg_map = df.groupby(prov_col)[amt_col].mean().to_dict()

                spec_col = "ProviderSpecialty" if "ProviderSpecialty" in df.columns else ("provider_specialty" if "provider_specialty" in df.columns else None)
                if spec_col and spec_col in df.columns:
                    self.specialty_avg_map = df.groupby(spec_col)[amt_col].mean().to_dict()
                    if proc_col in df.columns:
                        for spec, group in df.groupby(spec_col):
                            top_code = group[proc_col].mode()[0] if not group[proc_col].empty else "AA395"
                            code_avg = float(self.procedure_avg_map.get(top_code, group[amt_col].mean()))
                            self.specialty_to_code_map[str(spec).lower()] = (top_code, code_avg)

        self.procedure_codes_list = list(self.procedure_avg_map.keys())
        self.default_procedure = self.procedure_codes_list[0] if self.procedure_codes_list else "AA395"

    def resolve_procedure_code(self, user_input: str) -> tuple[str, str]:
        """
        Tokenized Fuzzy Matcher: Tokenizes prompt input to reliably match keywords
        like 'pedia', 'pediatric', 'neuro', 'cardio' to their dataset Specialty & ProcedureCode.
        """
        if not user_input or not user_input.strip():
            return self.default_procedure, "General Practice"

        query = user_input.strip().lower()

        # 1. Exact or Partial Match on ProcedureCode (e.g., 'AA395')
        for code in self.procedure_codes_list:
            if str(code).lower() in query:
                return str(code), "General Practice"

        # 2. Tokenized Prefix & Keyword Match for ProviderSpecialty (e.g. 'pedia' -> 'Pediatrics')
        tokens = [t for t in re.split(r'\W+', query) if len(t) >= 3]
        for spec_name, (top_code, spec_avg) in self.specialty_to_code_map.items():
            spec_lower = spec_name.lower()
            for token in tokens:
                if token in spec_lower or spec_lower.startswith(token):
                    return top_code, spec_name.title()

        return self.default_procedure, "General Practice"

    def _feature_pipeline(self, raw_claim: dict) -> tuple[pd.DataFrame, dict]:
        """Transforms raw claim parameters into exact feature vector schema expected by model."""
        amount = float(raw_claim.get("ClaimAmount", 0.0))
        income = float(raw_claim.get("PatientIncome", 35000.0))
        age = float(raw_claim.get("PatientAge", 35.0))
        
        patient_id = str(raw_claim.get("PatientID", ""))
        procedure_code = str(raw_claim.get("ProcedureCode", self.default_procedure))
        provider_id = str(raw_claim.get("ProviderID", ""))
        provider_specialty = str(raw_claim.get("ProviderSpecialty", "General Practice"))
        claim_type = str(raw_claim.get("ClaimType", "Outpatient"))

        baseline_peer_avg = self.procedure_avg_map.get(procedure_code, self.global_mean)
        baseline_patient_avg = self.patient_avg_map.get(patient_id, baseline_peer_avg)
        baseline_provider_avg = self.provider_avg_map.get(provider_id, baseline_peer_avg)

        peer_deviation_pct = ((amount - baseline_peer_avg) / baseline_peer_avg) * 100.0 if baseline_peer_avg > 0 else 0.0
        patient_ratio = amount / baseline_patient_avg if baseline_patient_avg > 0 else 1.0
        provider_ratio = amount / baseline_provider_avg if baseline_provider_avg > 0 else 1.0

        dispersion_metrics = {
            "procedure_code": procedure_code,
            "provider_specialty": provider_specialty,
            "peer_avg": baseline_peer_avg,
            "peer_deviation_pct": peer_deviation_pct,
            "patient_avg": baseline_patient_avg,
            "patient_ratio": patient_ratio,
            "provider_ratio": provider_ratio
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
            "customer_tenure": float(raw_claim.get("customer_tenure", 365.0)),
            "patient_previous_claim_count": float(raw_claim.get("patient_previous_claim_count", 1.0)),
            "avg_historical_claim_amount": baseline_patient_avg,
            "num_claims_last_12m": float(raw_claim.get("num_claims_last_12m", 1.0)),
            "previously_rejected_claims": float(raw_claim.get("previously_rejected_claims", 0.0)),
            "provider_claim_frequency": float(raw_claim.get("provider_claim_frequency", 10.0)),
            "provider_historical_avg_claim": baseline_provider_avg,
            "provider_historical_fraud_rate": float(raw_claim.get("provider_historical_fraud_rate", 0.02)),
            "deviation_from_peer_claims": (amount - baseline_peer_avg) / baseline_peer_avg if baseline_peer_avg > 0 else 0.0,
            "claim_amount_vs_patient_average": patient_ratio,
            "claim_amount_vs_provider_average": provider_ratio,
            "log_claim_amount": np.log1p(max(0.0, amount))
        }
        
        df = pd.DataFrame([processed])
        for col in self.required_features:
            if col not in df.columns:
                df[col] = 0.0
                
        return df[self.required_features], dispersion_metrics

    # =================================================================
    # ROUTER NODE
    # =================================================================
    def intent_router_node(self, state: ClaimState) -> Dict[str, Any]:
        raw_claim = state.get("raw_claim", {})
        try:
            claim_amount = float(raw_claim.get("ClaimAmount", 0.0))
        except (ValueError, TypeError):
            claim_amount = 0.0

        if not raw_claim or claim_amount <= 0:
            return {"target_route": "policy_only"}
            
        return {"target_route": "full_eval"}

    # =================================================================
    # AGENT 1: CLAIMS TRIAGE AGENT
    # =================================================================
    def claims_triage_agent(self, state: ClaimState) -> Dict[str, Any]:
        with Spinner("Agent 1 (Triage): Assessing ML Fraud Risk & Dataset Dispersion..."):
            df_features, dispersion_metrics = self._feature_pipeline(state["raw_claim"])
            X_trans = self.preprocessor.transform(df_features)
            
            fraud_prob = float(self.model.predict_proba(X_trans)[0][1])
            is_fraud = int(fraud_prob >= self.optimal_threshold)
            triage_status = "🚨 HIGH RISK (SUSPICIOUS)" if is_fraud else "✅ LOW RISK (NORMAL)"
            
            return {
                "ml_probability": fraud_prob,
                "ml_prediction": is_fraud,
                "triage_status": triage_status,
                "dispersion_metrics": dispersion_metrics
            }

    # =================================================================
    # AGENT 2: RISK ANALYSIS AGENT (SHAP EXPLAINABILITY)
    # =================================================================
    def risk_analysis_agent(self, state: ClaimState) -> Dict[str, Any]:
        with Spinner("Agent 2 (Risk Analysis): Calculating SHAP Feature Contributions..."):
            df_features, _ = self._feature_pipeline(state["raw_claim"])
            X_trans = self.preprocessor.transform(df_features)
            
            try:
                if self.explainer is not None:
                    shap_values = self.explainer.shap_values(X_trans)
                    sv = shap_values[1][0] if isinstance(shap_values, list) else (
                        shap_values[0] if shap_values.ndim == 2 else shap_values
                    )
                    feature_impacts = sorted(list(zip(self.feature_names, sv)), key=lambda x: abs(x[1]), reverse=True)
                    
                    explanation = "Key Model Feature Impact (SHAP Analysis):\n"
                    for feat, impact in feature_impacts[:3]:
                        direction = "pushed RISK UP" if impact > 0 else "pulled RISK DOWN"
                        explanation += f"  - {feat}: {direction} by {abs(impact):.3f}\n"
                else:
                    explanation = "SHAP explainer unavailable."
            except Exception as e:
                explanation = f"Risk explanation notice: {str(e)}"

            return {"risk_explanation": explanation}

    # =================================================================
    # AGENT 3: POLICY AGENT (STRICT GROUNDED POLICY RAG)
    # =================================================================
    def policy_agent(self, state: ClaimState) -> Dict[str, Any]:
        user_query = state.get("user_query", "").strip()
        raw_claim = state.get("raw_claim", {})
        claim_type = raw_claim.get("ClaimType", "Outpatient")
        procedure_code = raw_claim.get("ProcedureCode", self.default_procedure)
        
        if not user_query:
            user_query = f"What are the policy coverage rules and limits for a {claim_type} claim regarding {procedure_code}?"
            
        candidate_docs = []
        final_reranked_docs = []
        context_str = ""

        with Spinner("Agent 3 (Policy RAG): Executing Weaviate Hybrid Search (Dense + BM25)..."):
            query_vector = self.embedder.encode(user_query).tolist()
            
            try:
                with weaviate.connect_to_local() as client:
                    kb = client.collections.use("InsuranceKnowledge")
                    
                    meta_filter = None
                    if claim_type in ["Inpatient", "Outpatient"]:
                        try:
                            meta_filter = Filter.by_property("category").like(f"*{claim_type}*")
                        except Exception:
                            meta_filter = None

                    response = kb.query.hybrid(
                        query=user_query,
                        vector=query_vector,
                        alpha=0.5,
                        filters=meta_filter,
                        limit=10,
                        return_metadata=MetadataQuery(score=True)
                    )
                    
                    if not response.objects and meta_filter is not None:
                        response = kb.query.hybrid(
                            query=user_query,
                            vector=query_vector,
                            alpha=0.5,
                            limit=10,
                            return_metadata=MetadataQuery(score=True)
                        )

                    for obj in response.objects:
                        raw_content = obj.properties.get('content', '').strip()
                        snippet = " ".join(raw_content.split())
                        if len(snippet) > 220:
                            snippet = snippet[:220] + "..."

                        candidate_docs.append({
                            "source_file": obj.properties.get('source_file', 'Unknown PDF'),
                            "category": obj.properties.get('category', 'Policy Document'),
                            "full_text": raw_content,
                            "snippet": snippet,
                            "hybrid_score": obj.metadata.score if obj.metadata else 0.0
                        })
            except Exception as e:
                context_str = f"Weaviate Hybrid Search Notice: {str(e)}"

        if candidate_docs:
            with Spinner("Agent 3 (Policy RAG): Re-Ranking Top Candidates with Cross-Encoder..."):
                query_doc_pairs = [[user_query, doc["full_text"]] for doc in candidate_docs]
                cross_scores = self.reranker.predict(query_doc_pairs)
                
                for doc, score in zip(candidate_docs, cross_scores):
                    norm_score = float(1.0 / (1.0 + np.exp(-score))) * 100.0
                    doc["rerank_score"] = norm_score
                
                candidate_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
                final_reranked_docs = candidate_docs[:2]
                
                for doc in final_reranked_docs:
                    context_str += f"[{doc['source_file']}]:\n{doc['full_text']}\n\n"

        if self.use_local_llm and final_reranked_docs:
            with Spinner("Agent 3 (Policy RAG): Synthesizing Grounded LLM Response..."):
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict health insurance policy officer.\n"
                            "STRICT RULE: Summarize coverage limits and rules using ONLY the provided policy excerpts.\n"
                            "Do NOT ask generic or ungrounded questions about primary/secondary coverage or documents.\n"
                            "If excerpts lack specific sub-limits for the user's procedure, state general limits ($100,000 Inpatient / $10,000 Outpatient)."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Policy Excerpts:\n{context_str}\nQuestion: {user_query}\n\nAnswer:"
                    }
                ]
                
                output = self.llm_generator(
                    messages,
                    max_new_tokens=256,
                    temperature=0.1,
                    clean_up_tokenization_spaces=False
                )
                generated_answer = output[0]["generated_text"][-1]["content"].strip()
        else:
            generated_answer = f"Retrieved Context:\n{context_str}" if context_str else "No matching policy context found."

        return {
            "retrieved_docs": final_reranked_docs,
            "rag_generated_answer": generated_answer,
            "needs_clarification": False
        }

    # =================================================================
    # FORMATTING NODE
    # =================================================================
    def formatting_node(self, state: ClaimState) -> Dict[str, Any]:
        claim_id = state.get("raw_claim", {}).get("ClaimID", "").strip()
        disp = state.get("dispersion_metrics", {})
        raw = state.get("raw_claim", {})
        
        report = f"\n=======================================================================\n"
        if claim_id:
            report += f"🕵️  MULTI-AGENT INVESTIGATION REPORT | Claim ID: {claim_id}\n"
        else:
            report += f"🕵️  MULTI-AGENT INVESTIGATION REPORT\n"
        report += f"=======================================================================\n\n"
        
        if state.get("triage_status"):
            report += f"[AGENT 1: CLAIMS TRIAGE (Trained ML Model)]\n"
            report += f"  - Status: {state['triage_status']}\n"
            report += f"  - Fraud Score: {state['ml_probability']:.2%} (Decision Cutoff: {self.optimal_threshold:.2%})\n"
            
            if disp:
                claim_amt = raw.get("ClaimAmount", 0.0)
                proc_code = disp.get("procedure_code", "Procedure")
                spec_name = disp.get("provider_specialty", "General Practice")
                report += f"  - Dataset Statistical Dispersion Ratios:\n"
                report += f"    • Claim Amount (${claim_amt:,.2f}) vs. Dataset Peer Mean for Code '{proc_code}' ({spec_name}) (${disp['peer_avg']:,.2f}): "
                report += f"{disp['peer_deviation_pct']:+.2f}% deviation ({claim_amt/disp['peer_avg']:.2f}x peer mean)\n"
                report += f"    • Claim Amount vs. Dataset Patient Historical Baseline (${disp['patient_avg']:,.2f}): "
                report += f"{disp['patient_ratio']:.2f}x patient history\n\n"
            
        if state.get("risk_explanation"):
            report += f"[AGENT 2: RISK ANALYSIS (Model Explainability)]\n"
            report += f"{state['risk_explanation']}\n"
            
        if state.get("rag_generated_answer"):
            report += f"[AGENT 3: POLICY AGENT (Two-Stage Hybrid RAG Synthesis)]\n"
            report += f"{state['rag_generated_answer']}\n\n"
            
            if state.get("retrieved_docs"):
                report += f"📑 Referenced Document Excerpts & Re-Ranker Scores:\n"
                for idx, doc in enumerate(state["retrieved_docs"], 1):
                    report += f"  {idx}. Source: '{doc['source_file']}'\n"
                    report += f"     Category: {doc['category']} | Re-Ranker Match Score: {doc['rerank_score']:.2f}%\n"
                    report += f"     Exact Text Snippet: \"{doc['snippet']}\"\n\n"
            
        report += f"=======================================================================\n"
        return {"final_report": report}

    # =================================================================
    # DYNAMIC ROUTING
    # =================================================================
    def route_by_intent(self, state: ClaimState) -> Literal["policy_agent", "claims_triage_agent"]:
        if state.get("target_route") == "policy_only":
            return "policy_agent"
        return "claims_triage_agent"

    # =================================================================
    # LANGGRAPH CONSTRUCT
    # =================================================================
    def _build_graph(self):
        workflow = StateGraph(ClaimState)
        
        workflow.add_node("intent_router", self.intent_router_node)
        workflow.add_node("claims_triage_agent", self.claims_triage_agent)
        workflow.add_node("risk_analysis_agent", self.risk_analysis_agent)
        workflow.add_node("policy_agent", self.policy_agent)
        workflow.add_node("formatter", self.formatting_node)
        
        workflow.add_edge(START, "intent_router")
        
        workflow.add_conditional_edges(
            "intent_router",
            self.route_by_intent,
            {
                "policy_agent": "policy_agent",
                "claims_triage_agent": "claims_triage_agent"
            }
        )
        
        workflow.add_edge("claims_triage_agent", "risk_analysis_agent")
        workflow.add_edge("risk_analysis_agent", "policy_agent")
        workflow.add_edge("policy_agent", "formatter")
        workflow.add_edge("formatter", END)
        
        return workflow.compile()

    def process_claim(self, raw_claim: dict = None, user_query: str = "") -> dict:
        initial_state = {
            "raw_claim": raw_claim or {},
            "user_query": user_query,
            "target_route": "",
            "dispersion_metrics": {},
            "ml_probability": 0.0,
            "ml_prediction": 0,
            "triage_status": "",
            "risk_explanation": "",
            "retrieved_docs": [],
            "rag_generated_answer": "",
            "needs_clarification": False,
            "final_report": ""
        }
        
        final_state = self.graph.invoke(initial_state)
        print(final_state["final_report"])
        return final_state


# =====================================================================
# 3. INTERACTIVE CLI RUNNER WITH SLOT-FILLING VALIDATION
# =====================================================================
if __name__ == "__main__":
    system = InsuranceAgentSystem(use_local_llm=True)

    print("=" * 75)
    print("🚀 SINGLE-PROMPT MULTI-AGENT CLAIM INVESTIGATOR")
    print("=" * 75)

    def extract_and_validate_claim(prompt_text: str) -> dict:
        """
        Slot-Filling Extractor:
        1. Extracts ClaimAmount and Procedure/Specialty using Regex and Fuzzy Matching.
        2. Checks for missing parameters (PatientIncome, PatientAge, ClaimType).
        3. Prompts user for missing fields BEFORE assuming defaults.
        """
        data = {}
        
        # Extract Claim Amount
        amt_match = re.search(r'(?:[\$₹]\s*|\b)(\d[\d,]*)(?:\s*k\b|\b)', prompt_text, re.IGNORECASE)
        if amt_match:
            try:
                raw_val = amt_match.group(1).replace(',', '')
                val = float(raw_val)
                if 'k' in amt_match.group(0).lower() and val < 1000:
                    val *= 1000
                if val > 0:
                    data["ClaimAmount"] = val
            except ValueError:
                pass

        # If no claim amount is found, treat strictly as a Policy/RAG Query
        if data.get("ClaimAmount", 0) <= 0:
            return {}

        # Resolve Procedure Code and Specialty
        code, spec = system.resolve_procedure_code(prompt_text)
        data["ProcedureCode"] = code
        data["ProviderSpecialty"] = spec

        # Extract Income if explicitly stated
        inc_match = re.search(r'(?:income|salary|earning)[^\d]*([\$₹]?\s*\d[\d,]*)', prompt_text, re.IGNORECASE)
        if inc_match:
            try:
                data["PatientIncome"] = float(inc_match.group(1).replace('$', '').replace(',', ''))
            except ValueError:
                pass

        # Extract Age if explicitly stated
        age_match = re.search(r'(\d{1,2})\s*(?:years old|yo|age)', prompt_text, re.IGNORECASE)
        if age_match:
            try:
                data["PatientAge"] = float(age_match.group(1))
            except ValueError:
                pass

        # Extract Claim Type
        if "inpatient" in prompt_text.lower():
            data["ClaimType"] = "Inpatient"
        elif "outpatient" in prompt_text.lower():
            data["ClaimType"] = "Outpatient"

        # Check for missing parameters
        missing_slots = []
        if "PatientIncome" not in data:
            missing_slots.append("Patient Income")
        if "PatientAge" not in data:
            missing_slots.append("Patient Age")
        if "ClaimType" not in data:
            missing_slots.append("Claim Type (Inpatient/Outpatient)")

        # Interactive Slot-Filling Prompt
        if missing_slots:
            print(f"\n⚠️ Missing Claim Input(s) Detected: {', '.join(missing_slots)}")
            print("Provide missing details below (or press ENTER to use dataset baseline defaults):")
            
            if "Patient Income" in missing_slots:
                inc_in = input("  • Patient Annual Income [$35,000.00]: ").strip()
                if inc_in:
                    try:
                        data["PatientIncome"] = float(re.sub(r'[^\d.]', '', inc_in))
                    except ValueError:
                        data["PatientIncome"] = 35000.0
                else:
                    data["PatientIncome"] = 35000.0

            if "Patient Age" in missing_slots:
                age_in = input("  • Patient Age [45]: ").strip()
                if age_in:
                    try:
                        data["PatientAge"] = float(re.sub(r'[^\d.]', '', age_in))
                    except ValueError:
                        data["PatientAge"] = 45.0
                else:
                    data["PatientAge"] = 45.0

            if "Claim Type (Inpatient/Outpatient)" in missing_slots:
                type_in = input("  • Claim Type (Inpatient/Outpatient) [Outpatient]: ").strip()
                data["ClaimType"] = type_in.capitalize() if type_in else "Outpatient"

        return data

    while True:
        user_input = input("\n💬 Enter query or claim description: ").strip()
        if not user_input:
            continue

        # Extract and interactively validate claim slots
        claim_payload = extract_and_validate_claim(user_input)

        # Process through LangGraph system
        system.process_claim(raw_claim=claim_payload, user_query=user_input)

        cont = input("\nPerform another query? (y/n): ").strip().lower()
        if cont != 'y':
            print("Exiting Investigator. Goodbye!")
            break