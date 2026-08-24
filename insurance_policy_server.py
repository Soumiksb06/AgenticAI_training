import logging
import os
from typing import Any, Dict, Literal

from fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("insurance_policy_server")

mcp = FastMCP("Insurance Policy MCP")

_SYSTEM = None


class PolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_type: Literal["Inpatient", "Outpatient", "Emergency", "Other"] = "Outpatient"
    procedure_code: str = "AA395"
    provider_specialty: str = "General Practice"
    question: str = Field(default="What are the policy rules and limits for this claim?")
    claim_amount: float = Field(default=5000.0, gt=0)
    patient_age: int = Field(default=45, ge=18, le=100)
    patient_income: float = Field(default=35000.0, gt=0)
    patient_id: str = "P-DEFAULT"
    provider_id: str = "PRV-201"
    claim_status: str = "Submitted"
    diagnosis_code: str = "D001"
    provider_location: str = "Urban"
    claim_submission_method: str = "Electronic"


def _get_system():
    global _SYSTEM
    if _SYSTEM is None:
        from insurance_multi_agent_chunking_indexing_copy_2 import InsuranceAgentSystem

        logger.info("Initializing policy retrieval pipeline")
        _SYSTEM = InsuranceAgentSystem(use_local_llm=True)
    return _SYSTEM


@mcp.tool()
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "server": "Insurance Policy MCP",
        "service": "policy-guidance",
        "environment": os.environ.get("ENVIRONMENT", "development"),
    }


@mcp.tool()
def lookup_policy(request: PolicyRequest) -> Dict[str, Any]:
    """Return grounded policy guidance for the claim using the original retrieval pipeline."""
    system = _get_system()
    raw_claim = {
        "ClaimType": request.claim_type,
        "ClaimAmount": request.claim_amount,
        "ProcedureCode": request.procedure_code,
        "ProviderSpecialty": request.provider_specialty,
        "PatientIncome": request.patient_income,
        "PatientAge": request.patient_age,
        "PatientID": request.patient_id,
        "ProviderID": request.provider_id,
        "ClaimStatus": request.claim_status,
        "DiagnosisCode": request.diagnosis_code,
        "ProviderLocation": request.provider_location,
        "ClaimSubmissionMethod": request.claim_submission_method,
    }

    state = {
        "raw_claim": raw_claim,
        "user_query": request.question,
        "target_route": "policy_only",
        "dispersion_metrics": {},
        "ml_probability": 0.0,
        "ml_prediction": 0,
        "triage_status": "",
        "risk_explanation": "",
        "retrieved_docs": [],
        "rag_generated_answer": "",
        "needs_clarification": False,
        "final_report": "",
    }

    result = system.policy_agent(state)
    return {
        "claim_type": request.claim_type,
        "procedure_code": request.procedure_code,
        "question": request.question,
        "answer": result.get("rag_generated_answer", ""),
        "retrieved_docs": result.get("retrieved_docs", []),
    }


if __name__ == "__main__":
    port = int(os.getenv("POLICY_MCP_PORT", "8013"))
    mcp.run(transport="http", host="0.0.0.0", port=port)
