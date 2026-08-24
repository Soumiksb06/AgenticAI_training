import logging
import os
from typing import Any, Dict, List, Literal

from fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("insurance_risk_server")

mcp = FastMCP("Insurance Risk MCP")

_SYSTEM = None


class ClaimInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(..., min_length=1)
    claim_amount: float = Field(..., gt=0)
    claim_type: Literal["Inpatient", "Outpatient", "Emergency", "Other"] = "Outpatient"
    procedure_code: str = "AA395"
    provider_specialty: str = "General Practice"
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

        logger.info("Initializing fraud model pipeline")
        _SYSTEM = InsuranceAgentSystem(use_local_llm=False)
    return _SYSTEM


@mcp.tool()
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "server": "Insurance Risk MCP",
        "service": "fraud-triage",
        "environment": os.environ.get("ENVIRONMENT", "development"),
    }


@mcp.tool()
def score_claim(claim: ClaimInput) -> Dict[str, Any]:
    """Triage a claim and return model-based probability and decision."""
    system = _get_system()
    raw_claim = claim.model_dump()

    # keep original project schema and naming aligned
    raw_claim["ClaimAmount"] = raw_claim.pop("claim_amount")
    raw_claim["ClaimType"] = raw_claim.pop("claim_type")
    raw_claim["ProcedureCode"] = raw_claim.pop("procedure_code")
    raw_claim["ProviderSpecialty"] = raw_claim.pop("provider_specialty")
    raw_claim["PatientAge"] = raw_claim.pop("patient_age")
    raw_claim["PatientIncome"] = raw_claim.pop("patient_income")
    raw_claim["PatientID"] = raw_claim.pop("patient_id")
    raw_claim["ProviderID"] = raw_claim.pop("provider_id")
    raw_claim["ClaimStatus"] = raw_claim.pop("claim_status")
    raw_claim["DiagnosisCode"] = raw_claim.pop("diagnosis_code")
    raw_claim["ProviderLocation"] = raw_claim.pop("provider_location")
    raw_claim["ClaimSubmissionMethod"] = raw_claim.pop("claim_submission_method")

    state = system.process_claim(raw_claim=raw_claim, user_query=f"Risk triage for claim {claim.claim_id}")

    probability = float(state.get("ml_probability", 0.0))
    risk_level = "HIGH_RISK" if state.get("ml_prediction", 0) else "LOW_RISK"
    return {
        "claim_id": claim.claim_id,
        "risk_level": risk_level,
        "risk_score": round(probability, 6),
        "decision_cutoff": round(system.optimal_threshold, 6),
        "triage_status": state.get("triage_status", ""),
        "risk_explanation": state.get("risk_explanation", ""),
        "final_report": state.get("final_report", ""),
    }


if __name__ == "__main__":
    port = int(os.getenv("RISK_MCP_PORT", "8011"))
    mcp.run(transport="http", host="0.0.0.0", port=port)
