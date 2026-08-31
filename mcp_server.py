"""FastMCP server exposing domain capabilities as tools.

Important architectural rule:
    MCP tools do NOT route between agents and do NOT call agents.
    They execute domain capabilities and return structured results.

Flow:
    Specialist Agent -> MCP Tool -> Capability Layer -> ML/RAG -> result
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from fastmcp import FastMCP

from backend_new import (
    lookup_policy_capability,
    score_claim_capability,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("insurance_mcp_server")

mcp = FastMCP("Insurance Domain Capabilities")


@mcp.tool()
def score_claim(
    claim_amount: float,
    patient_income: float = 35000.0,
    patient_age: int = 45,
    claim_type: str = "Outpatient",
    claim_id: str = "CLM-AUTO",
    procedure_code: str = "AA395",
    patient_id: str = "P-DEFAULT",
    provider_id: str = "PRV-201",
    provider_specialty: str = "General Practice",
    diagnosis_code: str = "D001",
    provider_location: str = "Urban",
    claim_status: str = "Submitted",
    claim_submission_method: str = "Electronic",
    previously_rejected_claims: float = 0.0,
    num_claims_last_12m: float = 1.0,
) -> Dict[str, Any]:
    """Score a health insurance claim for fraud risk and explain the model decision.

    The tool performs model inference, historical-baseline comparisons and SHAP
    explainability. It does not decide which agent should run next.
    """
    raw_claim = {
        "ClaimID": claim_id,
        "ClaimAmount": claim_amount,
        "PatientIncome": patient_income,
        "PatientAge": patient_age,
        "ClaimType": claim_type,
        "ProcedureCode": procedure_code,
        "PatientID": patient_id,
        "ProviderID": provider_id,
        "ProviderSpecialty": provider_specialty,
        "DiagnosisCode": diagnosis_code,
        "ProviderLocation": provider_location,
        "ClaimStatus": claim_status,
        "ClaimSubmissionMethod": claim_submission_method,
        "previously_rejected_claims": previously_rejected_claims,
        "num_claims_last_12m": num_claims_last_12m,
    }
    try:
        result = score_claim_capability(raw_claim)
        result["claim_id"] = claim_id
        return result
    except Exception as exc:
        logger.exception("score_claim failed")
        return {"error": str(exc), "claim_id": claim_id}


@mcp.tool()
def lookup_policy(
    question: str,
    claim_type: str = "Outpatient",
    procedure_code: str = "AA395",
) -> Dict[str, Any]:
    """Retrieve and synthesize grounded insurance policy information.

    The tool performs hybrid retrieval, cross-encoder reranking and grounded
    answer generation. It does not route between agents.
    """
    try:
        return lookup_policy_capability(
            question=question,
            claim_type=claim_type,
            procedure_code=procedure_code,
        )
    except Exception as exc:
        logger.exception("lookup_policy failed")
        return {"error": str(exc), "question": question}


if __name__ == "__main__":
    target_port = int(os.getenv("MCP_PORT", "8011"))
    logger.info("Starting Insurance Domain MCP server on port %s", target_port)
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=target_port,
        stateless_http=True,
    )
