import logging
import os
import platform
import signal
import subprocess
import threading
import time
from typing import Any, Dict, Literal

from fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("insurance_unified_server")

# Unified MCP Server
mcp = FastMCP("Insurance Unified MCP")

# 🛑 THE FIX: A single, unified system instance for BOTH tools
_SYSTEM = None

# A lock to prevent PyTorch tensor corruption from concurrent execution
_ML_LOCK = threading.Lock()


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


class PolicyRequest(ClaimInput):
    question: str = Field(default="What are the policy rules and limits for this claim?")
    claim_id: str = Field(default="CLM-DEFAULT", min_length=1)


def _get_system():
    global _SYSTEM
    if _SYSTEM is None:
        from insurance_multi_agent_chunking_indexing import InsuranceAgentSystem
        logger.info("Initializing Unified Pipeline (Risk ML + Policy LLM)")
        # Set use_local_llm=False to stop Hugging Face weights from loading into Python memory
        _SYSTEM = InsuranceAgentSystem(use_local_llm=False)
    return _SYSTEM


def clear_port(port: int):
    """Kills any process currently listening on the target port."""
    logger.info(f"Checking if port {port} is blocked...")
    try:
        if platform.system() == "Windows":
            command = f"netstat -ano | findstr :{port}"
            try:
                output = subprocess.check_output(command, shell=True, text=True)
                for line in output.strip().split('\n'):
                    if f":{port}" in line and "LISTENING" in line:
                        pid = line.strip().split()[-1]
                        if pid and pid != "0":
                            logger.warning(f"Killing Windows process {pid} blocking port {port}")
                            subprocess.call(["taskkill", "/F", "/PID", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                pass
        else:
            command = f"lsof -t -i:{port}"
            try:
                output = subprocess.check_output(command, shell=True, text=True)
                pids = output.strip().split('\n')
                for pid in pids:
                    if pid:
                        logger.warning(f"Killing Unix process {pid} blocking port {port}")
                        os.kill(int(pid), signal.SIGKILL)
            except subprocess.CalledProcessError:
                pass
        time.sleep(0.5)
    except Exception as e:
        logger.error(f"Failed to clear port {port}: {e}")


@mcp.tool()
def health_check() -> Dict[str, Any]:
    """Check the health of the Unified Insurance MCP."""
    return {
        "status": "ok",
        "server": "Insurance Unified MCP",
        "services": ["policy-guidance", "fraud-triage"]
    }


@mcp.tool()
def lookup_policy(request: PolicyRequest) -> Dict[str, Any]:
    """Search insurance policies, limits, and rules for a given claim."""
    system = _get_system() # Uses the Singleton
    
    raw_claim = request.model_dump()
    mapped_claim = {
        "ClaimType": request.claim_type, "ClaimAmount": request.claim_amount,
        "ProcedureCode": request.procedure_code, "ProviderSpecialty": request.provider_specialty,
        "PatientIncome": request.patient_income, "PatientAge": request.patient_age,
        "PatientID": request.patient_id, "ProviderID": request.provider_id,
        "ClaimStatus": request.claim_status, "DiagnosisCode": request.diagnosis_code,
        "ProviderLocation": request.provider_location, "ClaimSubmissionMethod": request.claim_submission_method,
    }

    state = {
        "raw_claim": mapped_claim, "user_query": request.question, "target_route": "policy_only",
        "dispersion_metrics": {}, "ml_probability": 0.0, "ml_prediction": 0, "triage_status": "",
        "risk_explanation": "", "retrieved_docs": [], "rag_generated_answer": "", 
        "needs_clarification": False, "final_report": "",
    }
    
    with _ML_LOCK:
        try:
            result = system.policy_agent(state)
            return {
                "claim_type": request.claim_type, "question": request.question,
                "answer": result.get("rag_generated_answer", ""),
                "retrieved_docs": result.get("retrieved_docs", []),
            }
        except Exception as e:
            logger.error(f"RAG Engine Error: {e}")
            return {"error": f"Model error: {str(e)}"}


@mcp.tool()
def score_claim(claim: ClaimInput) -> Dict[str, Any]:
    """Calculate fraud risk, model-based probability, and triage decisions."""
    system = _get_system() 
    
    raw_claim = claim.model_dump()
    mapped_claim = {
        "ClaimType": raw_claim.pop("claim_type"), 
        "ClaimAmount": raw_claim.pop("claim_amount"),
        "ProcedureCode": raw_claim.pop("procedure_code"), 
        "ProviderSpecialty": raw_claim.pop("provider_specialty"),
        "PatientAge": raw_claim.pop("patient_age"), 
        "PatientIncome": raw_claim.pop("patient_income"),
        "PatientID": raw_claim.pop("patient_id"), 
        "ProviderID": raw_claim.pop("provider_id"),
        "ClaimStatus": raw_claim.pop("claim_status"), 
        "DiagnosisCode": raw_claim.pop("diagnosis_code"),
        "ProviderLocation": raw_claim.pop("provider_location"), 
        "ClaimSubmissionMethod": raw_claim.pop("claim_submission_method")
    }

    state = {
        "raw_claim": mapped_claim,
        "user_query": f"Risk triage for {claim.claim_id}",
        "target_route": "claims_triage_agent",
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

    with _ML_LOCK:
        try:
            # 1. Execute Agent 1: ML Triage Math
            triage_res = system.claims_triage_agent(state)
            state.update(triage_res)
            
            # 2. Execute Agent 2: SHAP Feature Importance
            shap_res = system.risk_analysis_agent(state)
            state.update(shap_res)
            
            # Return pure ML results without invoking policy_agent (RAG)
            return {
                "claim_id": claim.claim_id,
                "risk_level": "HIGH_RISK" if state.get("ml_prediction", 0) else "LOW_RISK",
                "risk_score": round(float(state.get("ml_probability", 0.0)), 6),
                "decision_cutoff": round(system.optimal_threshold, 6),
                "triage_status": state.get("triage_status", ""),
                "risk_explanation": state.get("risk_explanation", ""),
            }
        except Exception as e:
            logger.error(f"Risk Engine Error: {e}")
            return {"error": f"Model error: {str(e)}"}


if __name__ == "__main__":
    target_port = int(os.getenv("MCP_PORT", "8011"))
    clear_port(target_port)
    mcp.run(transport="http", host="0.0.0.0", port=target_port, stateless_http=True)