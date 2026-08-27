import logging
import os
import platform
import signal
import subprocess
import threading
import time
from typing import Any, Dict

from fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("insurance_unified_server")

mcp = FastMCP("Insurance Unified MCP")
_SYSTEM = None
_ML_LOCK = threading.Lock()

def _get_system():
    global _SYSTEM
    if _SYSTEM is None:
        from insurance_multi_agent_chunking_indexing import InsuranceAgentSystem
        logger.info("Initializing Unified Pipeline (Risk ML + Policy LLM)")
        _SYSTEM = InsuranceAgentSystem(use_local_llm=False)
    return _SYSTEM

def clear_port(port: int):
    try:
        if platform.system() == "Windows":
            command = f"netstat -ano | findstr :{port}"
            output = subprocess.check_output(command, shell=True, text=True)
            for line in output.strip().split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    if pid and pid != "0":
                        subprocess.call(["taskkill", "/F", "/PID", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            command = f"lsof -t -i:{port}"
            output = subprocess.check_output(command, shell=True, text=True)
            for pid in output.strip().split('\n'):
                if pid:
                    os.kill(int(pid), signal.SIGKILL)
        time.sleep(0.5)
    except Exception:
        pass

@mcp.tool()
def lookup_policy(question: str) -> Dict[str, Any]:
    """Search insurance policies, coverage limits, rules, and SOPs for a given query."""
    system = _get_system()
    mapped_claim = {
        "ClaimType": "Outpatient", "ClaimAmount": 1.0, "ProcedureCode": "AA395",
        "ProviderSpecialty": "General Practice", "PatientIncome": 35000.0, "PatientAge": 45,
        "PatientID": "P-DEFAULT", "ProviderID": "PRV-201", "ClaimStatus": "Submitted",
        "DiagnosisCode": "D001", "ProviderLocation": "Urban", "ClaimSubmissionMethod": "Electronic"
    }
    state = {
        "raw_claim": mapped_claim, "user_query": question, "target_route": "policy_only",
        "dispersion_metrics": {}, "ml_probability": 0.0, "ml_prediction": 0, "triage_status": "",
        "risk_explanation": "", "retrieved_docs": [], "rag_generated_answer": "", 
        "needs_clarification": False, "final_report": ""
    }
    with _ML_LOCK:
        try:
            result = system.policy_agent(state)
            return {
                "question": question,
                "answer": result.get("rag_generated_answer", ""),
                "retrieved_docs": result.get("retrieved_docs", [])
            }
        except Exception as e:
            logger.error(f"RAG Engine Error: {e}")
            return {"error": f"Model error: {str(e)}"}

@mcp.tool()
def score_claim(
    claim_amount: float,
    patient_income: float = 35000.0,
    patient_age: int = 45,
    claim_type: str = "Outpatient",
    claim_id: str = "CLM-AUTO",
    procedure_code: str = "AA395",
    patient_id: str = "P-DEFAULT"
) -> Dict[str, Any]:
    """Calculate fraud risk, model-based probability, and triage decisions for a medical claim."""
    system = _get_system()
    mapped_claim = {
        "ClaimType": claim_type, "ClaimAmount": claim_amount, "ProcedureCode": procedure_code,
        "ProviderSpecialty": "General Practice", "PatientAge": patient_age, "PatientIncome": patient_income,
        "PatientID": patient_id, "ProviderID": "PRV-201", "ClaimStatus": "Submitted",
        "DiagnosisCode": "D001", "ProviderLocation": "Urban", "ClaimSubmissionMethod": "Electronic"
    }
    state = {
        "raw_claim": mapped_claim,
        "user_query": f"Risk triage for {claim_id}",
        "target_route": "claims_triage_agent",
        "dispersion_metrics": {}, "ml_probability": 0.0, "ml_prediction": 0,
        "triage_status": "", "risk_explanation": "", "retrieved_docs": [],
        "rag_generated_answer": "", "needs_clarification": False, "final_report": ""
    }
    with _ML_LOCK:
        try:
            # 1. Execute ML Triage Math
            triage_res = system.claims_triage_agent(state)
            state.update(triage_res)
            
            # Ensure raw_claim is retained for SHAP analysis
            state["raw_claim"] = mapped_claim
            
            # 2. Execute SHAP Feature Importance
            shap_res = system.risk_analysis_agent(state)
            state.update(shap_res)
            
            return {
                "claim_id": claim_id,
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