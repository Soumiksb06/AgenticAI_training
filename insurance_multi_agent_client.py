import asyncio
import json
import os
from typing import Any, Dict, Callable, TypeVar

from fastmcp import Client

RISK_SERVER = "http://127.0.0.1:" + str(os.getenv("RISK_MCP_PORT", "8011")) + "/mcp"
POLICY_SERVER = "http://127.0.0.1:" + str(os.getenv("POLICY_MCP_PORT", "8012")) + "/mcp"
T = TypeVar("T")


def _as_dict(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content
    if hasattr(result, "data") and isinstance(result.data, dict):
        return result.data
    if hasattr(result, "content") and result.content:
        for item in result.content:
            if hasattr(item, "text"):
                try:
                    parsed = json.loads(item.text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
    return {}


async def call_server(url: str, tool_name: str, payload: Dict[str, Any]):
    async with Client(url) as client:
        result = await client.call_tool(tool_name, payload)
        return _as_dict(result)


async def investigate_claim(claim: Dict[str, Any]):
    risk_claim = {
        key: value for key, value in claim.items() if key != "policy_question"
    }
    risk_task = call_server(
        RISK_SERVER,
        "score_claim",
        {"claim": risk_claim},
    )

    policy_task = call_server(
        POLICY_SERVER,
        "lookup_policy",
        {
            "request": {
                "claim_type": claim.get("claim_type", "Outpatient"),
                "procedure_code": claim.get("procedure_code", "AA395"),
                "provider_specialty": claim.get("provider_specialty", "General Practice"),
                "claim_amount": claim.get("claim_amount", 5000.0),
                "patient_age": claim.get("patient_age", 45),
                "patient_income": claim.get("patient_income", 35000.0),
                "patient_id": claim.get("patient_id", "P-DEFAULT"),
                "provider_id": claim.get("provider_id", "PRV-201"),
                "claim_status": claim.get("claim_status", "Submitted"),
                "diagnosis_code": claim.get("diagnosis_code", "D001"),
                "provider_location": claim.get("provider_location", "Urban"),
                "claim_submission_method": claim.get("claim_submission_method", "Electronic"),
                "question": claim.get(
                    "policy_question",
                    f"What policy rules apply to a {claim.get('claim_type', 'Outpatient')} claim "
                    f"for {claim.get('procedure_code', 'AA395')} valued at ${claim.get('claim_amount', 5000.0):,.2f}?",
                ),
            }
        },
    )

    risk_result, policy_result = await asyncio.gather(risk_task, policy_task)

    risk_payload = risk_result
    policy_payload = policy_result

    risk_level = str(risk_payload.get("risk_level", "UNKNOWN")).upper()
    risk_score = float(risk_payload.get("risk_score", 0.0))
    policy_answer = str(policy_payload.get("answer", "No policy answer available."))

    if risk_level == "HIGH_RISK" or risk_score >= 0.6:
        final_verdict = "Escalate for manual review"
    elif risk_level == "LOW_RISK" and risk_score < 0.4:
        final_verdict = "Auto-approve"
    else:
        final_verdict = "Review with policy validation"

    return {
        "claim_id": claim.get("claim_id", "UNKNOWN"),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "decision_cutoff": risk_payload.get("decision_cutoff", 0.0),
        "policy_answer": policy_answer,
        "final_verdict": final_verdict,
        "triage_status": risk_payload.get("triage_status", ""),
        "risk_explanation": risk_payload.get("risk_explanation", ""),
    }


def _prompt(label: str, default: T, converter: Callable[[str], T] = str) -> T:
    value = input(f"{label} [{default}]: ").strip()
    if not value:
        return default
    try:
        return converter(value)
    except (TypeError, ValueError):
        print(f"Invalid value. Using {default}.")
        return default


def collect_claim() -> Dict[str, Any]:
    print("\nEnter claim details. Press ENTER to use the displayed default.")
    claim_type = _prompt("Claim type (Inpatient/Outpatient/Emergency/Other)", "Outpatient")
    if claim_type not in {"Inpatient", "Outpatient", "Emergency", "Other"}:
        print("Invalid claim type. Using Outpatient.")
        claim_type = "Outpatient"

    claim_amount = _prompt("Claim amount", 24500.0, float)
    if claim_amount <= 0:
        print("Claim amount must be greater than zero. Using 24500.0.")
        claim_amount = 24500.0

    patient_age = _prompt("Patient age", 42, int)
    if not 18 <= patient_age <= 100:
        print("Patient age must be between 18 and 100. Using 42.")
        patient_age = 42

    patient_income = _prompt("Patient annual income", 80000.0, float)
    if patient_income <= 0:
        print("Patient income must be greater than zero. Using 80000.0.")
        patient_income = 80000.0

    return {
        "claim_id": _prompt("Claim ID", "CLM-1042"),
        "claim_amount": claim_amount,
        "claim_type": claim_type,
        "procedure_code": _prompt("Procedure code", "AA395"),
        "provider_specialty": _prompt("Provider specialty", "Pediatrics"),
        "patient_age": patient_age,
        "patient_income": patient_income,
        "patient_id": _prompt("Patient ID", "PAT-441"),
        "provider_id": _prompt("Provider ID", "PROV-88"),
        "claim_status": _prompt("Claim status", "Submitted"),
        "diagnosis_code": _prompt("Diagnosis code", "D003"),
        "provider_location": _prompt("Provider location", "Urban"),
        "claim_submission_method": _prompt("Submission method", "Electronic"),
        "policy_question": _prompt(
            "Policy question",
            "What policy rules apply to this claim?",
        ),
    }


async def main():
    print("=" * 75)
    print("MULTI-AGENT CLAIM INVESTIGATOR")
    print("=" * 75)
    while True:
        claim = collect_claim()
        try:
            result = await investigate_claim(claim)
            print("\nInvestigation result:")
            print(json.dumps(result, indent=2))
        except Exception as exc:
            print(f"\nInvestigation failed: {exc}")

        again = input("\nInvestigate another claim? (y/n) [n]: ").strip().lower()
        if again != "y":
            print("Exiting Investigator.")
            break


if __name__ == "__main__":
    asyncio.run(main())
