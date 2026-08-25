import asyncio
import json
import os
import re
from typing import Any, Dict, Callable, TypeVar

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
    from fastmcp import Client
    async with Client(url) as client:
        result = await client.call_tool(tool_name, payload)
        return _as_dict(result)


async def investigate_claim(claim: Dict[str, Any]):
    is_policy_only = claim.get("mode") == "policy_only" or float(claim.get("claim_amount", 0.0)) <= 0

    # Auto-Route 1: Standalone Policy RAG Query (Skips Risk MCP Server)
    if is_policy_only:
        policy_payload = await call_server(
            POLICY_SERVER,
            "lookup_policy",
            {
                "request": {
                    "claim_type": claim.get("claim_type", "Outpatient"),
                    "procedure_code": claim.get("procedure_code", "AA395"),
                    "provider_specialty": claim.get("provider_specialty", "General Practice"),
                    "claim_amount": 5000.0,
                    "patient_age": 45,
                    "patient_income": 35000.0,
                    "question": claim.get("policy_question", "What are the policy rules for this treatment?"),
                }
            },
        )
        return {
            "mode": "POLICY_QUERY_ONLY",
            "question": claim.get("policy_question"),
            "policy_answer": policy_payload.get("answer", "No policy answer available."),
            "retrieved_docs": policy_payload.get("retrieved_docs", []),
        }

    # Auto-Route 2: Dual Agent Evaluation (Risk MCP + Policy MCP concurrently)
    risk_claim = {
        key: value for key, value in claim.items() if key not in {"policy_question", "mode"}
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

    risk_level = str(risk_result.get("risk_level", "UNKNOWN")).upper()
    risk_score = float(risk_result.get("risk_score", 0.0))
    policy_answer = str(policy_result.get("answer", "No policy answer available."))

    if risk_level == "HIGH_RISK" or risk_score >= 0.6:
        final_verdict = "Escalate for manual review"
    elif risk_level == "LOW_RISK" and risk_score < 0.4:
        final_verdict = "Auto-approve"
    else:
        final_verdict = "Review with policy validation"

    return {
        "mode": "FULL_CLAIM_INVESTIGATION",
        "claim_id": claim.get("claim_id", "UNKNOWN"),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "decision_cutoff": risk_result.get("decision_cutoff", 0.0),
        "policy_answer": policy_answer,
        "final_verdict": final_verdict,
        "triage_status": risk_result.get("triage_status", ""),
        "risk_explanation": risk_result.get("risk_explanation", ""),
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


def _extract_all_fields_from_text(text: str) -> Dict[str, Any]:
    """Dynamically parses and maps all detected claim entities from user query text."""
    extracted = {}
    text_lower = text.lower()

    # 1. Patient Annual Income (e.g., 'income 70000', 'salary 85000', 'income 30000')
    inc_match = re.search(
        r'(?:income|salary|earning|makes?|pay)\s*(?:of\s*)?[\$₹]?\s*(\d[\d,]*(?:\.\d+)?)'
        r'|(\d[\d,]*)\s*(?:income|salary|annual)',
        text, re.IGNORECASE
    )
    if inc_match:
        for group in inc_match.groups():
            if group:
                try:
                    val = float(group.replace(',', ''))
                    if 'k' in inc_match.group(0).lower() and val < 1000:
                        val *= 1000
                    if val > 0:
                        extracted['patient_income'] = val
                        break
                except ValueError:
                    pass

    # 2. Patient Age (e.g., 'age 42', '42 yo', '42 years old', '42y')
    age_match = re.search(r'\b(?:age\s*(\d{1,2})|(\d{1,2})\s*(?:years?\s*old|yo|y/o|yr|yrs))\b', text, re.IGNORECASE)
    if age_match:
        raw_age = age_match.group(1) or age_match.group(2)
        if raw_age:
            try:
                age_val = int(raw_age)
                if 18 <= age_val <= 100:
                    extracted['patient_age'] = age_val
            except ValueError:
                pass

    # 3. Claim Amount (e.g., 'need 10000', 'claim 7000', '$7000', 'cost of 5000', '10000 required')
    amt_match = re.search(
        r'(?:claim\s*(?:of\s*)?|cost\s*(?:of\s*)?|amount\s*(?:of\s*)?|bill\s*(?:of\s*)?|fee\s*(?:of\s*)?|need\s*(?:s|ed)?\s*(?:of\s*)?|require\s*(?:s|d)?\s*(?:of\s*)?|total\s*(?:of\s*)?|treatment\s*(?:of\s*)?|surgery\s*(?:of\s*)?|for\s+|[\$₹]\s*)(\d[\d,]*(?:\.\d+)?)\s*(?:k\b|dollars?|usd)?'
        r'|(\d[\d,]*(?:\.\d+)?)\s*(?:k\b|dollars?|usd)\b'
        r'|(\d[\d,]*)\s*(?:claim|cost|bill|fee|need|needed|required|total|expense)',
        text, re.IGNORECASE
    )
    if amt_match:
        for group in amt_match.groups():
            if group:
                try:
                    val = float(group.replace(',', ''))
                    full_match = amt_match.group(0).lower()
                    if 'k' in full_match and val < 1000:
                        val *= 1000
                    if val > 0:
                        extracted['claim_amount'] = val
                        break
                except ValueError:
                    pass

    # Fallback heuristic: If claim_amount not found, find any remaining unassigned number (excluding income & age)
    if 'claim_amount' not in extracted:
        all_numbers = re.findall(r'\b\d[\d,]*\b', text)
        for num_str in all_numbers:
            try:
                val = float(num_str.replace(',', ''))
                if val == extracted.get('patient_income') or val == extracted.get('patient_age'):
                    continue
                if 100 <= val <= 500000:
                    extracted['claim_amount'] = val
                    break
            except ValueError:
                pass

    # 4. Claim Type
    if "inpatient" in text_lower or "hospitalized" in text_lower or "admitted" in text_lower or "tumor" in text_lower or "surgery" in text_lower:
        extracted['claim_type'] = "Inpatient"
    elif "emergency" in text_lower or "er " in text_lower or "e.r." in text_lower:
        extracted['claim_type'] = "Emergency"
    elif "outpatient" in text_lower or "clinic" in text_lower or "ambulatory" in text_lower:
        extracted['claim_type'] = "Outpatient"

    # 5. Provider Specialty
    specialties = {
        "brain": "Neurology",
        "neuro": "Neurology",
        "neurology": "Neurology",
        "tumor": "Oncology",
        "cancer": "Oncology",
        "oncology": "Oncology",
        "ortho": "Orthopedics",
        "orthopedic": "Orthopedics",
        "pedia": "Pediatrics",
        "pediatric": "Pediatrics",
        "cardio": "Cardiology",
        "cardiology": "Cardiology",
        "general": "General Practice",
        "gp": "General Practice",
        "surg": "Surgery",
        "surgical": "Surgery"
    }
    for key, spec in specialties.items():
        if re.search(rf'\b{key}\b', text_lower):
            extracted['provider_specialty'] = spec
            break

    # 6. Procedure Code (e.g., AA395, PROC-101)
    proc_match = re.search(r'\b([A-Za-z]{2,4}-?\d{3,4})\b', text)
    if proc_match:
        extracted['procedure_code'] = proc_match.group(1).upper()

    # 7. Diagnosis Code (e.g., D003, DIAG-01)
    diag_match = re.search(r'\b(D\d{3,4}|DIAG-?\d+)\b', text, re.IGNORECASE)
    if diag_match:
        extracted['diagnosis_code'] = diag_match.group(1).upper()

    # 8. Provider Location
    if "rural" in text_lower:
        extracted['provider_location'] = "Rural"
    elif "suburban" in text_lower:
        extracted['provider_location'] = "Suburban"
    elif "urban" in text_lower:
        extracted['provider_location'] = "Urban"

    # 9. Submission Method
    if "paper" in text_lower or "mail" in text_lower:
        extracted['claim_submission_method'] = "Paper"
    elif "electronic" in text_lower or "online" in text_lower or "portal" in text_lower:
        extracted['claim_submission_method'] = "Electronic"

    # 10. IDs
    claim_id_m = re.search(r'\b(CLM-\d+)\b', text, re.IGNORECASE)
    if claim_id_m:
        extracted['claim_id'] = claim_id_m.group(1).upper()

    patient_id_m = re.search(r'\b(PAT-\d+|P-\d+)\b', text, re.IGNORECASE)
    if patient_id_m:
        extracted['patient_id'] = patient_id_m.group(1).upper()

    provider_id_m = re.search(r'\b(PROV-\d+|PRV-\d+)\b', text, re.IGNORECASE)
    if provider_id_m:
        extracted['provider_id'] = provider_id_m.group(1).upper()

    return extracted


def _parse_claim_type(input_str: str) -> str:
    """Normalizes variations like 'out', 'in', 'er' into valid schema types."""
    val = input_str.strip().lower()
    if val in {"out", "outpatient", "op"}:
        return "Outpatient"
    if val in {"in", "inpatient", "ip"}:
        return "Inpatient"
    if val in {"er", "emergency", "em"}:
        return "Emergency"
    if val in {"other"}:
        return "Other"
    return "Outpatient"


def collect_claim() -> Dict[str, Any]:
    print("\nEnter input parameters (Press ENTER to accept defaults).")
    policy_question = _prompt("Policy Question / Prompt", "What policy rules apply to this treatment?")
    
    # Auto-extract ALL slots present in user prompt text
    extracted = _extract_all_fields_from_text(policy_question)

    default_amount = extracted.get("claim_amount", 0.0)
    claim_amount = _prompt("Claim amount ($0.0 to auto-route to policy only)", default_amount, float)

    # AUTO-ROUTE: If claim amount is $0.0, return immediately and SKIP ALL remaining prompts
    if claim_amount <= 0:
        print("ℹ️ General policy query detected ($0.0 claim). Auto-routing strictly to Policy Agent (RAG).\n")
        return {
            "claim_amount": 0.0,
            "policy_question": policy_question,
            "mode": "policy_only",
            "claim_type": extracted.get("claim_type", "Outpatient"),
            "procedure_code": extracted.get("procedure_code", "AA395"),
        }

    # Populate all remaining prompts with extracted values when detected
    print(f"\n📋 Claim detected (${claim_amount:,.2f}). Collecting additional claim parameters for Risk Evaluation:")
    
    default_type = extracted.get("claim_type", "Outpatient")
    raw_type = _prompt("Claim type (Inpatient/Outpatient/Emergency/Other)", default_type)
    claim_type = _parse_claim_type(raw_type)

    default_age = extracted.get("patient_age", 42)
    patient_age = _prompt("Patient age", default_age, int)
    if not 18 <= patient_age <= 100:
        print(f"Patient age must be between 18 and 100. Using {default_age}.")
        patient_age = default_age

    default_income = extracted.get("patient_income", 80000.0)
    patient_income = _prompt("Patient annual income", default_income, float)
    if patient_income <= 0:
        print(f"Patient income must be greater than zero. Using {default_income}.")
        patient_income = default_income

    return {
        "claim_id": _prompt("Claim ID", extracted.get("claim_id", "CLM-1042")),
        "claim_amount": claim_amount,
        "claim_type": claim_type,
        "procedure_code": _prompt("Procedure code", extracted.get("procedure_code", "AA395")),
        "provider_specialty": _prompt("Provider specialty", extracted.get("provider_specialty", "Pediatrics")),
        "patient_age": patient_age,
        "patient_income": patient_income,
        "patient_id": _prompt("Patient ID", extracted.get("patient_id", "PAT-441")),
        "provider_id": _prompt("Provider ID", extracted.get("provider_id", "PROV-88")),
        "claim_status": _prompt("Claim status", extracted.get("claim_status", "Submitted")),
        "diagnosis_code": _prompt("Diagnosis code", extracted.get("diagnosis_code", "D003")),
        "provider_location": _prompt("Provider location", extracted.get("provider_location", "Urban")),
        "claim_submission_method": _prompt("Submission method", extracted.get("claim_submission_method", "Electronic")),
        "policy_question": policy_question,
    }


async def main():
    print("=" * 75)
    print("MULTI-AGENT CLAIM INVESTIGATOR & POLICY ADVISOR")
    print("=" * 75)
    while True:
        claim = collect_claim()
        try:
            result = await investigate_claim(claim)
            print("\nInvestigation result:")
            print(json.dumps(result, indent=2))
        except Exception as exc:
            print(f"\nInvestigation failed: {exc}")

        again = input("\nPerform another query? (y/n) [n]: ").strip().lower()
        if again != "y":
            print("Exiting Investigator.")
            break


if __name__ == "__main__":
    asyncio.run(main())