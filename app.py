import asyncio
import json
import os
import re
from typing import Any, Dict

import streamlit as st
from fastmcp import Client

# Config & Ports
RISK_SERVER_PORT = os.getenv("RISK_MCP_PORT", "8011")
POLICY_SERVER_PORT = os.getenv("POLICY_MCP_PORT", "8012")

RISK_SERVER_URL = f"http://127.0.0.1:{RISK_SERVER_PORT}/mcp"
POLICY_SERVER_URL = f"http://127.0.0.1:{POLICY_SERVER_PORT}/mcp"

st.set_page_config(
    page_title="Insurance MCP Claim Investigator",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# 1. FIXED PRECEDENCE NLP EXTRACTION ENGINE
# =====================================================================
def extract_slots_from_text(prompt_text: str) -> Dict[str, Any]:
    if not prompt_text:
        return {}

    text = prompt_text.lower().strip()
    slots = {}

    # 1. EXPLICIT INCOME
    income_match = re.search(
        r"(?:income|salary|earning|earns?)\s*[\$]?\s*(\d[\d,]*)", text
    )
    if income_match:
        try:
            slots["patient_income"] = float(income_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # 2. EXPLICIT AGE
    age_match = re.search(
        r"(?:age\s*|aged?\s*)(\d{1,2})\b|\b(\d{1,2})\s*(?:years?|yo)\b", text
    )
    if age_match:
        try:
            val = age_match.group(1) or age_match.group(2)
            slots["patient_age"] = int(val)
        except ValueError:
            pass

    # 3. EXPLICIT CLAIM AMOUNT
    amount_match = re.search(
        r"(?:need|claim|requiring|total|for|amount|val|value)\s*[\$]?\s*(\d[\d,]*)",
        text,
    )
    if amount_match:
        try:
            slots["claim_amount"] = float(amount_match.group(1).replace(",", ""))
        except ValueError:
            pass
    else:
        # Fallback: Currency patterns ($5000, 5k)
        curr_match = re.search(
            r"[\$]\s*(\d[\d,]*)(?:\s*k\b|\b)|(\b\d[\d,]*\s*k\b)", text
        )
        if curr_match:
            raw_str = curr_match.group(0).replace("$", "").replace(",", "").strip()
            multiplier = 1000 if "k" in raw_str else 1
            clean_num = float(re.sub(r"[^\d.]", "", raw_str)) * multiplier
            slots["claim_amount"] = clean_num
        else:
            # Fallback: Find unassigned numbers not matched to income or age
            all_nums = re.findall(r"\b\d[\d,]*\b", text)
            assigned_nums = set()
            if "patient_income" in slots:
                assigned_nums.add(str(int(slots["patient_income"])))
            if "patient_age" in slots:
                assigned_nums.add(str(slots["patient_age"]))

            for n_str in all_nums:
                clean_n = n_str.replace(",", "")
                if clean_n not in assigned_nums:
                    val = float(clean_n)
                    if val > 100 and val != slots.get("patient_income", 0):
                        slots["claim_amount"] = val
                        break

    # 4. EXPLICIT CLAIM TYPE
    if re.search(r"\b(outp|outpatient|clinic|daycare)\b", text):
        slots["claim_type"] = "Outpatient"
    elif re.search(r"\b(inp|inpatient|admitted|hospitalized)\b", text):
        slots["claim_type"] = "Inpatient"
    elif re.search(r"\b(emergency|er|urgent)\b", text):
        slots["claim_type"] = "Emergency"
    elif "brain tumor" in text or "surgery" in text:
        slots["claim_type"] = "Inpatient"

    # 5. EXPLICIT SPECIALTY
    if re.search(r"\b(ortho|orthopedics?|bone)\b", text):
        slots["provider_specialty"] = "Orthopedics"
    elif re.search(r"\b(neuro|neurology)\b", text):
        slots["provider_specialty"] = "Neurology"
    elif re.search(r"\b(pedia|pediatric|child)\b", text):
        slots["provider_specialty"] = "Pediatrics"
    elif re.search(r"\b(cardio|cardiology|heart)\b", text):
        slots["provider_specialty"] = "Cardiology"
    elif re.search(r"\b(onco|oncology|cancer|tumor)\b", text):
        slots["provider_specialty"] = "Oncology"
    elif "brain" in text:
        slots["provider_specialty"] = "Neurology"

    # 6. LOCATION
    if "rural" in text:
        slots["provider_location"] = "Rural"
    elif "urban" in text:
        slots["provider_location"] = "Urban"

    return slots


# =====================================================================
# 2. ASYNC MCP CLIENT WITH LIVE PROGRESS TRACKING
# =====================================================================
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


async def call_mcp_tool(url: str, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        async with Client(url) as client:
            res = await client.call_tool(tool_name, payload)
            return _as_dict(res)
    except Exception as exc:
        return {"error": f"Failed to connect to {url}: {str(exc)}"}


async def run_mcp_investigation(
    claim_payload: Dict[str, Any], policy_query: str, status_box
):
    claim_amount = claim_payload.get("claim_amount", 0.0)

    # Step 1: Auto-Routing Decision
    status_box.update(
        label="⚙️ Step 1/3: Analyzing Intent & Routing Query...", state="running"
    )
    await asyncio.sleep(0.3)

    if claim_amount <= 0.0:
        # Policy-Only Route
        status_box.write("🔀 **Auto-Routing Verdict**: Claim amount = `$0.00`. Routing strictly to **Policy Agent**.")
        status_box.write(f"📡 **MCP Server**: `Insurance Policy MCP` (`{POLICY_SERVER_URL}`)")
        status_box.write("🛠️ **MCP Tool Called**: `lookup_policy`")

        status_box.update(
            label=f"📡 Step 2/2: Executing Policy MCP Tool ('lookup_policy' on Port {POLICY_SERVER_PORT})...",
            state="running",
        )

        policy_res = await call_mcp_tool(
            POLICY_SERVER_URL,
            "lookup_policy",
            {
                "request": {
                    "claim_type": claim_payload.get("claim_type", "Outpatient"),
                    "procedure_code": claim_payload.get("procedure_code", "AA395"),
                    "provider_specialty": claim_payload.get("provider_specialty", "General Practice"),
                    "question": policy_query or "What policy rules apply to this treatment?",
                    "claim_amount": 5000.0,
                    "patient_age": claim_payload.get("patient_age", 45),
                    "patient_income": claim_payload.get("patient_income", 35000.0),
                }
            },
        )

        status_box.update(
            label="✅ Policy Guidance Retrieved Successfully!", state="complete"
        )
        return {"mode": "POLICY_ONLY", "risk": None, "policy": policy_res}

    else:
        # Dual Execution Route
        status_box.write(f"🔀 **Auto-Routing Verdict**: Claim amount = `${claim_amount:,.2f}` (> $0.00). Triggering **Dual Agent Pipeline**.")
        status_box.write(f"📡 **Task 1**: Calling `Insurance Risk MCP` (`{RISK_SERVER_URL}`) | 🛠️ Tool: `score_claim`")
        status_box.write(f"📡 **Task 2**: Calling `Insurance Policy MCP` (`{POLICY_SERVER_URL}`) | 🛠️ Tool: `lookup_policy`")

        status_box.update(
            label="⚡ Step 2/3: Concurrently Dispatching Requests via asyncio.gather...",
            state="running",
        )

        risk_task = call_mcp_tool(
            RISK_SERVER_URL,
            "score_claim",
            {"claim": claim_payload},
        )
        policy_task = call_mcp_tool(
            POLICY_SERVER_URL,
            "lookup_policy",
            {
                "request": {
                    "claim_type": claim_payload.get("claim_type", "Outpatient"),
                    "procedure_code": claim_payload.get("procedure_code", "AA395"),
                    "provider_specialty": claim_payload.get("provider_specialty", "General Practice"),
                    "question": policy_query or f"Policy rules for {claim_payload.get('claim_type')} claim ${claim_amount:,.2f}",
                    "claim_amount": claim_amount,
                    "patient_age": claim_payload.get("patient_age", 45),
                    "patient_income": claim_payload.get("patient_income", 35000.0),
                    "patient_id": claim_payload.get("patient_id", "P-DEFAULT"),
                    "provider_id": claim_payload.get("provider_id", "PRV-201"),
                }
            },
        )

        risk_res, policy_res = await asyncio.gather(risk_task, policy_task)

        status_box.update(
            label="✅ Dual MCP Tasks Completed & Synthesized!", state="complete"
        )
        return {"mode": "DUAL_EXECUTION", "risk": risk_res, "policy": policy_res}


# =====================================================================
# 3. STREAMLIT UI DASHBOARD
# =====================================================================
st.title("🕵️ Insurance Claim Investigator & Policy Advisor")
st.caption("Powered by FastMCP, LangGraph Auto-Routing, and Multi-Agent RAG")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ MCP Server Cluster")
    st.text_input("Risk MCP Endpoint", RISK_SERVER_URL, disabled=True)
    st.text_input("Policy MCP Endpoint", POLICY_SERVER_URL, disabled=True)
    st.markdown("---")
    st.markdown("### 💡 Quick Examples")
    if st.button("Example 1: Brain Tumor Claim ($5k)"):
        st.session_state["user_prompt"] = (
            "I am having brain tumor, need 5000 income 30000 from rural ortho age 21 outp"
        )
        st.rerun()
    if st.button("Example 2: General Policy Query"):
        st.session_state["user_prompt"] = "best insurance to take?"
        st.rerun()

# User Text Input
default_prompt = st.session_state.get("user_prompt", "best insurance to take?")

user_query = st.text_area(
    "💬 Enter Natural Language Query or Claim Details:",
    value=default_prompt,
    height=90,
)

# Extract slots
extracted_slots = extract_slots_from_text(user_query)

# Display Detected Slot Badges
st.markdown("##### 🔍 Detected Parameters from Prompt:")
if extracted_slots:
    badge_cols = st.columns(len(extracted_slots))
    for idx, (k, v) in enumerate(extracted_slots.items()):
        badge_cols[idx].info(f"**{k}**: `{v}`")
else:
    st.caption("No specific claim attributes detected. Query will route to Policy RAG.")

# Determine if this prompt has explicit claim details
has_claim_details = bool(
    extracted_slots and extracted_slots.get("claim_amount", 0.0) > 0.0
)

# Refinement Form (Auto-expands ONLY if claim parameters were detected)
with st.expander(
    "⚙️ Refine Extracted Parameters (Slot-Filling Form)",
    expanded=has_claim_details,
):
    col1, col2, col3, col4 = st.columns(4)

    # Dynamic key based on prompt to force value refresh on query change
    prompt_hash = str(hash(user_query))

    with col1:
        form_amount = st.number_input(
            "Claim Amount ($)",
            min_value=0.0,
            value=float(extracted_slots.get("claim_amount", 0.0)),
            step=500.0,
            key=f"amt_{prompt_hash}",
        )
        claim_type_opts = ["Outpatient", "Inpatient", "Emergency", "Other"]
        default_type = extracted_slots.get("claim_type", "Outpatient")
        form_type = st.selectbox(
            "Claim Type",
            claim_type_opts,
            index=claim_type_opts.index(default_type) if default_type in claim_type_opts else 0,
            key=f"type_{prompt_hash}",
        )

    with col2:
        form_age = st.number_input(
            "Patient Age",
            min_value=18,
            max_value=100,
            value=int(extracted_slots.get("patient_age", 42)),
            key=f"age_{prompt_hash}",
        )
        form_income = st.number_input(
            "Patient Income ($)",
            min_value=0.0,
            value=float(extracted_slots.get("patient_income", 35000.0)),
            step=1000.0,
            key=f"inc_{prompt_hash}",
        )

    with col3:
        form_specialty = st.text_input(
            "Provider Specialty",
            value=extracted_slots.get("provider_specialty", "General Practice"),
            key=f"spec_{prompt_hash}",
        )
        location_opts = ["Urban", "Rural"]
        default_loc = extracted_slots.get("provider_location", "Urban")
        form_location = st.selectbox(
            "Provider Location",
            location_opts,
            index=location_opts.index(default_loc) if default_loc in location_opts else 0,
            key=f"loc_{prompt_hash}",
        )

    with col4:
        form_proc_code = st.text_input("Procedure Code", value="AA395", key=f"proc_{prompt_hash}")
        form_claim_id = st.text_input("Claim ID", value="CLM-1042", key=f"cid_{prompt_hash}")

st.markdown("<br>", unsafe_allow_html=True)

# Execution Action Button
if st.button("🚀 Run Multi-Agent Investigation", type="primary", use_container_width=True):
    # Construct Payload
    claim_payload = {
        "claim_id": form_claim_id,
        "claim_amount": form_amount,
        "claim_type": form_type,
        "procedure_code": form_proc_code,
        "provider_specialty": form_specialty,
        "patient_age": form_age,
        "patient_income": form_income,
        "patient_id": "PAT-441",
        "provider_id": "PROV-88",
        "claim_status": "Submitted",
        "diagnosis_code": "D001",
        "provider_location": form_location,
        "claim_submission_method": "Electronic",
    }

    # Interactive Status Tracker
    status_box = st.status("🚀 Initializing MCP Agent Pipeline...", expanded=True)

    # Run Investigation
    results = asyncio.run(
        run_mcp_investigation(claim_payload, user_query, status_box)
    )

    st.markdown("---")

    # Display Auto-Routing Banner
    if results["mode"] == "POLICY_ONLY":
        st.info("ℹ️ **Auto-Routing**: Claim Amount = `$0.00`. Routed strictly to **Policy Agent (RAG)**.")
    else:
        st.success("⚡ **Auto-Routing**: Claim Amount > `$0.00`. Executed **Dual Agent Concurrent Pipeline**.")

    res_col1, res_col2 = st.columns(2)

    # Risk Agent Results Column
    with res_col1:
        st.subheader("🛡️ Risk Analysis & Triage")
        st.caption(f"Server: `Insurance Risk MCP` | Tool: `score_claim`")
        risk_data = results.get("risk")

        if not risk_data:
            st.warning("Risk assessment bypassed (Policy-only mode).")
        elif "error" in risk_data:
            st.error(risk_data["error"])
        else:
            risk_level = risk_data.get("risk_level", "UNKNOWN")
            score = risk_data.get("risk_score", 0.0)
            cutoff = risk_data.get("decision_cutoff", 0.0)

            if risk_level == "HIGH_RISK":
                st.error(f"**Verdict**: {risk_level} (Score: {score:.2%})")
            else:
                st.success(f"**Verdict**: {risk_level} (Score: {score:.2%})")

            st.progress(min(score, 1.0))
            st.caption(f"Decision Threshold Cutoff: {cutoff:.2%}")

            with st.expander("🔬 SHAP Feature Contributions", expanded=True):
                st.text(risk_data.get("risk_explanation", "No explanation available."))

    # Policy Agent Results Column
    with res_col2:
        st.subheader("📜 Policy Guidance (RAG)")
        st.caption(f"Server: `Insurance Policy MCP` | Tool: `lookup_policy`")
        policy_data = results.get("policy")

        if not policy_data:
            st.warning("No policy response available.")
        elif "error" in policy_data:
            st.error(policy_data["error"])
        else:
            st.markdown(f"**Grounded Answer:**\n{policy_data.get('answer', 'N/A')}")

            retrieved_docs = policy_data.get("retrieved_docs", [])
            if retrieved_docs:
                with st.expander(f"📑 Grounded Policy Excerpts ({len(retrieved_docs)})", expanded=True):
                    for doc in retrieved_docs:
                        st.markdown(f"- **Source**: `{doc.get('source_file')}` | **Re-Rank Score**: {doc.get('rerank_score', 0):.2f}%")
                        st.caption(f"\"{doc.get('snippet')}\"")