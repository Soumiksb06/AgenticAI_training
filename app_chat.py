"""ClaimsAI Streamlit UI with true orchestrator -> specialist agents -> MCP tools.

Architecture
------------
User/UI
   -> Orchestrator Agent
      -> Risk Specialist Agent -> score_claim MCP tool
      -> Policy Specialist Agent -> lookup_policy MCP tool
   -> Orchestrator synthesis
   -> User

The orchestrator never calls the domain MCP tools directly.
Each specialist receives only the MCP capability relevant to its role.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, Literal, Optional

import streamlit as st
from fastmcp import Client
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MCP_SERVER_PORT = os.getenv("MCP_PORT", "8011")
MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL", f"http://127.0.0.1:{MCP_SERVER_PORT}/mcp"
)

st.set_page_config(
    page_title="ClaimsAI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero-container { text-align:center; padding:1rem 1rem .5rem; margin-bottom:1rem; }
    .main-title { font-size:2.35rem; font-weight:800; margin-bottom:.2rem; }
    .sub-title { color:#555E6C; font-size:1rem; font-weight:600; margin-bottom:1rem; }
    .badge-container { display:flex; justify-content:center; gap:10px; margin-bottom:.8rem; }
    .badge { background:#EBF3FE; color:#1E3C72; padding:5px 12px; border-radius:20px;
             font-size:.8rem; font-weight:700; border:1px solid #C6DCFA; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# MCP transport helpers
# ---------------------------------------------------------------------------
async def _call_mcp(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call one MCP tool and normalize FastMCP content into a Python dict."""
    try:
        async with Client(MCP_SERVER_URL) as client:
            result = await client.call_tool(tool_name, payload)
            if hasattr(result, "content"):
                text_parts = []
                for item in result.content or []:
                    text = getattr(item, "text", None)
                    if text is not None:
                        text_parts.append(text)
                if text_parts:
                    raw = "\n".join(text_parts)
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        return {"result": raw}
            if isinstance(result, dict):
                return result
            return {"result": str(result)}
    except Exception as exc:
        return {"error": f"MCP {tool_name} call failed: {exc}"}


async def _discover_mcp_tools() -> Dict[str, Any]:
    try:
        async with Client(MCP_SERVER_URL) as client:
            tools = await client.list_tools()
            return {t.name: t for t in tools}
    except Exception as exc:
        return {"__error__": exc}


@st.cache_resource(show_spinner=False)
def get_mcp_tool_catalog() -> Dict[str, Any]:
    try:
        return asyncio.run(_discover_mcp_tools())
    except Exception as exc:
        return {"__error__": exc}


# ---------------------------------------------------------------------------
# MCP-backed specialist tools
# ---------------------------------------------------------------------------
class RiskToolInput(BaseModel):
    claim_amount: float = Field(..., description="Claim amount in the claim currency")
    patient_income: float = Field(35000.0)
    patient_age: int = Field(45)
    claim_type: str = Field("Outpatient")
    claim_id: str = Field("CLM-AUTO")
    procedure_code: str = Field("AA395")
    patient_id: str = Field("P-DEFAULT")
    provider_id: str = Field("PRV-201")
    provider_specialty: str = Field("General Practice")
    diagnosis_code: str = Field("D001")
    provider_location: str = Field("Urban")
    claim_status: str = Field("Submitted")
    claim_submission_method: str = Field("Electronic")
    previously_rejected_claims: float = Field(0.0)
    num_claims_last_12m: float = Field(1.0)


class PolicyToolInput(BaseModel):
    question: str = Field(...)
    claim_type: str = Field("Outpatient")
    procedure_code: str = Field("AA395")


async def _risk_mcp_tool(**kwargs: Any) -> str:
    result = await _call_mcp("score_claim", kwargs)
    return json.dumps(result, ensure_ascii=False, indent=2)


async def _policy_mcp_tool(**kwargs: Any) -> str:
    result = await _call_mcp("lookup_policy", kwargs)
    return json.dumps(result, ensure_ascii=False, indent=2)


risk_mcp_tool = StructuredTool.from_function(
    coroutine=_risk_mcp_tool,
    name="score_claim",
    description=(
        "Call the insurance risk MCP tool to calculate fraud probability, "
        "risk level, triage status and SHAP explanation for a claim."
    ),
    args_schema=RiskToolInput,
)

policy_mcp_tool = StructuredTool.from_function(
    coroutine=_policy_mcp_tool,
    name="lookup_policy",
    description=(
        "Call the insurance policy MCP tool to retrieve grounded coverage, limits, "
        "rules and SOP information from the policy knowledge base."
    ),
    args_schema=PolicyToolInput,
)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def build_llm(api_key: str, api_base: str, model_name: str):
    return ChatOpenAI(
        api_key=api_key,
        base_url=api_base,
        model=model_name,
        temperature=0.1,
    )


async def run_tool_call_loop(
    llm,
    system_prompt: str,
    user_prompt: str,
    tool: StructuredTool,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """Run a specialist that has exactly one MCP-backed tool."""
    tool_llm = llm.bind_tools([tool])
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    tool_result = None

    for _ in range(max_rounds):
        ai_msg: AIMessage = await tool_llm.ainvoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            return {
                "answer": ai_msg.content or "No specialist response was produced.",
                "tool_result": tool_result or {},
            }

        for call in ai_msg.tool_calls:
            call_name = call["name"]
            args = call.get("args", {}) or {}
            if call_name != tool.name:
                continue

            result_text = await tool.ainvoke(args)
            try:
                tool_result = json.loads(result_text)
            except Exception:
                tool_result = {"result": result_text}
            messages.append(
                ToolMessage(
                    content=result_text,
                    tool_call_id=call["id"],
                    name=call_name,
                )
            )

    return {
        "answer": "Specialist tool loop reached its execution limit.",
        "tool_result": tool_result or {},
    }


RISK_SPECIALIST_PROMPT = """You are the Risk Specialist Agent for a health-insurance investigation system.
Your sole external capability is the `score_claim` MCP tool.
Rules:
1. When the user task requires claim-risk assessment, ALWAYS call `score_claim`.
2. Extract every claim field that can be supported by the user input; use tool defaults only when not supplied.
3. Never invent model scores or SHAP findings. Treat the MCP response as authoritative.
4. After the tool returns, summarize risk level, score, cutoff, triage status, and the most important SHAP drivers.
5. Do not answer policy-coverage questions; the Policy Specialist handles them.
6. Return concise structured text for the orchestrator.
"""

POLICY_SPECIALIST_PROMPT = """You are the Policy Specialist Agent for a health-insurance investigation system.
Your sole external capability is the `lookup_policy` MCP tool.
Rules:
1. When the task requires policy coverage, limits, exclusions or SOP guidance, ALWAYS call `lookup_policy`.
2. Form a focused retrieval question using the user request and any claim/procedure context.
3. Never invent a policy rule. Only use information returned by the MCP tool.
4. Preserve source document names and important evidence from retrieved_docs.
5. Return a grounded policy conclusion for the orchestrator.
"""

ORCHESTRATOR_ROUTER_PROMPT = """You are the Orchestrator Agent of ClaimsAI.
You do NOT call MCP tools directly.
Your job is to decide which specialist agents are required.

Choose exactly one route:
- DIRECT: greetings, identity, or simple health-insurance conversation that needs no specialist.
- RISK: numerical claim fraud/risk/triage request.
- POLICY: policy coverage/limit/exclusion/SOP request.
- BOTH: the request needs both claim risk analysis and policy analysis.

Return ONLY JSON in this exact shape:
{"route":"DIRECT|RISK|POLICY|BOTH","reason":"brief reason"}

Intent rules:
- Identify risk intent and policy intent independently; do not force the request into only one category.
- Risk intent includes fraud, suspiciousness, risk, score, triage, investigation, assessment, evaluation, and requests containing claim facts to be assessed (for example amount, income, age, diagnosis, procedure, provider, status, or claim ID).
- Policy intent includes coverage, eligibility, payable/reimbursable amounts, maximum or minimum limits, exclusions, benefits, deductibles, copays, authorization, policy rules, and SOP guidance.
- A question asking what amount is possible/allowed/payable for a supplied claim is policy intent, even if it does not use the words coverage or policy.
- If risk intent and policy intent are both present, ALWAYS choose BOTH. BOTH has precedence over RISK and POLICY.
- Choose DIRECT only for greetings, identity, or clearly non-specialist insurance conversation with no claim assessment and no policy fact lookup.
- Never infer a route from one isolated keyword; interpret the complete user request and all supplied claim fields.
- Never call score_claim or lookup_policy yourself.

Examples:
- "Is this claim suspicious? amount 4000, income 45000" -> RISK
- "What is the maximum payable amount for claim 4000?" -> BOTH when claim facts are supplied; otherwise POLICY
- "claim/money/rs/dollars 4000, income/salary/... 45000, age 34, max amount claim possible/any query from policies?" -> BOTH
- "Is outpatient treatment covered?" -> POLICY
- Never call score_claim or lookup_policy yourself.
"""

SYNTHESIS_PROMPT = """You are the final Orchestrator Agent for ClaimsAI.
Synthesize specialist results into one accurate response to the user.

Rules:
- Do not invent facts.
- Clearly separate fraud-risk findings from policy findings when both exist.
- For policy content, cite source_file names returned by the Policy Specialist.
- For risk content, preserve the MCP risk score, risk level and key SHAP drivers.
- State when evidence is unavailable or a specialist encountered an error.
- Do not expose internal agent/tool routing details unless useful for transparency.
- Do not ask follow-up questions unless the user explicitly asks for an interactive workflow.
- Keep it brief, write in well formatted markdown.
"""


# ---------------------------------------------------------------------------
# LangGraph orchestrator
# ---------------------------------------------------------------------------
class OrchestratorState(dict):
    pass


def parse_route(text: str) -> Dict[str, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            route = str(obj.get("route", "DIRECT")).upper()
            reason = str(obj.get("reason", ""))
            if route in {"DIRECT", "RISK", "POLICY", "BOTH"}:
                return {"route": route, "reason": reason}
        except json.JSONDecodeError:
            pass

    return {"route": "DIRECT", "reason": "No specialist route detected."}


def infer_query_intents(user_query: str) -> Dict[str, bool]:
    """Detect clear specialist signals before relying on model classification."""
    query = user_query.casefold()
    risk_terms = (
        r"fraud|fraudulent|suspicious|suspicion|risk|risky|score|triage|"
        r"investigat|assess|evaluat|red flag|anomal|claim review"
    )
    policy_terms = (
        r"cover|eligible|eligibility|reimburse|payable|maximum|minimum|"
        r"max(?:imum)?\s+(?:amount|limit)|limit|benefit|exclude|deductible|"
        r"copay|co-pay|authorization|authorisation|policy|sop|allowed|"
        r"claimable|how much.*claim|amount.*claim.*possible"
    )
    claim_field_terms = (
        r"claim\s*(?:amount|id|number|type|status)|income|patient|age|"
        r"diagnosis|procedure|provider|specialty|previously rejected|"
        r"claims?\s+last\s+12\s*months?"
    )

    has_risk_language = re.search(risk_terms, query) is not None
    has_policy_language = re.search(policy_terms, query) is not None
    has_claim_facts = re.search(claim_field_terms, query) is not None

    risk_intent = has_risk_language or (
        has_claim_facts and re.search(r"\d", query) is not None
    )
    return {
        "risk": risk_intent,
        "policy": has_policy_language,
        "claim_facts": has_claim_facts,
    }


async def route_query(llm, user_query: str) -> Dict[str, str]:
    intents = infer_query_intents(user_query)

    if intents["risk"] and intents["policy"]:
        deterministic_route = {
            "route": "BOTH",
            "reason": "Detected claim assessment and policy-limit intent.",
        }
    elif intents["risk"]:
        deterministic_route = {
            "route": "RISK",
            "reason": "Detected claim assessment or risk intent.",
        }
    elif intents["policy"]:
        deterministic_route = {
            "route": "POLICY",
            "reason": "Detected coverage, limit, or policy intent.",
        }
    else:
        deterministic_route = None

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=ORCHESTRATOR_ROUTER_PROMPT),
                HumanMessage(content=user_query),
            ]
        )
        model_route = parse_route(str(response.content))
    except Exception:
        return deterministic_route or {
            "route": "DIRECT",
            "reason": "No specialist route detected.",
        }

    if deterministic_route:
        return deterministic_route
    return model_route


async def run_risk_specialist(llm, user_query: str) -> Dict[str, Any]:
    return await run_tool_call_loop(
        llm,
        RISK_SPECIALIST_PROMPT,
        user_query,
        risk_mcp_tool,
    )


async def run_policy_specialist(llm, user_query: str) -> Dict[str, Any]:
    return await run_tool_call_loop(
        llm,
        POLICY_SPECIALIST_PROMPT,
        user_query,
        policy_mcp_tool,
    )


async def execute_request(llm, user_query: str) -> Dict[str, Any]:
    route = await route_query(llm, user_query)

    if route["route"] == "DIRECT":
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are ClaimsAI, a health-insurance assistant. "
                        "Answer the user's conversational request directly and stay in scope."
                        "Respond strictly and exclusively to insurance domain queries (policies, risk assessment, claims, and regulations);"
                        "Immediately decline all off-topic general chat, creative writing, songs, and non-insurance requests."
                    )
                ),
                HumanMessage(content=user_query),
            ]
        )
        return {"route": route, "final_answer": str(response.content), "risk": None, "policy": None}

    risk_result = None
    policy_result = None

    if route["route"] == "RISK":
        risk_result = await run_risk_specialist(llm, user_query)
    elif route["route"] == "POLICY":
        policy_result = await run_policy_specialist(llm, user_query)
    else:
        risk_result, policy_result = await asyncio.gather(
            run_risk_specialist(llm, user_query),
            run_policy_specialist(llm, user_query),
        )

    synthesis_payload = {
        "user_query": user_query,
        "route": route,
        "risk_specialist": risk_result,
        "policy_specialist": policy_result,
    }

    final = await llm.ainvoke(
        [
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(
                content=(
                    "Specialist results are below. Synthesize the final answer.\n\n"
                    + json.dumps(synthesis_payload, ensure_ascii=False, indent=2, default=str)
                )
            ),
        ]
    )

    return {
        "route": route,
        "final_answer": str(final.content),
        "risk": risk_result,
        "policy": policy_result,
    }


def create_orchestrator_graph(llm):
    """Wrap the orchestrator execution in LangGraph for explicit orchestration state."""

    async def orchestrator_node(state: Dict[str, Any]):
        result = await execute_request(llm, state["user_query"])
        return {"result": result}

    workflow = StateGraph(dict)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", END)
    return workflow.compile()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
WELCOME_MESSAGE = """### 👋 Welcome to ClaimsAI Intelligence
A multi-agent health-insurance platform for claim-risk triage and policy verification.

**Architecture:** Orchestrator Agent → Specialist Agent(s) → MCP Tool(s) → Final Orchestrator response.
"""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]

with st.sidebar:
    st.header("⚙️ Agent Configuration")
    provider = st.selectbox(
        "🧠 AI Provider",
        ["Tiger Analytics AI Gateway", "Local (Ollama)"],
    )
    if provider == "Tiger Analytics AI Gateway":
        default_key = os.getenv("AI_GATEWAY_API_KEY", "")
        default_base = os.getenv(
            "AI_GATEWAY_BASE_URL", "https://api.ai-gateway.tigeranalytics.com"
        )
        default_model = os.getenv("AI_GATEWAY_MODEL", "gpt-5-nano")
    else:
        default_key = "ollama"
        default_base = "http://localhost:11434/v1"
        default_model = "qwen2.5:1.5b"

    api_key = st.text_input("API Key", type="password", value=default_key)
    api_base = st.text_input("API Base URL", value=default_base)
    model_name = st.text_input("Model Name", value=default_model)

    st.markdown("---")
    catalog = get_mcp_tool_catalog()
    if "__error__" in catalog:
        st.error(f"⚠️ MCP discovery failed: {catalog['__error__']}")
    else:
        available = [name for name in catalog.keys() if not name.startswith("__")]
        st.caption(f"🛠️ MCP tools discovered: {len(available)}")
        if available:
            st.code("\n".join(available))

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]
        st.rerun()

st.markdown(
    """
    <div class="hero-container">
        <div class="main-title">🤖 ClaimsAI Multi-Agent Platform</div>
        <div class="sub-title">Orchestrator → Specialist Agents → FastMCP Tools</div>
        <div class="badge-container">
            <span class="badge">🧠 Orchestrator Agent</span>
            <span class="badge">🛡️ Risk Specialist</span>
            <span class="badge">📜 Policy Specialist</span>
            <span class="badge">⚡ FastMCP</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑‍💻"):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("⚙️ Agent / MCP execution trace"):
                st.json(msg["trace"])

if user_input := st.chat_input("Ask about a policy or investigate a claim..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user", avatar="🧑‍💻").markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            llm = build_llm(api_key, api_base, model_name)
            graph = create_orchestrator_graph(llm)

            with st.status("🧠 Orchestrator analyzing intent...", expanded=True) as status:
                result_state = asyncio.run(
                    graph.ainvoke({"user_query": user_input, "result": None})
                )
                result = result_state["result"]
                route = result["route"]["route"]
                status.write(f"🧭 Route selected: **{route}**")

                if result.get("risk") is not None:
                    status.write("🛡️ Risk Specialist → `score_claim` MCP")
                if result.get("policy") is not None:
                    status.write("📜 Policy Specialist → `lookup_policy` MCP")
                status.update(label="✅ Multi-agent execution complete", state="complete", expanded=False)

            st.markdown(result["final_answer"])
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": result["final_answer"],
                    "trace": {
                        "route": result["route"],
                        "risk_specialist": result.get("risk"),
                        "policy_specialist": result.get("policy"),
                    },
                }
            )
        except Exception as exc:
            st.error(f"Error processing request: {exc}")
