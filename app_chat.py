import asyncio
import os
from typing import Annotated, Any, Dict, TypedDict

import streamlit as st
from fastmcp import Client
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import create_model

# =====================================================================
# CONFIGURATION & FAST-MCP SETUP
# =====================================================================
MCP_SERVER_PORT = os.getenv("MCP_PORT", "8011")
MCP_SERVER_URL = f"http://127.0.0.1:{MCP_SERVER_PORT}/mcp"

st.set_page_config(
    page_title="ClaimsAI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# MODERN CUSTOM CSS (CENTERED TYPOGRAPHY & CLEAN CARDS)
# =====================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        text-align: center;
        padding: 1rem 1rem 0.5rem 1rem;
        margin-bottom: 1rem;
    }
    
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3C72 0%, #2A5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #555E6C;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 0.8rem;
    }
    
    .badge {
        background-color: #EBF3FE;
        color: #1E3C72;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid #C6DCFA;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================================
# DYNAMIC MCP TOOL DISCOVERY
# =====================================================================
async def _call_mcp(tool_name: str, payload: Dict[str, Any]) -> str:
    try:
        async with Client(MCP_SERVER_URL) as client:
            res = await client.call_tool(tool_name, payload)
            if hasattr(res, "content") and res.content:
                for item in res.content:
                    if hasattr(item, "text"):
                        return item.text
            return str(res)
    except Exception as exc:
        return f"Error executing {tool_name}: {str(exc)}"

def build_pydantic_schema(tool_name: str, input_schema: dict):
    fields = {}
    props = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    type_map = {"string": str, "number": float, "integer": int, "boolean": bool}
    
    for field_name, field_info in props.items():
        field_type = type_map.get(field_info.get("type"), Any)
        default_val = ... if field_name in required else field_info.get("default", None)
        fields[field_name] = (field_type, default_val)
        
    return create_model(f"{tool_name}_input", **fields)

def load_dynamic_mcp_tools() -> list:
    tools_list = []
    
    async def fetch_tools():
        async with Client(MCP_SERVER_URL) as client:
            return await client.list_tools()
            
    try:
        mcp_tools = asyncio.run(fetch_tools())
        for mcp_tool in mcp_tools:
            name = mcp_tool.name
            desc = mcp_tool.description
            schema_dict = getattr(mcp_tool, "inputSchema", {}) or getattr(mcp_tool, "parameters", {})
            args_schema = build_pydantic_schema(name, schema_dict)

            def make_runner(tool_name):
                def runner(**kwargs):
                    return asyncio.run(_call_mcp(tool_name, kwargs))
                return runner

            tools_list.append(
                StructuredTool.from_function(
                    func=make_runner(name),
                    name=name,
                    description=desc,
                    args_schema=args_schema
                )
            )
    except Exception as e:
        st.sidebar.error(f"⚠️ Dynamic tool discovery failed: {e}")
        
    return tools_list

tools = load_dynamic_mcp_tools()
tool_node = ToolNode(tools)

# =====================================================================
# LANGGRAPH ENGINE
# =====================================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def create_agent_graph(llm):
    llm_with_tools = llm.bind_tools(tools)
    def call_model(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    def should_continue(state: AgentState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    return workflow.compile()

# =====================================================================
# SYSTEM PROMPT
# =====================================================================
SYSTEM_PROMPT = """You are ClaimsAI, an enterprise health insurance & policy intelligence assistant.

### 1. DUAL INTENT ROUTING MANDATES
- **Dual-Intent Queries**: If a query asks about policy limits/coverage AND fraud risk/escalation, YOU MUST INVOKE BOTH TOOLS:
  1. `score_claim` to assess fraud risk and triage status.
  2. `lookup_policy` to retrieve policy coverage limits and SOP guidelines.
- **Single Policy Queries**: Call `lookup_policy` for policy coverage rules, limits, or SOPs.
- **Single Triage Queries**: Call `score_claim` when numerical claim amounts are provided for risk scoring.
- **General Chat**: Respond directly without tools ONLY for greetings ("hi") or identity queries ("who are you").

### 2. STRICT SCOPE GUARDRAIL
- Decline non-health-insurance queries in one sentence: "I am specialized exclusively in health insurance policy guidance and claim risk triage."

### 3. OUTPUT & CITATION FORMATTING
- **Synthesize Both Tools**: If both tools were called, present the output in two clear sections: **🛡️ Fraud Risk Triage** and **📜 Policy Coverage & Limits**.
- **Citations**: Include brief document citations for policy lookups (e.g., `📁 Source: [Document / Clause Name]`, top 3 sources.).
- **No Unsolicited Offers**: NEVER append follow-up questions or offers (e.g., "Would you like me to...")."""

WELCOME_MESSAGE = """### 👋 Welcome to ClaimsAI Intelligence
Your multi-agent platform for health insurance risk scoring and policy verification.

* 🛡️ **Fraud Risk Scoring:** Input claim amounts and patient financial details for ML triage and SHAP explanations.
* 📜 **Policy Guidance:** Query policy rules, coverage limits, and operational SOPs.
* 💬 **Direct Support:** Ask general health insurance questions anytime.
"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        AIMessage(content=WELCOME_MESSAGE)
    ]

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("⚙️ Agent Configuration")
    provider = st.selectbox("🧠 AI Provider", ["Tiger Analytics AI Gateway", "Local (Ollama)"])
    if provider == "Tiger Analytics AI Gateway":
        default_key, default_base, default_model = "sk-samplekey123", "https://api.ai-gateway.tigeranalytics.com", "gpt-5-nano"
    else:
        default_key, default_base, default_model = "ollama", "http://localhost:11434/v1", "qwen2.5:1.5b"

    api_key = st.text_input("API Key", type="password", value=default_key)
    api_base = st.text_input("API Base URL", value=default_base)
    model_name = st.text_input("Model Name", value=default_model)
    st.markdown("---")
    st.caption(f"🛠️ Auto-Discovered Tools: `{len(tools)} loaded`")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            AIMessage(content=WELCOME_MESSAGE)
        ]
        st.rerun()

# =====================================================================
# CENTER-ORIENTED HEADER
# =====================================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="main-title">🤖 ClaimsAI Multi-Agent Platform</div>
        <div class="sub-title">Automated Risk Triage & Policy Intelligence</div>
        <div class="badge-container">
            <span class="badge">FastMCP Auto-Discovery</span>
            <span class="badge">SHAP Explainability</span>
            <span class="badge">RAG Vector Engine</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Render Chat History
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    elif isinstance(msg, HumanMessage):
        st.chat_message("user", avatar="🧑‍💻").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.content:
            st.chat_message("assistant", avatar="🤖").markdown(msg.content)
    elif isinstance(msg, ToolMessage):
        with st.expander(f"⚙️ FastMCP Tool Execution Trace: `{msg.name}`"):
            st.code(msg.content, language="json")

# =====================================================================
# SYNCHRONIZED REAL-TIME CHAT EXECUTION LOOP
# =====================================================================
if user_input := st.chat_input("Ask about a policy or score a claim..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user", avatar="🧑‍💻").markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        status_container = st.container()
        text_placeholder = st.empty()
        
        try:
            llm = ChatOpenAI(api_key=api_key, base_url=api_base, model=model_name, temperature=0.1)
            app = create_agent_graph(llm)

            final_content = ""
            executed_tools = []

            # Immediate UI feedback upon submitting query
            with status_container.status("🧠 **Analyzing intent & routing query...**", expanded=True) as status_box:
                for event in app.stream({"messages": st.session_state.messages}, stream_mode="values"):
                    last_msg = event["messages"][-1]
                    
                    # 1. Tool Call Triggered
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            t_name = tc["name"]
                            if t_name not in executed_tools:
                                executed_tools.append(t_name)
                                status_box.write(f"⚙️ Executing `{t_name}` on FastMCP backend...")
                        
                        status_box.update(
                            label=f"⚡ **Running Agent Tools: `{', '.join(executed_tools)}`...**", 
                            state="running", 
                            expanded=True
                        )

                    # 2. Tool Execution Finished -> Transitioning to Synthesis
                    elif isinstance(last_msg, ToolMessage):
                        status_box.write(f"✅ FastMCP `{last_msg.name}` step complete. Synthesizing final response...")
                        status_box.update(
                            label=f"🧠 **Synthesizing multi-agent analysis...**", 
                            state="running", 
                            expanded=True
                        )

                    # 3. Final Content Received
                    elif isinstance(last_msg, AIMessage) and last_msg.content:
                        final_content = last_msg.content
                        text_placeholder.markdown(final_content)

                # Close status box ONLY AFTER the entire streaming loop finishes and text is rendered
                if executed_tools:
                    status_box.update(
                        label=f"✅ **Execution Complete ({len(executed_tools)} tool{'s' if len(executed_tools)>1 else ''} used)**", 
                        state="complete", 
                        expanded=False
                    )
                else:
                    # Clear status box for pure chat queries
                    status_container.empty()

            st.session_state.messages = event["messages"]
            
        except Exception as e:
            status_container.empty()
            text_placeholder.error(f"Error processing request: {str(e)}")