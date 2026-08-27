import asyncio
import os
import subprocess
import time
from typing import Annotated, Any, Dict, TypedDict

import streamlit as st
from fastmcp import Client
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import signal

def clear_port(port: int):
    """Finds and terminates any lingering process running on the specified port."""
    try:
        # Find PIDs using lsof
        result = subprocess.check_output(["lsof", "-t", f"-i:{port}"]).decode().strip()
        if result:
            pids = result.split("\n")
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
            print(f"🧹 Successfully cleared port {port} (Killed PID(s): {', '.join(pids)})")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Port is already free
        pass

# Clear ports before initializing the UI and backend connections
clear_port(8501)

# =====================================================================
# CONFIGURATION & AUTO-START OLLAMA
# =====================================================================
MCP_SERVER_PORT = os.getenv("MCP_PORT", "8011")
MCP_SERVER_URL = f"http://127.0.0.1:{MCP_SERVER_PORT}/mcp"

st.set_page_config(
    page_title="Insurance Multi-Agent AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def start_ollama_background(model_name: str):
    """Starts Ollama in the background silently."""
    try:
        process = subprocess.Popen(
            ["ollama", "run", model_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.PIPE
        )
        time.sleep(3)
        return process
    except FileNotFoundError:
        st.error("⚠️ Ollama is not installed or not found in system PATH.")
        return None

# Ensure Ollama is running in background
ollama_process = start_ollama_background("qwen2.5:1.5b")

# =====================================================================
# ASYNC MCP CLIENT HELPER
# =====================================================================
async def _call_mcp(tool_name: str, payload: Dict[str, Any]) -> str:
    """Connects to FastMCP server and executes the requested tool."""
    try:
        async with Client(MCP_SERVER_URL) as client:
            mcp_payload = {"request": payload} if tool_name == "lookup_policy" else {"claim": payload}
            res = await client.call_tool(tool_name, mcp_payload)
            
            if hasattr(res, "content") and res.content:
                for item in res.content:
                    if hasattr(item, "text"):
                        return item.text
            return str(res)
    except Exception as exc:
        return f"Error executing {tool_name}: {str(exc)}"

# =====================================================================
# LANGCHAIN TOOLS (Wrappers for FastMCP)
# =====================================================================
@tool
def lookup_policy(
    question: str, 
    claim_amount: float = 1.0, 
    claim_type: str = "Outpatient", 
    procedure_code: str = "AA395",
    patient_id: str = "P-DEFAULT",
    patient_age: int = 45,
    patient_income: float = 35000.0
) -> str:
    """Search insurance policies, coverage limits, and rules for a given claim.
    Use this strictly if the user asks a question about policy terms, limits, or rules.
    """
    payload = {
        "question": question,
        "claim_amount": claim_amount,
        "claim_type": claim_type,
        "procedure_code": procedure_code,
        "patient_id": patient_id,
        "patient_age": patient_age,
        "patient_income": patient_income
    }
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(_call_mcp("lookup_policy", payload))
    loop.close()
    return result

@tool
def score_claim(
    claim_amount: float, 
    patient_id: str = "P-DEFAULT",
    claim_id: str = "CLM-AUTO",
    claim_type: str = "Outpatient", 
    procedure_code: str = "AA395",
    patient_age: int = 45,
    patient_income: float = 35000.0
) -> str:
    """Calculate fraud risk, model-based probability, and triage decisions for a medical claim.
    Always map claim_amount, patient_id, patient_age, and patient_income if provided.
    """
    payload = {
        "claim_id": claim_id,
        "patient_id": patient_id,
        "claim_amount": claim_amount,
        "claim_type": claim_type,
        "procedure_code": procedure_code,
        "patient_age": patient_age,
        "patient_income": patient_income
    }
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(_call_mcp("score_claim", payload))
    loop.close()
    return result

tools = [lookup_policy, score_claim]
tool_node = ToolNode(tools)

# =====================================================================
# LANGGRAPH STATE & WORKFLOW
# =====================================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def create_agent_graph(llm):
    llm_with_tools = llm.bind_tools(tools)
    
    def call_model(state: AgentState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# =====================================================================
# SYSTEM PROMPT & INITIALIZATION
# =====================================================================
SYSTEM_PROMPT = """You are an Intelligent Insurance Claims & Policy AI Assistant.

ROUTING RULES:
1. Conversational Chat: If the user says "hi", asks general questions, or makes small talk, DO NOT use any tools. Respond conversationally using your internal knowledge.
2. Fraud/Risk Assessment: If the user provides numerical claim details (e.g., claim amount, patient ID, age, income) or asks for fraud triage, call `score_claim` ONLY.
3. Policy Verification: If the user asks explicitly about insurance rules, coverage limits, or SOPs, call `lookup_policy` ONLY.
4. Never ask for missing details like procedure codes or claim IDs—rely on tool defaults automatically.

Synthesize tool outputs into concise, professional, human-readable answers."""

# Welcome Message Setup
INITIAL_ASSISTANT_MESSAGE = (
    "Hi, I am your AI Insurance Claims & Policy Assistant! 🤖\n\n"
    "I am powered by a local Qwen model and a multi-agent backend. I can help you with:\n"
    "* 💬 **Conversational Support**: Answering general insurance questions.\n"
    "* 🛡️ **Fraud Risk Triage**: Scoring claims and explaining risk via SHAP analysis.\n"
    "* 📜 **Policy Guidance**: Retrieving policy limits and operational SOPs.\n\n"
    "How can I assist you with your claims or policy queries today?"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        AIMessage(content=INITIAL_ASSISTANT_MESSAGE)
    ]

# =====================================================================
# UI LAYOUT
# =====================================================================
with st.sidebar:
    st.header("⚙️ Agent Configuration")
    api_key = st.text_input("API Key", type="password", value="ollama")
    api_base = st.text_input("API Base URL", value="http://localhost:11434/v1")
    model_name = st.text_input("Model Name", value="qwen2.5:1.5b")
    st.markdown("---")
    st.caption(f"🔗 FastMCP Endpoint: `{MCP_SERVER_URL}`")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            AIMessage(content=INITIAL_ASSISTANT_MESSAGE)
        ]
        st.rerun()

st.title("🤖 Intelligent Insurance Agent")
st.caption("Orchestrated via LangGraph, FastMCP, and Local Qwen 2.5")

# Render existing chat history
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.content:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg.content)
    elif isinstance(msg, ToolMessage):
        with st.expander(f"⚙️ Executed Tool: {msg.name}"):
            st.code(msg.content, language="json")

# =====================================================================
# CHAT EXECUTION LOOP
# =====================================================================
if user_input := st.chat_input("Ask about a policy, analyze a claim, or chat..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        status_box = st.empty()
        status_box.markdown("💭 *Thinking...*")

        try:
            llm = ChatOpenAI(
                api_key=api_key, 
                base_url=api_base, 
                model=model_name,
                temperature=0.1
            )
            app = create_agent_graph(llm)

            final_text = ""
            for event in app.stream({"messages": st.session_state.messages}, stream_mode="values"):
                last_message = event["messages"][-1]
                
                if isinstance(last_message, AIMessage):
                    if last_message.tool_calls:
                        tool_names = ", ".join([tc["name"] for tc in last_message.tool_calls])
                        status_box.markdown(f"⚡ *Routing to tool: `{tool_names}`...*")
                    elif last_message.content:
                        final_text = last_message.content
                        status_box.markdown(final_text)

            st.session_state.messages = event["messages"]
            
        except Exception as e:
            status_box.error(f"Error executing agent request: {str(e)}")