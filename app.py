import os
import re
import json
import warnings
import pandas as pd
import numpy as np
import streamlit as st

# Import your multi-agent class from your existing code script
# (Assuming your backend script is named insurance_multi_agent_chunking_indexing_copy_2.py)
try:
    from insurance_multi_agent_chunking_indexing_copy_2 import InsuranceAgentSystem
except ImportError:
    st.error("⚠️ Backend module not found. Ensure `insurance_multi_agent_chunking_indexing_copy_2.py` is in the same directory.")
    st.stop()

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================================================================
# STREAMLIT PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Multi-Agent Claim Investigator",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .card { background-color: #F8FAFC; border-radius: 10px; padding: 1.2rem; border: 1px solid #E2E8F0; margin-bottom: 1rem; }
    .metric-title { font-weight: 600; font-size: 0.9rem; color: #64748B; }
    .high-risk { color: #DC2626; font-weight: 700; }
    .low-risk { color: #16A34A; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)


# =====================================================================
# CACHED SYSTEM INITIALIZATION
# =====================================================================
@st.cache_resource(show_spinner="🚀 Initializing Multi-Agent RAG & ML System (Loading LLM, Embeddings, Weaviate)...")
def load_investigator_system():
    return InsuranceAgentSystem(use_local_llm=True)

try:
    system = load_investigator_system()
except Exception as e:
    st.error(f"Failed to load backend system: {str(e)}")
    st.stop()


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield-search.png", width=70)
    st.title("System Status")
    
    st.success("🟢 ML Fraud Model Loaded")
    st.success("🟢 Weaviate Knowledge Base Ready")
    st.success("🟢 Local Qwen2.5-1.5B LLM Ready")
    
    st.divider()
    st.markdown("**Model Parameters:**")
    
    # Safe attribute access with fallbacks
    threshold = getattr(system, 'optimal_threshold', 0.9803)
    global_mean = getattr(system, 'global_mean', 5014.20)
    proc_map = getattr(system, 'procedure_avg_map', {})
    spec_map = getattr(system, 'specialty_avg_map', {})

    st.write(f"• **Decision Threshold:** `{threshold:.2%}`")
    st.write(f"• **Global Dataset Mean:** `${global_mean:,.2f}`")
    st.write(f"• **Indexed Procedures:** `{len(proc_map)}`")
    st.write(f"• **Indexed Specialties:** `{len(spec_map)}`")
    
    st.divider()
    if st.button("🔄 Clear System Cache / Reset", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()


# =====================================================================
# HEADER
# =====================================================================
st.markdown('<div class="main-header">🚀 Dynamic Multi-Agent Claim Investigator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production RAG & Tabular ML System for Insurance Fraud Triage, Explainability & Policy QA</div>', unsafe_allow_html=True)


# =====================================================================
# INPUT INTERFACE (TABS)
# =====================================================================
tab1, tab2 = st.tabs(["💬 Single-Prompt Investigator", "📝 Structured Claim Form"])

claim_payload = {}
user_query = ""

with tab1:
    st.markdown("Enter a natural language description of a claim or a general policy question.")
    prompt_input = st.text_area(
        "Claim Description / Policy Question:",
        placeholder="e.g., 'claim $10,000 for pedia, income 80000, age 21, outpatient' or 'What are the annual coverage limits for inpatient stay?'",
        height=100
    )
    
    if prompt_input:
        user_query = prompt_input.strip()
        # Parse claim details from text
        amt_match = re.search(r'(?:[\$₹]\s*|\b)(\d[\d,]*)(?:\s*k\b|\b)', user_query, re.IGNORECASE)
        if amt_match:
            try:
                raw_val = amt_match.group(1).replace(',', '')
                val = float(raw_val)
                if 'k' in amt_match.group(0).lower() and val < 1000:
                    val *= 1000
                if val > 0:
                    claim_payload["ClaimAmount"] = val
            except ValueError:
                pass

        if claim_payload.get("ClaimAmount", 0) > 0:
            code, spec = system.resolve_procedure_code(user_query)
            claim_payload["ProcedureCode"] = code
            claim_payload["ProviderSpecialty"] = spec
            
            # Extract Income
            inc_match = re.search(r'(?:income|salary|earning)[^\d]*([\$₹]?\s*\d[\d,]*)', user_query, re.IGNORECASE)
            if inc_match:
                try:
                    claim_payload["PatientIncome"] = float(inc_match.group(1).replace('$', '').replace(',', ''))
                except ValueError:
                    claim_payload["PatientIncome"] = 35000.0
            else:
                claim_payload["PatientIncome"] = 35000.0

            # Extract Age
            age_match = re.search(r'(\d{1,2})\s*(?:years old|yo|age)', user_query, re.IGNORECASE)
            if age_match:
                try:
                    claim_payload["PatientAge"] = float(age_match.group(1))
                except ValueError:
                    claim_payload["PatientAge"] = 45.0
            else:
                claim_payload["PatientAge"] = 45.0

            claim_payload["ClaimType"] = "Inpatient" if "inpatient" in user_query.lower() else "Outpatient"

with tab2:
    st.markdown("Fill in specific claim attributes directly for precise machine learning evaluation.")
    c1, c2, c3 = st.columns(3)
    with c1:
        f_amt = st.number_input("Claim Amount ($)", min_value=0.0, value=10000.0, step=500.0)
        f_type = st.selectbox("Claim Type", ["Outpatient", "Inpatient"])
    with c2:
        f_spec = st.text_input("Procedure / Specialty", value="Pediatrics")
        f_income = st.number_input("Patient Annual Income ($)", min_value=0.0, value=80000.0, step=5000.0)
    with c3:
        f_age = st.number_input("Patient Age", min_value=0, max_value=120, value=21)
        f_pid = st.text_input("Patient ID", value="PAT-101")

    f_query = st.text_input("Policy Question (Optional):", value="What are the coverage limits for this procedure?")

    if st.button("Submit Form Claim", type="secondary"):
        code, spec = system.resolve_procedure_code(f_spec)
        claim_payload = {
            "ClaimAmount": f_amt,
            "ClaimType": f_type,
            "ProcedureCode": code,
            "ProviderSpecialty": spec,
            "PatientIncome": f_income,
            "PatientAge": f_age,
            "PatientID": f_pid
        }
        user_query = f_query

# =====================================================================
# INVESTIGATION EXECUTION & DASHBOARD DISPLAY
# =====================================================================
st.divider()

if st.button("🚀 Run Investigation Pipeline", type="primary", use_container_width=True):
    if not user_query and not claim_payload:
        st.warning("Please enter a query or claim details to investigate.")
    else:
        with st.spinner("Multi-Agent System Processing (Router ➔ ML Triage ➔ SHAP ➔ Hybrid RAG)..."):
            # Execute LangGraph Pipeline
            initial_state = {
                "raw_claim": claim_payload,
                "user_query": user_query,
                "target_route": "",
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
            results = system.graph.invoke(initial_state)

        # -------------------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------------------
        st.subheader("🕵️ Investigation Results")

        # AGENT 1 & 2: ML TRIAGE & RISK EXPLAINABILITY (Only if Claim Evaluated)
        if results.get("triage_status"):
            st.markdown("### Agent 1 & 2: ML Triage & Risk Analysis")
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                prob = results["ml_probability"]
                cutoff = system.optimal_threshold
                
                if prob >= cutoff:
                    st.error(f"### {results['triage_status']}")
                else:
                    st.success(f"### {results['triage_status']}")

                st.metric("Fraud Probability Score", f"{prob:.2%}", delta=f"{prob - cutoff:+.2%} vs Cutoff")
                st.progress(min(prob, 1.0))
                
                disp = results.get("dispersion_metrics", {})
                if disp:
                    st.markdown("**Statistical Dispersion Metrics:**")
                    st.write(f"• **Code/Specialty:** `{disp.get('procedure_code')}` ({disp.get('provider_specialty')})")
                    st.write(f"• **Dataset Peer Mean:** `${disp.get('peer_avg'):,.2f}`")
                    st.write(f"• **Peer Deviation:** `{disp.get('peer_deviation_pct'):+.2f}%` ({claim_payload.get('ClaimAmount',0)/disp.get('peer_avg',1):.2f}x mean)")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Agent 2: SHAP Feature Attributions")
                st.text(results.get("risk_explanation", "No SHAP explanation available."))
                st.markdown('</div>', unsafe_allow_html=True)

        # AGENT 3: POLICY AGENT (TWO-STAGE HYBRID RAG)
        st.markdown("### Agent 3: Policy RAG & Guidance")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(results.get("rag_generated_answer", "No policy answer generated."))
        st.markdown('</div>', unsafe_allow_html=True)

        # REFERENCED DOCUMENTS ACCORDION
        docs = results.get("retrieved_docs", [])
        if docs:
            with st.expander("📑 View Referenced Policy Documents & Cross-Encoder Scores", expanded=False):
                for idx, doc in enumerate(docs, 1):
                    st.markdown(f"**Document #{idx}: {doc['source_file']}** (Category: `{doc['category']}`)")
                    st.progress(float(doc['rerank_score']) / 100.0)
                    st.caption(f"Cross-Encoder Match Score: {doc['rerank_score']:.2f}% | Hybrid Search Score: {doc['hybrid_score']:.4f}")
                    st.info(f'"{doc["snippet"]}"')
                    st.divider()