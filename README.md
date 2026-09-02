# Multi-Agent Insurance MCP Architecture

![alt text](image-1.png)
![alt text](image.png)
![alt text](image-2.png)
![alt text](image-3.png)
![alt text](image-4.png)

<video controls src="streamlit-app_chat-2026-09-02-20-08-31.webm" title="Title"></video>

An end-to-end **Insurance Claims Risk & Investigation Assistant** that combines:

* Multi-agent orchestration with **LangGraph**
* Specialized **Risk** and **Policy** agents
* **FastMCP** capability services
* ML-based fraud detection
* **SHAP** explainability
* Hybrid **RAG** with Weaviate
* Cross-encoder reranking
* Local **Qwen** grounded generation
* Streamlit-based chat interface

The system follows a clear architectural separation:

> **Orchestrator Agent decides → Specialist Agent solves → MCP Tool invokes → Backend Capability executes → Specialist interprets → Orchestrator synthesizes**

---

# 1. Architecture

```mermaid
flowchart LR

    USER["👤 User"]
    UI["🖥️ Streamlit<br/>app_chat.py"]

    ORCH["🧠 Orchestrator Agent<br/><br/>Understands intent<br/>Routes work<br/>Coordinates agents<br/>Synthesizes answer"]

    RISK["🛡️ Risk Specialist Agent<br/><br/>Handles fraud-risk tasks"]
    POLICY["📚 Policy Specialist Agent<br/><br/>Handles coverage/policy tasks"]

    RMCP["Risk MCP<br/><b>score_claim</b>"]
    PMCP["Policy MCP<br/><b>lookup_policy</b>"]

    RISKENG["⚙️ Fraud Risk Capability<br/><br/>ML Model<br/>Historical Baselines<br/>SHAP"]

    POLICYENG["📚 Policy RAG Capability<br/><br/>Weaviate Hybrid Search<br/>Cross-Encoder<br/>Qwen"]

    FINAL["📋 Final Orchestrated Response"]

    USER --> UI
    UI --> ORCH

    ORCH --> RISK
    ORCH --> POLICY

    RISK -->|score_claim| RMCP
    POLICY -->|lookup_policy| PMCP

    RMCP --> RISKENG
    PMCP --> POLICYENG

    RISKENG -->|Risk Result| RISK
    POLICYENG -->|Policy Result| POLICY

    RISK --> ORCH
    POLICY --> ORCH

    ORCH --> FINAL
    FINAL --> UI

    classDef user fill:#4F46E5,color:#fff,stroke:#312E81,stroke-width:2px;
    classDef client fill:#E0F2FE,color:#0C4A6E,stroke:#0284C7,stroke-width:2px;
    classDef orch fill:#EDE9FE,color:#4C1D95,stroke:#7C3AED,stroke-width:2px;
    classDef risk fill:#DCFCE7,color:#14532D,stroke:#16A34A,stroke-width:2px;
    classDef policy fill:#FEF3C7,color:#78350F,stroke:#D97706,stroke-width:2px;
    classDef mcp fill:#DBEAFE,color:#1E3A8A,stroke:#2563EB,stroke-width:2px;
    classDef backend fill:#FEE2E2,color:#7F1D1D,stroke:#DC2626,stroke-width:2px;
    classDef backend2 fill:#FFF7ED,color:#9A3412,stroke:#EA580C,stroke-width:2px;
    classDef final fill:#F3E8FF,color:#581C87,stroke:#9333EA,stroke-width:2px;

    class USER user;
    class UI client;
    class ORCH orch;
    class RISK risk;
    class POLICY policy;
    class RMCP,PMCP mcp;
    class RISKENG backend;
    class POLICYENG backend2;
    class FINAL final;
```

---

# 2. Architectural Principles

The architecture separates **reasoning**, **tool access**, and **domain execution**.

## Orchestrator Agent

The Orchestrator is the central decision-maker.

It is responsible for:

* Understanding the user's request
* Determining whether the request is:

  * `DIRECT`
  * `RISK`
  * `POLICY`
  * `BOTH`
* Selecting the appropriate specialist agent(s)
* Running Risk and Policy specialists in parallel when both are required
* Receiving specialist results
* Producing the final answer

The Orchestrator **does not directly execute the domain capabilities**.

---

## Risk Specialist Agent

The Risk Specialist handles:

* Fraud detection
* Claim risk scoring
* Triage
* Risk explanation
* SHAP interpretation

It receives:

* User request
* Claim details
* Relevant claim context

It calls:

```text
score_claim
```

through the Risk MCP service.

---

## Policy Specialist Agent

The Policy Specialist handles:

* Coverage questions
* Policy limits
* Exclusions
* Procedure-related rules
* Policy/SOP guidance

It receives:

* User policy question
* Claim type
* Procedure information
* Optional claim context

It calls:

```text
lookup_policy
```

through the Policy MCP service.

---

# 3. Runtime Request Flow

## 3.1 Risk-only request

Example:

> Is this $25,000 claim suspicious?

```mermaid
flowchart LR

    A["User"] --> B["Streamlit"]
    B --> C["Orchestrator"]
    C --> D["Risk Specialist"]
    D --> E["Risk MCP<br/>score_claim"]
    E --> F["Fraud ML + SHAP"]
    F --> D
    D --> C
    C --> G["Final Response"]
```

---

## 3.2 Policy-only request

Example:

> Is outpatient treatment covered under the policy?

```mermaid
flowchart LR

    A["User"] --> B["Streamlit"]
    B --> C["Orchestrator"]
    C --> D["Policy Specialist"]
    D --> E["Policy MCP<br/>lookup_policy"]
    E --> F["Hybrid RAG"]
    F --> G["Cross-Encoder"]
    G --> H["Qwen"]
    H --> D
    D --> C
    C --> I["Final Response"]
```

---

## 3.3 Dual-intent request

Example:

> Assess whether this claim is fraudulent and tell me whether the treatment is covered.

The Orchestrator can invoke both specialists independently.

```mermaid
flowchart TD

    A["User Request"] --> B["Orchestrator"]

    B --> C["Risk Specialist"]
    B --> D["Policy Specialist"]

    C --> E["Risk MCP<br/>score_claim"]
    E --> F["Fraud ML + SHAP"]
    F --> C

    D --> G["Policy MCP<br/>lookup_policy"]
    G --> H["Hybrid RAG + Reranking + Qwen"]
    H --> D

    C --> B
    D --> B

    B --> I["Final Investigation Response"]
```

This allows the two specialist paths to execute concurrently when appropriate.

---

# 4. Agent Routing

The Orchestrator uses four logical routes.

```mermaid
flowchart TD

    A["User Request"] --> B{"Determine Intent"}

    B -->|General in-scope conversation| C["DIRECT"]
    B -->|Fraud / risk / triage| D["RISK"]
    B -->|Coverage / policy / SOP| E["POLICY"]
    B -->|Risk + policy required| F["BOTH"]

    C --> G["Respond Directly"]
    D --> H["Risk Specialist"]
    E --> I["Policy Specialist"]
    F --> J["Risk + Policy Specialists"]
```

Examples:

| User request                                               | Route    |
| ---------------------------------------------------------- | -------- |
| `"Hi"`                                                     | `DIRECT` |
| `"Is this claim suspicious?"`                              | `RISK`   |
| `"What is the outpatient coverage limit?"`                 | `POLICY` |
| `"Is this claim suspicious and is the treatment covered?"` | `BOTH`   |

---

# 5. Specialist Agent Responsibilities

```mermaid
flowchart LR

    O["Orchestrator"]

    R["Risk Specialist"]
    P["Policy Specialist"]

    RT["score_claim"]
    PT["lookup_policy"]

    RE["Risk Capability"]
    PE["Policy Capability"]

    O --> R
    O --> P

    R --> RT
    RT --> RE

    P --> PT
    PT --> PE

    RE --> R
    PE --> P

    R --> O
    P --> O
```

### Risk Specialist

**Input**

```text
User task
+
Claim context
```

**Action**

```text
Call score_claim
Interpret ML + SHAP result
```

**Output**

```text
Risk level
Risk score
Decision cutoff
Triage status
Risk explanation
```

### Policy Specialist

**Input**

```text
Policy question
+
Claim type
+
Procedure/context
```

**Action**

```text
Call lookup_policy
Interpret retrieved evidence
```

**Output**

```text
Grounded policy answer
Relevant documents
Source information
Reranking information
```

---

# 6. MCP Architecture

MCP is the **capability access layer**.

```mermaid
flowchart LR

    A["Specialist Agent"]

    B["FastMCP"]
    C["MCP Tool"]

    D["Backend Capability"]

    A --> B
    B --> C
    C --> D
    D --> C
    C --> B
    B --> A
```

MCP is responsible for:

* Tool discovery
* Tool schemas
* Input validation
* Transport
* Capability invocation
* Structured responses

MCP is **not responsible for orchestration**.

The MCP server does not decide:

* Which agent should run
* Which specialist should run next
* Whether Risk or Policy is required
* How the final response should be synthesized

Those decisions belong to the agents.

---

# 7. Risk Capability

The Risk MCP exposes:

```text
score_claim
```

The underlying capability performs:

```mermaid
flowchart LR

    A["Claim Input"]
    B["Feature Engineering"]
    C["Historical Baselines"]
    D["Fraud ML Model"]
    E["Fraud Probability"]
    F["Decision Threshold"]
    G["SHAP Explainability"]
    H["Risk Result"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
```

The risk capability uses information such as:

* Claim amount
* Patient income
* Patient age
* Claim type
* Procedure code
* Patient history
* Provider history
* Provider specialty
* Historical claim averages
* Peer deviation
* Previous rejected claims
* Claim frequency

---

# 8. Risk MCP Response

Conceptual response:

```json
{
  "claim_id": "CLM-123",
  "risk_level": "HIGH_RISK",
  "risk_score": 0.87,
  "decision_cutoff": 0.42,
  "triage_status": "HIGH RISK (SUSPICIOUS)",
  "risk_explanation": "Key model feature impact..."
}
```

The actual values are generated by the trained fraud model and SHAP engine.

---

# 9. Policy Capability

The Policy MCP exposes:

```text
lookup_policy
```

The policy capability performs:

```mermaid
flowchart LR

    A["Policy Question"]
    B["SentenceTransformer"]
    C["Weaviate Hybrid Search"]
    D["Candidate Documents"]
    E["Cross-Encoder Reranking"]
    F["Relevant Passages"]
    G["Qwen Grounded Generation"]
    H["Policy Result"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
```

The policy retrieval system combines:

* Dense semantic retrieval
* BM25 / keyword retrieval
* Cross-encoder reranking
* Grounded LLM generation

The knowledge collection is:

```text
InsuranceKnowledge
```

---

# 10. Policy MCP Response

Conceptual response:

```json
{
  "question": "Is outpatient treatment covered?",
  "answer": "Grounded policy answer...",
  "retrieved_docs": [
    {
      "source_file": "coverage_rules.pdf",
      "category": "Outpatient",
      "snippet": "...",
      "rerank_score": 91.4
    }
  ]
}
```

Policy answers should be grounded in retrieved policy evidence.

---

# 11. Final Response Synthesis

Specialist outputs return to the Orchestrator.

```mermaid
flowchart TD

    A["Risk Specialist Result"]
    B["Policy Specialist Result"]

    C["Orchestrator"]

    D["Final Investigation Response"]

    A --> C
    B --> C
    C --> D
```

For a dual-intent investigation, the final answer can contain:

```text
Fraud Risk Triage
-----------------
Risk Level: HIGH_RISK
Risk Score: 87%

Key Risk Drivers:
...

Policy Coverage & Limits
------------------------
Coverage:
...

Sources:
- coverage_rules.pdf
- policy_guidelines.pdf
```

---

# 12. Technology Stack

| Layer               | Technology                     |
| ------------------- | ------------------------------ |
| UI                  | Streamlit                      |
| Agent orchestration | LangGraph                      |
| Agent LLM           | OpenAI-compatible LLM / Ollama |
| MCP framework       | FastMCP                        |
| Fraud model         | XGBoost / LightGBM             |
| Explainability      | SHAP                           |
| Feature engineering | Pandas / NumPy                 |
| Embeddings          | Sentence Transformers          |
| Vector database     | Weaviate                       |
| Retrieval           | Hybrid Dense + BM25            |
| Reranking           | Cross-Encoder                  |
| Policy generation   | Qwen2.5-1.5B-Instruct          |
| Model artifacts     | Joblib                         |
| Environment         | Python / Docker                |

---

# 13. Repository Structure

```text
AgenticAI_training/
│
├── Health Insurance Fraud Claims.xlsx
│
├── train_fraud_model.py
├── insurance_multi_agent_chunking_indexing.py
├── insurance_mcp_server.py
├── app_chat.py
├── generate_rag_pdfs.py
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── rag/
│   └── documents/
│       └── *.pdf
│
└── output/
    ├── fraud_detection_model.pkl
    └── processed_claim_features.csv
```

---

# 14. File Responsibilities

| File                                         | Responsibility                                                                                        |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `app_chat.py`                                | Streamlit UI, Orchestrator Agent, Risk Specialist, Policy Specialist, MCP clients and final synthesis |
| `insurance_mcp_server.py`                    | FastMCP server exposing `score_claim` and `lookup_policy`                                             |
| `insurance_multi_agent_chunking_indexing.py` | Fraud and policy capability implementations                                                           |
| `train_fraud_model.py`                       | Fraud-model training, feature engineering and SHAP artifact generation                                |
| `generate_rag_pdfs.py`                       | Generates sample policy documents for RAG                                                             |
| `Health Insurance Fraud Claims.xlsx`         | Historical claims dataset                                                                             |
| `output/fraud_detection_model.pkl`           | Trained ML runtime artifact                                                                           |
| `output/processed_claim_features.csv`        | Processed claims data                                                                                 |
| `rag/documents/`                             | Policy documents used by RAG                                                                          |

---

# 15. Installation

Clone the repository:

```bash
git clone https://github.com/tigersb06/AgenticAI_training.git
cd AgenticAI_training
```

Create a virtual environment.

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 16. Start Weaviate

The Policy capability requires Weaviate.

Using Docker Compose:

```bash
docker compose up -d
```

Verify:

```bash
docker ps
```

The policy RAG system expects:

```text
InsuranceKnowledge
```

as the knowledge collection.

---

# 17. Prepare Policy Documents

Generate the sample policy PDFs:

```bash
python generate_rag_pdfs.py
```

Or place your own documents in:

```text
rag/documents/
```

Example:

```text
rag/
└── documents/
    ├── coverage_rules.pdf
    ├── exclusions.pdf
    ├── claim_limits.pdf
    ├── policy_guidelines.pdf
    └── documentation_requirements.pdf
```

---

# 18. Train the Fraud Model

Run:

```bash
python train_fraud_model.py
```

Input:

```text
Health Insurance Fraud Claims.xlsx
```

Generated runtime artifacts:

```text
output/
├── fraud_detection_model.pkl
└── processed_claim_features.csv
```

The trained artifact must exist before Risk MCP is used.

---

# 19. Build the RAG Index

Run:

```bash
python insurance_multi_agent_chunking_indexing.py
```

Conceptually:

```mermaid
flowchart LR

    A["Policy PDFs"]
    B["Text Extraction"]
    C["Chunking"]
    D["Embeddings"]
    E["Weaviate"]

    A --> B
    B --> C
    C --> D
    D --> E
```

Runtime retrieval:

```mermaid
flowchart LR

    A["Policy Question"]
    B["Hybrid Retrieval"]
    C["Cross-Encoder"]
    D["Qwen"]
    E["Grounded Answer"]

    A --> B
    B --> C
    C --> D
    D --> E
```

---

# 20. Start FastMCP

Start the MCP service:

```bash
python insurance_mcp_server.py
```

Default endpoint:

```text
http://127.0.0.1:8011/mcp
```

Available tools:

```text
score_claim
lookup_policy
```

The MCP server should remain running while the application is active.

---

# 21. Start Streamlit

In another terminal:

```bash
streamlit run app_chat.py
```

The UI provides:

* Natural-language interaction
* Agent routing
* Risk analysis
* Policy analysis
* Dual-intent execution
* MCP tool execution tracing
* Final response synthesis

---

# 22. Recommended Startup Order

```mermaid
flowchart TD

    A["1. Activate Python Environment"]
    B["2. Start Weaviate"]
    C["3. Train Fraud Model"]
    D["4. Prepare Policy Documents"]
    E["5. Build RAG Index"]
    F["6. Start Insurance MCP Server"]
    G["7. Start Streamlit"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
```

Commands:

```bash
source venv/bin/activate

docker compose up -d

python train_fraud_model.py

python generate_rag_pdfs.py

python insurance_multi_agent_chunking_indexing.py

python insurance_mcp_server.py
```

Then, in another terminal:

```bash
source venv/bin/activate
streamlit run app_chat.py
```

---

# 23. MCP Tools

## `score_claim`

Purpose:

```text
Fraud risk scoring and explainability
```

Typical inputs include:

```text
claim_amount
patient_income
patient_age
claim_type
claim_id
procedure_code
patient_id
provider_id
provider_specialty
diagnosis_code
provider_location
claim_status
claim_submission_method
previously_rejected_claims
num_claims_last_12m
```

Returns:

```text
risk_level
risk_score
decision_cutoff
triage_status
risk_explanation
```

---

## `lookup_policy`

Purpose:

```text
Policy retrieval and grounded policy synthesis
```

Typical inputs:

```text
question
claim_type
procedure_code
```

Returns:

```text
answer
retrieved_docs
source information
reranking information
```

---

# 24. Agent Input / Output Summary

| Agent             | Input                      | MCP Tool          | Output                            |
| ----------------- | -------------------------- | ----------------- | --------------------------------- |
| Orchestrator      | User query + claim context | Specialist agents | Final response                    |
| Risk Specialist   | Risk task + claim details  | `score_claim`     | Risk result + explanation         |
| Policy Specialist | Policy question + context  | `lookup_policy`   | Grounded policy result + evidence |

---

# 25. Architecture Rule

The core design principle is:

```mermaid
flowchart LR

    A["Orchestrator"]
    B["Specialist Agent"]
    C["MCP Tool"]
    D["Backend Capability"]

    A -->|"Decides WHO"| B
    B -->|"Decides WHICH capability"| C
    C -->|"Invokes"| D
    D -->|"Returns result"| C
    C --> B
    B --> A
```

### In one line:

> **Agent decides → MCP invokes → backend executes → Agent interprets → Orchestrator synthesizes**

The architecture intentionally avoids:

```text
MCP → Agent
```

and instead uses:

```text
Orchestrator → Specialist Agent → MCP → Capability
```

---

# 26. Error Handling

## Missing Fraud Model

If the application reports that:

```text
output/fraud_detection_model.pkl
```

is missing, run:

```bash
python train_fraud_model.py
```

---

## MCP Connection Error

Ensure:

```bash
python insurance_mcp_server.py
```

is running.

Verify the endpoint:

```text
http://127.0.0.1:8011/mcp
```

---

## Weaviate Connection Error

Check Docker:

```bash
docker ps
```

Ensure the Weaviate container is running.

---

## Empty Policy Results

Check:

* Weaviate is running
* Policy documents exist
* RAG indexing has been executed
* `InsuranceKnowledge` exists
* `lookup_policy` is available through MCP

---

# 27. Production Considerations

For production deployment, consider:

* MCP authentication
* Tool-level authorization
* API-key / secret management
* Request IDs and correlation IDs
* Structured logging
* MCP audit logging
* Rate limiting
* Model version tracking
* RAG document versioning
* Model drift monitoring
* Retrieval-quality monitoring
* Human approval for high-risk claims
* Data privacy controls

The current implementation is primarily an end-to-end architecture and working prototype.

---

# 28. Final Architecture Summary

```mermaid
flowchart LR

    U["👤 User"]
    UI["🖥️ Streamlit"]

    O["🧠 Orchestrator"]

    R["🛡️ Risk Specialist"]
    P["📚 Policy Specialist"]

    RM["Risk MCP<br/>score_claim"]
    PM["Policy MCP<br/>lookup_policy"]

    RF["Fraud ML + SHAP"]
    PR["Policy RAG + Reranking + Qwen"]

    O --> R
    O --> P

    R --> RM
    P --> PM

    RM --> RF
    PM --> PR

    RF --> R
    PR --> P

    R --> O
    P --> O

    U --> UI
    UI --> O
    O --> UI
```

## Core Principle

> **The Orchestrator decides which specialist should work.
> The Specialist Agent decides which capability it needs.
> MCP exposes that capability as a tool.
> The backend performs the work.
> The result returns to the Specialist, then to the Orchestrator for final synthesis.**
