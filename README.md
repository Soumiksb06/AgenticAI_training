# AgenticAI_training

```mermaid
graph TD
    A[User Prompt / Query] --> B[Slot-Filling & Regex Parsing]

    subgraph Pre-Processing Validation
        B --> C{Missing Required Slots?}
        C -- Yes --> D[User Interactive Prompt / Baseline Defaults]
        C -- No --> E[Initialize ClaimState Node]
        D --> E
    end

    subgraph LangGraph Multi-Agent Execution
        E --> F{Intent Router Node}

        F -- Claim Amount = $0 <br/> Policy Only --> I[Agent 3: Policy Agent]
        F -- Claim Amount > $0 <br/> Full Evaluation --> G[Agent 1: Claims Triage ML]

        subgraph ML & Explainability Subsystem
            G --> G1[Feature Pipeline & Baseline Ratios]
            G1 --> G2[XGBoost Model Inference]
            G2 --> G3[Triage: Fraud Score vs. Optimal Cutoff]
            G3 --> H[Agent 2: Risk Analysis Agent]
            H --> H1[SHAP TreeExplainer Attributions]
        end

        H1 --> I

        subgraph Two-Stage Hybrid RAG Engine
            I --> I1[Weaviate Hybrid Search <br/> Dense Vectors + BM25]
            I1 --> I2[Cross-Encoder Re-Ranking <br/> ms-marco-MiniLM-L-6-v2]
            I2 --> I3[Qwen-2.5-1.5B Grounded Synthesis]
        end

        I3 --> J[Formatting Node]
    end

    J --> K[Final Investigation Report Output]
