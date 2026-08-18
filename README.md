# AgenticAI_training

```mermaid
graph TD
    subgraph Data Stores & Model Artifacts
        DS1[(Policy PDFs)] --> DS2[Chunking & Embedding Pipeline]
        DS2 --> DS3[Bi-Encoder: all-MiniLM-L6-v2]
        DS3 --> DB1[(Weaviate Vector DB <br/> InsuranceKnowledge Collection)]
        
        DS4[(Historical Claims Dataset <br/> Excel / CSV)] --> DS5[Pre-calculated Baselines <br/> Peer/Patient/Provider Means]
        DS6[(Model Artifacts Store <br/> fraud_detection_model.pkl)] --> DS7[XGBoost, Preprocessor, <br/> SHAP Explainer & Cutoff]
    end

    A[User Prompt / Claim Description] --> B[Slot-Filling Extractor <br/> Regex & Tokenized Fuzzy Matcher]

    subgraph Interactive Pre-Processing
        B --> C{Missing Required Slots? <br/> Income / Age / Type}
        C -- Yes --> D[CLI Prompt User for Input / <br/> Apply Dataset Defaults]
        C -- No --> E[Initialize ClaimState Node]
        D --> E
    end

    subgraph LangGraph Multi-Agent Execution
        E --> F{Intent Router Node}

        F -- Claim Amount = $0 <br/> Policy Only --> I[Agent 3: Policy Agent]
        F -- Claim Amount > $0 <br/> Full Evaluation --> G[Agent 1: Claims Triage ML]

        subgraph ML & Explainability Subsystem
            DS5 -. Baselines .-> G1
            DS7 -. Model Artifacts .-> G2
            G --> G1[Feature Pipeline & <br/> Statistical Dispersion Ratios]
            G1 --> G2[XGBoost Model Inference]
            G2 --> G3[Triage Decision: <br/> Fraud Prob vs. Optimal Cutoff]
            G3 --> H[Agent 2: Risk Analysis Agent]
            DS7 -. SHAP Explainer .-> H1
            H --> H1[SHAP TreeExplainer Attributions]
        end

        H1 --> I

        subgraph Two-Stage Hybrid RAG Engine
            I --> I1[Weaviate Hybrid Search <br/> Dense + BM25 + Category Filter]
            DB1 -. Document Retrieval .-> I1
            I1 --> I2[Cross-Encoder Re-Ranking <br/> ms-marco-MiniLM-L-6-v2]
            I2 --> I3[Local LLM Synthesis <br/> Qwen-2.5-1.5B-Instruct]
        end

        I3 --> J[Formatting Node]
    end

    J --> K[Final Investigation Report Output <br/> CLI / Streamlit Dashboard]
