```mermaid
graph TD
    %% Define Visual Styles
    classDef user fill:#f9d0c4,stroke:#333,stroke-width:2px,color:#000;
    classDef client fill:#d4e157,stroke:#333,stroke-width:2px,color:#000;
    classDef mcpServer fill:#81d4fa,stroke:#333,stroke-width:2px,color:#000;
    classDef coreSys fill:#ce93d8,stroke:#333,stroke-width:2px,color:#000;
    classDef aiModel fill:#ffb74d,stroke:#333,stroke-width:1px,color:#000;
    classDef db fill:#bcaaa4,stroke:#333,stroke-width:1px,color:#000;

    %% Process Orchestration
    Launcher["MCP Launcher<br/><i>start_insurance_mcp.py</i>"]:::mcpServer
    Launcher -.->|"Spawns and Monitors Port 8011"| RiskMCP
    Launcher -.->|"Spawns and Monitors Port 8012"| PolicyMCP

    %% User and Client Entry
    User((User)):::user
    User -->|"1. Submits Natural Language Query / Claim"| CLI

    subgraph ClientSubsystem ["Async MCP Client Subsystem (insurance_multi_agent_client.py)"]
        direction TB
        CLI["Async MCP Client"]:::client
        NLP["Entity Extractor and Slot-Filler<br/><i>_extract_all_fields_from_text()</i><br/>(Extracts Income, Age, Amount, Specialty, Medical Verbs)"]:::client
        AutoRoute{"Auto-Routing Engine<br/>Is Claim Amount > $0.00?"}:::client

        CLI --> NLP
        NLP --> AutoRoute
    end

    %% Routing Paths
    AutoRoute -->|"NO ($0.00 - Policy Only Mode)"| DirectPolicyCall["Single MCP Request<br/>Tool: lookup_policy"]:::client
    AutoRoute -->|"YES (> $0.00 - Full Claim Evaluation)"| DualGather["Concurrent asyncio.gather"]:::client

    %% MCP Network Layer
    subgraph ConcurrentExecution ["MCP Server Infrastructure (FastMCP)"]
        direction LR
        RiskMCP["Risk MCP Server<br/><i>Port: 8011+</i><br/>Pydantic: ClaimInput"]:::mcpServer
        PolicyMCP["Policy MCP Server<br/><i>Port: 8012+</i><br/>Pydantic: PolicyRequest"]:::mcpServer
    end

    DirectPolicyCall -->|"Direct FastMCP Call"| PolicyMCP
    DualGather -->|"Tool: score_claim"| RiskMCP
    DualGather -->|"Tool: lookup_policy"| PolicyMCP

    %% Risk Subsystem
    subgraph RiskSubsystem ["Risk Analysis Subsystem"]
        RiskSys["InsuranceAgentSystem<br/><i>use_local_llm=False</i><br/>(Full LangGraph process_claim Traversal)"]:::coreSys
        ML[("Trained ML Model<br/>and SHAP Explainer")]:::aiModel
        
        RiskMCP -->|"process_claim()<br/>intent_router -> triage -> SHAP"| RiskSys
        RiskSys <-->|"Feature Vector, Dispersion Ratios,<br/>Predict Proba and SHAP"| ML
    end

    %% Policy Subsystem
    subgraph PolicySubsystem ["Policy RAG Subsystem"]
        PolicySys["InsuranceAgentSystem<br/><i>use_local_llm=True</i><br/>(Direct policy_agent execution)"]:::coreSys
        Weaviate[("Weaviate Local<br/>Vector DB")]:::db
        Embedder["SentenceTransformer<br/><i>all-MiniLM-L6-v2</i>"]:::aiModel
        Reranker["CrossEncoder<br/><i>ms-marco-MiniLM-L-6-v2</i>"]:::aiModel
        LLM["Local LLM Generator<br/><i>Qwen2.5-1.5B-Instruct</i>"]:::aiModel
        
        PolicyMCP -->|"policy_agent()<br/>target_route='policy_only'"| PolicySys
        PolicySys <-->|"Dense Embeddings"| Embedder
        PolicySys <-->|"Hybrid Search (BM25 + Vector)"| Weaviate
        PolicySys <-->|"Cross-Encoder Reranking"| Reranker
        PolicySys <-->|"Grounded Answer Synthesis"| LLM
    end

    %% Return Payload Flow
    RiskSys -->|"Risk Level, Score, Cutoff,<br/>SHAP Explanation"| RiskMCP
    PolicySys -->|"Grounded Policy Answer,<br/>Retrieved Docs"| PolicyMCP

    RiskMCP -->|"Risk Payload"| CLI
    PolicyMCP -->|"Policy Payload"| CLI

    %% Client Decision Engine
    subgraph VerdictEngine ["Client Decision Engine"]
        VerdictLogic{"Evaluate Verdict Rules<br/>HIGH_RISK or Score >= 0.6 -> Escalate<br/>LOW_RISK and Score < 0.4 -> Auto-approve<br/>Else -> Review with policy validation"}:::client
    end

    CLI --> VerdictLogic
    VerdictLogic -->|"Formatted JSON Output"| User
