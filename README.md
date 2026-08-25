```mermaid
graph TD
    %% Define Styles
    classDef user fill:#f9d0c4,stroke:#333,stroke-width:2px,color:#000;
    classDef client fill:#d4e157,stroke:#333,stroke-width:2px,color:#000;
    classDef mcpServer fill:#81d4fa,stroke:#333,stroke-width:2px,color:#000;
    classDef coreSys fill:#ce93d8,stroke:#333,stroke-width:2px,color:#000;
    classDef aiModel fill:#ffb74d,stroke:#333,stroke-width:1px,color:#000;
    classDef db fill:#bcaaa4,stroke:#333,stroke-width:1px,color:#000;

    %% User Interaction & Orchestration
    User((User)):::user
    Launcher[MCP Launcher<br/><i>start_insurance_mcp.py</i>]:::mcpServer
    CLI[Async MCP Client<br/><i>insurance_multi_agent_client.py</i><br/>(Slot-Filling & Validation)]:::client
    
    Launcher -.->|Spawns & Monitors Ports| RiskMCP
    Launcher -.->|Spawns & Monitors Ports| PolicyMCP
    
    User -->|Inputs Claim Details<br/>via Interactive CLI| CLI

    %% MCP Network Layer
    subgraph Concurrent MCP Execution
        direction LR
        RiskMCP[Risk MCP Server<br/><i>Port: 8011+ (FastMCP)</i><br/>Pydantic: ClaimInput]:::mcpServer
        PolicyMCP[Policy MCP Server<br/><i>Port: 8012+ (FastMCP)</i><br/>Pydantic: PolicyRequest]:::mcpServer
    end

    CLI -->|asyncio.gather<br/>Tool: score_claim| RiskMCP
    CLI -->|asyncio.gather<br/>Tool: lookup_policy| PolicyMCP

    %% Risk Environment
    subgraph Risk Environment [Risk Analysis Subsystem]
        RiskSys[InsuranceAgentSystem<br/><i>use_local_llm=False</i>]:::coreSys
        ML[(Trained ML Model<br/>& SHAP Explainer)]:::aiModel
        
        RiskMCP -->|process_claim()| RiskSys
        RiskSys <-->|Feature Extraction &<br/>Predict Proba| ML
    end

    %% Policy Environment
    subgraph Policy Environment [Policy RAG Subsystem]
        PolicySys[InsuranceAgentSystem<br/><i>use_local_llm=True</i>]:::coreSys
        Weaviate[(Weaviate Local<br/>Vector DB)]:::db
        Embedder[SentenceTransformer<br/><i>all-MiniLM-L6-v2</i>]:::aiModel
        Reranker[CrossEncoder<br/><i>ms-marco-MiniLM-L-6-v2</i>]:::aiModel
        LLM[Local LLM Generator<br/><i>Qwen2.5-1.5B-Instruct</i>]:::aiModel
        
        PolicyMCP -->|policy_agent()| PolicySys
        PolicySys <-->|Vectorize Query| Embedder
        PolicySys <-->|Hybrid Search| Weaviate
        PolicySys <-->|Re-Rank Docs| Reranker
        PolicySys <-->|Synthesize Answer| LLM
    end

    %% Return Data Flow
    RiskSys -->|Risk Level, Score,<br/>SHAP Explanation| RiskMCP
    PolicySys -->|Grounded Policy Answer,<br/>Retrieved Docs| PolicyMCP
    
    RiskMCP -->|Risk Payload| CLI
    PolicyMCP -->|Policy Payload| CLI

    %% Final Verdict
    Decision{Final Verdict Logic<br/><i>Escalate / Auto-Approve / Review</i>}:::client
    CLI --> Decision
    Decision -->|Consolidated Report| User
