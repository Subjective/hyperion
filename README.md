# hyperion

An agentic framework for hyperparameter tuning that runs parallel experiments using an exploration tree, while maintaining a transparent reasoning trace

## Overview

Hyperion is an open, agentic framework for hyperparameter optimization in the PyTorch ecosystem. Unlike traditional tuning libraries, Hyperion is built to orchestrate and reason about long-running, parallel model training experiments using an event-driven, agent-based approach. Experiments are represented as a dynamic graph, allowing the system to efficiently explore, branch, and prune hyperparameter configurations across multiple GPUs or nodes. Leveraging intelligent reasoning and traceable decision-making, Hyperion aims to automate the iterative process of finding optimal hyperparameters, reduce manual overhead, and provide interpretable insights for researchers and practitioners. The framework is highly modular and model-agnostic, supporting arbitrary architectures and training routines via a flexible interface.

## System Architecture

```mermaid
flowchart TD
    subgraph User
        A[User Model & Hyperparam Definition] --> B[Hyperion API]
    end
    subgraph Hyperion Core
        B --> C[Experiment Orchestrator]
        C --> D[Resource Manager]
        C --> E[Event Queue]
        E --> F[Agentic Reasoning Module]
        F --> G[Search/Branching Logic]
        G -->|Propose New Trials| C
        C --> H[Experiment Database]
        H <--> F
        H <--> G
        H --> I[Trace/Context Store]
        H <--> J[Checkpoint Store]
    end
    subgraph Executors
        D --> K[Experiment Workers/Executors]
        K --> L[PyTorch Training Jobs]
        L -->|Results/Checkpoints| J
        L -->|Metrics/Logs| M[Logging & Visualization - WandB/MLflow]
        K -->|Results| C
    end
    style B fill:#C6E2FF,stroke:#333,stroke-width:2px
    style F fill:#FFE4C4,stroke:#333,stroke-width:2px
    style G fill:#FFFACD,stroke:#333,stroke-width:2px
    style K fill:#D3FFD3,stroke:#333,stroke-width:2px
    style H fill:#F8E9A1,stroke:#333,stroke-width:2px
    style L fill:#F7CAC9,stroke:#333,stroke-width:2px
    style M fill:#B5EAD7,stroke:#333,stroke-width:2px
    style J fill:#F1C0E8,stroke:#333,stroke-width:2px

```

## Experiment Graph with Beam Search (Branching & Pruning)

```mermaid
graph TD
    A(["Root Experiment<br/>(Initial Hyperparams)"])
    A -->|"Branch 1"| B1(["Exp 1: lr ↑"])
    A -->|"Branch 2"| B2(["Exp 2: lr ↓"])
    A -->|"Branch 3"| B3(["Exp 3: batch ↑"])
    B1 -->|"Branch"| C1(["Exp 4: lr ↑↑"])
    B1 -->|"Branch"| C2(["Exp 5: lr ↓"])
    B2 -->|"Branch"| C3(["Exp 6: batch ↑"])
    C1 -.->|Pruned| X1(("X"))
    C2 -->|"Best"| D1(["Best Path"])
    C3 -.->|Pruned| X2(("X"))

    style A fill:#C6E2FF,stroke:#333,stroke-width:2px
    style D1 fill:#a2e6a8,stroke:#333,stroke-width:2px
    style X1 fill:#ffdddd,stroke:#333,stroke-width:2px
    style X2 fill:#ffdddd,stroke:#333,stroke-width:2px
```

## Event-drive Agent Reasoning Loop

```mermaid
sequenceDiagram
    participant Orchestrator
    participant Agent
    participant ExperimentDB
    participant Executor
    participant User

    Executor->>Orchestrator: Trial Complete (Event)
    Orchestrator->>ExperimentDB: Store Results
    Orchestrator->>Agent: Notify (event, context)
    Agent->>ExperimentDB: Fetch Trial Data/Trace
    Agent->>Agent: Analyze Results & Trace
    Agent->>Orchestrator: Propose New Trials (w/ Rationale)
    Orchestrator->>Executor: Launch New Trials
    Note over Agent,Orchestrator: Loop continues until stop criteria
    Orchestrator->>User: (Optional) Explanations & Progress
```

## Experiment/Trial Lifecycle

```mermaid
flowchart TD
    A["Trial Proposed (by Agent/Search)"]
    B["Trial Scheduled by Orchestrator"]
    C["Trial Dispatched to Executor/Worker"]
    D["PyTorch Model Trained<br/>(Logs, Metrics, Checkpoint)"]
    E["Results Sent to Orchestrator"]
    F["Results Stored in DB"]
    G["Agent Consumes Results/Event"]

    A --> B --> C --> D --> E --> F --> G
```

## Parallel Resource Management

```mermaid
flowchart LR
    subgraph "Resource Pool (e.g. GPUs)"
        GPU1["GPU 1"]
        GPU2["GPU 2"]
        GPU3["GPU 3"]
        GPU4["GPU N"]
    end

    subgraph Orchestrator
        Q1["Pending Trials Queue"]
        Q2["Running Trials"]
    end

    Q1 -->|Assign| GPU1
    Q1 -->|Assign| GPU2
    Q1 -->|Assign| GPU3
    Q1 -->|Assign| GPU4

    GPU1 -->|"Results"| Q2
    GPU2 -->|"Results"| Q2
    GPU3 -->|"Results"| Q2
    GPU4 -->|"Results"| Q2
```

## Trace/Context Store Visualization

```mermaid
flowchart TD
    Start(["Root"])
    Start --> N1(["Trial 1"])
    N1 --> N2(["Trial 2"])
    N2 --> N3(["Trial 3"])
    N3 --> N4(["Trial 4"])

    subgraph "Trace for Trial 4"
        direction LR
        T_Start(["Root"]) -.-> T_N1(["Trial 1"]) -.-> T_N2(["Trial 2"]) -.-> T_N3(["Trial 3"]) -.-> T_N4(["Trial 4"])
    end

    style N4 fill:#ffeb3b,stroke:#333,stroke-width:2px
    style T_N4 fill:#ffeb3b,stroke:#333,stroke-width:2px
```

## Trace Graph with Merging

Beam search, PBT, and most HPO strategies produce many parallel branches/lineages. Good configurations may arise in separate "islands." It’s often suboptimal to pick just the best path and ignore others—sometimes combining the best of multiple lines yields a better result (e.g., learning rate from one, regularization from another). Hence, we should devise a solutions for achieving cross-branch context. Some approaches from the current literature involve:

- **Population Based Training (PBT)**: Routinely "steals" hyperparameters from better-performing members of the population, sometimes across very different branches. The population is constantly cross-pollinated .
- **Evolutionary HPO**: Crossover operators explicitly merge parts of two different parameter vectors (i.e., "children" inherit from two "parents").
- **AgentHPO and LLM-based agents**: Instruct the agent to consider the entire set of experiment traces/results when deciding next experiments—not just a single lineage .

```mermaid
graph TD
    Start([Root])
    Start --> A1([Trial A1])
    Start --> B1([Trial B1])
    A1 --> A2([Trial A2])
    B1 --> B2([Trial B2])
    A2 --> C([Trial C: Merge A2+B2])
    B2 --> C
```
