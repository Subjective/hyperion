# Hyperion

Hyperion is a modern hyperparameter optimization framework built for the agentic era. Unlike conventional libraries, it orchestrates and reasons about long-running, parallel experiments through an event-driven, agent-based architecture. Experiments are modeled as a dynamic exploration tree, enabling efficient branching, pruning, and adaptation across parallel runs while maintaining a transparent reasoning trace.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/dashboard-dark.png">
  <img src="docs/images/dashboard-light.png" alt="Hyperion dashboard (lineage graph)">
</picture>

At its core, Hyperion is designed to be model-agnostic and support any training routine through a flexible interface: if you can wrap your training loop in Python, you can optimize it here. While still in early days, the aim is to make optimization composable, observable, and scalable—with faster results, less manual effort, and interpretable outcomes.

## Key Features

- 🎯 **Multiple Search Strategies**: Random, Grid, Beam Search, Bayesian Optimization, Population-Based Training
- 🤖 **Agent Integration**: LLM-driven and rule-based agents for intelligent optimization
- 🌳 **Lineage-Aware Trials**: First-class support for branching search with trial ancestry tracking
- 📊 **Full Observability**: Complete event log with decision rationale and reproducible experiments
- 🚀 **Progressive Scaling**: From in-memory prototypes to distributed execution
- 🔧 **Ergonomic API**: High-level `tune()` API with progressive disclosure to framework internals

## Quick Start

### Basic Optimization

```python
from hyperion import tune, Float, Choice, ObjectiveResult

def objective(ctx, lr: float, batch_size: int) -> ObjectiveResult:
    # Your training code here
    score = train_model(lr=lr, batch_size=batch_size)

    # Report progress during training
    ctx.report(step=1, loss=0.5, accuracy=0.8)

    # Check if we should stop early
    if ctx.should_stop():
        return ObjectiveResult(score=score)

    return ObjectiveResult(score=score)

# Run optimization
result = tune(
    objective=objective,
    space={
        "lr": Float(0.001, 0.1, log=True),  # Log-scale sampling
        "batch_size": Choice([32, 64, 128])
    },
    strategy="random",
    max_trials=50,
    max_concurrent=4,
    show_progress=True,  # Display live progress
    show_summary=True,   # Show final summary
)

print(f"Best params: {result['best']}")
```

### Using Different Search Strategies

```python
# Beam Search - Tree-based exploration with pruning
result = tune(
    objective=objective,
    space=space,
    strategy="beam_search",
    strategy_kwargs={
        "K": 3,           # Keep top 3 trials per depth
        "width": 2,       # Generate 2 children per parent
        "max_depth": 4,   # Maximum search tree depth
    },
    max_trials=100,
    max_concurrent=4,
)

# Grid Search - Exhaustive search over discrete values
from hyperion import Int

result = tune(
    objective=objective,
    space={
        "lr": Choice([0.001, 0.01, 0.1]),
        "batch_size": Choice([32, 64]),
        "layers": Int(2, 4),
    },
    strategy="grid",
    max_concurrent=4,
)

# LLM Agent - AI-powered optimization (provider-agnostic)
result = tune(
    objective=objective,
    space=space,
    strategy="llm_agent",
    strategy_kwargs={
        "llm": ...,  # Provide any LLM via a simple callable (prompt: str) -> str
        "max_history": 20,
    },
    max_trials=50,
    max_concurrent=2,
)
```

### With Persistent Storage

```python
# Use SQLite to persist experiment data
result = tune(
    objective=objective,
    space=space,
    strategy="random",
    max_trials=100,
    storage="sqlite:///experiments.db",  # Save to database
)

# Results are automatically saved and can be analyzed later
```

## Web UI

Hyperion includes a web dashboard for real-time experiment monitoring and visualization. The dashboard provides:

- **Live Experiment Tracking**: Monitor running experiments with real-time updates via WebSocket
- **Interactive Lineage Graph**: Visualize trial relationships and branching patterns
- **Metrics Dashboard**: Track performance metrics, compare trials, and identify trends
- **Event Timeline**: Audit trail of all decisions and actions with full context

### Running the Dashboard

```bash
# Start both backend and frontend
mise run ui

# Or run them separately:
mise run ui-backend   # FastAPI server on port 8000
mise run ui-frontend  # React app on port 5173
```

The dashboard automatically connects to your SQLite database and provides both live monitoring (when running in-process) and historical analysis capabilities.

## Architecture

Hyperion follows a layered architecture so you can use as much or as little of it as you like:

1. **Core Layer**: Events, commands, controller, executors, and storage primitives
2. **Framework Layer**: Experiments, policies, search spaces, and callbacks
3. **API Layer**: High-level functions like `tune()` and `optimize()`
4. **Interface Layer**: CLI and optional web UI

This separation keeps the internals composable while letting you choose your level of control. Most users will start at the API and only drop down when necessary.

## Documentation

- **[API Guide](docs/api-guide.md)** - Complete guide to using the high-level API
- **[Framework Guide](docs/framework-guide.md)** - Advanced usage and customization
- **[Examples](examples/)** - Runnable example scripts for various use cases

## Installation

### From PyPI

```bash
# Install base package
pip install hyperion-opt
```

Note: The web UI (backend + React frontend) is currently developed and run from the repository. Installing from PyPI does not include the UI app itself; to run the dashboard, clone the repo and use the commands in the Web UI section below.

### From Source

For development or to use the latest unreleased features:

```bash
# Clone the repository
git clone https://github.com/Subjective/hyperion.git
cd hyperion

# Install in editable mode with development dependencies
mise run install-dev

# Or using pip directly
pip install -e ".[dev]"
```

## Development

This project uses:

- **mise** for environment management
- **uv** for fast package management
- **ruff** for linting and formatting
- **pyright** for type checking
- **pytest** for testing

Common development tasks:

```bash
mise run fix          # Fix lint issues and format code
mise run check        # Run all checks (lint, type-check, test)
mise run test         # Run tests
mise run install-dev  # Install with dev dependencies
```

## License

This project is licensed under the [MIT License](LICENSE).

## Status

Hyperion is currently in active development. Expect breaking changes as the framework evolves. Contributions and feedback are welcome.
