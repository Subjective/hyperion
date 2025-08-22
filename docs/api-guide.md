# Hyperion API Guide

This guide covers the high-level API for hyperparameter optimization using Hyperion. The API is designed to be simple yet powerful, with progressive disclosure of advanced features.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Search Spaces](#search-spaces)
3. [Objective Functions](#objective-functions)
4. [Search Strategies](#search-strategies)
5. [Storage Options](#storage-options)
6. [Monitoring and Visualization](#monitoring-and-visualization)
7. [Understanding Results](#understanding-results)

## Quick Start

The simplest way to use Hyperion is through the `tune()` function:

```python
from hyperion import tune, Float, Choice, ObjectiveResult

def objective(ctx, learning_rate: float, optimizer: str) -> ObjectiveResult:
    """Your training function."""
    # Train your model
    score = train_model(lr=learning_rate, opt=optimizer)

    # Return the result
    return ObjectiveResult(score=score)

# Run optimization
result = tune(
    objective=objective,
    space={
        "learning_rate": Float(0.0001, 0.1, log=True),
        "optimizer": Choice(["adam", "sgd", "rmsprop"])
    },
    strategy="random",
    max_trials=50,
    max_concurrent=4
)

print(f"Best configuration: {result['best']}")
```

## Search Spaces

Hyperion provides several primitives for defining search spaces:

### Continuous Distributions

```python
from hyperion import Float

# Linear uniform distribution
space = {
    "dropout": Float(0.0, 0.5),  # Samples uniformly between 0.0 and 0.5
}

# Log-uniform distribution (useful for learning rates)
space = {
    "learning_rate": Float(0.0001, 0.1, log=True),  # Samples in log space
}
```

### Integer Distributions

```python
from hyperion import Int

space = {
    "batch_size": Int(32, 256),  # Samples integers from 32 to 256
    "n_layers": Int(2, 8),        # Number of layers
    "hidden_units": Int(64, 512, log=True),  # Log-scale sampling
}
```

### Categorical Choices

```python
from hyperion import Choice

space = {
    "optimizer": Choice(["adam", "sgd", "rmsprop"]),
    "activation": Choice(["relu", "tanh", "sigmoid"]),
    "architecture": Choice(["resnet", "vgg", "densenet"])
}
```

### Boolean Flags

```python
from hyperion import Bool

space = {
    "use_batch_norm": Bool(),  # 50/50 chance of True/False
    "use_dropout": Bool(),
}
```

### Nested Search Spaces

Search spaces can be nested arbitrarily:

```python
space = {
    "model": {
        "n_layers": Int(2, 5),
        "layer_size": Int(32, 128),
        "activation": Choice(["relu", "tanh"])
    },
    "training": {
        "learning_rate": Float(0.001, 0.1, log=True),
        "batch_size": Choice([32, 64, 128]),
        "optimizer": Choice(["adam", "sgd"])
    }
}
```

## Objective Functions

Objective functions receive a `TrialContext` and hyperparameters, and return an `ObjectiveResult`.

### Basic Objective

```python
from hyperion import ObjectiveResult

def objective(ctx, **params) -> ObjectiveResult:
    # Use params to configure your model
    model = create_model(**params)

    # Train and evaluate
    score = train_and_evaluate(model)

    # Return the result
    return ObjectiveResult(score=score)
```

### Reporting Progress

Use `ctx.report()` to log metrics during training:

```python
def objective(ctx, **params) -> ObjectiveResult:
    model = create_model(**params)

    for epoch in range(10):
        train_loss = train_one_epoch(model)
        val_score = validate(model)

        # Report progress
        ctx.report(
            step=epoch,
            train_loss=train_loss,
            val_score=val_score
        )

    return ObjectiveResult(score=val_score)
```

### Early Stopping

Check `ctx.should_stop()` to handle early termination. This returns `True` when the trial has been killed by an early stopping policy:

```python
def objective(ctx, **params) -> ObjectiveResult:
    model = create_model(**params)
    best_score = 0.0

    for epoch in range(100):
        score = train_and_evaluate(model)
        best_score = max(best_score, score)

        # Report progress - enables early stopping policies to monitor performance
        ctx.report(step=epoch, score=score, val_loss=validation_loss)

        # Check if killed by early stopping policy
        if ctx.should_stop():
            break

    return ObjectiveResult(score=best_score)
```

**Note**: The `ctx.report()` calls are essential for early stopping policies to work. They update the trial's metrics which policies use to decide which trials to terminate.

### Returning Additional Metrics

Include additional metrics and artifacts in the result:

```python
def objective(ctx, **params) -> ObjectiveResult:
    model = create_model(**params)
    score = train_model(model)

    return ObjectiveResult(
        score=score,
        metrics={
            "final_accuracy": 0.95,
            "final_loss": 0.05,
            "training_time": 120.5,
            "model_size": 1024000
        },
        artifacts={
            "model_path": "/path/to/saved/model.pt",
            "tensorboard_logs": "/path/to/logs/"
        }
    )
```

## Search Strategies

Hyperion supports multiple search strategies:

### Random Search

Samples randomly from the search space:

```python
result = tune(
    objective=objective,
    space=space,
    strategy="random",
    max_trials=100
)
```

### Grid Search

Exhaustive search over all combinations:

```python
# Define discrete values for grid search
space = {
    "learning_rate": Choice([0.001, 0.01, 0.1]),
    "batch_size": Choice([32, 64, 128]),
    "layers": Choice([2, 3, 4])
}

result = tune(
    objective=objective,
    space=space,
    strategy="grid",
    max_concurrent=4
)
```

### Beam Search

Tree-based exploration with pruning:

```python
result = tune(
    objective=objective,
    space=space,
    strategy="beam_search",
    strategy_kwargs={
        "K": 3,           # Keep top K trials per depth
        "width": 2,       # Generate width children per parent
        "max_depth": 4,   # Maximum tree depth
        "prune": True,    # Kill underperforming branches
    },
    max_trials=100,
    max_concurrent=4
)
```

### LLM Agent

AI-powered optimization using Large Language Models (provider-agnostic):

```python
# Example using Ollama as a provider
import ollama

def ollama_llm(prompt: str) -> str:
    resp = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": "Return RAW JSON only."},
            {"role": "user", "content": prompt},
        ],
        options={},
    )
    return resp.get("message", {}).get("content", "{}")

result = tune(
    objective=objective,
    space=space,
    strategy="llm_agent",
    strategy_kwargs={
        "llm": ollama_llm,           # Provide any LLM via a simple callable (prompt: str) -> str
        "max_history": 20,           # Number of trials to show LLM
        "exploration_rate": 0.3,     # Balance exploration/exploitation
    },
    max_trials=50,
    max_concurrent=2,  # LLM agents work best with fewer concurrent trials
)
```

The LLM agent:

- Analyzes trial history to identify patterns
- Suggests promising parameter combinations
- Provides detailed rationale for decisions
- Automatically validates suggestions against search space constraints
- Falls back to random sampling if LLM fails

### LLM Branching Agent

Tree-based exploration with LLM-guided branching decisions (provider-agnostic):

```python
result = tune(
    objective=objective,
    space=space,
    strategy="llm_branching_agent",
    strategy_kwargs={
        "llm": ollama_llm,           # Provide any LLM via a simple callable (prompt: str) -> str
        "max_depth": 5,              # Maximum tree depth
        "beam_width": 3,             # Keep top 3 trials per depth
        "branch_factor": 3,          # Up to 3 branches per parent
        "enable_pruning": True,      # Prune weak branches
    },
    max_trials=100,
    max_concurrent=4
)
```

Combines systematic tree exploration with intelligent parameter variation based on lineage analysis.

### Custom Strategy Configuration

Pass additional arguments via `strategy_kwargs`:

```python
result = tune(
    objective=objective,
    space=space,
    strategy="custom_strategy",
    strategy_kwargs={
        "param1": value1,
        "param2": value2,
    }
)
```

### Early Stopping Policies

Terminate underperforming trials early to save computational resources:

```python
# Median Early Stopping - Stop bottom 50% at checkpoints
result = tune(
    objective=objective,
    space=space,
    strategy="random",
    early_stopping="median",  # Kill trials below median
    early_stopping_kwargs={
        "check_interval": 100,    # Check every 100 progress reports
        "min_trials": 3,          # Need at least 3 trials to compare
    },
    metric="val_loss",            # Metric to monitor
    mode="min",                   # Minimize the metric
    max_trials=100,
    max_concurrent=8,
)

# Aggressive Early Stopping - Check more frequently (every 25 steps)
result = tune(
    objective=objective,
    space=space,
    strategy="bayesian",
    early_stopping="aggressive",
    max_trials=100,
)

# Patient Early Stopping - Check less frequently (every 200 steps)
result = tune(
    objective=objective,
    space=space,
    strategy="random",
    early_stopping="patient",
    max_trials=100,
)
```

Early stopping policies monitor the metrics reported via `ctx.report()` and terminate trials that are underperforming relative to others. The `median` policy kills the bottom 50% of trials at each checkpoint, while `aggressive` and `patient` variants adjust the checking frequency.

## Storage Options

### In-Memory Storage (Default)

By default, results are stored in memory:

```python
result = tune(
    objective=objective,
    space=space,
    max_trials=50
    # storage=None  # Default: in-memory
)
```

### SQLite Storage

Persist results to a SQLite database:

```python
result = tune(
    objective=objective,
    space=space,
    storage="sqlite:///experiments.db",
    max_trials=100
)

# Results are saved and can be analyzed later
```

### Benefits of Persistent Storage

- **Resume experiments**: Continue from where you left off
- **Share results**: Database file can be shared with others
- **Post-hoc analysis**: Query and analyze results later
- **Fault tolerance**: Data persists even if the process crashes

## Monitoring and Visualization

### Live Progress

Show real-time progress during optimization:

```python
result = tune(
    objective=objective,
    space=space,
    show_progress=True,  # Display live progress
    max_trials=50
)
```

### Summary Statistics

Display a summary after optimization:

```python
result = tune(
    objective=objective,
    space=space,
    show_summary=True,  # Show final summary
    max_trials=50
)
```

### Both Progress and Summary

```python
result = tune(
    objective=objective,
    space=space,
    show_progress=True,
    show_summary=True,
    max_trials=50
)
```

## Understanding Results

The `tune()` function returns a dictionary with optimization results:

### Result Structure

```python
result = tune(objective=objective, space=space, max_trials=50)

# Access the best trial information
best_info = result["best"]
```

### Best Trial Information

The `best` dictionary contains:

```python
{
    "trial_id": "trial_abc123",           # Unique trial identifier
    "params": {                           # Best hyperparameters
        "learning_rate": 0.01,
        "batch_size": 64
    },
    "score": 0.95,                        # Objective score
    "metrics": {                          # Additional metrics
        "final_accuracy": 0.95,
        "training_time": 120.5
    }
}
```

### Optimization Modes

Control whether to minimize or maximize:

```python
# Maximize score (default)
result = tune(
    objective=objective,
    space=space,
    metric="accuracy",
    mode="max"
)

# Minimize loss
result = tune(
    objective=objective,
    space=space,
    metric="loss",
    mode="min"
)
```

## Advanced Options

### Resource Limits

Control execution time and resources:

```python
result = tune(
    objective=objective,
    space=space,
    max_trials=100,        # Maximum number of trials
    max_concurrent=4,      # Parallel trials
    max_time_s=3600,       # Maximum time in seconds
)
```

### Custom Metrics

Optimize for specific metrics:

```python
def objective(ctx, **params) -> ObjectiveResult:
    # ... training code ...

    return ObjectiveResult(
        score=validation_score,  # Primary metric
        metrics={
            "accuracy": acc,
            "f1_score": f1,      # Can optimize for any metric
            "latency": latency
        }
    )

# Optimize for f1_score instead of score
result = tune(
    objective=objective,
    space=space,
    metric="f1_score",  # Specify which metric to optimize
    mode="max"
)
```

### Executor Selection

Choose between thread-based or process-based execution:

```python
# Thread-based executor (default, good for I/O-bound tasks)
result = tune(
    objective=objective,
    space=space,
    executor="thread"
)

# Process-based executor (better isolation, good for CPU-bound tasks)
result = tune(
    objective=objective,
    space=space,
    executor="process"
)
```

## Best Practices

1. **Start Simple**: Begin with random search and a small number of trials
2. **Use Log Scale**: For learning rates and similar parameters, use `log=True`
3. **Report Progress**: Use `ctx.report()` to track training progress
4. **Handle Early Stopping**: Check `ctx.should_stop()` to save computation
5. **Persist Results**: Use SQLite storage for important experiments
6. **Monitor Progress**: Enable `show_progress=True` for interactive sessions
7. **Choose the Right Strategy**:
   - Random for initial exploration
   - Grid for exhaustive search of discrete spaces
   - Beam search for hierarchical exploration
   - LLM agent for intelligent, pattern-aware optimization

## Common Patterns

### Hyperparameter Tuning for ML Models

```python
from hyperion import tune, Float, Choice, Int, ObjectiveResult

def train_neural_network(ctx, **params) -> ObjectiveResult:
    model = build_model(
        layers=params["layers"],
        units=params["units"],
        dropout=params["dropout"]
    )

    optimizer = create_optimizer(
        name=params["optimizer"],
        lr=params["learning_rate"]
    )

    best_val_score = 0.0
    for epoch in range(params["epochs"]):
        train_loss = train_epoch(model, optimizer)
        val_score = validate(model)
        best_val_score = max(best_val_score, val_score)

        ctx.report(
            step=epoch,
            train_loss=train_loss,
            val_score=val_score
        )

        if ctx.should_stop():
            break

    return ObjectiveResult(score=best_val_score)

result = tune(
    objective=train_neural_network,
    space={
        "layers": Int(2, 5),
        "units": Int(32, 256, log=True),
        "dropout": Float(0.0, 0.5),
        "optimizer": Choice(["adam", "sgd", "rmsprop"]),
        "learning_rate": Float(0.0001, 0.1, log=True),
        "epochs": 50  # Fixed parameter
    },
    strategy="random",
    max_trials=100,
    max_concurrent=4,
    show_progress=True,
    storage="sqlite:///nn_tuning.db"
)
```

### A/B Testing Different Configurations

```python
def test_configuration(ctx, **config) -> ObjectiveResult:
    # Run your system with the given configuration
    performance = run_system(**config)

    return ObjectiveResult(
        score=performance["throughput"],
        metrics={
            "latency": performance["latency"],
            "error_rate": performance["error_rate"],
            "cpu_usage": performance["cpu_usage"]
        }
    )

result = tune(
    objective=test_configuration,
    space={
        "cache_size": Choice([128, 256, 512, 1024]),
        "num_workers": Int(1, 16),
        "batch_timeout": Float(0.1, 2.0),
        "algorithm": Choice(["v1", "v2", "v3"])
    },
    strategy="grid",  # Test all combinations
    metric="throughput",
    mode="max"
)
```

## Next Steps

- See the [Framework Guide](framework-guide.md) for advanced usage
- Check out [Examples](../examples/) for complete working examples
- Read the [Design Document](design-doc.md) for architecture details
