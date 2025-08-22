# Hyperion Framework Guide

This guide covers advanced usage of Hyperion's framework layer for building custom optimization workflows, implementing policies, and leveraging the event-driven architecture.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Custom Experiments with ExperimentSpec](#custom-experiments-with-experimentspec)
3. [Resources and Budgets](#resources-and-budgets)
4. [Implementing Custom Policies](#implementing-custom-policies)
5. [Event-Driven Architecture](#event-driven-architecture)
6. [Pipeline Composition](#pipeline-composition)
7. [Storage Backends](#storage-backends)
8. [Lineage and Tree-Based Search](#lineage-and-tree-based-search)
9. [Multi-Experiment Orchestration](#multi-experiment-orchestration)

## Architecture Overview

Hyperion follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────┐
│           API Layer (tune())            │
├─────────────────────────────────────────┤
│     Framework Layer (ExperimentSpec)    │
├─────────────────────────────────────────┤
│  Core Layer (Events, Bus, Controller)   │
├─────────────────────────────────────────┤
│    Storage Layer (Stores, EventLog)     │
└─────────────────────────────────────────┘
```

### Key Components

- **EventBus**: Central message bus for all components
- **Controller**: Enforces commands, schedules trials, manages state
- **Policies**: Strategies and agents that make optimization decisions
- **Executors**: Run trials in isolated processes/threads
- **Stores**: Persist experiments, trials, and events

## Custom Experiments with ExperimentSpec

For advanced control, use `ExperimentSpec` directly instead of `tune()`:

```python
from hyperion.framework import (
    ExperimentSpec,
    Resources,
    Budget,
    Pipeline,
    Monitoring,
    ExperimentRunner
)
from hyperion.policies import RandomSearchPolicy

# Define the experiment specification
spec = ExperimentSpec(
    name="my_experiment",
    objective=my_objective_function,
    search_space={
        "param1": Float(0.0, 1.0),
        "param2": Choice(["a", "b", "c"])
    },
    pipeline=Pipeline(steps=[
        RandomSearchPolicy(space=search_space, experiment_id="temp")
    ]),
    resources=Resources(
        max_concurrent=8,
        timeout_s=3600  # 1 hour timeout
    ),
    budget=Budget(
        max_trials=100,
        max_time_s=7200,  # 2 hours
        target=0.95,      # Stop if we reach this score
        metric="accuracy",
        mode="max"
    ),
    monitoring=Monitoring(
        callbacks=[MyCustomCallback()]
    ),
    tags={"project": "research", "version": "v1"}
)

# Option 1: Use convenience function (recommended)
from hyperion.framework import create_services

services = create_services(
    storage="memory",     # or "sqlite:///experiments.db"
    max_concurrent=8,
    executor="async"      # or "process" for better isolation
)

# Option 2: Manual service creation (advanced)
# When you need full control over service configuration
from hyperion.core.bus import InMemoryBus
from hyperion.core.executor import LocalAsyncExecutor
from hyperion.core.capacity import CapacityManager
from hyperion.storage.memory import InMemoryStores

bus = InMemoryBus()
services = {
    "bus": bus,
    "stores": InMemoryStores(),
    "executor": LocalAsyncExecutor(bus),
    "capacity": CapacityManager(max_concurrent=8)
}

# Run the experiment
runner = ExperimentRunner(spec, services=services)
result = await runner.run()

# Option 3: One-liner execution (simplest)
from hyperion.framework import run_experiment

result = await run_experiment(spec)  # Uses default services automatically
```

## Resources and Budgets

### Resource Configuration

Control computational resources and scheduling:

```python
from hyperion.framework import Resources

resources = Resources(
    max_concurrent=4,      # Maximum parallel trials
    quotas={               # Per-experiment quotas (for multi-experiment)
        "exp1": 2,
        "exp2": 2
    },
    weights={              # Fair-share weights
        "exp1": 1.0,
        "exp2": 2.0        # exp2 gets 2x priority
    },
    timeout_s=3600         # Global timeout in seconds
)
```

### Budget Configuration

Define stopping conditions:

```python
from hyperion.framework import Budget

budget = Budget(
    max_trials=100,        # Stop after 100 trials
    max_time_s=3600,       # Stop after 1 hour
    target=0.99,           # Stop if target is reached
    metric="f1_score",     # Which metric to check
    mode="max"             # Maximize or minimize
)
```

The experiment stops when ANY condition is met.

## Implementing Custom Policies

Policies are the decision-making components that determine which trials to run.

### Policy Interface

```python
from typing import List, Optional
from hyperion.framework import Policy, StartTrial, KillTrial, Action
from hyperion.core.state import ReadableState
from hyperion.core.events import Event

class MyCustomPolicy(Policy):
    def __init__(self, space, experiment_id, **kwargs):
        self.space = space
        self.experiment_id = experiment_id
        self.history = []

    async def on_events(self, events: List[Event]) -> None:
        """Process incoming events to update internal state."""
        for event in events:
            if event.type == "TRIAL_COMPLETED":
                self.history.append(event.data)

    async def decide(self, state: ReadableState) -> List[Action]:
        """Make optimization decisions based on current state."""
        actions = []

        # Check capacity
        if state.capacity_free() > 0:
            # Generate new trial parameters
            params = self._generate_params()

            actions.append(StartTrial(
                experiment_id=self.experiment_id,
                params=params,
                parent_trial_id=None,  # Set for tree-based search
                tags={"generation": len(self.history)}
            ))

        # Optionally kill underperforming trials
        for trial in state.running_trials(self.experiment_id):
            if self._should_kill(trial):
                actions.append(KillTrial(
                    trial_id=trial.trial_id,
                    reason="Underperforming"
                ))

        return actions

    async def rationale(self) -> Optional[str]:
        """Provide reasoning for decisions (for audit trail)."""
        return f"Generated trial based on {len(self.history)} completed trials"

    def _generate_params(self):
        # Your parameter generation logic
        pass

    def _should_kill(self, trial):
        # Your early stopping logic
        return False
```

### Registering Custom Policies

Add your policy to the strategy registry:

```python
# In your code or a custom module
from hyperion.policies import REGISTRY

def create_my_policy(space, **kwargs):
    return MyCustomPolicy(space, **kwargs)

REGISTRY["my_custom"] = create_my_policy

# Now you can use it
result = tune(
    objective=objective,
    space=space,
    strategy="my_custom",
    strategy_kwargs={"param": value}
)
```

## Event-Driven Architecture

Hyperion uses events for all communication between components.

### Event Types

```python
from hyperion.core.events import EventType

# Lifecycle events
EventType.EXPERIMENT_STARTED
EventType.TRIAL_STARTED
EventType.TRIAL_PROGRESS
EventType.TRIAL_COMPLETED
EventType.TRIAL_FAILED
EventType.TRIAL_KILLED

# Control events
EventType.CAPACITY_AVAILABLE
EventType.DECISION_RECORDED
```

### Subscribing to Events

```python
from hyperion.core.bus import InMemoryBus

bus = InMemoryBus()

# Subscribe to specific event type
def on_trial_completed(event):
    print(f"Trial {event.data['trial_id']} completed with score {event.data['score']}")

bus.subscribe(EventType.TRIAL_COMPLETED, on_trial_completed)

# Subscribe to all events
def log_all_events(event):
    print(f"[{event.type}] {event.data}")

bus.subscribe("*", log_all_events)
```

### Custom Callbacks

Implement callbacks to react to events:

```python
class MetricsLogger:
    def __init__(self, log_file):
        self.log_file = log_file

    async def on_event(self, event):
        if event.type == EventType.TRIAL_PROGRESS:
            # Log metrics to file
            with open(self.log_file, "a") as f:
                f.write(f"{event.ts},{event.data}\n")

# Use in experiment
spec = ExperimentSpec(
    # ...
    monitoring=Monitoring(callbacks=[
        MetricsLogger("metrics.csv")
    ])
)
```

## Pipeline Composition

Combine multiple policies in a pipeline:

```python
from hyperion.framework import Pipeline
from hyperion.policies import RandomSearchPolicy, MedianEarlyStoppingPolicy

pipeline = Pipeline(steps=[
    # First: Random exploration
    RandomSearchPolicy(space=space, experiment_id="temp"),

    # Second: Early stopping to kill underperformers
    MedianEarlyStoppingPolicy(
        metric="val_loss",
        mode="min",
        check_interval=50,  # Check every 50 progress reports
        min_trials=3
    ),

    # Third: Custom refinement policy
    RefinementPolicy(top_k=5)
])

spec = ExperimentSpec(
    # ...
    pipeline=pipeline
)
```

Policies in the pipeline run concurrently and can interact through events. Early stopping policies monitor `TRIAL_PROGRESS` events and issue `KillTrial` actions for underperforming trials.

## Storage Backends

### In-Memory Storage

Fast, ephemeral storage for development:

```python
from hyperion.storage.memory import InMemoryStores

stores = InMemoryStores()
```

### SQLite Storage

Persistent storage with SQL queries:

```python
from hyperion.storage.sql import SQLiteStores

stores = SQLiteStores("sqlite:///experiments.db")

# Access individual stores
event_log = stores.events
trial_store = stores.trials
experiment_store = stores.experiments
```

### Custom Storage Implementation

Implement the storage protocols for custom backends:

```python
from hyperion.core.state import EventLog, TrialStore, ExperimentStore
from typing import List, Optional

class MyEventLog(EventLog):
    async def append(self, evt: Event) -> None:
        # Store event
        pass

    async def tail(self, n: int, aggregate_id: Optional[str] = None) -> List[Event]:
        # Retrieve recent events
        pass

class MyTrialStore(TrialStore):
    def create(self, experiment_id: str, params: dict, lineage: dict) -> Trial:
        # Create new trial
        pass

    def get(self, trial_id: str) -> Trial:
        # Retrieve trial
        pass

    def update(self, trial_id: str, **fields) -> None:
        # Update trial fields
        pass

    def running(self, experiment_id: Optional[str] = None) -> List[Trial]:
        # Get running trials
        pass

    def best_of(self, experiment_id: str, metric: str, mode: str) -> dict:
        # Get best trial
        pass

class MyDecisionStore(DecisionStore):
    def create(self, decision: Decision) -> None:
        # Store decision record
        pass

    def get(self, decision_id: str) -> Decision | None:
        # Retrieve decision
        pass

    def list_by_experiment(self, experiment_id: str) -> List[Decision]:
        # Get experiment decisions
        pass

# Similar for ExperimentStore

class MyStores:
    def __init__(self):
        self.events = MyEventLog()
        self.trials = MyTrialStore()
        self.experiments = MyExperimentStore()
        self.decisions = MyDecisionStore()
```

## Lineage and Tree-Based Search

Hyperion natively supports tree-based search with trial lineage tracking.

### Using Lineage in Policies

```python
class TreeSearchPolicy(Policy):
    async def decide(self, state: ReadableState) -> List[Action]:
        actions = []

        # Get best trials from previous generation
        best_parents = state.best_trials(
            experiment_id=self.experiment_id,
            top_n=3,
            key="score",
            mode="max"
        )

        for parent in best_parents:
            # Generate children by mutating parent
            for i in range(2):  # 2 children per parent
                child_params = self._mutate(parent.params)

                actions.append(StartTrial(
                    experiment_id=self.experiment_id,
                    params=child_params,
                    parent_trial_id=parent.trial_id,  # Set lineage
                    tags={
                        "generation": parent.depth + 1,
                        "branch": f"{parent.branch_id}-{i}"
                    }
                ))

        return actions

    def _mutate(self, params):
        # Mutation logic
        mutated = params.copy()
        # Add noise, crossover, etc.
        return mutated
```

### Querying by Lineage

```python
# Get trials at specific depth
generation_2 = state.trials_by_depth(experiment_id, depth=2)

# Get trial ancestry
trial = state.trial(trial_id)
parent = state.trial(trial.parent_trial_id) if trial.parent_trial_id else None

# Track branches
same_branch = [
    t for t in state.running_trials()
    if t.branch_id == trial.branch_id
]
```

## Multi-Experiment Orchestration

Run multiple experiments concurrently with resource sharing:

```python
import asyncio
from hyperion.framework import ExperimentRunner, ExperimentSpec, create_services

async def run_multiple_experiments():
    # Option 1: Use convenience function (recommended)
    services = create_services(
        storage="sqlite:///multi_exp.db",
        max_concurrent=16,
        executor="async"
    )

    # Option 2: Manual service creation for custom quotas
    from hyperion.core.bus import InMemoryBus
    from hyperion.core.executor import LocalAsyncExecutor
    from hyperion.core.capacity import CapacityManager
    from hyperion.storage.sql import SQLiteStores

    bus = InMemoryBus()
    services = {
        "bus": bus,
        "stores": SQLiteStores("sqlite:///multi_exp.db"),
        "executor": LocalAsyncExecutor(bus),
        "capacity": CapacityManager(
            max_concurrent=16,
            quotas={"exp1": 8, "exp2": 8}
        )
    }

    # Define experiments
    exp1 = ExperimentSpec(
        name="exp1",
        objective=objective1,
        # ...
    )

    exp2 = ExperimentSpec(
        name="exp2",
        objective=objective2,
        # ...
    )

    # Run concurrently
    runner1 = ExperimentRunner(exp1, services=services)
    runner2 = ExperimentRunner(exp2, services=services)

    results = await asyncio.gather(
        runner1.run(),
        runner2.run()
    )

    return results
```

## Advanced Patterns

### Warm Starting

Continue from previous results:

```python
class WarmStartPolicy(Policy):
    def __init__(self, previous_best, **kwargs):
        super().__init__(**kwargs)
        self.previous_best = previous_best

    async def decide(self, state: ReadableState) -> List[Action]:
        # Start with variations of previous best
        actions = []
        for i in range(5):
            params = self._perturb(self.previous_best)
            actions.append(StartTrial(
                experiment_id=self.experiment_id,
                params=params
            ))
        return actions
```

### Adaptive Strategies

Switch strategies based on progress:

```python
class AdaptivePolicy(Policy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exploration_phase = True
        self.trials_count = 0

    async def decide(self, state: ReadableState) -> List[Action]:
        self.trials_count += 1

        # Switch to exploitation after exploration
        if self.trials_count > 50 and self.exploration_phase:
            self.exploration_phase = False
            print("Switching to exploitation phase")

        if self.exploration_phase:
            # Random exploration
            params = sample_space(self.space)
        else:
            # Exploit best regions
            best = state.best_trials(self.experiment_id, top_n=1, key="score", mode="max")[0]
            params = self._local_search(best.params)

        return [StartTrial(self.experiment_id, params)]
```

### Checkpointing and Recovery

Save and restore experiment state:

```python
# Save checkpoint
async def save_checkpoint(stores, experiment_id):
    events = await stores.events.tail(n=10000, aggregate_id=experiment_id)
    trials = stores.trials.list_by_experiment(experiment_id)

    checkpoint = {
        "events": events,
        "trials": trials,
        "timestamp": datetime.now()
    }

    with open(f"checkpoint_{experiment_id}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)

# Restore from checkpoint
def restore_checkpoint(experiment_id):
    with open(f"checkpoint_{experiment_id}.pkl", "rb") as f:
        return pickle.load(f)
```

## Best Practices

1. **Event Processing**: Keep event handlers lightweight and async
2. **State Queries**: Use ReadableState for consistent views
3. **Lineage Tracking**: Always set parent_trial_id for tree search
4. **Resource Management**: Set appropriate quotas for multi-experiment runs
5. **Error Handling**: Policies should handle failures gracefully
6. **Audit Trail**: Use rationale() to document decision logic
7. **Testing**: Test policies with mock state before integration

## Debugging Tips

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("hyperion")
```

### Monitor Event Flow

```python
def debug_events(event):
    print(f"[{event.ts}] {event.type}: {event.data}")

bus.subscribe("*", debug_events)
```

### Inspect State

```python
# In your policy
async def decide(self, state: ReadableState) -> List[Action]:
    print(f"Free capacity: {state.capacity_free()}")
    print(f"Running trials: {len(state.running_trials())}")

    # Your decision logic
```

## Next Steps

- Review the [API Guide](api-guide.md) for high-level usage
- Check [Examples](../examples/) for complete implementations
- Read the [Design Document](design-doc.md) for architectural details
- Explore the source code for deeper understanding
