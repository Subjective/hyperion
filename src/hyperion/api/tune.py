"""High-level API for hyperparameter optimization."""

import asyncio
from typing import Any, Literal

import nest_asyncio  # type: ignore[import-untyped]

from hyperion.core.bus import InMemoryBus
from hyperion.core.capacity import CapacityManager
from hyperion.core.executor import LocalAsyncExecutor, LocalProcessExecutor
from hyperion.framework.experiment import (
    Budget,
    ExperimentRunner,
    ExperimentSpec,
    Monitoring,
    Pipeline,
    Resources,
)
from hyperion.framework.policy import Policy
from hyperion.policies import resolve
from hyperion.storage.memory import InMemoryStores
from hyperion.storage.sql import SQLiteStores
from hyperion.visualization.terminal import LiveEventLogger, TerminalVisualizer


def tune(
    objective: Any,
    space: dict[str, Any],
    *,
    strategy: str | Policy = "random",
    strategy_kwargs: dict[str, Any] | None = None,
    early_stopping: str | None = None,
    early_stopping_kwargs: dict[str, Any] | None = None,
    max_trials: int = 50,
    max_concurrent: int = 4,
    max_time_s: int | None = None,
    metric: str = "score",
    mode: Literal["min", "max"] = "max",
    storage: str | None = None,
    return_state: bool = False,
    executor: Literal["thread", "process"] = "thread",
    show_progress: bool = False,
    show_summary: bool = False,
) -> dict[str, Any]:
    """Run a single experiment end-to-end and return best params/score.

    This is the main entry point for simple hyperparameter optimization tasks.
    It creates all necessary services, runs the experiment, and returns the
    best result found.

    Args:
        objective: Callable that takes (ctx, **params) and returns ObjectiveResult
        space: Search space specification with distributions (optional - some policies don't need it)
        strategy: Strategy name (e.g., "random") or Policy instance
        strategy_kwargs: Additional arguments for strategy construction
        early_stopping: Early stopping strategy name (e.g., "median", "aggressive", "patient") or None
        early_stopping_kwargs: Additional arguments for early stopping construction
        max_trials: Maximum number of trials to run
        max_concurrent: Maximum concurrent trials
        max_time_s: Maximum time in seconds
        metric: Metric to optimize
        mode: Optimization mode - "max" or "min"
        storage: Storage backend URL (e.g., "sqlite:///hyperion.db")
        return_state: Whether to return full state (not implemented)
        executor: Executor type - "thread" or "process"
        show_progress: Whether to show live progress during execution
        show_summary: Whether to show summary at end

    Returns:
        Dict with "best" key containing best trial info
    """
    # Resolve strategy if it's a string
    if isinstance(strategy, str):
        kw = dict(strategy_kwargs or {})
        policy = resolve(
            strategy,
            space=space,
            metric=metric,
            mode=mode,
            **kw,
        )
    else:
        # Assume it's already a Policy instance
        policy = strategy

    # Build list of policies
    policies = [policy]

    # Add early stopping if requested
    if early_stopping:
        es_kw = dict(early_stopping_kwargs or {})
        # Prepend "stop:" if not already present for clarity
        es_name = (
            early_stopping
            if early_stopping.startswith("stop:")
            else f"stop:{early_stopping}"
        )
        es_policy = resolve(
            es_name,
            space=space,  # Pass space even though early stopping doesn't need it
            metric=metric,
            mode=mode,
            **es_kw,
        )
        policies.append(es_policy)

    # Set up monitoring callbacks if requested
    callbacks: list[LiveEventLogger] = []
    if show_progress:
        callbacks.append(LiveEventLogger(verbose=False))

    # Build experiment spec
    spec = ExperimentSpec(
        name=strategy if isinstance(strategy, str) else "tune",
        objective=objective,
        search_space=space,
        pipeline=Pipeline(steps=policies),
        resources=Resources(max_concurrent=max_concurrent),
        budget=Budget(
            max_trials=max_trials, max_time_s=max_time_s, metric=metric, mode=mode
        ),
        monitoring=Monitoring(callbacks=callbacks),
    )

    # Create services
    services = _create_default_services(storage, max_concurrent, executor)

    # Create and run runner
    runner = ExperimentRunner(spec, services=services)

    # Run the experiment (handle both regular and notebook/async contexts)
    try:
        loop = asyncio.get_running_loop()
        # Already in an event loop (e.g., Jupyter). Make loop re-entrant and run.
        nest_asyncio.apply()  # type: ignore[no-untyped-call]
        result = loop.run_until_complete(_async_tune(runner))
    except RuntimeError:
        # No running loop: create one with asyncio.run
        result = asyncio.run(_async_tune(runner))

    # Show summary if requested
    if show_summary and runner.experiment_id:
        visualizer = TerminalVisualizer(services["stores"])
        visualizer.print_summary(runner.experiment_id, detailed=True)

    # Include minimal state for callers (keep it simple for now)
    if return_state:
        result = {
            **result,
            "experiment_id": runner.experiment_id,
        }

    return result


async def _async_tune(runner: ExperimentRunner) -> dict[str, Any]:
    """Async wrapper for tune."""
    return await runner.run()


def _create_default_services(
    storage: str | None,
    max_concurrent: int,
    executor: Literal["thread", "process"],
) -> dict[str, Any]:
    """Create default services for tune().

    Args:
        storage: Storage backend URL (e.g., "sqlite:///hyperion.db")
        max_concurrent: Maximum concurrent trials
        executor: Executor type - "thread" or "process"

    Returns:
        Dict with bus, stores, executor, capacity
    """
    bus = InMemoryBus()

    # Create appropriate storage backend
    stores = (
        SQLiteStores(storage)
        if storage and storage.startswith("sqlite://")
        else InMemoryStores()
    )

    exec_impl = (
        LocalProcessExecutor(bus) if executor == "process" else LocalAsyncExecutor(bus)
    )
    capacity = CapacityManager(max_concurrent=max_concurrent)

    return {
        "bus": bus,
        "stores": stores,
        "executor": exec_impl,
        "capacity": capacity,
    }
