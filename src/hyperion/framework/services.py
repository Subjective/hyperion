"""Convenience factory functions for creating framework services."""

from typing import Any, Literal

from hyperion.core.bus import InMemoryBus
from hyperion.core.capacity import CapacityManager
from hyperion.core.executor import (
    LocalAsyncExecutor,
    LocalProcessExecutor,
    TrialExecutor,
)
from hyperion.core.state import Stores
from hyperion.storage.memory import InMemoryStores
from hyperion.storage.sql import SQLiteStores


def create_services(
    storage: Literal["memory", "sqlite"] | str = "memory",
    max_concurrent: int = 4,
    executor: Literal["async", "process"] = "async",
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a standard set of services for framework experiments.

    Args:
        storage: Storage type. "memory" for in-memory, "sqlite" for SQLite,
                or a connection string like "sqlite:///path/to/db.sqlite"
        max_concurrent: Maximum number of concurrent trials
        executor: Executor type. "async" for faster startup, "process" for isolation
        **kwargs: Additional arguments passed to storage (e.g., connection_string)

    Returns:
        Dictionary of services ready for use with ExperimentRunner

    Example:
        ```python
        from hyperion.framework import create_services, ExperimentSpec, ExperimentRunner

        # Simple usage
        services = create_services()

        # Custom configuration
        services = create_services(
            storage="sqlite:///experiments.db",
            max_concurrent=8,
            executor="process"
        )

        runner = ExperimentRunner(spec, services=services)
        result = await runner.run()
        ```
    """
    # Create event bus
    bus = InMemoryBus()

    # Create storage
    # TODO: Fix protocol variance issue - mutable attributes cause type checker
    # issues with concrete implementations). We can consider making these read-only
    # properties or using a different approach that maintains backwards compatibility.
    stores: Stores
    if storage == "memory":
        stores = InMemoryStores()  # type: ignore[assignment]
    elif storage == "sqlite":
        stores = SQLiteStores("sqlite:///hyperion.db", **kwargs)  # type: ignore[assignment]
    elif storage.startswith("sqlite://"):
        stores = SQLiteStores(storage, **kwargs)  # type: ignore[assignment]
    else:
        raise ValueError(f"Unsupported storage type: {storage}")

    # Create executor
    executor_instance: TrialExecutor
    if executor == "async":
        executor_instance = LocalAsyncExecutor(bus)
    elif executor == "process":
        executor_instance = LocalProcessExecutor(bus)
    else:
        raise ValueError(f"Unsupported executor type: {executor}")

    # Create capacity manager
    capacity = CapacityManager(max_concurrent=max_concurrent)

    return {
        "bus": bus,
        "stores": stores,
        "executor": executor_instance,
        "capacity": capacity,
    }


async def run_experiment(
    spec: Any, services: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run an experiment with automatic service creation.

    Args:
        spec: ExperimentSpec defining the experiment
        services: Optional services dict. If None, creates default services

    Returns:
        Experiment result dictionary

    Example:
        ```python
        from hyperion.framework import ExperimentSpec, run_experiment

        # Automatic service creation
        result = await run_experiment(spec)

        # Custom services
        services = create_services(storage="sqlite:///my_exp.db")
        result = await run_experiment(spec, services)
        ```
    """
    from hyperion.framework.experiment import ExperimentRunner

    if services is None:
        services = create_services()

    runner = ExperimentRunner(spec, services=services)
    return await runner.run()
