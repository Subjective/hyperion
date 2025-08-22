"""Built-in optimization strategies registry."""

from collections.abc import Callable
from typing import Any, Literal

from hyperion.framework.policy import Policy
from hyperion.policies.bayesian_optimization import BayesianOptimizationPolicy
from hyperion.policies.beam_search import BeamSearchPolicy
from hyperion.policies.early_stopping import (
    MedianEarlyStoppingPolicy,
    NoEarlyStoppingPolicy,
)
from hyperion.policies.grid_search import GridSearchPolicy
from hyperion.policies.llm_agent import LLMAgentPolicy
from hyperion.policies.llm_branching_agent import LLMBranchingAgent
from hyperion.policies.population_based_training import PopulationBasedTrainingPolicy
from hyperion.policies.random_search import RandomSearchPolicy

# Factory type for creating Policy instances
StrategyFactory = Callable[..., Policy]


def _random_factory(
    *, space: dict[str, Any], experiment_id: str | None = None, **_: Any
) -> Policy:
    return RandomSearchPolicy(space=space, experiment_id=experiment_id)


# Explicit registry of built-in strategy factories
def _grid_factory(
    *,
    space: dict[str, Any],
    experiment_id: str | None = None,
    grid_values: dict[str, list[Any]] | None = None,
    **_: Any,
) -> Policy:
    return GridSearchPolicy(
        space=space, experiment_id=experiment_id, grid_values=grid_values
    )


def _beam_factory(
    *,
    space: dict[str, Any],
    experiment_id: str | None = None,
    K: int = 2,
    width: int = 2,
    max_depth: int = 3,
    metric: str = "score",
    mode: Literal["min", "max"] = "max",
    prune: bool = False,
    **_: Any,
) -> Policy:
    # BeamSearchPolicy expects Literal["min","max"] for mode
    return BeamSearchPolicy(
        space=space,
        experiment_id=experiment_id,
        K=K,
        width=width,
        max_depth=max_depth,
        metric=metric,
        mode=mode,
        prune=prune,
    )


def _llm_agent_factory(
    *,
    space: dict[str, Any],
    experiment_id: str | None = None,
    llm: Callable[[str], str] | None = None,
    metric: str = "score",
    mode: Literal["min", "max"] = "max",
    max_history: int = 20,
    exploration_rate: float = 0.3,
    **kwargs: Any,
) -> Policy:
    # LLMAgentPolicy expects Literal["min","max"] for mode
    return LLMAgentPolicy(
        space=space,
        experiment_id=experiment_id,
        llm=llm,
        metric=metric,
        mode=mode,
        max_history=max_history,
        exploration_rate=exploration_rate,
        **kwargs,
    )


def _llm_branching_factory(
    *,
    space: dict[str, Any],
    experiment_id: str | None = None,
    llm: Callable[[str], str] | None = None,
    metric: str = "score",
    mode: Literal["min", "max"] = "max",
    max_depth: int = 5,
    beam_width: int = 3,
    branch_factor: int = 3,
    prune_ratio: float = 0.5,
    enable_pruning: bool = True,
    **kwargs: Any,
) -> Policy:
    # LLMBranchingAgent expects Literal["min","max"] for mode
    return LLMBranchingAgent(
        space=space,
        experiment_id=experiment_id,
        llm=llm,
        metric=metric,
        mode=mode,
        max_depth=max_depth,
        beam_width=beam_width,
        branch_factor=branch_factor,
        prune_ratio=prune_ratio,
        enable_pruning=enable_pruning,
        **kwargs,
    )


def _bayesian_factory(
    *,
    space: dict[str, Any],
    experiment_id: str | None = None,
    n_initial: int = 10,
    acquisition: str = "ei",
    xi: float = 0.01,
    metric: str = "score",
    mode: Literal["min", "max"] = "max",
    **kwargs: Any,
) -> Policy:
    # BayesianOptimizationPolicy expects Literal["min","max"] for mode
    return BayesianOptimizationPolicy(
        space=space,
        experiment_id=experiment_id,
        n_initial=n_initial,
        acquisition=acquisition,
        xi=xi,
        metric=metric,
        mode=mode,
        **kwargs,
    )


def _pbt_factory(
    *,
    space: dict[str, Any],
    experiment_id: str | None = None,
    population_size: int = 8,
    perturbation_factor: float = 1.2,
    metric: str = "score",
    mode: Literal["min", "max"] = "max",
    **kwargs: Any,
) -> Policy:
    # PopulationBasedTrainingPolicy expects Literal["min","max"] for mode
    return PopulationBasedTrainingPolicy(
        space=space,
        experiment_id=experiment_id,
        population_size=population_size,
        perturbation_factor=perturbation_factor,
        metric=metric,
        mode=mode,
        **kwargs,
    )


def _no_stop_factory(
    *,
    space: dict[str, Any] | None = None,
    experiment_id: str | None = None,
    **kwargs: Any,
) -> Policy:
    # NoEarlyStoppingPolicy doesn't need space or other args
    # Remove metric/mode from kwargs since NoEarlyStoppingPolicy doesn't use them
    kwargs.pop("metric", None)
    kwargs.pop("mode", None)
    return NoEarlyStoppingPolicy(experiment_id=experiment_id, **kwargs)


def _median_stop_factory(
    *,
    space: dict[str, Any] | None = None,
    experiment_id: str | None = None,
    metric: str = "val_loss",
    mode: Literal["min", "max"] = "min",
    check_interval: int = 100,
    **kwargs: Any,
) -> Policy:
    # MedianEarlyStoppingPolicy doesn't need space
    return MedianEarlyStoppingPolicy(
        experiment_id=experiment_id,
        metric=metric,
        mode=mode,
        check_interval=check_interval,
        **kwargs,
    )


def _aggressive_stop_factory(
    *,
    space: dict[str, Any] | None = None,
    experiment_id: str | None = None,
    metric: str = "val_loss",
    mode: Literal["min", "max"] = "min",
    **kwargs: Any,
) -> Policy:
    # Aggressive uses shorter interval
    check_interval = kwargs.pop("check_interval", 25)
    return MedianEarlyStoppingPolicy(
        experiment_id=experiment_id,
        metric=metric,
        mode=mode,
        check_interval=check_interval,
        **kwargs,
    )


def _patient_stop_factory(
    *,
    space: dict[str, Any] | None = None,
    experiment_id: str | None = None,
    metric: str = "val_loss",
    mode: Literal["min", "max"] = "min",
    **kwargs: Any,
) -> Policy:
    # Patient uses longer interval
    check_interval = kwargs.pop("check_interval", 200)
    return MedianEarlyStoppingPolicy(
        experiment_id=experiment_id,
        metric=metric,
        mode=mode,
        check_interval=check_interval,
        **kwargs,
    )


BUILTIN: dict[str, StrategyFactory] = {
    # Search strategies
    "random": _random_factory,
    "grid": _grid_factory,
    "beam_search": _beam_factory,
    "bayesian": _bayesian_factory,
    "pbt": _pbt_factory,
    "llm_agent": _llm_agent_factory,
    "llm_branching_agent": _llm_branching_factory,  # Alias for compatibility
    # Early stopping strategies (with stop: prefix)
    "stop:none": _no_stop_factory,
    "stop:median": _median_stop_factory,
    "stop:aggressive": _aggressive_stop_factory,
    "stop:patient": _patient_stop_factory,
    # Backwards compatibility aliases (without prefix)
    "none": _no_stop_factory,
    "median": _median_stop_factory,
    "aggressive": _aggressive_stop_factory,
    "patient": _patient_stop_factory,
}


def resolve(
    name: str,
    *,
    space: dict[str, Any] | None = None,
    metric: str = "score",
    mode: str = "max",
    **kwargs: Any,
) -> Policy:
    """Resolve any policy name to a Policy instance.

    Args:
        name: Policy name (strategy or early stopping)
        space: Search space specification (optional - only required by some policies)
        metric: Metric to optimize
        mode: Optimization mode - "min" or "max"
        **kwargs: Additional user-provided arguments

    Returns:
        Policy instance configured with the provided arguments
    """
    try:
        factory = BUILTIN[name]
    except KeyError as e:
        available = sorted(BUILTIN.keys())
        raise ValueError(
            f"Unknown policy '{name}'. Available policies: {available}"
        ) from e

    # Pass all args; policies decide what they need
    return factory(space=space, metric=metric, mode=mode, **kwargs)
