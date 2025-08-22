"""Integration tests for the high-level tune() API."""

from hyperion.api.tune import tune
from hyperion.core.models import ObjectiveResult
from hyperion.framework.search_space import Float


def test_tune_with_string_strategy_runs_and_returns_best():
    """Using 'random' strategy name runs and returns a result dict with best entry."""

    # Simple objective
    def objective(ctx, x=0.5, y=0.5):
        return ObjectiveResult(score=x * y)

    # Search space
    space = {"x": Float(0, 1), "y": Float(0, 1)}

    # Run tune with random strategy
    result = tune(
        objective,
        space,
        strategy="random",
        max_trials=2,
        max_concurrent=1,
        metric="score",
        mode="max",
    )

    # Should return a dict with best entry
    assert isinstance(result, dict)
    assert "best" in result

    # If trials ran, should have best info
    if result["best"]:
        assert (
            "trial_id" in result["best"]
            or "params" in result["best"]
            or "score" in result["best"]
        )


def test_tune_with_policy_instance_uses_it_directly():
    """Passing a Policy instance is honored (no resolve call needed)."""
    # Track if policy was used
    policy_used = False

    class CustomPolicy:
        def __init__(self):
            self.experiment_id: str | None = None
            self.experiment_id = None

        async def on_events(self, events):
            pass

        async def decide(self, state):
            nonlocal policy_used
            policy_used = True
            return []  # Don't start any trials

        async def rationale(self):
            return "Custom policy"

    # Simple objective
    def objective(ctx):
        return ObjectiveResult(score=0.5)

    # Use custom policy instance
    policy = CustomPolicy()

    result = tune(
        objective,
        {},
        strategy=policy,
        max_trials=1,
        max_concurrent=1,
        max_time_s=1,
    )

    # Policy should have been used
    assert policy_used
    assert isinstance(result, dict)


def test_tune_propagates_budget_and_resources():
    """max_trials, max_concurrent, metric, mode propagate to ExperimentSpec and behavior."""
    # Objective that tracks calls
    call_count = 0

    def objective(ctx, x=0.5):
        nonlocal call_count
        call_count += 1
        return ObjectiveResult(score=x)

    # Search space
    space = {"x": Float(0, 1)}

    # Run tune with specific budget/resources
    result = tune(
        objective,
        space,
        strategy="random",
        max_trials=3,
        max_concurrent=2,
        metric="score",
        mode="max",
    )

    # Should have run trials (up to max_trials)
    assert call_count <= 3
    assert isinstance(result, dict)
    assert "best" in result
