"""Tests for strategy registry and resolution."""

import pytest

from hyperion.framework.search_space import Float
from hyperion.policies import BUILTIN, resolve
from hyperion.policies.random_search import RandomSearchPolicy


def test_resolve_random_returns_policy_instance():
    """Resolving 'random' returns a RandomSearchPolicy configured with provided args."""
    space = {"x": Float(0, 1), "y": Float(-1, 1)}

    policy = resolve(
        "random",
        space=space,
        metric="score",
        mode="max",
        custom_ignored="value",
    )

    assert isinstance(policy, RandomSearchPolicy)
    # After isinstance check, we know it's RandomSearchPolicy
    random_policy = policy  # type: RandomSearchPolicy
    assert random_policy.space == space
    # experiment_id is set later by the runner
    assert random_policy.experiment_id is None


def test_resolve_unknown_raises_value_error():
    """Unknown strategy name raises with helpful message."""
    space = {"x": Float(0, 1)}

    with pytest.raises(ValueError) as exc_info:
        resolve("unknown_strategy", space=space)

    assert "unknown_strategy" in str(exc_info.value)
    assert "Available" in str(exc_info.value)


def test_registry_contains_expected_strategies():
    """Registry should contain at least the random strategy."""
    assert "random" in BUILTIN
    assert callable(BUILTIN["random"])
