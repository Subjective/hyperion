"""Integration test for LLM agent policy."""

from hyperion.api.tune import tune
from hyperion.core.models import ObjectiveResult
from hyperion.framework.search_space import Float
from hyperion.policies.llm_agent import LLMAgentPolicy


def test_llm_agent_can_be_created():
    """Test that LLMAgentPolicy can be instantiated."""
    space = {"x": Float(0, 1), "y": Float(0, 1)}

    # Provide a minimal llm callable that returns valid JSON
    def fake_llm(prompt: str) -> str:  # pragma: no cover - trivial
        return '{"rationale":"test","suggestions":[{"x":0.1,"y":0.2}]}'

    policy = LLMAgentPolicy(
        space=space,
        experiment_id="test_exp",
        llm=fake_llm,
        metric="score",
        mode="max",
    )

    assert policy is not None
    assert policy.space == space
    assert policy.experiment_id == "test_exp"
    # No provider-specific attributes are required


def test_llm_agent_fallback_to_random():
    """Test that LLM agent falls back to random sampling when LLM is unavailable."""

    def objective(ctx, x=0.5, y=0.5):
        return ObjectiveResult(score=x * y)

    space = {"x": Float(0, 1), "y": Float(0, 1)}

    # Use a bogus llm output to trigger fallback to random
    result = tune(
        objective,
        space,
        strategy="llm_agent",
        strategy_kwargs={
            "llm": lambda prompt: "not-json",  # invalid JSON triggers fallback
        },
        max_trials=2,
        max_concurrent=1,
        metric="score",
        mode="max",
        executor="thread",
    )

    # Should still return a result (using fallback)
    assert isinstance(result, dict)
    assert "best" in result
