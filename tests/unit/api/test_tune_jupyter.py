"""Tests for tune() API compatibility with Jupyter notebooks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyperion.api.tune import tune
from hyperion.core.models import ObjectiveResult
from hyperion.framework.search_space import Float


@pytest.mark.asyncio
async def test_tune_works_with_existing_event_loop():
    """Test that tune() works when called from within an async context (like Jupyter)."""

    # Simple objective
    def objective(ctx, x=0.5):
        return ObjectiveResult(score=x)

    space = {"x": Float(0, 1)}

    # This simulates the Jupyter environment where we already have a running loop
    # The test itself runs in an async context via pytest.mark.asyncio
    assert asyncio.get_running_loop() is not None

    # Mock the runner to avoid actually running trials
    with patch("hyperion.api.tune.ExperimentRunner") as mock_runner_class:
        mock_runner = MagicMock()
        mock_runner.experiment_id = "test-exp-123"
        mock_runner.services = {}  # Add services attribute

        async def mock_run():
            return {"experiment": "tune", "best": {"score": 0.9}}

        mock_runner.run = mock_run
        mock_runner_class.return_value = mock_runner

        with patch("hyperion.api.tune._create_default_services") as mock_services:
            mock_stores = MagicMock()
            mock_stores.close = AsyncMock()
            mock_services.return_value = {"stores": mock_stores}

            # This should work even though we're in an async context
            result = tune(
                objective,
                space,
                max_trials=1,
                max_concurrent=1,
            )

            assert result == {"experiment": "tune", "best": {"score": 0.9}}


def test_tune_still_works_without_event_loop():
    """Test that tune() still works in regular synchronous contexts."""

    # Ensure we're not in an async context
    try:
        asyncio.get_running_loop()
        pytest.skip("This test requires no running event loop")
    except RuntimeError:
        pass  # Good, no loop running

    # Simple objective
    def objective(ctx, x=0.5):
        return ObjectiveResult(score=x)

    space = {"x": Float(0, 1)}

    # Mock the runner
    with patch("hyperion.api.tune.ExperimentRunner") as mock_runner_class:
        mock_runner = MagicMock()
        mock_runner.experiment_id = "test-exp-456"
        mock_runner.services = {}  # Add services attribute

        async def mock_run():
            return {"experiment": "tune", "best": {"score": 0.8}}

        mock_runner.run = mock_run
        mock_runner_class.return_value = mock_runner

        with patch("hyperion.api.tune._create_default_services") as mock_services:
            mock_stores = MagicMock()
            mock_stores.close = AsyncMock()
            mock_services.return_value = {"stores": mock_stores}

            # This should work in a regular sync context
            result = tune(
                objective,
                space,
                max_trials=1,
                max_concurrent=1,
            )

            assert result == {"experiment": "tune", "best": {"score": 0.8}}


@pytest.mark.asyncio
async def test_nest_asyncio_applied_only_once():
    """Test that nest_asyncio.apply() is idempotent and doesn't cause issues."""

    # Simple objective
    def objective(ctx, x=0.5):
        return ObjectiveResult(score=x)

    space = {"x": Float(0, 1)}

    # Mock the runner
    with patch("hyperion.api.tune.ExperimentRunner") as mock_runner_class:
        mock_runner = MagicMock()
        mock_runner.experiment_id = "test-exp-789"
        mock_runner.services = {}  # Add services attribute

        async def mock_run():
            return {"experiment": "tune", "best": {"score": 0.7}}

        mock_runner.run = mock_run
        mock_runner_class.return_value = mock_runner

        with patch("hyperion.api.tune._create_default_services") as mock_services:
            mock_stores = MagicMock()
            mock_stores.close = AsyncMock()
            mock_services.return_value = {"stores": mock_stores}

            # Call tune() multiple times in the same async context
            result1 = tune(objective, space, max_trials=1)
            result2 = tune(objective, space, max_trials=1)

            # Both should work
            assert result1 == {"experiment": "tune", "best": {"score": 0.7}}
            assert result2 == {"experiment": "tune", "best": {"score": 0.7}}


@pytest.mark.asyncio
async def test_tune_with_real_simple_execution():
    """Integration test with real execution in async context."""

    # Simple objective that actually runs
    def simple_objective(ctx, x=0.5):
        ctx.report(1, score=x)
        return ObjectiveResult(score=x)

    space = {"x": Float(0, 1)}

    # This should complete without errors in an async context
    result = tune(
        simple_objective,
        space,
        strategy="random",
        max_trials=1,
        max_concurrent=1,
        show_progress=False,
        show_summary=False,
    )

    # Should have a result
    assert "best" in result
    assert result["experiment"] == "random"
