"""Tests for the high-level tune() API."""

from unittest.mock import MagicMock, patch

from hyperion.api.tune import tune
from hyperion.core.models import ObjectiveResult
from hyperion.framework.search_space import Float


@patch("hyperion.api.tune.resolve")
@patch("hyperion.api.tune.ExperimentRunner")
@patch("hyperion.api.tune._create_default_services")
@patch("hyperion.api.tune._async_tune")
def test_tune_resolves_string_strategy_with_correct_args(
    mock_async_tune_with_cleanup, mock_services, mock_runner_class, mock_resolve
):
    """Strategy string gets resolved with metric/mode and budget/resource hints."""
    # Setup mocks
    mock_policy = MagicMock()
    mock_resolve.return_value = mock_policy

    mock_runner = MagicMock()
    mock_runner_class.return_value = mock_runner

    mock_services.return_value = {"stores": MagicMock()}
    mock_async_tune_with_cleanup.return_value = {"best": {"score": 0.9}}

    # Simple objective and space
    def objective(ctx, x=0.5):
        return ObjectiveResult(score=x)

    space = {"x": Float(0, 1)}

    # Call tune
    result = tune(
        objective,
        space,
        strategy="random",
        strategy_kwargs={"custom": "arg"},
        max_trials=10,
        max_concurrent=3,
        metric="val_loss",
        mode="min",
    )

    # Verify strategy resolution was called correctly
    mock_resolve.assert_called_once_with(
        "random",
        space=space,
        metric="val_loss",
        mode="min",
        custom="arg",  # from strategy_kwargs
    )

    # Verify ExperimentSpec was built correctly
    mock_runner_class.assert_called_once()
    args, kwargs = mock_runner_class.call_args
    spec = args[0]

    assert spec.name == "random"
    assert spec.objective == objective
    assert spec.search_space == space
    assert spec.pipeline.steps == [mock_policy]
    assert spec.resources.max_concurrent == 3
    assert spec.budget.max_trials == 10
    assert spec.budget.metric == "val_loss"
    assert spec.budget.mode == "min"

    # Verify services were created
    mock_services.assert_called_once_with(None, 3, "thread")

    # Verify result was returned
    assert result == {"best": {"score": 0.9}}


@patch("hyperion.api.tune.ExperimentRunner")
@patch("hyperion.api.tune._create_default_services")
@patch("hyperion.api.tune._async_tune")
def test_tune_passes_policy_instance_directly(
    mock_async_tune_with_cleanup, mock_services, mock_runner_class
):
    """Policy instance is passed through as-is."""
    # Setup mocks
    mock_runner = MagicMock()
    mock_runner_class.return_value = mock_runner

    mock_services.return_value = {"stores": MagicMock()}
    mock_async_tune_with_cleanup.return_value = {"best": {}}

    # Create custom policy
    custom_policy = MagicMock()

    # Simple objective
    def objective(ctx):
        return ObjectiveResult(score=0.5)

    # Call tune with policy instance
    tune(
        objective,
        {},
        strategy=custom_policy,  # Policy instance, not string
        max_trials=5,
        max_concurrent=2,
    )

    # Verify ExperimentSpec was built with the policy instance
    mock_runner_class.assert_called_once()
    args, _ = mock_runner_class.call_args
    spec = args[0]

    assert spec.pipeline.steps == [custom_policy]
    assert spec.budget.max_trials == 5
    assert spec.resources.max_concurrent == 2


@patch("hyperion.api.tune.resolve")
@patch("hyperion.api.tune.ExperimentRunner")
@patch("hyperion.api.tune._create_default_services")
@patch("hyperion.api.tune._async_tune")
def test_tune_spec_construction_propagates_all_params(
    mock_async_tune_with_cleanup, mock_services, mock_runner_class, mock_resolve
):
    """Spec construction propagates max_trials/max_concurrent/metric/mode."""
    # Setup mocks
    mock_resolve.return_value = MagicMock()
    mock_runner_class.return_value = MagicMock()
    mock_services.return_value = {"stores": MagicMock()}
    mock_async_tune_with_cleanup.return_value = {"best": {}}

    # Call tune with all parameters
    tune(
        lambda ctx: ObjectiveResult(score=1.0),
        {"x": Float(0, 1)},
        strategy="random",
        max_trials=100,
        max_concurrent=8,
        metric="accuracy",
        mode="max",
        storage="sqlite:///test.db",
        return_state=True,
    )

    # Check ExperimentSpec construction
    args, _ = mock_runner_class.call_args
    spec = args[0]

    # Verify all parameters propagated correctly
    assert spec.budget.max_trials == 100
    assert spec.budget.metric == "accuracy"
    assert spec.budget.mode == "max"
    assert spec.resources.max_concurrent == 8

    # Verify services creation
    mock_services.assert_called_once_with("sqlite:///test.db", 8, "thread")
