"""Tests for warm start support in TrialContext."""

from hyperion.core.context import TrialContext


class MockEvent:
    """Mock event for testing."""

    def is_set(self) -> bool:
        return False


def test_trial_context_with_warm_start():
    """TrialContext should support warm start checkpoint info."""

    def mock_report(trial_id: str, step: int | str, metrics: dict[str, float]):
        pass

    # Test without warm start
    ctx = TrialContext(
        trial_id="trial-1",
        _report=mock_report,
        _stop_event=MockEvent(),
    )

    assert ctx.warm_start_checkpoint is None

    # Test with warm start checkpoint
    ctx_warm = TrialContext(
        trial_id="trial-2",
        _report=mock_report,
        _stop_event=MockEvent(),
        warm_start_checkpoint="/path/to/checkpoint.pt",
    )

    assert ctx_warm.warm_start_checkpoint == "/path/to/checkpoint.pt"

    # Objective can check if warm starting
    if ctx_warm.warm_start_checkpoint:
        # Would load from checkpoint
        assert ctx_warm.warm_start_checkpoint == "/path/to/checkpoint.pt"


def test_trial_context_backward_compatibility():
    """TrialContext should maintain backward compatibility."""

    def mock_report(trial_id: str, step: int | str, metrics: dict[str, float]):
        pass

    # Existing code should still work
    ctx = TrialContext(
        trial_id="trial-1",
        _report=mock_report,
        _stop_event=MockEvent(),
    )

    # Should have all expected methods and attributes
    assert hasattr(ctx, "trial_id")
    assert hasattr(ctx, "report")
    assert hasattr(ctx, "should_stop")
    assert hasattr(ctx, "warm_start_checkpoint")

    # Default warm start should be None
    assert ctx.warm_start_checkpoint is None
