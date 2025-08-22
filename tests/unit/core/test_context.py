"""Tests for trial context functionality."""

import asyncio

from hyperion.core.context import TrialContext


def test_trial_context_report():
    """TrialContext should report progress via callback."""
    reported = []

    def report_fn(trial_id: str, step: int | str, metrics: dict[str, float]):
        reported.append((trial_id, step, metrics))

    ctx = TrialContext(
        trial_id="trial-123", _report=report_fn, _stop_event=asyncio.Event()
    )

    # Report some progress
    ctx.report(0, loss=0.5, accuracy=0.8)
    ctx.report(1, loss=0.4, accuracy=0.85)
    ctx.report("final", loss=0.3, accuracy=0.9)

    assert len(reported) == 3
    assert reported[0] == ("trial-123", 0, {"loss": 0.5, "accuracy": 0.8})
    assert reported[1] == ("trial-123", 1, {"loss": 0.4, "accuracy": 0.85})
    assert reported[2] == ("trial-123", "final", {"loss": 0.3, "accuracy": 0.9})


def test_trial_context_should_stop():
    """TrialContext should check stop signal."""
    stop_event = asyncio.Event()

    ctx = TrialContext(
        trial_id="trial-456", _report=lambda *args: None, _stop_event=stop_event
    )

    # Initially not stopped
    assert ctx.should_stop() is False

    # Set stop signal
    stop_event.set()
    assert ctx.should_stop() is True


def test_trial_context_report_with_single_metric():
    """TrialContext should handle single metric reports."""
    reported = []

    def report_fn(trial_id: str, step: int | str, metrics: dict[str, float]):
        reported.append(metrics)

    ctx = TrialContext(
        trial_id="trial-789", _report=report_fn, _stop_event=asyncio.Event()
    )

    # Report single metric
    ctx.report(0, score=0.95)

    assert reported[0] == {"score": 0.95}
