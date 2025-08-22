"""Trial context for cooperative cancellation and progress reporting."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

# Type for the report callback function
ReportFn = Callable[[str, int | str, dict[str, float]], None]


class EventLike(Protocol):
    """Protocol for event-like objects (e.g., asyncio.Event, multiprocessing.Event)."""

    def is_set(self) -> bool:
        """Check if the event is set."""
        ...


@dataclass
class TrialContext:
    """Context provided to objective functions for progress reporting and cancellation.

    This allows trials to:
    - Report intermediate metrics during execution
    - Check if they should stop early (cooperative cancellation)
    - Access warm start checkpoint information if available
    """

    trial_id: str
    _report: ReportFn
    _stop_event: EventLike
    warm_start_checkpoint: str | None = None  # Path to checkpoint for warm starting

    def report(self, step: int | str, **metrics: float) -> None:
        """Report progress metrics for this trial.

        Args:
            step: Step number or label (e.g., 0, 1, "final")
            **metrics: Metric values as keyword arguments
        """
        self._report(self.trial_id, step, metrics)

    def should_stop(self) -> bool:
        """Check if this trial should stop early.

        Returns:
            True if a stop signal has been sent, False otherwise
        """
        return self._stop_event.is_set()
