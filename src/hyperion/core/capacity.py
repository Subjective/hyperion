"""Capacity management for controlling concurrent trial execution."""

from typing import Any


class CapacityManager:
    """Manages execution capacity and admission control.

    Tracks running trials and enforces global concurrency limits.
    Future versions will support per-experiment quotas and weights.
    """

    def __init__(
        self,
        max_concurrent: int,
        quotas: dict[str, int] | None = None,  # TODO: Still need to implement
        weights: dict[str, float] | None = None,  # TODO: Still need to implement
    ):
        """Initialize capacity manager.

        Args:
            max_concurrent: Maximum concurrent trials globally
            quotas: Per-experiment quotas
            weights: Fair-share weights
        """
        self.max_concurrent = max_concurrent
        self.quotas = quotas or {}
        self.weights = weights or {}
        self._running_count = 0
        self._running_by_experiment: dict[str, int] = {}

    def can_admit(self, experiment_id: str) -> bool:
        """Check if a new trial can be admitted.

        Args:
            experiment_id: ID of experiment requesting admission

        Returns:
            True if capacity available, False otherwise
        """
        # NOTE: experiment_id is reserved for per-experiment quotas/fairness.
        # Enforcement is not wired yet; only a global cap is applied.
        return self._running_count < self.max_concurrent

    def on_start(self, experiment_id: str) -> None:
        """Record that a trial has started.

        Args:
            experiment_id: ID of experiment that started a trial
        """
        if self._running_count >= self.max_concurrent:
            # NOTE: caller should check can_admit() before on_start().
            # This guard prevents internal counters from drifting.
            return
        self._running_count += 1
        self._running_by_experiment[experiment_id] = (
            self._running_by_experiment.get(experiment_id, 0) + 1
        )

    def on_end(self, experiment_id: str) -> None:
        """Record that a trial has ended.

        Args:
            experiment_id: ID of experiment that ended a trial
        """
        if self._running_count > 0:
            self._running_count -= 1

        if experiment_id in self._running_by_experiment:
            count = self._running_by_experiment[experiment_id]
            if count > 1:
                self._running_by_experiment[experiment_id] = count - 1
            else:
                del self._running_by_experiment[experiment_id]

    def free_slots(self) -> int:
        """Number of free capacity slots available globally.

        Returns:
            Remaining slots (never negative)
        """
        return max(0, self.max_concurrent - self._running_count)

    def next_queued(self) -> dict[str, Any] | None:
        """Get information about next queued admission.

        Returns:
            Info about next queued trial or None
        """
        # TODO: implement admission ordering and per-experiment queues.
        return None
