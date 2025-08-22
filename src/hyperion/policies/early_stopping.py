"""Early stopping policies for terminating underperforming trials.

These policies monitor trial progress and can terminate trials that are
underperforming to free up computational resources for more promising trials.
"""

import logging
from collections.abc import Iterable
from typing import Any, Literal

from hyperion.core.events import Event, EventType
from hyperion.core.state import ReadableState
from hyperion.framework.policy import Action, KillTrial, Policy

logger = logging.getLogger(__name__)


class NoEarlyStoppingPolicy(Policy):
    """Early stopping policy that never stops trials early (default behavior).

    This policy allows all trials to run to completion naturally without
    any early termination. It's useful when you want complete information
    about all hyperparameter configurations.
    """

    def __init__(self, experiment_id: str | None = None):
        """Initialize the no-op early stopping policy.

        Args:
            experiment_id: ID of experiment to monitor (set by framework)
        """
        self.experiment_id = experiment_id

    async def on_events(self, events: Iterable[Event]) -> None:
        """Process events (no-op for this policy).

        Args:
            events: Stream of events from the event bus
        """
        pass  # No tracking needed

    async def decide(self, state: ReadableState) -> list[Action]:
        """Never terminate any trials.

        Args:
            state: Current system state

        Returns:
            Empty list (no actions)
        """
        return []

    async def rationale(self) -> str | None:
        """Explain the policy's strategy.

        Returns:
            Description of the no-op strategy
        """
        return "No early stopping: allowing all trials to complete naturally"


class MedianEarlyStoppingPolicy(Policy):
    """Stops trials performing below median at evaluation checkpoints.

    This policy periodically evaluates all running trials and terminates
    those performing below the median on a specified metric. This helps
    focus computational resources on more promising trials.
    """

    def __init__(
        self,
        *,
        metric: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        check_interval: int = 100,
        min_trials: int = 2,
        experiment_id: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the median early stopping policy.

        Args:
            metric: Metric to monitor for early stopping decisions
            mode: Whether to minimize or maximize the metric
            check_interval: Number of progress events between evaluations
            min_trials: Minimum number of running trials before stopping any
            experiment_id: ID of experiment to monitor (set by framework)
        """
        self.metric = metric
        self.mode = mode
        self.check_interval = check_interval
        self.min_trials = min_trials
        self.experiment_id = experiment_id

        # Track progress for evaluation timing
        self.steps_since_check = 0
        self.total_checks = 0

    async def on_events(self, events: Iterable[Event]) -> None:
        """Track trial progress events.

        Args:
            events: Stream of events from the event bus
        """
        for event in events:
            if event.type == EventType.TRIAL_PROGRESS:
                self.steps_since_check += 1

    async def decide(self, state: ReadableState) -> list[Action]:
        """Decide which trials to terminate based on median performance.

        Args:
            state: Current system state

        Returns:
            List of KillTrial actions for underperforming trials
        """
        if not self.experiment_id:
            return []

        actions: list[Action] = []

        # Check if it's time to evaluate
        if self.steps_since_check >= self.check_interval:
            running = state.running_trials(self.experiment_id)

            # Need minimum number of trials to make meaningful comparisons
            if len(running) >= self.min_trials:
                # Get metric values for all running trials
                values: list[tuple[Any, float]] = []
                for trial in running:
                    if self.metric in trial.metrics_last:
                        value = trial.metrics_last[self.metric]
                        values.append((trial, value))

                if len(values) >= self.min_trials:
                    # Calculate median
                    sorted_values = sorted(values, key=lambda x: x[1])
                    median_idx = len(sorted_values) // 2
                    median_value = sorted_values[median_idx][1]

                    # Determine which trials are below median
                    for trial, value in values:
                        should_stop = (
                            value > median_value
                            if self.mode == "min"
                            else value < median_value
                        )

                        if should_stop:
                            actions.append(
                                KillTrial(
                                    trial_id=trial.trial_id,
                                    reason=(
                                        f"Below median {self.metric}={value:.4f} "
                                        f"(median={median_value:.4f}) at checkpoint {self.total_checks}"
                                    ),
                                )
                            )

                    if actions:
                        logger.debug(
                            f"MedianEarlyStopping: Stopping {len(actions)} trials "
                            f"below median {self.metric}={median_value:.4f}"
                        )

                self.total_checks += 1

            # Reset counter
            self.steps_since_check = 0

        return actions

    async def rationale(self) -> str | None:
        """Explain the policy's current strategy.

        Returns:
            Description of the median-based early stopping strategy
        """
        return (
            f"Median early stopping: Monitor {self.metric} ({'minimize' if self.mode == 'min' else 'maximize'}), "
            f"stop bottom 50% every {self.check_interval} steps. "
            f"Performed {self.total_checks} evaluations."
        )
