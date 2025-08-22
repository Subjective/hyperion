"""Random search strategy for hyperparameter optimization."""

from collections.abc import Iterable
from typing import Any

from hyperion.core.events import Event
from hyperion.core.state import ReadableState
from hyperion.framework.policy import Action, Policy, StartTrial
from hyperion.framework.search_space import sample_space


class RandomSearchPolicy(Policy):
    """Random search policy that samples uniformly from the search space.

    This is the simplest optimization strategy - it randomly samples
    parameters without considering previous results.
    """

    def __init__(
        self, *, space: dict[str, Any], experiment_id: str | None = None, **_: Any
    ):
        """Initialize random search policy.

        Args:
            space: Search space specification
            experiment_id: ID of experiment to run trials for (set later by runner)
        """
        self.space = space
        self.experiment_id = experiment_id

    async def on_events(self, events: Iterable[Event]) -> None:
        """Process incoming events (currently unused)."""
        pass

    async def decide(self, state: ReadableState) -> list[Action]:
        """Decide to start new trials if budget and capacity allow.

        Args:
            state: Current system state

        Returns:
            List of StartTrial actions
        """
        if not self.experiment_id:
            return []

        free_slots = state.capacity_free()
        if free_slots <= 0:
            return []

        actions: list[Action] = []
        for _ in range(free_slots):
            params = sample_space(self.space)
            actions.append(
                StartTrial(
                    experiment_id=self.experiment_id,
                    params=params,
                    tags={
                        "strategy": "random",
                        "rationale": "Uniform random sampling from search space",
                    },
                )
            )

        return actions

    async def rationale(self) -> str | None:
        """Explain the random search strategy."""
        return "Random search: sampling uniformly from search space"
