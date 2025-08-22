"""Population-Based Training (PBT) strategy for hyperparameter optimization.

PBT maintains a population of trials and evolves them over generations by
creating new trials from the best performers with perturbed parameters.
"""

import logging
import random
from collections.abc import Iterable
from typing import Any, Literal

from hyperion.core.events import Event, EventType
from hyperion.core.state import ReadableState
from hyperion.framework.policy import Action, Policy, StartTrial
from hyperion.framework.search_space import Bool, Choice, Float, Int, sample_space

logger = logging.getLogger(__name__)


class PopulationBasedTrainingPolicy(Policy):
    """Population-Based Training policy for evolutionary hyperparameter optimization.

    This policy evolves a population of trials over generations. When enough trials
    complete, it selects the best performers as parents and creates new trials with
    perturbed parameters from those parents, forming a lineage tree.
    """

    def __init__(
        self,
        *,
        space: dict[str, Any],
        experiment_id: str | None = None,
        population_size: int = 8,
        perturbation_factor: float = 1.2,
        metric: str = "score",
        mode: Literal["min", "max"] = "max",
        **kwargs: Any,
    ):
        """Initialize Population-Based Training policy.

        Args:
            space: Search space specification
            experiment_id: ID of experiment to run trials for
            population_size: Number of trials per generation
            perturbation_factor: Factor for perturbing parameters (multiply or divide)
            metric: Metric to optimize
            mode: Optimization mode - "min" or "max"
        """
        self.space = space
        self.experiment_id = experiment_id
        self.population_size = population_size
        self.perturbation_factor = perturbation_factor
        self.metric = metric
        self.mode: Literal["min", "max"] = mode

        # Track generations
        self.generation = 0
        self.last_evolution_count = 0

    async def on_events(self, events: Iterable[Event]) -> None:
        """Track trial lifecycle events."""
        # We only need to track experiment started to get experiment_id if needed
        for event in events:
            if event.type == EventType.EXPERIMENT_STARTED and not self.experiment_id:
                self.experiment_id = event.data.get("experiment_id")

    async def decide(self, state: ReadableState) -> list[Action]:
        """Decide actions for population management.

        Args:
            state: Current system state

        Returns:
            List of StartTrial actions for new trials
        """
        if not self.experiment_id:
            return []

        actions: list[Action] = []

        # Get current population state
        all_trials = state.all_trials(self.experiment_id)
        completed = [t for t in all_trials if t.status == "COMPLETED"]
        total_started = len(all_trials)

        # Bootstrap initial population if needed
        if total_started < self.population_size:
            free_slots = min(
                state.capacity_free(), self.population_size - total_started
            )

            logger.debug(
                f"PBT: Bootstrapping initial population "
                f"({total_started}/{self.population_size} started)"
            )

            for i in range(free_slots):
                params = sample_space(self.space)
                tags = {
                    "strategy": "pbt",
                    "pbt_generation": 0,
                    "rationale": f"Initial population member {total_started + i + 1}/{self.population_size}",
                }

                actions.append(
                    StartTrial(
                        experiment_id=self.experiment_id,
                        params=params,
                        parent_trial_id=None,  # No parent for initial population
                        tags=tags,
                    )
                )

        # Evolve new generation from completed trials
        elif (
            len(completed) > self.last_evolution_count
            and len(completed) >= self.population_size // 2
        ):
            # We have new completed trials and enough to select from

            # Select best performers as parents
            sorted_completed = sorted(
                completed,
                key=lambda t: t.score if t.score is not None else float("-inf"),
                reverse=(self.mode == "max"),
            )

            # Take top half as potential parents
            n_parents = max(1, self.population_size // 2)
            best_parents = sorted_completed[:n_parents]

            # Create new trials if we have capacity
            available_slots = state.capacity_free()
            if available_slots > 0:
                self.generation += 1
                logger.debug(
                    f"PBT: Starting generation {self.generation} "
                    f"({len(completed)} completed trials, selecting top {n_parents} as parents)"
                )

                # Create new trials from best parents
                for i in range(min(available_slots, self.population_size)):
                    # Select parent (cycle through best parents)
                    parent = best_parents[i % len(best_parents)]

                    # Perturb parameters
                    new_params = self._perturb_params(parent.params)

                    tags = {
                        "strategy": "pbt",
                        "pbt_generation": self.generation,
                        "pbt_parent": parent.trial_id,
                        "parent_score": parent.score,
                        "rationale": (
                            f"Generation {self.generation}: "
                            f"evolved from {parent.trial_id[:8]} (score={parent.score:.4f})"
                        ),
                    }

                    actions.append(
                        StartTrial(
                            experiment_id=self.experiment_id,
                            params=new_params,
                            parent_trial_id=parent.trial_id,  # This creates the lineage!
                            tags=tags,
                        )
                    )

                # Remember how many completed trials we've processed
                self.last_evolution_count = len(completed)

        return actions

    def _perturb_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Perturb parameters for exploration.

        Args:
            params: Original parameters

        Returns:
            Perturbed parameters
        """
        new_params: dict[str, Any] = {}

        for param_name, param_value in params.items():
            if param_name not in self.space:
                new_params[param_name] = param_value
                continue

            spec = self.space[param_name]

            if isinstance(spec, Float):
                # Perturb continuous values
                if random.random() < 0.5:
                    # Multiply
                    new_val = param_value * self.perturbation_factor
                else:
                    # Divide
                    new_val = param_value / self.perturbation_factor

                # Clip to bounds
                if spec.min is not None and spec.max is not None:
                    new_val = spec.clip(new_val)
                    # Add some noise
                    noise = random.gauss(0, 0.1 * (spec.max - spec.min))
                    new_val = spec.clip(new_val + noise)

                new_params[param_name] = new_val

            elif isinstance(spec, Int):
                # Perturb integer values
                if spec.min is not None and spec.max is not None:
                    if random.random() < 0.5:
                        # Increase
                        step = max(1, int((spec.max - spec.min) * 0.1))
                        new_val = param_value + random.randint(1, step)
                    else:
                        # Decrease
                        step = max(1, int((spec.max - spec.min) * 0.1))
                        new_val = param_value - random.randint(1, step)

                    # Clip to bounds
                    new_val = spec.clip(new_val)
                else:
                    # No bounds, just perturb
                    new_val = param_value + random.randint(-5, 5)
                new_params[param_name] = new_val

            elif isinstance(spec, Choice):
                # Occasionally resample categorical
                if random.random() < 0.3:  # 30% chance to change
                    new_params[param_name] = random.choice(spec.options)
                else:
                    new_params[param_name] = param_value

            elif isinstance(spec, Bool):
                # Flip boolean with some probability
                if random.random() < 0.2:  # 20% chance to flip
                    new_params[param_name] = not param_value
                else:
                    new_params[param_name] = param_value
            else:
                # Keep unchanged for unknown types
                new_params[param_name] = param_value

        return new_params

    async def rationale(self) -> str | None:
        """Explain the PBT strategy."""
        return (
            f"Population-Based Training: Generation {self.generation}, "
            f"population size {self.population_size}. "
            f"Evolving from best completed trials with "
            f"perturbation factor {self.perturbation_factor}."
        )
