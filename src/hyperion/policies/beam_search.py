"""Beam search strategy for hyperparameter optimization.

Maintains a frontier of top-K trials per depth up to max_depth. Expands by
mutating parent params, and can prune underperforming branches.

This is a simple deterministic implementation aligned with the design doc.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from hyperion.core.events import Event, EventType
from hyperion.core.state import ReadableState, TrialView
from hyperion.framework.policy import Action, KillTrial, Policy, StartTrial
from hyperion.framework.search_space import Choice, Float, Int, sample_space


@dataclass
class _ScoredTrial:
    trial: TrialView
    score: float


class BeamSearchPolicy(Policy):
    """Beam search policy for structured exploration of hyperparameter space.

    This policy maintains a frontier of top-K trials at each depth level,
    expanding the search tree by creating child trials from the best performers.
    It supports optional pruning of underperforming trials to free capacity.
    """

    def __init__(
        self,
        *,
        space: dict[str, Any],
        experiment_id: str | None = None,
        K: int = 2,
        width: int = 2,
        max_depth: int = 3,
        metric: str = "score",
        mode: Literal["min", "max"] = "max",
        prune: bool = False,
    ) -> None:
        """Initialize beam search policy.

        Args:
            space: Search space specification
            experiment_id: ID of experiment to run trials for (set later by runner)
            K: Number of top trials to keep per depth level
            width: Number of child trials to spawn from each parent
            max_depth: Maximum depth of the search tree
            metric: Metric to optimize
            mode: Optimization mode - "min" or "max"
            prune: Whether to prune underperforming trials
        """
        self.space = space
        self.experiment_id = experiment_id
        self.K = K
        self.width = width
        self.max_depth = max_depth
        self.metric = metric
        self.mode = mode
        self.prune = prune

        # Cache of latest best scores by trial_id for quick sorting
        self._last_scores: dict[str, float] = {}

    async def on_events(self, events: Iterable[Event]) -> None:
        """Process incoming events to track trial scores.

        Args:
            events: Stream of events from the event bus
        """
        for evt in events:
            if evt.type == EventType.TRIAL_COMPLETED:
                trial_id = evt.data.get("trial_id")
                score = evt.data.get("score")
                if trial_id and isinstance(score, int | float):
                    self._last_scores[trial_id] = float(score)

    async def decide(self, state: ReadableState) -> list[Action]:
        """Decide which trials to start based on beam search strategy.

        Args:
            state: Current system state

        Returns:
            List of StartTrial and optionally KillTrial actions
        """
        if not self.experiment_id:
            return []

        free = state.capacity_free()
        if free <= 0:
            return []

        actions: list[Action] = []

        # Gather trials by depth for this experiment
        running = state.running_trials(self.experiment_id)
        by_depth: dict[int, list[TrialView]] = {}
        for t in running:
            by_depth.setdefault(t.depth, []).append(t)

        # Find frontier: top K per depth
        frontier: list[TrialView] = []
        for _, trials in sorted(by_depth.items()):
            scored = [_ScoredTrial(trial=t, score=self._score_of(t)) for t in trials]
            scored.sort(key=lambda s: s.score, reverse=self.mode == "max")
            frontier.extend([s.trial for s in scored[: self.K]])

        # Expand children from deepest frontier first
        expansions: list[StartTrial] = []
        if not frontier:
            # Bootstrap depth 0: start width trials sampled from space
            to_emit = min(free, self.width)
            for _ in range(to_emit):
                params = sample_space(self.space)
                expansions.append(
                    StartTrial(
                        experiment_id=self.experiment_id,
                        params=params,
                        parent_trial_id=None,
                        tags={
                            "strategy": "beam",
                            "depth": 0,
                            "rationale": f"Bootstrap trial {_ + 1}/{to_emit} at depth 0",
                        },
                    )
                )
        else:
            # Expand each frontier node with `width` children
            for parent in frontier:
                if parent.depth >= self.max_depth:
                    continue
                for _ in range(self.width):
                    child_params = self._mutate(parent.params)
                    expansions.append(
                        StartTrial(
                            experiment_id=self.experiment_id,
                            params=child_params,
                            parent_trial_id=parent.trial_id,
                            tags={
                                "strategy": "beam",
                                "parent": parent.trial_id,
                                "depth": parent.depth + 1,
                                "rationale": f"Child of {parent.trial_id[:8]}... at depth {parent.depth + 1}",
                            },
                        )
                    )

        # Only start up to free capacity
        for st in expansions[:free]:
            actions.append(st)

        # Optional pruning: kill running trials not in frontier if we need capacity
        if self.prune and free == 0 and frontier:
            frontier_ids = {t.trial_id for t in frontier}
            for t in running:
                if t.trial_id not in frontier_ids:
                    actions.append(KillTrial(trial_id=t.trial_id, reason="prune"))

        return actions

    def _score_of(self, t: TrialView) -> float:
        """Get the score of a trial for ranking.

        Args:
            t: Trial to score

        Returns:
            Score value, using infinity for missing scores
        """
        # Use score when metric=="score"; otherwise use metrics_last
        if self.metric == "score":
            return float(t.score or float("-inf" if self.mode == "max" else "inf"))
        val = t.metrics_last.get(self.metric)
        if val is None:
            return float("-inf" if self.mode == "max" else "inf")
        return float(val)

    def _mutate(self, base: dict[str, Any]) -> dict[str, Any]:
        """Mutate parameters to create child trials.

        Args:
            base: Parent trial parameters to mutate

        Returns:
            Mutated parameters for child trial
        """
        import random

        params = dict(base)

        # Select a random parameter to mutate
        if not self.space:
            return params

        param_name = random.choice(list(self.space.keys()))
        spec = self.space[param_name]
        current_value = params.get(param_name)

        if isinstance(spec, Float):
            # Gaussian perturbation for float parameters
            if (
                current_value is not None
                and spec.min is not None
                and spec.max is not None
            ):
                # Use 10% of range as standard deviation
                std = (spec.max - spec.min) * 0.1
                new_val = current_value + random.gauss(0, std)
                params[param_name] = spec.clip(new_val)
            else:
                # If no bounds or current value, sample randomly
                params[param_name] = spec.sample()

        elif isinstance(spec, Int):
            # Step mutation for integers
            if (
                current_value is not None
                and spec.min is not None
                and spec.max is not None
            ):
                # Randomly step up or down
                step = random.choice([-2, -1, 1, 2])  # Variable step size
                new_val = current_value + step
                params[param_name] = spec.clip(new_val)
            else:
                params[param_name] = spec.sample()

        elif isinstance(spec, Choice):
            # Pick a different option for categorical
            if current_value is not None:
                others = [opt for opt in spec.options if opt != current_value]
                if others:
                    params[param_name] = random.choice(others)
                else:
                    # Only one option, can't mutate
                    params[param_name] = current_value
            else:
                params[param_name] = spec.sample()

        else:
            # For other types, sample randomly
            params[param_name] = sample_space(spec)

        return params

    async def rationale(self) -> str | None:
        """Explain the beam search strategy."""
        prune_str = ", with pruning" if self.prune else ""
        return (
            f"Beam search: keeping top {self.K} trials per depth, "
            f"expanding {self.width} children per parent, "
            f"max depth {self.max_depth}{prune_str}"
        )
