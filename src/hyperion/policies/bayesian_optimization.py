"""Bayesian Optimization strategy for hyperparameter optimization.

Uses a lightweight TPE-style density ratio to guide the search. It
splits completed trials into "good" and "bad" sets and prefers
parameters with higher p(x|good)/p(x|bad).
"""

import logging
import math
from collections.abc import Iterable
from typing import Any, Literal

from hyperion.core.events import Event
from hyperion.core.state import ReadableState, TrialView
from hyperion.framework.policy import Action, Policy, StartTrial
from hyperion.framework.search_space import Bool, Choice, Float, Int, sample_space

logger = logging.getLogger(__name__)


class BayesianOptimizationPolicy(Policy):
    """Bayesian Optimization policy using a TPE-style surrogate.

    This policy models per-parameter densities for "good" and "bad"
    observations and selects candidates that maximize the density ratio.
    """

    def __init__(
        self,
        *,
        space: dict[str, Any],
        experiment_id: str | None = None,
        n_initial: int = 10,
        acquisition: str = "tpe",
        xi: float = 0.01,
        gamma: float = 0.2,
        metric: str = "score",
        mode: Literal["min", "max"] = "max",
        **kwargs: Any,
    ):
        """Initialize Bayesian Optimization policy.

        Args:
            space: Search space specification
            experiment_id: ID of experiment to run trials for
            n_initial: Number of random initial points before using BO
            acquisition: Acquisition function name
            xi: Exploration parameter (ignored; kept for compatibility)
            gamma: Quantile used to split good/bad sets (0 < gamma < 1)
            metric: Metric to optimize
            mode: Optimization mode - "min" or "max"
        """
        self.space = space
        self.experiment_id = experiment_id
        self.n_initial = int(max(1, n_initial))
        self.acquisition = acquisition
        self.xi = xi
        self.gamma = float(gamma)
        self.metric = metric
        self.mode = mode
        if not (0.0 < self.gamma < 1.0):
            raise ValueError("gamma must be in (0,1)")

    async def on_events(self, events: Iterable[Event]) -> None:
        """Process incoming events (currently unused)."""
        pass

    async def decide(self, state: ReadableState) -> list[Action]:
        """Decide next trials to run based on Bayesian Optimization.

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

        # Get completed trials
        completed = state.completed_trials(self.experiment_id)

        actions: list[Action] = []

        # Use random sampling for initial exploration
        if len(completed) < self.n_initial:
            logger.debug(
                f"Bayesian Optimization: Random exploration phase "
                f"({len(completed)}/{self.n_initial} initial trials)"
            )
            for _ in range(min(free_slots, self.n_initial - len(completed))):
                params = sample_space(self.space)
                actions.append(
                    StartTrial(
                        experiment_id=self.experiment_id,
                        params=params,
                        tags={
                            "strategy": "bayesian",
                            "phase": "exploration",
                            "rationale": f"Random exploration ({len(completed)}/{self.n_initial} initial trials)",
                        },
                    )
                )
        else:
            # Use TPE-style density ratio for guided search
            good, bad = self._split_good_bad(completed)
            if good and bad:
                for _ in range(min(free_slots, 1)):
                    next_params = self._tpe_propose(good, bad)
                    actions.append(
                        StartTrial(
                            experiment_id=self.experiment_id,
                            params=next_params,
                            tags={
                                "strategy": "bayesian",
                                "phase": "exploitation",
                                "rationale": f"TPE-guided selection (gamma={self.gamma:.2f})",
                            },
                        )
                    )
            else:
                # Fallback to random if split is degenerate
                logger.warning("TPE split failed; falling back to random sampling")
                params = sample_space(self.space)
                actions.append(
                    StartTrial(
                        experiment_id=self.experiment_id,
                        params=params,
                        tags={
                            "strategy": "bayesian",
                            "phase": "fallback",
                            "rationale": "Insufficient data for TPE split; using random sampling",
                        },
                    )
                )

        return actions

    def _split_good_bad(
        self, trials: list[TrialView]
    ) -> tuple[list[TrialView], list[TrialView]]:
        """Split completed trials into good and bad sets by gamma quantile.

        Uses score (or metric) and mode to order trials.
        """
        # Extract usable value
        vals: list[tuple[TrialView, float]] = []
        for t in trials:
            if self.metric == "score":
                if t.score is not None:
                    vals.append((t, float(t.score)))
            else:
                m = t.metrics_last.get(self.metric)
                if isinstance(m, int | float):
                    vals.append((t, float(m)))

        if not vals:
            return [], []

        reverse = self.mode == "max"
        ordered = sorted(vals, key=lambda tv: tv[1], reverse=reverse)
        n = len(ordered)
        n_good = max(1, min(n - 1, int(math.floor(self.gamma * n)))) if n > 1 else 1
        good = [tv[0] for tv in ordered[:n_good]]
        bad = [tv[0] for tv in ordered[n_good:]]
        return good, bad

    def _tpe_propose(
        self, good: list[TrialView], bad: list[TrialView]
    ) -> dict[str, Any]:
        """Sample candidates and select the one maximizing log p_good - log p_bad."""
        # Build observation buckets per dimension
        obs_good: dict[str, list[Any]] = {}
        obs_bad: dict[str, list[Any]] = {}
        for t in good:
            for k, v in t.params.items():
                obs_good.setdefault(k, []).append(v)
        for t in bad:
            for k, v in t.params.items():
                obs_bad.setdefault(k, []).append(v)

        k_candidates = self._num_candidates()
        best: tuple[float, dict[str, Any]] | None = None

        for _ in range(k_candidates):
            params = sample_space(self.space)
            lr = 0.0
            for name, spec in self.space.items():
                x = params.get(name)
                pg = self._p_dim(spec, x, obs_good.get(name, []))
                pb = self._p_dim(spec, x, obs_bad.get(name, []))
                pg = max(pg, 1e-12)
                pb = max(pb, 1e-12)
                lr += math.log(pg) - math.log(pb)
            if best is None or lr > best[0]:
                best = (lr, params)

        return best[1] if best else sample_space(self.space)

    def _num_candidates(self) -> int:
        d = max(1, len(self.space))
        return max(128, min(2048, 64 * d))

    def _p_dim(self, spec: Any, x: Any, obs: list[Any]) -> float:
        # Float
        if isinstance(spec, Float):
            return self._p_continuous(float(x), obs, spec.min, spec.max)
        # Int
        if isinstance(spec, Int):
            return self._p_discrete(
                int(x), [int(v) for v in obs if isinstance(v, int | float)]
            )
        # Choice/Bool
        if isinstance(spec, Choice):
            return self._p_categorical(x, obs, len(spec.options))
        if isinstance(spec, Bool):
            return self._p_categorical(bool(x), [bool(v) for v in obs], 2)
        # Fallback small constant
        return 1e-6

    def _p_continuous(
        self, x: float, obs: list[Any], vmin: float | None, vmax: float | None
    ) -> float:
        vals = [float(v) for v in obs if isinstance(v, int | float)]
        n = len(vals)
        if n == 0:
            if vmin is not None and vmax is not None and vmax > vmin:
                return 1.0 / (vmax - vmin)
            return 1e-3
        vmin_eff = min(vals) if vmin is None else vmin
        vmax_eff = max(vals) if vmax is None else vmax
        rng = max(1e-12, float(vmax_eff - vmin_eff))
        h = max(rng * 0.01, rng / (3.0 * math.sqrt(n)))
        s = 0.0
        for v in vals:
            u = abs(x - v) / h
            k = max(0.0, 1.0 - u)  # triangular kernel
            s += k
        return s / (n * h)

    def _p_discrete(self, x: int, vals: list[int]) -> float:
        n = len(vals)
        if n == 0:
            return 0.5
        s = 0.0
        for v in vals:
            d = abs(x - v)
            s += 1.0 if d == 0 else (0.5 if d == 1 else 0.0)
        return (s + 1.0) / (n + 2.0)

    def _p_categorical(self, x: Any, vals: list[Any], k: int) -> float:
        n = len(vals)
        if n == 0:
            return 1.0 / k if k > 0 else 0.5
        alpha = 1.0
        count = sum(1 for v in vals if v == x)
        return (count + alpha) / (n + alpha * k)

    async def rationale(self) -> str | None:
        """Explain the Bayesian Optimization strategy."""
        return (
            f"Bayesian Optimization using Gaussian Process with "
            f"{self.acquisition.upper()} acquisition function. "
            f"Initial exploration: {self.n_initial} trials, "
            f"xi={self.xi} for exploration-exploitation balance."
        )
