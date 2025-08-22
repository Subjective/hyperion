"""Grid search strategy for hyperparameter optimization.

Enumerates the Cartesian product of discrete parameter values.

Supported primitives out of the box:
- Choice: uses provided options
- Bool: [True, False]
- Int: inclusive range [low, high] with step 1 when log=False

Uniform (continuous) and Float(log=True) require explicit discretization via
the optional grid_values argument, mapping parameter names to a list of values
to use in the grid.
"""

from collections.abc import Iterable
from itertools import product
from typing import Any

from hyperion.core.events import Event
from hyperion.core.state import ReadableState
from hyperion.framework.policy import Action, Policy, StartTrial
from hyperion.framework.search_space import Bool, Choice, Float, Int


class GridSearchPolicy(Policy):
    """Deterministic grid search policy."""

    def __init__(
        self,
        *,
        space: dict[str, Any],
        experiment_id: str | None = None,
        grid_values: dict[str, list[Any]] | None = None,
        **_: Any,
    ) -> None:
        self.space = space
        self.experiment_id = experiment_id
        self.grid_values = grid_values or {}

        # Precompute grid combinations (ordered by dict key order)
        self._keys: list[str] = list(space.keys())
        value_lists: list[list[Any]] = []

        for key in self._keys:
            values = self._enumerate_values(key, space[key])
            if len(values) == 0:
                raise ValueError(f"Grid for '{key}' produced no values")
            value_lists.append(list(values))

        # Cartesian product of per-key values
        self._grid: list[dict[str, Any]] = [
            {k: v for k, v in zip(self._keys, combo, strict=False)}
            for combo in product(*value_lists)
        ]
        self._next_index: int = 0

    def _enumerate_values(self, key: str, spec: Any) -> list[Any]:
        # User overrides take precedence
        if key in self.grid_values:
            return list(self.grid_values[key])

        if isinstance(spec, Choice):
            return list(spec.options)
        if isinstance(spec, Bool):
            return [True, False]
        if isinstance(spec, Int):
            if spec.log:
                raise ValueError(
                    f"Int(log=True) requires explicit grid_values for '{key}'"
                )
            if not spec.min or not spec.max:
                raise ValueError(
                    f"Grid search requires bounded Int with min and max for '{key}'"
                )
            return list(range(spec.min, spec.max + 1))
        if isinstance(spec, Float):
            raise ValueError(
                f"Float requires explicit grid_values for '{key}' (continuous parameter)"
            )

        # Plain value: treat as fixed singleton
        return [spec]

    async def on_events(self, events: Iterable[Event]) -> None:
        """Process incoming events (currently unused)."""
        pass

    async def decide(self, state: ReadableState) -> list[Action]:
        if not self.experiment_id:
            return []

        free = state.capacity_free()
        if free <= 0:
            return []

        actions: list[Action] = []
        remaining = len(self._grid) - self._next_index
        to_start = min(free, remaining)
        for _ in range(to_start):
            params = self._grid[self._next_index]
            self._next_index += 1
            actions.append(
                StartTrial(
                    experiment_id=self.experiment_id,
                    params=params,
                    tags={
                        "strategy": "grid",
                        "grid_index": self._next_index - 1,
                        "grid_size": len(self._grid),
                        "rationale": f"Grid point {self._next_index}/{len(self._grid)}",
                    },
                )
            )

        return actions

    async def rationale(self) -> str | None:
        total = len(self._grid)
        return f"Grid search: systematically evaluating {total} parameter combinations"
