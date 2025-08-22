"""Parameter schemas for defining search spaces."""

import random
from dataclasses import dataclass
from math import exp, log
from typing import Any, Protocol, cast


class Serializable(Protocol):
    """Protocol for objects that can be serialized to JSON."""

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        ...


@dataclass(frozen=True)
class Float:
    """Float parameter with optional bounds."""

    min: float | None = None
    max: float | None = None
    log: bool = False

    def validate(self, value: float) -> bool:
        """Check if value is within bounds."""
        if self.min is not None and value < self.min:
            return False
        return not (self.max is not None and value > self.max)

    def clip(self, value: float) -> float:
        """Clip value to valid range."""
        result = value
        if self.min is not None:
            result = max(self.min, result)
        if self.max is not None:
            result = min(self.max, result)
        return result

    def sample(self) -> float:
        """Random sample (kept for convenience)."""
        if self.min is None or self.max is None:
            raise ValueError("Cannot sample from unbounded Float")
        if self.log:
            log_min = log(self.min) if self.min > 0 else log(1e-10)
            log_max = log(self.max) if self.max > 0 else log(1e-10)
            return exp(random.uniform(log_min, log_max))
        return random.uniform(self.min, self.max)

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {"_type": "Float", "min": self.min, "max": self.max, "log": self.log}


@dataclass(frozen=True)
class Int:
    """Integer parameter with optional bounds."""

    min: int | None = None
    max: int | None = None
    log: bool = False

    def validate(self, value: int) -> bool:
        """Check if value is within bounds."""
        if self.min is not None and value < self.min:
            return False
        return not (self.max is not None and value > self.max)

    def clip(self, value: int) -> int:
        """Clip value to valid range."""
        result = value
        if self.min is not None:
            result = max(self.min, result)
        if self.max is not None:
            result = min(self.max, result)
        return result

    def sample(self) -> int:
        """Random sample (kept for convenience)."""
        if self.min is None or self.max is None:
            raise ValueError("Cannot sample from unbounded Int")
        if self.log:
            # Sample in log space then round
            log_min = log(max(1, self.min))  # Avoid log(0)
            log_max = log(self.max) if self.max > 0 else log(1)
            return round(exp(random.uniform(log_min, log_max)))
        else:
            return random.randint(self.min, self.max)

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {"_type": "Int", "min": self.min, "max": self.max, "log": self.log}


@dataclass(frozen=True)
class Choice:
    """Categorical parameter."""

    options: list[Any]

    def validate(self, value: Any) -> bool:
        """Check if value is a valid option."""
        return value in self.options

    def sample(self) -> Any:
        """Sample one option from the list."""
        return random.choice(self.options)

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {"_type": "Choice", "options": self.options}


@dataclass(frozen=True)
class Bool:
    """Boolean parameter."""

    def validate(self, value: Any) -> bool:
        """Check if value is boolean."""
        return isinstance(value, bool)

    def sample(self) -> bool:
        """Sample True or False."""
        return random.random() < 0.5

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {"_type": "Bool"}


@dataclass(frozen=True)
class When:
    """Conditional sampling based on a predicate.

    For future use in conditional search spaces.
    """

    pred: Any  # Callable[[dict[str, Any]], bool]
    spec: Any


def sample_space(space: Any) -> Any:
    """Sample from a search space specification.

    Args:
        space: A search space primitive, dict of primitives, or plain value

    Returns:
        Sampled value(s) with same structure as input
    """
    # Handle primitives
    if isinstance(space, Float | Int | Choice | Bool):
        return space.sample()

    # Handle dict of spaces
    if isinstance(space, dict):
        space_dict = cast(dict[str, Any], space)
        return {key: sample_space(value) for key, value in space_dict.items()}

    # Handle list of spaces
    if isinstance(space, list):
        space_list = cast(list[Any], space)
        return [sample_space(item) for item in space_list]

    # Plain values pass through
    return space


def validate_params(params: dict[str, Any], space: dict[str, Any]) -> bool:
    """Validate parameters against search space.

    Args:
        params: Parameters to validate
        space: Search space specification

    Returns:
        True if all parameters are valid
    """
    for name, value in params.items():
        if name in space:
            spec = space[name]
            if hasattr(spec, "validate") and not spec.validate(value):
                return False
    return True


def clip_params(params: dict[str, Any], space: dict[str, Any]) -> dict[str, Any]:
    """Clip parameters to valid ranges.

    Args:
        params: Parameters to clip
        space: Search space specification

    Returns:
        Clipped parameters
    """
    clipped: dict[str, Any] = {}
    for name, value in params.items():
        if name in space:
            spec = space[name]
            if hasattr(spec, "clip"):
                clipped[name] = spec.clip(value)
            else:
                clipped[name] = value
        else:
            clipped[name] = value
    return clipped
