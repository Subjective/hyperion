"""Hyperion - Event-driven agentic hyperparameter optimization framework."""

from hyperion.api.tune import tune
from hyperion.core.context import TrialContext
from hyperion.core.models import ObjectiveResult
from hyperion.framework.search_space import (
    Bool,
    Choice,
    Float,
    Int,
    When,
    sample_space,
)

__version__ = "0.1.0"
__all__ = [
    "tune",
    "ObjectiveResult",
    "TrialContext",
    "Float",
    "Int",
    "Choice",
    "Bool",
    "When",
    "sample_space",
]
