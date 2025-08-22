"""Core data models for experiments and trials."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TrialStatus(str, Enum):
    """Trial lifecycle states."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    KILLED = "KILLED"
    LOST = "LOST"


class ExperimentStatus(str, Enum):
    """Experiment lifecycle states."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ObjectiveResult:
    """Result returned by an objective function.

    Frozen to ensure results are immutable once created.
    """

    score: float
    metrics: dict[str, float] = field(default_factory=lambda: dict[str, float]())
    artifacts: dict[str, str] = field(
        default_factory=lambda: dict[str, str]()
    )  # uri or path


@dataclass
class Trial:
    """Represents a single hyperparameter optimization trial."""

    id: str
    experiment_id: str
    params: dict[str, Any]

    # Status and results
    status: TrialStatus = TrialStatus.PENDING
    score: float | None = None
    metrics_last: dict[str, float] = field(default_factory=lambda: dict[str, float]())

    # Timing
    started_at: datetime | None = None
    ended_at: datetime | None = None

    # Lineage for branching strategies (beam search, PBT, etc.)
    parent_trial_id: str | None = None
    depth: int = 0
    branch_id: str | None = None
    mutation_op: str | None = None

    # Metadata
    tags: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())


@dataclass
class Experiment:
    """Represents a hyperparameter optimization experiment."""

    id: str
    name: str
    created_at: datetime

    # Status and configuration
    status: ExperimentStatus = ExperimentStatus.PENDING
    config: dict[str, Any] = field(
        default_factory=lambda: dict[str, Any]()
    )  # search space, budgets
    tags: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())


@dataclass
class Decision:
    """Represents a recorded decision made by a policy or agent."""

    id: str
    timestamp: datetime
    experiment_id: str

    # Actor information
    actor_type: str  # "strategy" or "agent"
    actor_id: str  # Policy class name or identifier

    # Decision details
    actions: list[dict[str, Any]]  # Serialized actions (StartTrial, KillTrial, etc.)
    rationale: str | None  # Human-readable explanation
    trace: dict[str, Any] | None = None  # Optional detailed trace (e.g., LLM prompts)

    # Event correlation
    correlation_id: str | None = None
    causation_id: str | None = None
