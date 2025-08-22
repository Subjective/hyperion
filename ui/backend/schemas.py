from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Experiment(BaseModel):
    id: str
    name: str
    created_at: int
    status: str
    config: dict[str, Any] | None = None
    tags: dict[str, Any] | None = None


class Trial(BaseModel):
    id: str
    experiment_id: str
    status: str
    score: float | None = None
    params: dict[str, Any]
    metrics_last: dict[str, Any] | None = None
    depth: int = 0
    parent_trial_id: str | None = None
    branch_id: str | None = None
    mutation_op: str | None = None
    tags: dict[str, Any] | None = None


class Event(BaseModel):
    id: str
    type: str
    ts: int
    aggregate_id: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    data: dict[str, Any]


class Lineage(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, str]]
