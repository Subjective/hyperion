from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request

from ..schemas import Event

router = APIRouter(tags=["events"])


def _db(request: Request):
    return request.app.state.db


@router.get("/events", response_model=list[Event])
async def tail_events(
    request: Request,
    experiment_id: str | None = None,
    since_ts: int | None = None,
    limit: int | None = None,
):
    db = _db(request)
    rows = db.tail_events(limit=limit, experiment_id=experiment_id, since_ts=since_ts)
    return [
        Event(
            id=r["id"],
            type=r["type"],
            ts=r["ts"],
            aggregate_id=r.get("aggregate_id"),
            correlation_id=r.get("correlation_id"),
            causation_id=r.get("causation_id"),
            data=json.loads(r.get("data_json") or "{}"),
        )
        for r in rows
    ]


@router.get("/experiments/{exp_id}/decisions")
async def list_decisions(
    request: Request,
    exp_id: str,
    limit: int = 100,
    offset: int = 0,
    resolve: bool = True,
):
    db = _db(request)
    rows = db.list_decisions(exp_id, limit=limit, offset=offset)
    out: list[dict[str, Any]] = []
    for r in rows:
        actions = json.loads(r.get("actions_json") or "[]")
        if resolve:
            for a in actions:
                cmd_id = a.get("command_id")
                if cmd_id:
                    a["trial_id"] = db.resolve_action_trial(cmd_id)
        out.append(
            {
                "id": r["id"],
                "ts": r["ts"],
                "experiment_id": r["experiment_id"],
                "actor_type": r["actor_type"],
                "actor_id": r["actor_id"],
                "actions": actions,
                "rationale": r.get("rationale"),
                "trace": json.loads(r.get("trace_json") or "{}"),
            }
        )
    return out
