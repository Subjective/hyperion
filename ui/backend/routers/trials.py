from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request

from ..schemas import Lineage, Trial

router = APIRouter(tags=["trials"])


def _db(request: Request):
    return request.app.state.db


@router.get("/experiments/{exp_id}/trials", response_model=list[Trial])
async def list_trials(
    request: Request,
    exp_id: str,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    db = _db(request)
    rows = db.list_trials(exp_id, status=status, limit=limit, offset=offset)
    return [
        Trial(
            id=r["id"],
            experiment_id=r["experiment_id"],
            status=r["status"],
            score=r.get("score"),
            params=json.loads(r.get("params_json") or "{}"),
            metrics_last=json.loads(r.get("metrics_last_json") or "{}"),
            depth=int(r.get("depth") or 0),
            parent_trial_id=r.get("parent_trial_id"),
            branch_id=r.get("branch_id"),
            mutation_op=r.get("mutation_op"),
            tags=json.loads(r.get("tags_json") or "{}"),
        )
        for r in rows
    ]


@router.get("/experiments/{exp_id}/best")
async def best_trial(
    request: Request,
    exp_id: str,
    metric: str = "score",
    mode: str = "max",
) -> dict[str, Any]:
    db = _db(request)
    r = db.best_trial(exp_id, metric=metric, mode=mode)
    if not r:
        return {}
    return {
        "trial_id": r["id"],
        metric: r["score"]
        if metric == "score"
        else json.loads(r.get("metrics_last_json") or "{}").get(metric),
        "params": json.loads(r.get("params_json") or "{}"),
    }


@router.get("/experiments/{exp_id}/lineage", response_model=Lineage)
async def lineage(request: Request, exp_id: str):
    db = _db(request)
    return db.lineage(exp_id)
