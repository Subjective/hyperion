from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..schemas import Experiment

router = APIRouter(tags=["experiments"])


def _db(request: Request):
    return request.app.state.db


@router.get("/experiments", response_model=list[Experiment])
async def list_experiments(request: Request, limit: int = 50, offset: int = 0):
    db = _db(request)
    rows = db.list_experiments(limit=limit, offset=offset)
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "name": r["name"],
                "created_at": r["created_at"],
                "status": r["status"],
                "config": json.loads(r.get("config_json") or "{}"),
                "tags": json.loads(r.get("tags_json") or "{}"),
            }
        )
    return out


@router.get("/experiments/{exp_id}", response_model=Experiment)
async def get_experiment(request: Request, exp_id: str):
    db = _db(request)
    r = db.get_experiment(exp_id)
    if not r:
        raise HTTPException(status_code=404, detail="experiment not found")
    return Experiment(
        id=r["id"],
        name=r["name"],
        created_at=r["created_at"],
        status=r["status"],
        config=json.loads(r.get("config_json") or "{}"),
        tags=json.loads(r.get("tags_json") or "{}"),
    )
