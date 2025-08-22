from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn  # type: ignore[import-untyped]
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .db import DB
from .routers import events, experiments, trials


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Initialize DB connection (shared)
    db_url = os.getenv("HYPERION_DB_URL", "sqlite:///hyperion.db")
    app.state.db = DB(db_url)
    try:
        yield
    finally:
        app.state.db.close()


app = FastAPI(title="Hyperion Dashboard API", version="0.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("HYPERION_CORS_ORIGIN", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Routers
app.include_router(experiments.router, prefix="/api")
app.include_router(trials.router, prefix="/api")
app.include_router(events.router, prefix="/api")


# Simple WebSocket event streaming that tails SQLite events table
@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    await ws.accept()
    db: DB = app.state.db
    qp = ws.scope.get("query_string", b"").decode()
    params = dict([p.split("=", 1) for p in qp.split("&") if "=" in p]) if qp else {}
    experiment_id = params.get("experiment_id")
    poll_ms = int(os.getenv("HYPERION_WS_POLL_MS", "300"))

    # Use SQLite rowid as a robust monotonic cursor to avoid timestamp ties/skew
    # Start from current end so we don't replay history (REST supplies initial snapshot)
    last_rowid = db.max_rowid(experiment_id)

    try:
        while True:
            rows = db.tail_events_after_rowid(
                experiment_id=experiment_id, after_rowid=last_rowid, limit=200
            )
            for r in rows:
                last_rowid = int(r["rowid"])  # advance cursor
                msg = {
                    "id": r["id"],
                    "type": r["type"],
                    "ts": r["ts"],
                    "aggregate_id": r["aggregate_id"],
                    "correlation_id": r["correlation_id"],
                    "causation_id": r["causation_id"],
                    "data": json.loads(r["data_json"] or "{}"),
                }
                await ws.send_text(json.dumps(msg))
            await asyncio.sleep(poll_ms / 1000.0)
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    uvicorn.run("ui.backend.app:app", host="0.0.0.0", port=8000, reload=True)
