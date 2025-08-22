from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any, cast


def _connect_sqlite(url: str) -> sqlite3.Connection:
    # Expect format sqlite:///path
    path = url[len("sqlite:///") :] if url.startswith("sqlite:///") else url
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


class DB:
    def __init__(self, url: str) -> None:
        self.conn = _connect_sqlite(url)

    def close(self) -> None:
        with suppress(Exception):
            self.conn.close()

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        cur: sqlite3.Cursor = self.conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    # Experiments
    def list_experiments(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT id,
                       name,
                       CAST(unixepoch(created_at) * 1000 AS INTEGER) AS created_at,
                       status,
                       config_json,
                       tags_json
                FROM experiments
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = [dict(r) for r in cur.fetchall()]
            return rows

    def get_experiment(self, exp_id: str) -> dict[str, Any] | None:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT id,
                       name,
                       CAST(unixepoch(created_at) * 1000 AS INTEGER) AS created_at,
                       status,
                       config_json,
                       tags_json
                FROM experiments
                WHERE id=?
                """,
                (exp_id,),
            )
            r = cur.fetchone()
            return dict(r) if r else None

    # Trials
    def list_trials(
        self, exp_id: str, status: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            if status:
                cur.execute(
                    """
                    SELECT id,experiment_id,params_json,status,score,metrics_last_json,
                           depth,parent_trial_id,branch_id,mutation_op,tags_json
                    FROM trials WHERE experiment_id=? AND status=?
                    ORDER BY id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (exp_id, status, limit, offset),
                )
            else:
                cur.execute(
                    """
                    SELECT id,experiment_id,params_json,status,score,metrics_last_json,
                           depth,parent_trial_id,branch_id,mutation_op,tags_json
                    FROM trials WHERE experiment_id=?
                    ORDER BY id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (exp_id, limit, offset),
                )
            rows: list[dict[str, Any]] = [dict(r) for r in cur.fetchall()]
            return rows

    # Best trial (simple)
    def best_trial(
        self, exp_id: str, metric: str = "score", mode: str = "max"
    ) -> dict[str, Any] | None:
        rows = self.list_trials(exp_id, status="COMPLETED", limit=10000, offset=0)
        if not rows:
            return None

        def key_fn(r: dict[str, Any]) -> float:
            if metric == "score":
                return float(r.get("score") or float("-inf"))
            try:
                m = json.loads(r.get("metrics_last_json") or "{}")
                return float(m.get(metric, float("-inf")))
            except Exception:
                return float("-inf")

        best = max(rows, key=key_fn) if mode == "max" else min(rows, key=key_fn)
        return best

    # Events
    def tail_events(
        self,
        *,
        limit: int | None = 200,
        experiment_id: str | None = None,
        since_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return events in ascending timestamp order.

        - If since_ts is provided, return events with ts > since_ts (chronological).
        - If experiment_id is provided without since_ts, return that experiment's history.
        - If limit is None or <= 0, do not apply a SQL LIMIT clause.
        """
        with self.cursor() as cur:
            params: list[Any] = []

            def _limit_clause() -> str:
                return "" if (limit is None or limit <= 0) else " LIMIT ?"

            def _maybe_add_limit() -> None:
                if not (limit is None or limit <= 0):
                    params.append(limit)

            if experiment_id and since_ts:
                sql = (
                    "SELECT id,type,CAST(unixepoch(ts) * 1000 AS INTEGER) AS ts,"
                    "correlation_id,causation_id,aggregate_id,data_json "
                    "FROM events WHERE aggregate_id=? AND CAST(unixepoch(ts) * 1000 AS INTEGER) > ? ORDER BY ts ASC"
                    + _limit_clause()
                )
                params.extend([experiment_id, since_ts])
                _maybe_add_limit()
                cur.execute(sql, tuple(params))
            elif experiment_id:
                sql = (
                    "SELECT id,type,CAST(unixepoch(ts) * 1000 AS INTEGER) AS ts,"
                    "correlation_id,causation_id,aggregate_id,data_json "
                    "FROM events WHERE aggregate_id=? ORDER BY ts ASC" + _limit_clause()
                )
                params.append(experiment_id)
                _maybe_add_limit()
                cur.execute(sql, tuple(params))
            elif since_ts:
                sql = (
                    "SELECT id,type,CAST(unixepoch(ts) * 1000 AS INTEGER) AS ts,"
                    "correlation_id,causation_id,aggregate_id,data_json "
                    "FROM events WHERE CAST(unixepoch(ts) * 1000 AS INTEGER) > ? ORDER BY ts ASC"
                    + _limit_clause()
                )
                params.append(since_ts)
                _maybe_add_limit()
                cur.execute(sql, tuple(params))
            else:
                sql = (
                    "SELECT id,type,CAST(unixepoch(ts) * 1000 AS INTEGER) AS ts,"
                    "correlation_id,causation_id,aggregate_id,data_json "
                    "FROM events ORDER BY ts ASC" + _limit_clause()
                )
                _maybe_add_limit()
                cur.execute(sql, tuple(params))
            rows: list[dict[str, Any]] = [dict(r) for r in cur.fetchall()]
            return rows

    # Rowid-based streaming helpers (robust against timestamp ties and skew)
    def max_rowid(self, experiment_id: str | None = None) -> int:
        with self.cursor() as cur:
            if experiment_id:
                cur.execute(
                    "SELECT COALESCE(MAX(rowid), 0) FROM events WHERE aggregate_id=?",
                    (experiment_id,),
                )
            else:
                cur.execute("SELECT COALESCE(MAX(rowid), 0) FROM events")
            r = cur.fetchone()
            return int(r[0] if r and r[0] is not None else 0)

    def max_rowid_at_or_before_ts(self, experiment_id: str | None, ts: str) -> int:
        with self.cursor() as cur:
            if experiment_id:
                cur.execute(
                    "SELECT COALESCE(MAX(rowid), 0) FROM events WHERE aggregate_id=? AND ts <= ?",
                    (experiment_id, ts),
                )
            else:
                cur.execute(
                    "SELECT COALESCE(MAX(rowid), 0) FROM events WHERE ts <= ?", (ts,)
                )
            r = cur.fetchone()
            return int(r[0] if r and r[0] is not None else 0)

    def tail_events_after_rowid(
        self, *, experiment_id: str | None, after_rowid: int, limit: int = 200
    ) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            if experiment_id:
                cur.execute(
                    (
                        "SELECT rowid, id, type, CAST(unixepoch(ts) * 1000 AS INTEGER) AS ts, "
                        "correlation_id, causation_id, aggregate_id, data_json "
                        "FROM events WHERE aggregate_id=? AND rowid > ? ORDER BY rowid ASC LIMIT ?"
                    ),
                    (experiment_id, after_rowid, limit),
                )
            else:
                cur.execute(
                    (
                        "SELECT rowid, id, type, CAST(unixepoch(ts) * 1000 AS INTEGER) AS ts, "
                        "correlation_id, causation_id, aggregate_id, data_json "
                        "FROM events WHERE rowid > ? ORDER BY rowid ASC LIMIT ?"
                    ),
                    (after_rowid, limit),
                )
            rows: list[dict[str, Any]] = [dict(r) for r in cur.fetchall()]
            return rows

    # Decisions
    def list_decisions(
        self, exp_id: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM decisions
                WHERE experiment_id=?
                ORDER BY ts DESC
                LIMIT ? OFFSET ?
                """,
                (exp_id, limit, offset),
            )
            rows: list[dict[str, Any]] = [dict(r) for r in cur.fetchall()]
            return rows

    def resolve_action_trial(self, command_id: str) -> str | None:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT json_extract(data_json, '$.trial_id') as trial_id
                FROM events
                WHERE type='TRIAL_STARTED' AND correlation_id=?
                ORDER BY ts ASC LIMIT 1
                """,
                (command_id,),
            )
            r = cur.fetchone()
            return r["trial_id"] if r and r["trial_id"] else None

    # Lineage
    def lineage(self, exp_id: str) -> dict[str, list[dict[str, Any]]]:
        with self.cursor() as cur:
            # Trials
            cur.execute(
                """
                SELECT id,parent_trial_id,depth,status,score,params_json,tags_json,branch_id,mutation_op
                FROM trials WHERE experiment_id=?
                """,
                (exp_id,),
            )
            rows = [dict(r) for r in cur.fetchall()]

            # Map trial_id -> (correlation_id, started_at timestamp) from TRIAL_STARTED
            cur.execute(
                """
                SELECT json_extract(data_json, '$.trial_id') as tid, 
                       correlation_id,
                       CAST(unixepoch(ts) * 1000 AS INTEGER) as started_at
                FROM events
                WHERE type='TRIAL_STARTED' AND aggregate_id=?
                """,
                (exp_id,),
            )
            tid_to_info: dict[str, dict[str, Any]] = {}
            for r in cur.fetchall():
                if r["tid"] is not None:
                    tid_to_info[str(r["tid"])] = {
                        "correlation_id": str(r["correlation_id"])
                        if r["correlation_id"]
                        else None,
                        "started_at": r["started_at"],
                    }

            # Build command_id -> (rationale, actor_type, actor_id)
            cur.execute(
                """
                SELECT rationale, actor_type, actor_id, actions_json
                FROM decisions
                WHERE experiment_id=?
                """,
                (exp_id,),
            )
            cmd_to_rationale: dict[str, dict[str, Any]] = {}
            for d in cur.fetchall():
                try:
                    actions_raw: Any = json.loads(d["actions_json"]) or []
                except Exception:
                    actions_raw = []
                actions_list: list[dict[str, Any]] = []
                if isinstance(actions_raw, list):
                    for item_any in cast(list[Any], actions_raw):
                        if isinstance(item_any, dict):
                            item = cast(dict[str, Any], item_any)
                            actions_list.append(item)
                for a in actions_list:
                    cid_val: Any = a.get("command_id")
                    if not isinstance(cid_val, str | int):
                        continue
                    cid = str(cid_val)
                    cmd_to_rationale[cid] = {
                        "rationale": d["rationale"],
                        "actorType": d["actor_type"],
                        "actorId": d["actor_id"],
                    }
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        ids = {r["id"] for r in rows}
        for r in rows:
            rat = None
            actor_type = None
            actor_id = None
            started_at = None

            # Get info from tid_to_info
            info = tid_to_info.get(r["id"]) if r.get("id") else None
            if info:
                cmd = info.get("correlation_id")
                started_at = info.get("started_at")
                if cmd and cmd in cmd_to_rationale:
                    rat = cmd_to_rationale[cmd].get("rationale")
                    actor_type = cmd_to_rationale[cmd].get("actorType")
                    actor_id = cmd_to_rationale[cmd].get("actorId")

            nodes.append(
                {
                    "id": r["id"],
                    "depth": r.get("depth") or 0,
                    "status": r.get("status"),
                    "score": r.get("score"),
                    "params": json.loads(r.get("params_json") or "{}"),
                    "tags": json.loads(r.get("tags_json") or "{}"),
                    "branchId": r.get("branch_id"),
                    "mutationOp": r.get("mutation_op"),
                    "rationale": rat,
                    "actorType": actor_type,
                    "actorId": actor_id,
                    "startedAt": started_at,
                }
            )
            parent = r.get("parent_trial_id")
            if parent and parent in ids:
                edges.append(
                    {"id": f"{parent}-{r['id']}", "source": parent, "target": r["id"]}
                )
        return {"nodes": nodes, "edges": edges}
