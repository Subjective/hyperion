"""SQLite/SQL storage implementations using SQLAlchemy."""

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    create_engine,
    desc,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from hyperion.core.events import Event
from hyperion.core.models import (
    Decision,
    Experiment,
    ExperimentStatus,
    Trial,
    TrialStatus,
)
from hyperion.core.state import DecisionStore, EventLog, ExperimentStore, TrialStore

logger = logging.getLogger(__name__)


def _convert_for_json(obj: Any) -> Any:
    """Convert non-JSON-serializable objects for storage."""
    if hasattr(obj, "to_json") and callable(obj.to_json):
        return obj.to_json()
    elif isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}  # type: ignore[misc]
    elif isinstance(obj, list):
        return [_convert_for_json(item) for item in obj]  # type: ignore[misc]
    return obj


# SQLAlchemy Models with full type annotations
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class ExperimentRow(Base):
    """SQLAlchemy model for experiments table."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    tags_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class TrialRow(Base):
    """SQLAlchemy model for trials table."""

    __tablename__ = "trials"
    __table_args__ = (Index("idx_trial_lookup", "experiment_id", "status"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("experiments.id"), nullable=False, index=True
    )
    params_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_last_json: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    parent_trial_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    branch_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mutation_op: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tags_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class EventRow(Base):
    """SQLAlchemy model for events table."""

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    causation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    aggregate_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    data_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class DecisionRow(Base):
    """SQLAlchemy model for decisions table."""

    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    experiment_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("experiments.id"), nullable=False, index=True
    )
    actor_type: Mapped[str] = mapped_column(String(16), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    actions_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    rationale: Mapped[str | None] = mapped_column(String, nullable=True)
    trace_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    causation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


# Store implementations
class SQLiteEventLog(EventLog):
    """SQLite event log implementation.

    Note: The EventLog protocol expects async methods, but Python allows
    synchronous implementations of async protocols. This simplifies the code
    significantly.
    """

    def __init__(self, engine: Engine):
        """Initialize with SQLAlchemy engine."""
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=engine)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        """Get a database session."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def append(self, evt: Event) -> None:
        """Append event to the log."""
        # Serialize event data for JSON storage
        serialized_data = _convert_for_json(evt.data)
        serialized_metadata = _convert_for_json(evt.metadata)

        with self._session() as session:
            row = EventRow(
                id=evt.id,
                type=evt.type,
                ts=evt.ts,
                correlation_id=evt.correlation_id,
                causation_id=evt.causation_id,
                aggregate_id=evt.aggregate_id,
                version=evt.version,
                data_json=serialized_data,
                metadata_json=serialized_metadata,
            )
            session.add(row)
            logger.debug(f"Appended event {evt.type} to SQLite log")

    async def tail(self, n: int, *, aggregate_id: str | None = None) -> list[Event]:
        """Get last N events, optionally filtered by aggregate ID."""
        with self._session() as session:
            query = select(EventRow).order_by(desc(EventRow.ts)).limit(n)
            if aggregate_id:
                query = query.where(EventRow.aggregate_id == aggregate_id)

            rows = session.execute(query).scalars().all()

            # Convert rows to Event objects in chronological order (oldest first)
            return [
                Event(
                    type=row.type,
                    data=row.data_json,
                    id=row.id,
                    ts=row.ts,
                    correlation_id=row.correlation_id,
                    causation_id=row.causation_id,
                    aggregate_id=row.aggregate_id,
                    version=row.version,
                    metadata=row.metadata_json,
                )
                for row in reversed(rows)  # Reverse to get chronological order
            ]


class SQLiteTrialStore(TrialStore):
    """SQLite trial storage implementation."""

    def __init__(self, engine: Engine):
        """Initialize with SQLAlchemy engine."""
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=engine)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        """Get a database session."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create(
        self, experiment_id: str, params: dict[str, Any], lineage: dict[str, Any]
    ) -> Trial:
        """Create a new trial."""
        trial_id = f"trial-{uuid.uuid4().hex[:8]}"

        trial = Trial(
            id=trial_id,
            experiment_id=experiment_id,
            params=params,
            parent_trial_id=lineage.get("parent_trial_id"),
            depth=lineage.get("depth", 0),
            branch_id=lineage.get("branch_id"),
            mutation_op=lineage.get("mutation_op"),
        )

        with self._session() as session:
            row = TrialRow(
                id=trial.id,
                experiment_id=trial.experiment_id,
                params_json=trial.params,
                status=trial.status.value,
                score=trial.score,
                metrics_last_json=trial.metrics_last,
                started_at=trial.started_at,
                ended_at=trial.ended_at,
                parent_trial_id=trial.parent_trial_id,
                depth=trial.depth,
                branch_id=trial.branch_id,
                mutation_op=trial.mutation_op,
                tags_json=trial.tags,
            )
            session.add(row)
            session.commit()

        logger.debug(f"Created trial {trial_id} for experiment {experiment_id}")
        return trial

    def get(self, trial_id: str) -> Trial | None:
        """Get trial by ID."""
        with self._session() as session:
            row = session.get(TrialRow, trial_id)
            if not row:
                return None

            return Trial(
                id=row.id,
                experiment_id=row.experiment_id,
                params=row.params_json,
                status=TrialStatus(row.status),
                score=row.score,
                metrics_last=row.metrics_last_json,
                started_at=row.started_at,
                ended_at=row.ended_at,
                parent_trial_id=row.parent_trial_id,
                depth=row.depth,
                branch_id=row.branch_id,
                mutation_op=row.mutation_op,
                tags=row.tags_json,
            )

    def update(self, trial_id: str, **fields: Any) -> None:
        """Update trial fields."""
        with self._session() as session:
            row = session.get(TrialRow, trial_id)
            if not row:
                logger.warning(f"Attempted to update non-existent trial {trial_id}")
                return

            for key, value in fields.items():
                if key == "status" and isinstance(value, TrialStatus):
                    row.status = value.value
                elif key == "metrics_last":
                    row.metrics_last_json = value
                elif key == "tags":
                    row.tags_json = value
                elif key in ("score", "started_at", "ended_at"):
                    setattr(row, key, value)

            session.commit()
            logger.debug(f"Updated trial {trial_id} with {list(fields.keys())}")

    def running(self, experiment_id: str | None = None) -> list[Trial]:
        """Get running trials."""
        with self._session() as session:
            query = select(TrialRow).where(TrialRow.status == TrialStatus.RUNNING.value)
            if experiment_id:
                query = query.where(TrialRow.experiment_id == experiment_id)

            rows = session.execute(query).scalars().all()
            return [self._row_to_trial(row) for row in rows]

    def list_by_experiment(self, experiment_id: str) -> list[Trial]:
        """List all trials for a given experiment."""
        with self._session() as session:
            rows = (
                session.execute(
                    select(TrialRow).where(TrialRow.experiment_id == experiment_id)
                )
                .scalars()
                .all()
            )
            return [self._row_to_trial(row) for row in rows]

    def best_of(
        self, experiment_id: str, metric: str, mode: Literal["min", "max"]
    ) -> dict[str, Any]:
        """Find best trial by metric using SQL aggregation."""
        with self._session() as session:
            # Build query based on metric type
            query = select(TrialRow).where(
                TrialRow.experiment_id == experiment_id,
                TrialRow.status == TrialStatus.COMPLETED.value,
            )

            # For score, we can use SQL ordering directly
            if metric == "score":
                query = query.where(TrialRow.score.isnot(None))
                query = (
                    query.order_by(desc(TrialRow.score))
                    if mode == "max"
                    else query.order_by(TrialRow.score)
                )

                row = session.execute(query).scalars().first()
                if not row:
                    return {}

                return {
                    "trial_id": row.id,
                    "params": row.params_json,
                    "score": row.score,
                    "metrics": row.metrics_last_json,
                }

            # For metrics_last, load and check in Python for portability
            rows = session.execute(query).scalars().all()
            if not rows:
                return {}

            # Filter trials with the metric
            candidates: list[tuple[TrialRow, float]] = []
            for row in rows:
                value = row.metrics_last_json.get(metric)
                if value is not None:
                    candidates.append((row, value))

            if not candidates:
                return {}

            # Find best
            if mode == "max":
                best_row, _ = max(candidates, key=lambda x: x[1])
            else:
                best_row, _ = min(candidates, key=lambda x: x[1])

            return {
                "trial_id": best_row.id,
                "params": best_row.params_json,
                "score": best_row.score,
                "metrics": best_row.metrics_last_json,
            }

    def _row_to_trial(self, row: TrialRow) -> Trial:
        """Convert database row to Trial object."""
        return Trial(
            id=row.id,
            experiment_id=row.experiment_id,
            params=row.params_json,
            status=TrialStatus(row.status),
            score=row.score,
            metrics_last=row.metrics_last_json,
            started_at=row.started_at,
            ended_at=row.ended_at,
            parent_trial_id=row.parent_trial_id,
            depth=row.depth,
            branch_id=row.branch_id,
            mutation_op=row.mutation_op,
            tags=row.tags_json,
        )


class SQLiteExperimentStore(ExperimentStore):
    """SQLite experiment storage implementation."""

    def __init__(self, engine: Engine):
        """Initialize with SQLAlchemy engine."""
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=engine)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        """Get a database session."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create(self, spec: dict[str, Any]) -> Experiment:
        """Create a new experiment."""
        exp_id = f"exp-{uuid.uuid4().hex[:8]}"

        exp = Experiment(
            id=exp_id,
            name=spec.get("name", "unnamed"),
            created_at=datetime.now(UTC),
            status=ExperimentStatus.PENDING,
            config=spec.get("config", {}),
            tags=spec.get("tags", {}),
        )

        # Serialize config for storage
        config_for_storage = _convert_for_json(exp.config)

        with self._session() as session:
            row = ExperimentRow(
                id=exp.id,
                name=exp.name,
                created_at=exp.created_at,
                status=exp.status.value,
                config_json=config_for_storage,
                tags_json=exp.tags,
            )
            session.add(row)
            session.commit()

        logger.debug(f"Created experiment {exp_id} with name '{exp.name}'")
        return exp

    def get(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        with self._session() as session:
            row = session.get(ExperimentRow, experiment_id)
            if not row:
                return None

            return Experiment(
                id=row.id,
                name=row.name,
                created_at=row.created_at,
                status=ExperimentStatus(row.status),
                config=row.config_json,
                tags=row.tags_json,
            )

    def update(self, experiment_id: str, **fields: Any) -> None:
        """Update experiment fields."""
        with self._session() as session:
            row = session.get(ExperimentRow, experiment_id)
            if not row:
                logger.warning(
                    f"Attempted to update non-existent experiment {experiment_id}"
                )
                return

            for key, value in fields.items():
                if key == "status" and isinstance(value, ExperimentStatus):
                    row.status = value.value
                elif key == "config":
                    row.config_json = value
                elif key == "tags":
                    row.tags_json = value
                elif key == "name":
                    row.name = value

            session.commit()
            logger.debug(
                f"Updated experiment {experiment_id} with {list(fields.keys())}"
            )


class SQLiteDecisionStore(DecisionStore):
    """SQLite decision storage implementation."""

    def __init__(self, engine: Engine):
        """Initialize with SQLAlchemy engine."""
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=engine)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        """Get a database session."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create(self, decision: Decision) -> None:
        """Store a new decision record."""
        with self._session() as session:
            row = DecisionRow(
                id=decision.id,
                ts=decision.timestamp,
                experiment_id=decision.experiment_id,
                actor_type=decision.actor_type,
                actor_id=decision.actor_id,
                actions_json=decision.actions,
                rationale=decision.rationale,
                trace_json=decision.trace,
                correlation_id=decision.correlation_id,
                causation_id=decision.causation_id,
            )
            session.add(row)
            session.commit()

        logger.debug(
            f"Stored decision {decision.id} by {decision.actor_id} "
            f"for experiment {decision.experiment_id}"
        )

    def get(self, decision_id: str) -> Decision | None:
        """Get decision by ID."""
        with self._session() as session:
            row = session.get(DecisionRow, decision_id)
            if not row:
                return None

            return Decision(
                id=row.id,
                timestamp=row.ts,
                experiment_id=row.experiment_id,
                actor_type=row.actor_type,
                actor_id=row.actor_id,
                actions=row.actions_json,
                rationale=row.rationale,
                trace=row.trace_json,
                correlation_id=row.correlation_id,
                causation_id=row.causation_id,
            )

    def list_by_experiment(self, experiment_id: str) -> list[Decision]:
        """List all decisions for a given experiment."""
        with self._session() as session:
            rows = (
                session.execute(
                    select(DecisionRow)
                    .where(DecisionRow.experiment_id == experiment_id)
                    .order_by(DecisionRow.ts)
                )
                .scalars()
                .all()
            )

            return [
                Decision(
                    id=row.id,
                    timestamp=row.ts,
                    experiment_id=row.experiment_id,
                    actor_type=row.actor_type,
                    actor_id=row.actor_id,
                    actions=row.actions_json,
                    rationale=row.rationale,
                    trace=row.trace_json,
                    correlation_id=row.correlation_id,
                    causation_id=row.causation_id,
                )
                for row in rows
            ]


class SQLiteStores:
    """Combined SQLite storage implementation."""

    def __init__(
        self,
        db_url: str = "sqlite:///hyperion.db",
        echo: bool = False,
    ):
        """Initialize SQLite stores.

        Args:
            db_url: SQLAlchemy database URL (e.g., "sqlite:///hyperion.db")
            echo: Whether to echo SQL statements (for debugging)
        """
        # Create engine with appropriate settings
        if "sqlite" in db_url:
            # SQLite-specific settings for better concurrency
            self.engine = create_engine(
                db_url,
                echo=echo,
                connect_args={
                    "check_same_thread": False,  # Allow multi-threaded access
                },
            )
        else:
            self.engine = create_engine(db_url, echo=echo)

        # Create all tables at initialization
        Base.metadata.create_all(self.engine)

        # Initialize individual stores
        self.events = SQLiteEventLog(self.engine)
        self.trials = SQLiteTrialStore(self.engine)
        self.experiments = SQLiteExperimentStore(self.engine)
        self.decisions = SQLiteDecisionStore(self.engine)

    def close(self) -> None:
        """Close database connections."""
        self.engine.dispose()
        logger.debug("Closed SQLite database connections")
