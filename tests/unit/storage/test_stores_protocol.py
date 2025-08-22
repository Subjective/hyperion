"""Abstract base tests for all storage implementations.

These tests verify that any storage implementation correctly
implements the storage protocols defined in core/state.py.
"""

from abc import ABC, abstractmethod
from datetime import UTC

import pytest

from hyperion.core.events import Event, EventType
from hyperion.core.models import Decision, ExperimentStatus, TrialStatus
from hyperion.core.state import (
    DecisionStore,
    EventLog,
    ExperimentStore,
    TrialStore,
)


class EventLogProtocolTests(ABC):
    """Base tests that all EventLog implementations must pass."""

    @abstractmethod
    def create_event_log(self) -> EventLog:
        """Create an EventLog instance for testing."""
        ...

    @pytest.mark.asyncio
    async def test_append_and_tail(self):
        """EventLog should append events and allow retrieval."""
        log = self.create_event_log()

        e1 = Event(type=EventType.TRIAL_STARTED, data={"trial_id": "t1"})
        e2 = Event(type=EventType.TRIAL_COMPLETED, data={"trial_id": "t1"})
        e3 = Event(type=EventType.TRIAL_STARTED, data={"trial_id": "t2"})

        await log.append(e1)
        await log.append(e2)
        await log.append(e3)

        # Get last 2 events
        recent = await log.tail(2)
        assert len(recent) == 2
        assert recent[0].data["trial_id"] == "t1"  # e2
        assert recent[1].data["trial_id"] == "t2"  # e3

    @pytest.mark.asyncio
    async def test_tail_by_aggregate_id(self):
        """EventLog should filter by aggregate_id when provided."""
        log = self.create_event_log()

        e1 = Event(type=EventType.TRIAL_STARTED, data={}, aggregate_id="exp-1")
        e2 = Event(type=EventType.TRIAL_STARTED, data={}, aggregate_id="exp-2")
        e3 = Event(type=EventType.TRIAL_COMPLETED, data={}, aggregate_id="exp-1")

        await log.append(e1)
        await log.append(e2)
        await log.append(e3)

        # Get events for exp-1 only
        exp1_events = await log.tail(10, aggregate_id="exp-1")
        assert len(exp1_events) == 2
        assert all(e.aggregate_id == "exp-1" for e in exp1_events)

    @pytest.mark.asyncio
    async def test_tail_empty_log(self):
        """EventLog should return empty list when no events."""
        log = self.create_event_log()
        events = await log.tail(5)
        assert events == []

    @pytest.mark.asyncio
    async def test_tail_more_than_available(self):
        """EventLog should return all events when n > total events."""
        log = self.create_event_log()

        e1 = Event(type=EventType.TRIAL_STARTED, data={"id": 1})
        await log.append(e1)

        events = await log.tail(10)
        assert len(events) == 1
        assert events[0].data["id"] == 1


class TrialStoreProtocolTests(ABC):
    """Base tests that all TrialStore implementations must pass."""

    @abstractmethod
    def create_trial_store(self) -> TrialStore:
        """Create a TrialStore instance for testing."""
        ...

    def test_create(self):
        """TrialStore should create trials with unique IDs."""
        store = self.create_trial_store()

        trial = store.create(
            experiment_id="exp-1",
            params={"lr": 0.01},
            lineage={"parent_trial_id": None, "depth": 0},
        )

        assert trial.id is not None
        assert trial.experiment_id == "exp-1"
        assert trial.params["lr"] == 0.01
        assert trial.status == TrialStatus.PENDING
        assert trial.depth == 0

    def test_get(self):
        """TrialStore should retrieve trials by ID."""
        store = self.create_trial_store()

        trial = store.create("exp-1", {"lr": 0.01}, {})
        retrieved = store.get(trial.id)

        assert retrieved is not None
        assert retrieved.id == trial.id
        assert retrieved.params["lr"] == 0.01

    def test_get_nonexistent(self):
        """TrialStore should return None for nonexistent IDs."""
        store = self.create_trial_store()
        assert store.get("nonexistent-id") is None

    def test_update(self):
        """TrialStore should update trial fields."""
        store = self.create_trial_store()

        trial = store.create("exp-1", {"lr": 0.01}, {})
        store.update(trial.id, status=TrialStatus.RUNNING, score=0.95)

        updated = store.get(trial.id)
        assert updated is not None
        assert updated.status == TrialStatus.RUNNING
        assert updated.score == 0.95

    def test_update_nonexistent(self):
        """TrialStore update should handle nonexistent IDs gracefully."""
        store = self.create_trial_store()
        # Should not raise exception
        store.update("nonexistent-id", status=TrialStatus.RUNNING)

    def test_running(self):
        """TrialStore should list running trials."""
        store = self.create_trial_store()

        t1 = store.create("exp-1", {}, {})
        t2 = store.create("exp-1", {}, {})
        t3 = store.create("exp-2", {}, {})

        store.update(t1.id, status=TrialStatus.RUNNING)
        store.update(t2.id, status=TrialStatus.COMPLETED)
        store.update(t3.id, status=TrialStatus.RUNNING)

        # All running trials
        running = store.running()
        assert len(running) == 2
        assert t1.id in [t.id for t in running]
        assert t3.id in [t.id for t in running]

        # Running trials for exp-1
        exp1_running = store.running(experiment_id="exp-1")
        assert len(exp1_running) == 1
        assert exp1_running[0].id == t1.id

    def test_list_by_experiment(self):
        """TrialStore should list all trials for an experiment."""
        store = self.create_trial_store()

        t1 = store.create("exp-1", {}, {})
        t2 = store.create("exp-1", {}, {})
        t3 = store.create("exp-2", {}, {})

        exp1_trials = store.list_by_experiment("exp-1")
        assert len(exp1_trials) == 2
        assert t1.id in [t.id for t in exp1_trials]
        assert t2.id in [t.id for t in exp1_trials]
        assert t3.id not in [t.id for t in exp1_trials]

    def test_best_of_score(self):
        """TrialStore should find best trial by score."""
        store = self.create_trial_store()

        t1 = store.create("exp-1", {"lr": 0.01}, {})
        t2 = store.create("exp-1", {"lr": 0.02}, {})
        t3 = store.create("exp-1", {"lr": 0.03}, {})

        store.update(t1.id, status=TrialStatus.COMPLETED, score=0.85)
        store.update(t2.id, status=TrialStatus.COMPLETED, score=0.95)
        store.update(t3.id, status=TrialStatus.COMPLETED, score=0.90)

        # Best score (max mode)
        best_max = store.best_of("exp-1", metric="score", mode="max")
        assert best_max["trial_id"] == t2.id
        assert best_max["score"] == 0.95
        assert best_max["params"]["lr"] == 0.02

        # Best score (min mode)
        best_min = store.best_of("exp-1", metric="score", mode="min")
        assert best_min["trial_id"] == t1.id
        assert best_min["score"] == 0.85

    def test_best_of_metrics(self):
        """TrialStore should find best trial by metrics_last fields."""
        store = self.create_trial_store()

        t1 = store.create("exp-1", {"lr": 0.01}, {})
        t2 = store.create("exp-1", {"lr": 0.02}, {})

        store.update(
            t1.id,
            status=TrialStatus.COMPLETED,
            metrics_last={"val_loss": 0.15, "accuracy": 0.85},
        )
        store.update(
            t2.id,
            status=TrialStatus.COMPLETED,
            metrics_last={"val_loss": 0.05, "accuracy": 0.95},
        )

        # Best by val_loss (min)
        best_val_loss = store.best_of("exp-1", metric="val_loss", mode="min")
        assert best_val_loss["trial_id"] == t2.id
        assert best_val_loss["metrics"]["val_loss"] == 0.05

        # Best by accuracy (max)
        best_accuracy = store.best_of("exp-1", metric="accuracy", mode="max")
        assert best_accuracy["trial_id"] == t2.id
        assert best_accuracy["metrics"]["accuracy"] == 0.95

    def test_best_of_empty(self):
        """TrialStore should return empty dict when no completed trials."""
        store = self.create_trial_store()

        # No trials at all
        result = store.best_of("exp-1", metric="score", mode="max")
        assert result == {}

        # Only running trials
        t1 = store.create("exp-1", {}, {})
        store.update(t1.id, status=TrialStatus.RUNNING, score=0.5)
        result = store.best_of("exp-1", metric="score", mode="max")
        assert result == {}

    def test_lineage_tracking(self):
        """TrialStore should properly track trial lineage."""
        store = self.create_trial_store()

        # Create parent trial
        parent = store.create("exp-1", {"lr": 0.01}, {})

        # Create child trial
        child = store.create(
            "exp-1",
            {"lr": 0.02},
            {
                "parent_trial_id": parent.id,
                "depth": 1,
                "branch_id": "branch-1",
                "mutation_op": "lr_increase",
            },
        )

        assert child.parent_trial_id == parent.id
        assert child.depth == 1
        assert child.branch_id == "branch-1"
        assert child.mutation_op == "lr_increase"


class ExperimentStoreProtocolTests(ABC):
    """Base tests that all ExperimentStore implementations must pass."""

    @abstractmethod
    def create_experiment_store(self) -> ExperimentStore:
        """Create an ExperimentStore instance for testing."""
        ...

    def test_create(self):
        """ExperimentStore should create experiments."""
        store = self.create_experiment_store()

        exp = store.create({"name": "test-exp", "config": {"max_trials": 100}})

        assert exp.id is not None
        assert exp.name == "test-exp"
        assert exp.config["max_trials"] == 100
        assert exp.created_at is not None
        assert exp.status == ExperimentStatus.PENDING

    def test_get(self):
        """ExperimentStore should retrieve experiments by ID."""
        store = self.create_experiment_store()

        exp = store.create({"name": "test"})
        retrieved = store.get(exp.id)

        assert retrieved is not None
        assert retrieved.id == exp.id
        assert retrieved.name == "test"

    def test_get_nonexistent(self):
        """ExperimentStore should return None for nonexistent IDs."""
        store = self.create_experiment_store()
        assert store.get("nonexistent-id") is None

    def test_update(self):
        """ExperimentStore should update experiment fields."""
        store = self.create_experiment_store()

        exp = store.create({"name": "test"})
        store.update(exp.id, status=ExperimentStatus.RUNNING, config={"updated": True})

        updated = store.get(exp.id)
        assert updated is not None
        assert updated.status == ExperimentStatus.RUNNING
        assert updated.config["updated"] is True

    def test_update_nonexistent(self):
        """ExperimentStore update should handle nonexistent IDs gracefully."""
        store = self.create_experiment_store()
        # Should not raise exception
        store.update("nonexistent-id", status=ExperimentStatus.RUNNING)

    def test_experiment_tags(self):
        """ExperimentStore should handle tags properly."""
        store = self.create_experiment_store()

        exp = store.create(
            {"name": "tagged-exp", "tags": {"team": "research", "priority": "high"}}
        )

        assert exp.tags["team"] == "research"
        assert exp.tags["priority"] == "high"

        # Update tags
        store.update(exp.id, tags={"team": "research", "priority": "low", "version": 2})
        updated = store.get(exp.id)
        assert updated is not None
        assert updated.tags["priority"] == "low"
        assert updated.tags["version"] == 2


class DecisionStoreProtocolTests(ABC):
    """Base tests that all DecisionStore implementations must pass."""

    @abstractmethod
    def create_decision_store(self) -> DecisionStore:
        """Create a DecisionStore instance for testing."""
        ...

    def test_create_and_get(self):
        """DecisionStore should store and retrieve decisions."""
        import uuid
        from datetime import datetime

        store = self.create_decision_store()

        decision = Decision(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            experiment_id="exp-1",
            actor_type="strategy",
            actor_id="RandomSearchPolicy",
            actions=[{"type": "StartTrial", "params": {"lr": 0.01}}],
            rationale="Random exploration",
            correlation_id="corr-123",
        )

        store.create(decision)
        retrieved = store.get(decision.id)

        assert retrieved is not None
        assert retrieved.id == decision.id
        assert retrieved.actor_id == "RandomSearchPolicy"
        assert retrieved.rationale == "Random exploration"

    def test_get_nonexistent(self):
        """DecisionStore should return None for nonexistent IDs."""
        store = self.create_decision_store()
        assert store.get("nonexistent-id") is None

    def test_list_by_experiment(self):
        """DecisionStore should list all decisions for an experiment."""
        import uuid
        from datetime import datetime

        store = self.create_decision_store()

        # Create decisions for different experiments
        d1 = Decision(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            experiment_id="exp-1",
            actor_type="strategy",
            actor_id="Policy1",
            actions=[],
            rationale=None,
        )
        d2 = Decision(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            experiment_id="exp-1",
            actor_type="agent",
            actor_id="Agent1",
            actions=[],
            rationale=None,
        )
        d3 = Decision(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            experiment_id="exp-2",
            actor_type="strategy",
            actor_id="Policy2",
            actions=[],
            rationale=None,
        )

        store.create(d1)
        store.create(d2)
        store.create(d3)

        exp1_decisions = store.list_by_experiment("exp-1")
        assert len(exp1_decisions) == 2
        assert d1.id in [d.id for d in exp1_decisions]
        assert d2.id in [d.id for d in exp1_decisions]
        assert d3.id not in [d.id for d in exp1_decisions]

    def test_decision_with_trace(self):
        """DecisionStore should handle decision traces properly."""
        import uuid
        from datetime import datetime

        store = self.create_decision_store()

        decision = Decision(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            experiment_id="exp-1",
            actor_type="agent",
            actor_id="LLMAgent",
            actions=[],
            rationale="Based on performance trends",
            trace={
                "prompts": ["Analyze the results..."],
                "responses": ["I recommend..."],
                "tool_calls": ["analyze_metrics"],
            },
        )

        store.create(decision)
        retrieved = store.get(decision.id)

        assert retrieved is not None
        assert retrieved.trace is not None
        assert "prompts" in retrieved.trace
        assert len(retrieved.trace["prompts"]) == 1
