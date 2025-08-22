"""Tests for SQLite/SQL storage implementations."""

import pytest
import pytest_asyncio
from tests.unit.storage.test_stores_protocol import (
    DecisionStoreProtocolTests,
    EventLogProtocolTests,
    ExperimentStoreProtocolTests,
    TrialStoreProtocolTests,
)

from hyperion.core.state import (
    DecisionStore,
    EventLog,
    ExperimentStore,
    TrialStore,
)
from hyperion.storage.sql import SQLiteStores


@pytest.fixture
def sqlite_stores(tmp_path):
    """Create SQLiteStores with temporary database."""
    db_path = tmp_path / "test.db"
    stores = SQLiteStores(f"sqlite:///{db_path}")
    yield stores
    # Cleanup is automatic via tmp_path fixture


class TestSQLiteEventLog(EventLogProtocolTests):
    """Test SQLiteEventLog implementation."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_stores(self, tmp_path):
        """Setup stores for each test."""
        db_path = tmp_path / "test.db"
        self.stores = SQLiteStores(f"sqlite:///{db_path}")
        yield
        # Cleanup connection
        self.stores.close()

    def create_event_log(self) -> EventLog:
        """Create a SQLiteEventLog instance."""
        return self.stores.events


class TestSQLiteTrialStore(TrialStoreProtocolTests):
    """Test SQLiteTrialStore implementation."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_stores(self, tmp_path):
        """Setup stores for each test."""
        db_path = tmp_path / "test.db"
        self.stores = SQLiteStores(f"sqlite:///{db_path}")
        yield
        # Cleanup connection
        self.stores.close()

    def create_trial_store(self) -> TrialStore:
        """Create a SQLiteTrialStore instance."""
        return self.stores.trials


class TestSQLiteExperimentStore(ExperimentStoreProtocolTests):
    """Test SQLiteExperimentStore implementation."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_stores(self, tmp_path):
        """Setup stores for each test."""
        db_path = tmp_path / "test.db"
        self.stores = SQLiteStores(f"sqlite:///{db_path}")
        yield
        # Cleanup connection
        self.stores.close()

    def create_experiment_store(self) -> ExperimentStore:
        """Create a SQLiteExperimentStore instance."""
        return self.stores.experiments


class TestSQLiteDecisionStore(DecisionStoreProtocolTests):
    """Test SQLiteDecisionStore implementation."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_stores(self, tmp_path):
        """Setup stores for each test."""
        db_path = tmp_path / "test.db"
        self.stores = SQLiteStores(f"sqlite:///{db_path}")
        yield
        # Cleanup connection
        self.stores.close()

    def create_decision_store(self) -> DecisionStore:
        """Create a SQLiteDecisionStore instance."""
        return self.stores.decisions


class TestSQLiteStoresIntegration:
    """Integration tests for SQLiteStores wrapper."""

    def test_stores_initialization(self, tmp_path):
        """Test that SQLiteStores properly initializes all components."""
        db_path = tmp_path / "test.db"
        stores = SQLiteStores(f"sqlite:///{db_path}")

        # Verify all stores are initialized
        assert stores.events is not None
        assert stores.trials is not None
        assert stores.experiments is not None
        assert stores.decisions is not None

        # Test basic operations work
        exp = stores.experiments.create({"name": "test-exp"})
        assert exp.id is not None

        trial = stores.trials.create(exp.id, {"lr": 0.01}, {})
        assert trial.experiment_id == exp.id

    @pytest.mark.asyncio
    async def test_cross_store_consistency(self, tmp_path):
        """Test that different stores can work together consistently."""
        db_path = tmp_path / "test.db"
        stores = SQLiteStores(f"sqlite:///{db_path}")

        # Create experiment
        exp = stores.experiments.create({"name": "consistency-test"})

        # Create trials
        t1 = stores.trials.create(exp.id, {"batch_size": 32}, {})
        stores.trials.create(exp.id, {"batch_size": 64}, {})

        # Log events
        from hyperion.core.events import Event, EventType

        await stores.events.append(
            Event(
                type=EventType.EXPERIMENT_STARTED,
                data={"experiment_id": exp.id},
                aggregate_id=exp.id,
            )
        )
        await stores.events.append(
            Event(
                type=EventType.TRIAL_STARTED,
                data={"trial_id": t1.id},
                aggregate_id=exp.id,
            )
        )

        # Verify consistency
        events = await stores.events.tail(10, aggregate_id=exp.id)
        assert len(events) == 2

        trials = stores.trials.list_by_experiment(exp.id)
        assert len(trials) == 2

        stores.close()
