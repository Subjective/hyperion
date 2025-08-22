"""Tests for in-memory storage implementations."""

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
from hyperion.storage.memory import (
    InMemoryDecisionStore,
    InMemoryEventLog,
    InMemoryExperimentStore,
    InMemoryTrialStore,
)


class TestInMemoryEventLog(EventLogProtocolTests):
    """Test InMemoryEventLog implementation."""

    def create_event_log(self) -> EventLog:
        """Create an InMemoryEventLog instance."""
        return InMemoryEventLog()


class TestInMemoryTrialStore(TrialStoreProtocolTests):
    """Test InMemoryTrialStore implementation."""

    def create_trial_store(self) -> TrialStore:
        """Create an InMemoryTrialStore instance."""
        return InMemoryTrialStore()


class TestInMemoryExperimentStore(ExperimentStoreProtocolTests):
    """Test InMemoryExperimentStore implementation."""

    def create_experiment_store(self) -> ExperimentStore:
        """Create an InMemoryExperimentStore instance."""
        return InMemoryExperimentStore()


class TestInMemoryDecisionStore(DecisionStoreProtocolTests):
    """Test InMemoryDecisionStore implementation."""

    def create_decision_store(self) -> DecisionStore:
        """Create an InMemoryDecisionStore instance."""
        return InMemoryDecisionStore()
