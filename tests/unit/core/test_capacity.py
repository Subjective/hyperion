"""Tests for capacity management."""

from hyperion.core.capacity import CapacityManager


def test_admit_up_to_max_concurrent():
    """Repeated can_admit()/on_start() admits exactly max_concurrent."""
    capacity = CapacityManager(max_concurrent=3)

    # Should admit first 3
    assert capacity.can_admit("exp-1") is True
    capacity.on_start("exp-1")

    assert capacity.can_admit("exp-1") is True
    capacity.on_start("exp-1")

    assert capacity.can_admit("exp-1") is True
    capacity.on_start("exp-1")

    # Should deny the 4th
    assert capacity.can_admit("exp-1") is False


def test_release_allows_new_admission():
    """After on_end(), a new can_admit() turns true."""
    capacity = CapacityManager(max_concurrent=2)

    # Fill capacity
    assert capacity.can_admit("exp-1") is True
    capacity.on_start("exp-1")
    assert capacity.can_admit("exp-1") is True
    capacity.on_start("exp-1")

    # At capacity
    assert capacity.can_admit("exp-1") is False

    # Release one slot
    capacity.on_end("exp-1")

    # Should be able to admit again
    assert capacity.can_admit("exp-1") is True


def test_per_experiment_does_not_exceed_global():
    """With multiple experiment_ids, total admissions never exceed global max_concurrent."""
    capacity = CapacityManager(max_concurrent=3)

    # Admit from different experiments
    assert capacity.can_admit("exp-1") is True
    capacity.on_start("exp-1")

    assert capacity.can_admit("exp-2") is True
    capacity.on_start("exp-2")

    assert capacity.can_admit("exp-1") is True
    capacity.on_start("exp-1")

    # Global limit reached
    assert capacity.can_admit("exp-3") is False
    assert capacity.can_admit("exp-1") is False
    assert capacity.can_admit("exp-2") is False
