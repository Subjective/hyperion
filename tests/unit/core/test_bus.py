"""Tests for the event bus implementation."""

import asyncio

import pytest

from hyperion.core.bus import InMemoryBus
from hyperion.core.events import Event, EventType


@pytest.mark.asyncio
async def test_subscribe_returns_unsubscribe_callable():
    """Subscribe should return a function to unsubscribe."""
    bus = InMemoryBus()
    received = []

    unsub = bus.subscribe("TEST_EVENT", lambda e: received.append(e))
    event = Event(type="TEST_EVENT", data={})

    # Should receive while subscribed
    await bus.publish(event)
    await bus.drain()
    assert len(received) == 1

    # Should not receive after unsubscribe
    unsub()
    await bus.publish(event)
    await bus.drain()
    assert len(received) == 1  # Still just 1


@pytest.mark.asyncio
async def test_publish_delivers_to_correct_subscribers():
    """Events should only go to subscribers of that event type."""
    bus = InMemoryBus()
    trial_events = []
    experiment_events = []

    bus.subscribe(EventType.TRIAL_STARTED, lambda e: trial_events.append(e))
    bus.subscribe(EventType.EXPERIMENT_STARTED, lambda e: experiment_events.append(e))

    trial_event = Event(type=EventType.TRIAL_STARTED, data={"trial_id": "t1"})
    exp_event = Event(type=EventType.EXPERIMENT_STARTED, data={"exp_id": "e1"})

    await bus.publish(trial_event)
    await bus.publish(exp_event)
    await bus.drain()

    assert len(trial_events) == 1
    assert trial_events[0].data["trial_id"] == "t1"
    assert len(experiment_events) == 1
    assert experiment_events[0].data["exp_id"] == "e1"


@pytest.mark.asyncio
async def test_wildcard_subscribers_receive_all_events():
    """Wildcard (*) subscribers should receive all event types."""
    bus = InMemoryBus()
    all_events = []

    bus.subscribe("*", lambda e: all_events.append(e))

    await bus.publish(Event(type=EventType.TRIAL_STARTED, data={}))
    await bus.publish(Event(type=EventType.TRIAL_COMPLETED, data={}))
    await bus.publish(Event(type="CUSTOM_EVENT", data={}))
    await bus.drain()

    assert len(all_events) == 3
    assert all_events[0].type == EventType.TRIAL_STARTED
    assert all_events[1].type == EventType.TRIAL_COMPLETED
    assert all_events[2].type == "CUSTOM_EVENT"


@pytest.mark.asyncio
async def test_handler_errors_dont_affect_other_handlers():
    """One handler failing shouldn't prevent others from receiving events."""
    bus = InMemoryBus()
    received = []

    def failing_handler(e):
        raise ValueError("Handler error!")

    def working_handler(e):
        received.append(e)

    bus.subscribe("TEST_EVENT", failing_handler)
    bus.subscribe("TEST_EVENT", working_handler)

    await bus.publish(Event(type="TEST_EVENT", data={}))
    await bus.drain()

    # Working handler should still receive the event
    assert len(received) == 1


@pytest.mark.asyncio
async def test_async_handlers_are_awaited():
    """Async handlers should be properly awaited."""
    bus = InMemoryBus()
    processed = []

    async def async_handler(event):
        await asyncio.sleep(0.01)  # Simulate async work
        processed.append(event)

    bus.subscribe("ASYNC_EVENT", async_handler)

    await bus.publish(Event(type="ASYNC_EVENT", data={"id": 1}))
    await bus.drain()  # Should wait for async handler to complete

    assert len(processed) == 1
    assert processed[0].data["id"] == 1


@pytest.mark.asyncio
async def test_multiple_subscribers_to_same_event():
    """Multiple handlers can subscribe to the same event type."""
    bus = InMemoryBus()
    handler1_received = []
    handler2_received = []

    bus.subscribe("SHARED_EVENT", lambda e: handler1_received.append(e))
    bus.subscribe("SHARED_EVENT", lambda e: handler2_received.append(e))

    event = Event(type="SHARED_EVENT", data={"value": 42})
    await bus.publish(event)
    await bus.drain()

    assert len(handler1_received) == 1
    assert len(handler2_received) == 1
    assert handler1_received[0].data["value"] == 42
    assert handler2_received[0].data["value"] == 42
