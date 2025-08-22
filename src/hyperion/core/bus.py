"""Event bus for decoupled communication between components."""

import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from hyperion.core.events import Envelope

logger = logging.getLogger(__name__)

# Handler can be sync or async
Handler = Callable[[Envelope], None] | Callable[[Envelope], Awaitable[None]]


class EventBus(Protocol):
    """Protocol for event bus implementations."""

    def subscribe(self, type: str, handler: Handler) -> Callable[[], None]:
        """Subscribe to events of a specific type.

        Args:
            type: Event type to subscribe to, or "*" for all events
            handler: Function to call when event is published

        Returns:
            Unsubscribe function
        """
        ...

    async def publish(self, msg: Envelope) -> None:
        """Publish an event to all subscribers.

        Args:
            msg: Event or command envelope to publish
        """
        ...


class InMemoryBus(EventBus):
    """In-memory event bus implementation for local development."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._wildcard: list[Handler] = []  # "*" subscribers
        self._tasks: set[asyncio.Task[Any]] = set()

    def subscribe(self, type: str, handler: Handler) -> Callable[[], None]:
        """Subscribe to events of a specific type."""
        if type == "*":
            self._wildcard.append(handler)
        else:
            self._subs[type].append(handler)

        def unsub():
            if type == "*":
                self._wildcard.remove(handler)
            else:
                self._subs[type].remove(handler)

        return unsub

    async def publish(self, msg: Envelope) -> None:
        """Publish event to all subscribers."""
        # Get type-specific subscribers and wildcard subscribers
        handlers = list(self._subs.get(msg.type, [])) + self._wildcard

        # Create tasks for all handlers
        logger.debug(f"Publishing {msg.type} to {len(handlers)} handlers")
        for h in handlers:
            t = asyncio.create_task(self._safe_handler(h, msg))
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait for all pending handler tasks to complete.

        Useful for testing to ensure all events are processed.
        """
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _safe_handler(self, handler: Handler, msg: Envelope) -> None:
        """Execute handler safely, catching and logging errors."""
        try:
            result = handler(msg)
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            logger.error(f"Handler error for {msg.type}: {e}")
