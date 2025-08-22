"""Trial executor implementations."""

import asyncio
import contextlib
import logging
import multiprocessing as mp
import queue
import traceback
from collections.abc import Callable
from typing import Any, Protocol

import cloudpickle  # type: ignore[import-untyped]

from hyperion.core.bus import EventBus
from hyperion.core.context import TrialContext
from hyperion.core.events import Event, EventType

logger = logging.getLogger(__name__)


class TrialExecutor(Protocol):
    """Protocol for trial executor implementations."""

    async def submit(
        self,
        trial_id: str,
        experiment_id: str,
        objective: Callable[..., Any],
        params: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Submit a trial for execution.

        Args:
            trial_id: Unique identifier for the trial
            experiment_id: Unique identifier for the experiment
            objective: Callable to execute with params
            params: Hyperparameters to pass to objective
            meta: Additional metadata for execution (optional)
        """
        ...

    async def kill(self, trial_id: str, reason: str = "") -> None:
        """Kill a running trial.

        Args:
            trial_id: ID of trial to kill
            reason: Optional reason for killing the trial
        """
        ...

    async def patch(self, trial_id: str, patch: dict[str, Any]) -> None:
        """Apply runtime parameter updates to a trial.

        Args:
            trial_id: ID of trial to patch
            patch: Parameter updates to apply
        """
        ...


class LocalAsyncExecutor(TrialExecutor):
    """Thread-backed executor using asyncio.to_thread.

    Preserves cooperative cancellation via TrialContext.should_stop and
    supports progress events while the objective runs.

    # TODO: propagate correlation_id/causation_id/aggregate_id consistently once
    # controller wiring is in place so lifecycle and progress events are chained.
    """

    def __init__(self, bus: EventBus, grace_sec: float = 1.0):
        """Initialize executor.

        Args:
            bus: Event bus for publishing trial lifecycle events
            grace_sec: Grace period before forced termination
                (not used in thread-based impl)
        """
        self.bus = bus
        self.grace_sec = grace_sec
        self._running_trials: dict[str, dict[str, Any]] = {}

    async def submit(
        self,
        trial_id: str,
        experiment_id: str,
        objective: Callable[..., Any],
        params: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Submit a trial for execution in a thread.

        Args:
            trial_id: Unique identifier for the trial
            experiment_id: Unique identifier for the experiment
            objective: Callable to execute with params
            params: Hyperparameters to pass to objective
            meta: Additional metadata for execution (optional)
        """
        # Create stop event for cooperative cancellation
        stop_event = asyncio.Event()
        # Capture the event loop to schedule publishes from the worker thread
        loop = asyncio.get_running_loop()

        # All events use experiment_id as aggregate_id for efficient querying
        # (trial_id is included in event data for filtering if needed)

        # Create report callback that publishes progress events
        def report_callback(tid: str, step: int | str, metrics: dict[str, float]):
            # Schedule publish back onto the captured event loop from the worker thread
            asyncio.run_coroutine_threadsafe(
                self.bus.publish(
                    Event(
                        type=EventType.TRIAL_PROGRESS,
                        data={"trial_id": tid, "step": step, "metrics": metrics},
                        aggregate_id=experiment_id,
                        correlation_id=(meta or {}).get("correlation_id"),
                        causation_id=(meta or {}).get("causation_id"),
                    )
                ),
                loop,
            )

        # Create trial context
        ctx = TrialContext(
            trial_id=trial_id, _report=report_callback, _stop_event=stop_event
        )

        # Store running trial info
        self._running_trials[trial_id] = {
            "stop_event": stop_event,
            "kill_reason": "",  # Will be set if kill() is called
            "task": None,  # TODO: wire this to a future task/handle if we
            # switch to process- or asyncio-based execution
        }

        # Emit TRIAL_STARTED
        await self.bus.publish(
            Event(
                type=EventType.TRIAL_STARTED,
                data={"trial_id": trial_id, "params": params},
                aggregate_id=experiment_id,  # Use experiment_id for aggregation
                correlation_id=(meta or {}).get("correlation_id"),
                causation_id=(meta or {}).get("causation_id"),
            )
        )

        try:
            # Run objective in thread
            # Pass ctx and unpacked params to objective
            result = await asyncio.to_thread(objective, ctx, **params)

            # Check if we were killed during execution
            if stop_event.is_set():
                # Get the kill reason if available
                kill_reason = ""
                if trial_id in self._running_trials:
                    kill_reason = self._running_trials[trial_id].get("kill_reason", "")
                if not kill_reason:
                    kill_reason = "Stopped by kill request"

                # Emit TRIAL_KILLED
                await self.bus.publish(
                    Event(
                        type=EventType.TRIAL_KILLED,
                        data={
                            "trial_id": trial_id,
                            "reason": kill_reason,
                        },
                        aggregate_id=experiment_id,
                        correlation_id=(meta or {}).get("correlation_id"),
                        causation_id=(meta or {}).get("causation_id"),
                    )
                )
            else:
                # Normal completion - emit TRIAL_COMPLETED
                await self.bus.publish(
                    Event(
                        type=EventType.TRIAL_COMPLETED,
                        data={
                            "trial_id": trial_id,
                            "score": result.score,
                            "metrics": result.metrics,
                            "artifacts": result.artifacts,
                        },
                        aggregate_id=experiment_id,
                        correlation_id=(meta or {}).get("correlation_id"),
                        causation_id=(meta or {}).get("causation_id"),
                    )
                )

        except Exception as e:
            # Emit TRIAL_FAILED on error
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Trial {trial_id} failed: {error_msg}")
            logger.debug(traceback.format_exc())

            await self.bus.publish(
                Event(
                    type=EventType.TRIAL_FAILED,
                    data={"trial_id": trial_id, "error": error_msg},
                    aggregate_id=experiment_id,
                    correlation_id=(meta or {}).get("correlation_id"),
                    causation_id=(meta or {}).get("causation_id"),
                )
            )
        finally:
            # Clean up running trial entry
            self._running_trials.pop(trial_id, None)

    async def kill(self, trial_id: str, reason: str = "") -> None:
        """Kill a running trial by setting its stop event.

        Args:
            trial_id: ID of trial to kill
            reason: Optional reason for killing the trial

        NOTE: Emission semantics — TRIAL_KILLED is published if the objective exits
        with the stop flag set; otherwise, if the objective completes first, it
        will emit TRIAL_COMPLETED. Forced termination is not performed in the
        thread-backed executor.
        """
        if trial_id in self._running_trials:
            # Store the kill reason
            self._running_trials[trial_id]["kill_reason"] = reason
            # Set stop event for cooperative cancellation
            self._running_trials[trial_id]["stop_event"].set()
            logger.debug(f"Set stop signal for trial {trial_id}: {reason}")

    async def patch(self, trial_id: str, patch: dict[str, Any]) -> None:
        """Apply runtime parameter updates to a trial.

        Not implemented for thread-based executor.

        Args:
            trial_id: ID of trial to patch
            patch: Parameter updates to apply
        """
        logger.warning(
            f"Patch not implemented for LocalAsyncExecutor (trial {trial_id})"
        )
        # Future: could use a queue to pass patches to running trials


class LocalProcessExecutor(TrialExecutor):
    """Process-backed executor using multiprocessing (spawn context).

    Provides stronger isolation and true parallelism. Progress is streamed via
    an inter-process queue; cooperative cancellation via a shared mp.Event.
    """

    def __init__(self, bus: EventBus, grace_sec: float = 5.0) -> None:
        self.bus = bus
        self.grace_sec = grace_sec
        self._ctx = mp.get_context("spawn")
        self._procs: dict[str, dict[str, Any]] = {}

    async def submit(
        self,
        trial_id: str,
        experiment_id: str,
        objective: Callable[..., Any],
        params: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Submit a trial for execution in a separate process.

        Args:
            trial_id: Unique identifier for the trial
            experiment_id: Unique identifier for the experiment
            objective: Callable to execute with params
            params: Hyperparameters to pass to objective
            meta: Additional metadata for execution (optional)
        """
        stop_event = self._ctx.Event()
        progress_q = self._ctx.Queue(maxsize=256)
        result_q = self._ctx.Queue(maxsize=1)

        # Serialize the objective using cloudpickle to support locals/closures
        pickled_obj = cloudpickle.dumps(objective)  # type: ignore[no-untyped-call]

        # NOTE: do not set daemon=True; daemonic processes cannot create child
        # processes (e.g., torch DataLoader workers). Keep default daemon=False.
        proc = self._ctx.Process(
            target=_worker_main,
            args=(pickled_obj, trial_id, params, stop_event, progress_q, result_q),
        )
        proc.start()

        self._procs[trial_id] = {
            "proc": proc,
            "stop_event": stop_event,
            "progress_q": progress_q,
            "result_q": result_q,
            "experiment_id": experiment_id,
            "meta": meta or {},
            "kill_reason": "",  # Will be set if kill() is called
        }

        # Emit TRIAL_STARTED
        await self.bus.publish(
            Event(
                type=EventType.TRIAL_STARTED,
                data={"trial_id": trial_id, "params": params},
                aggregate_id=experiment_id,  # Use experiment_id for aggregation
                correlation_id=(meta or {}).get("correlation_id"),
                causation_id=(meta or {}).get("causation_id"),
            )
        )

        # Start waiter that drains progress and awaits final result
        asyncio.create_task(self._wait_result(trial_id))

    async def _wait_result(self, trial_id: str) -> None:
        rec = self._procs.get(trial_id)
        if not rec:
            return
        proc: mp.Process = rec["proc"]
        stop_event = rec["stop_event"]
        result_q = rec["result_q"]
        progress_q = rec["progress_q"]
        experiment_id = rec.get("experiment_id", trial_id)
        meta: dict[str, Any] = rec.get("meta", {})

        result: Any | None = None
        # Drain progress while waiting for the final result
        while True:
            # Drain any available progress messages using a short timeout
            try:
                while True:
                    kind, step, metrics = await asyncio.to_thread(
                        progress_q.get, True, 0.01
                    )
                    if kind == "progress":
                        await self.bus.publish(
                            Event(
                                type=EventType.TRIAL_PROGRESS,
                                data={
                                    "trial_id": trial_id,
                                    "step": step,
                                    "metrics": metrics,
                                },
                                aggregate_id=experiment_id,
                                correlation_id=meta.get("correlation_id"),
                                causation_id=meta.get("causation_id"),
                            )
                        )
            except Exception:
                # No progress available right now
                pass

            # Try to obtain the final result with a small timeout
            try:
                kind, payload = await asyncio.to_thread(result_q.get, True, 0.05)
                result = (kind, payload)
                break
            except Exception:
                await asyncio.sleep(0.01)

        # Ensure process is joined/terminated
        if proc.is_alive():
            with contextlib.suppress(Exception):
                proc.join(timeout=self.grace_sec)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)

        # Final drain of any remaining progress messages
        try:
            while True:
                kind, step, metrics = progress_q.get_nowait()
                if kind == "progress":
                    await self.bus.publish(
                        Event(
                            type=EventType.TRIAL_PROGRESS,
                            data={
                                "trial_id": trial_id,
                                "step": step,
                                "metrics": metrics,
                            },
                            aggregate_id=experiment_id,
                            correlation_id=meta.get("correlation_id"),
                            causation_id=meta.get("causation_id"),
                        )
                    )
        except queue.Empty:
            pass

        # Emit lifecycle event
        if result is None:
            await self.bus.publish(
                Event(
                    type=EventType.TRIAL_FAILED,
                    data={"trial_id": trial_id, "error": "No result returned"},
                    aggregate_id=experiment_id,
                    correlation_id=meta.get("correlation_id"),
                    causation_id=meta.get("causation_id"),
                )
            )
        else:
            kind, payload = result
            if kind == "completed":
                if stop_event.is_set():
                    # Get the kill reason if available
                    kill_reason = rec.get("kill_reason", "")
                    if not kill_reason:
                        kill_reason = "Stopped by kill request"

                    await self.bus.publish(
                        Event(
                            type=EventType.TRIAL_KILLED,
                            data={
                                "trial_id": trial_id,
                                "reason": kill_reason,
                            },
                            aggregate_id=experiment_id,
                            correlation_id=meta.get("correlation_id"),
                            causation_id=meta.get("causation_id"),
                        )
                    )
                else:
                    await self.bus.publish(
                        Event(
                            type=EventType.TRIAL_COMPLETED,
                            data={
                                "trial_id": trial_id,
                                "score": payload.get("score"),
                                "metrics": payload.get("metrics", {}),
                                "artifacts": payload.get("artifacts", {}),
                            },
                            aggregate_id=experiment_id,
                            correlation_id=meta.get("correlation_id"),
                            causation_id=meta.get("causation_id"),
                        )
                    )
            elif kind == "failed":
                await self.bus.publish(
                    Event(
                        type=EventType.TRIAL_FAILED,
                        data={
                            "trial_id": trial_id,
                            "error": payload.get("error", "unknown"),
                        },
                        aggregate_id=experiment_id,
                        correlation_id=meta.get("correlation_id"),
                        causation_id=meta.get("causation_id"),
                    )
                )

        # Cleanup
        self._procs.pop(trial_id, None)

    async def kill(self, trial_id: str, reason: str = "") -> None:
        rec = self._procs.get(trial_id)
        if not rec:
            return
        # Store the kill reason
        rec["kill_reason"] = reason
        stop_event = rec["stop_event"]
        rec["proc"]
        stop_event.set()
        logger.debug(f"Set stop signal for trial {trial_id}: {reason}")
        # Allow the process to exit cooperatively; termination handled in _wait_result

    async def patch(self, trial_id: str, patch: dict[str, Any]) -> None:
        logger.warning(
            f"Patch not implemented for LocalProcessExecutor (trial {trial_id})"
        )


def _worker_main(
    pickled_objective: bytes,
    trial_id: str,
    params: dict[str, Any],
    stop_event: Any,  # mp.Event
    progress_q: Any,  # mp.Queue
    result_q: Any,  # mp.Queue
) -> None:
    """Child process entrypoint to execute the objective.

    Sends progress via progress_q and final result via result_q.
    """
    try:
        objective = cloudpickle.loads(pickled_objective)

        def report_callback(
            tid: str, step: int | str, metrics: dict[str, float]
        ) -> None:
            with contextlib.suppress(Exception):
                progress_q.put(("progress", step, metrics))

        ctx = TrialContext(
            trial_id=trial_id, _report=report_callback, _stop_event=stop_event
        )
        res = objective(ctx, **params)
        payload = {
            "score": getattr(res, "score", None),
            "metrics": getattr(res, "metrics", {}),
            "artifacts": getattr(res, "artifacts", {}),
        }
        result_q.put(("completed", payload))
    except Exception as e:  # noqa: BLE001 - we want to catch and report any child error
        err = f"{type(e).__name__}: {e}"
        with contextlib.suppress(Exception):
            result_q.put(("failed", {"error": err}))
