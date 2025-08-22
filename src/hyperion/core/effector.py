"""Typed command emitters for clean policy code."""

import logging
import uuid
from typing import Any

from hyperion.core.bus import EventBus
from hyperion.core.events import Command, CommandType

logger = logging.getLogger(__name__)


class Effector:
    """Provides typed methods for emitting commands to the event bus.

    This is a convenience wrapper that makes policy code cleaner and safer
    by providing typed methods instead of manual command construction.
    """

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus

    async def start_experiment(self, spec: dict[str, Any]) -> str:
        """Start a new experiment.

        Args:
            spec: Experiment specification including name, config, tags
        """
        logger.debug(
            f"Emitting START_EXPERIMENT command for '{spec.get('name', 'unnamed')}'"
        )
        cmd_id = uuid.uuid4().hex
        await self.bus.publish(
            Command(
                id=cmd_id,
                type=CommandType.START_EXPERIMENT,
                data=spec,
                correlation_id=cmd_id,
            )
        )
        return cmd_id

    async def start_trial(
        self,
        experiment_id: str,
        params: dict[str, Any],
        parent_trial_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> str:
        """Start a new trial.

        Args:
            experiment_id: ID of the experiment this trial belongs to
            params: Hyperparameters for this trial
            parent_trial_id: Optional parent for lineage tracking
            tags: Optional metadata tags
        """
        logger.debug(f"Emitting START_TRIAL command for experiment {experiment_id}")
        cmd_id = uuid.uuid4().hex
        await self.bus.publish(
            Command(
                id=cmd_id,
                type=CommandType.START_TRIAL,
                data={
                    "experiment_id": experiment_id,
                    "params": params,
                    "parent_trial_id": parent_trial_id,
                    "tags": tags or {},
                },
                correlation_id=cmd_id,
                aggregate_id=experiment_id,
            )
        )
        return cmd_id

    async def kill_trial(self, trial_id: str, reason: str = "") -> str:
        """Kill a running trial.

        Args:
            trial_id: ID of the trial to kill
            reason: Optional reason for killing the trial
        """
        logger.debug(f"Emitting KILL_TRIAL command for trial {trial_id}")
        cmd_id = uuid.uuid4().hex
        await self.bus.publish(
            Command(
                id=cmd_id,
                type=CommandType.KILL_TRIAL,
                data={"trial_id": trial_id, "reason": reason},
                correlation_id=cmd_id,
            )
        )
        return cmd_id

    async def patch_trial(self, trial_id: str, patch: dict[str, Any]) -> str:
        """Patch trial parameters at runtime.

        Args:
            trial_id: ID of the trial to patch
            patch: Parameter updates to apply
        """
        logger.debug(f"Emitting PATCH_TRIAL command for trial {trial_id}")
        cmd_id = uuid.uuid4().hex
        await self.bus.publish(
            Command(
                id=cmd_id,
                type=CommandType.PATCH_TRIAL,
                data={"trial_id": trial_id, "patch": patch},
                correlation_id=cmd_id,
            )
        )
        return cmd_id
