"""Terminal-based visualization for Hyperion experiments."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hyperion.core.events import Event, EventType
from hyperion.core.models import Decision, Trial, TrialStatus
from hyperion.core.state import Stores


@dataclass
class EventStats:
    """Statistics about events."""

    total_events: int = 0
    trials_started: int = 0
    trials_completed: int = 0
    trials_failed: int = 0
    decisions_made: int = 0
    last_event_time: datetime | None = None


class TerminalVisualizer:
    """Terminal-based visualization for experiments."""

    def __init__(self, stores: Stores):
        """Initialize with storage backends.

        Args:
            stores: Storage backends containing experiment data
        """
        self.stores = stores

    def print_summary(self, experiment_id: str, detailed: bool = True) -> None:
        """Print comprehensive experiment summary.

        Args:
            experiment_id: ID of experiment to summarize
            detailed: Whether to show detailed information
        """
        print("\n" + "=" * 80)
        print(f"📊 EXPERIMENT SUMMARY: {experiment_id}")
        print("=" * 80)

        # Get experiment info
        exp = self.stores.experiments.get(experiment_id)
        if exp:
            print(f"\n📋 Experiment: {exp.name}")
            print(f"   Status: {exp.status}")
            print(f"   Created: {exp.created_at}")

        # Print statistics
        self._print_statistics(experiment_id)

        if detailed:
            # Print decision history
            self._print_decisions(experiment_id)

            # Print trial tree
            self._print_trial_tree(experiment_id)

            # Print best trials
            self._print_best_trials(experiment_id)

        print("\n" + "=" * 80)

    def print_timeline(self, experiment_id: str, limit: int = 50) -> None:
        """Print chronological timeline of events.

        Args:
            experiment_id: ID of experiment
            limit: Maximum number of events to show
        """
        print(f"\n📅 TIMELINE for {experiment_id}")
        print("-" * 60)

        # Get events (handle async tail method)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        events = loop.run_until_complete(
            self.stores.events.tail(limit, aggregate_id=experiment_id)
        )
        decisions: dict[str, Decision] = {}
        if hasattr(self.stores, "decisions") and self.stores.decisions:
            for d in self.stores.decisions.list_by_experiment(experiment_id):
                decisions[d.id] = d

        # Sort by timestamp
        events = sorted(events, key=lambda e: e.ts)

        for event in events:
            time_str = event.ts.strftime("%H:%M:%S")
            icon = self._get_event_icon(event.type)

            if event.type == EventType.DECISION_RECORDED:
                decision_id = event.data.get("decision_id")
                if decision_id and decision_id in decisions:
                    decision: Decision = decisions[decision_id]
                    print(f"[{time_str}] {icon} DECISION by {decision.actor_id}")
                    print(f"            💭 {decision.rationale}")
                    print(f"            → {len(decision.actions)} actions")
            else:
                summary = self._summarize_event(event)
                print(f"[{time_str}] {icon} {event.type}")
                if summary:
                    print(f"            {summary}")

    def print_decisions(self, experiment_id: str) -> None:
        """Print decision audit trail.

        Args:
            experiment_id: ID of experiment
        """
        self._print_decisions(experiment_id)

    def print_tree(self, experiment_id: str) -> None:
        """Print trial lineage tree.

        Args:
            experiment_id: ID of experiment
        """
        self._print_trial_tree(experiment_id)

    def export_to_json(self, experiment_id: str) -> dict[str, Any]:
        """Export experiment data as JSON.

        Args:
            experiment_id: ID of experiment

        Returns:
            Dictionary with experiment data
        """
        data: dict[str, Any] = {
            "experiment_id": experiment_id,
            "trials": [],
            "decisions": [],
            "events": [],
        }

        # Export trials
        trials = self.stores.trials.list_by_experiment(experiment_id)
        for trial in trials:
            data["trials"].append(
                {
                    "id": trial.id,
                    "params": trial.params,
                    "status": trial.status.value,
                    "score": trial.score,
                    "metrics": trial.metrics_last,
                    "parent_id": trial.parent_trial_id,
                    "depth": trial.depth,
                }
            )

        # Export decisions
        if hasattr(self.stores, "decisions") and self.stores.decisions:
            decisions = self.stores.decisions.list_by_experiment(experiment_id)
            for decision in decisions:
                data["decisions"].append(
                    {
                        "id": decision.id,
                        "timestamp": decision.timestamp.isoformat(),
                        "actor": decision.actor_id,
                        "rationale": decision.rationale,
                        "actions": decision.actions,
                    }
                )

        # Export recent events (handle async tail method)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        events = loop.run_until_complete(
            self.stores.events.tail(1000, aggregate_id=experiment_id)
        )
        for event in events:
            data["events"].append(
                {
                    "type": event.type,
                    "timestamp": event.ts.isoformat(),
                    "data": event.data,
                }
            )

        return data

    def export_to_markdown(self, experiment_id: str) -> str:
        """Export experiment summary as Markdown.

        Args:
            experiment_id: ID of experiment

        Returns:
            Markdown-formatted report
        """
        lines: list[str] = []
        lines.append(f"# Experiment Report: {experiment_id}\n")

        # Experiment info
        exp = self.stores.experiments.get(experiment_id)
        if exp:
            lines.append("## Experiment Details\n")
            lines.append(f"- **Name**: {exp.name}")
            lines.append(f"- **Status**: {exp.status}")
            lines.append(f"- **Created**: {exp.created_at}\n")

        # Statistics
        stats = self._get_statistics(experiment_id)
        lines.append("## Statistics\n")
        lines.append(f"- Trials Started: {stats.trials_started}")
        lines.append(f"- Trials Completed: {stats.trials_completed}")
        lines.append(f"- Trials Failed: {stats.trials_failed}")
        lines.append(f"- Decisions Made: {stats.decisions_made}\n")

        # Best trials
        lines.append("## Best Trials\n")
        best = self.stores.trials.best_of(experiment_id, "score", "max")
        if best:
            lines.append(f"**Best Score**: {best.get('score', 'N/A')}\n")
            lines.append("**Parameters**:")
            for k, v in best.get("params", {}).items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        # Decision history
        if hasattr(self.stores, "decisions") and self.stores.decisions:
            decisions = self.stores.decisions.list_by_experiment(experiment_id)
            if decisions:
                lines.append("## Decision History\n")
                for decision in decisions:
                    lines.append(
                        f"### {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    lines.append(f"**Actor**: {decision.actor_id}\n")
                    lines.append(f"**Rationale**: {decision.rationale}\n")
                    lines.append(f"**Actions**: {len(decision.actions)} actions\n")

        return "\n".join(lines)

    # Private helper methods

    def _print_statistics(self, experiment_id: str) -> None:
        """Print experiment statistics."""
        stats = self._get_statistics(experiment_id)

        print("\n📈 Statistics:")
        print(f"   Total Events: {stats.total_events}")
        print(f"   Trials Started: {stats.trials_started}")
        print(f"   Trials Completed: {stats.trials_completed}")
        print(f"   Trials Failed: {stats.trials_failed}")
        print(f"   Decisions Made: {stats.decisions_made}")

    def _get_statistics(self, experiment_id: str) -> EventStats:
        """Calculate statistics for an experiment."""
        stats = EventStats()

        # Count events (handle async tail method)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        events = loop.run_until_complete(
            self.stores.events.tail(10000, aggregate_id=experiment_id)
        )
        stats.total_events = len(events)

        for event in events:
            if event.type == EventType.TRIAL_STARTED:
                stats.trials_started += 1
            elif event.type == EventType.TRIAL_COMPLETED:
                stats.trials_completed += 1
            elif event.type == EventType.TRIAL_FAILED:
                stats.trials_failed += 1
            elif event.type == EventType.DECISION_RECORDED:
                stats.decisions_made += 1

            if event.ts and (
                not stats.last_event_time or event.ts > stats.last_event_time
            ):
                stats.last_event_time = event.ts

        return stats

    def _print_decisions(self, experiment_id: str) -> None:
        """Print decision history."""
        if not hasattr(self.stores, "decisions") or not self.stores.decisions:
            return

        decisions = self.stores.decisions.list_by_experiment(experiment_id)
        if not decisions:
            return

        print("\n📝 Decision History:")
        print("-" * 60)

        for decision in sorted(decisions, key=lambda d: d.timestamp):
            time_str = decision.timestamp.strftime("%H:%M:%S")
            print(f"[{time_str}] {decision.actor_id}")
            print(f"   💭 {decision.rationale}")
            print(f"   → Actions: {self._summarize_actions(decision.actions)}")

    def _summarize_actions(self, actions: list[dict[str, Any]]) -> str:
        """Summarize a list of actions."""
        counts: defaultdict[str, int] = defaultdict(int)
        for action in actions:
            counts[action.get("type", "Unknown")] += 1

        parts: list[str] = []
        for action_type, count in counts.items():
            parts.append(f"{count} {action_type}")
        return ", ".join(parts)

    def _print_trial_tree(self, experiment_id: str) -> None:
        """Print trial lineage tree."""
        trials = self.stores.trials.list_by_experiment(experiment_id)
        if not trials:
            return

        print("\n🌳 Trial Lineage:")
        print("-" * 60)

        # Build tree structure
        children_map: defaultdict[str | None, list[Trial]] = defaultdict(list)
        for trial in trials:
            children_map[trial.parent_trial_id].append(trial)

        # Print tree recursively
        root_trials = children_map[None]  # Root trials
        for trial in root_trials:
            self._print_trial_node(trial, children_map, indent=0)

    def _print_trial_node(
        self, trial: Trial, children_map: dict[str | None, list[Trial]], indent: int
    ) -> None:
        """Print a single trial node and its children."""
        # Determine status icon
        if trial.status == TrialStatus.COMPLETED:
            icon = "✅"
        elif trial.status == TrialStatus.FAILED:
            icon = "❌"
        elif trial.status == TrialStatus.KILLED:
            icon = "⛔"
        elif trial.status == TrialStatus.RUNNING:
            icon = "🔄"
        else:
            icon = "⏳"

        # Format score
        score_str = f"{trial.score:.4f}" if trial.score is not None else "---"

        # Print node
        prefix = "  " * indent + ("├─ " if indent > 0 else "")
        print(f"{prefix}{icon} {trial.id[:8]}: score={score_str}")

        # Print children
        for child in children_map.get(trial.id, []):
            self._print_trial_node(child, children_map, indent + 1)

    def _print_best_trials(self, experiment_id: str, top_n: int = 5) -> None:
        """Print best trials."""
        print("\n🏆 Top Trials:")
        print("-" * 60)

        # Get completed trials
        all_trials = self.stores.trials.list_by_experiment(experiment_id)
        completed = [
            t for t in all_trials if t.status == TrialStatus.COMPLETED and t.score
        ]

        if not completed:
            print("   No completed trials yet")
            return

        # Sort by score
        completed.sort(key=lambda t: t.score or 0, reverse=True)

        # Print top N
        for i, trial in enumerate(completed[:top_n], 1):
            score_str = f"{trial.score:.4f}" if trial.score else "N/A"
            param_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in trial.params.items()
            )
            print(f"   {i}. Score: {score_str}")
            print(f"      Params: {param_str}")

    def _get_event_icon(self, event_type: str) -> str:
        """Get icon for event type."""
        icons = {
            EventType.EXPERIMENT_STARTED: "🚀",
            EventType.EXPERIMENT_COMPLETED: "🎯",
            EventType.TRIAL_STARTED: "▶️",
            EventType.TRIAL_COMPLETED: "✅",
            EventType.TRIAL_FAILED: "❌",
            EventType.TRIAL_KILLED: "⛔",
            EventType.TRIAL_PROGRESS: "📊",
            EventType.DECISION_RECORDED: "🤖",
            EventType.CAPACITY_AVAILABLE: "🔓",
        }
        return icons.get(event_type, "•")

    def _summarize_event(self, event: Event) -> str:
        """Create a brief summary of an event."""
        if event.type == EventType.TRIAL_COMPLETED:
            score = event.data.get("score")
            if score:
                return f"Score: {score:.4f}"
        elif event.type == EventType.TRIAL_STARTED:
            trial_id = event.data.get("trial_id", "unknown")
            return f"ID: {trial_id[:8]}"
        elif event.type == EventType.TRIAL_PROGRESS:
            step = event.data.get("step", "?")
            metrics = event.data.get("metrics", {})
            if metrics:
                metric_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in list(metrics.items())[:2]
                )
                return f"Step {step}: {metric_str}"
        elif event.type == EventType.CAPACITY_AVAILABLE:
            count = event.data.get("count", 0)
            return f"Free slots: {count}"

        return ""


class LiveEventLogger:
    """Callback for live event logging during execution."""

    def __init__(self, verbose: bool = False):
        """Initialize the logger.

        Args:
            verbose: Whether to show detailed information
        """
        self.verbose = verbose
        self.trial_count = 0
        self.completed_count = 0
        self.best_score = None
        self.start_time = None

    async def on_event(self, event: Event) -> None:
        """Handle incoming events.

        Args:
            event: Event to log
        """
        icon = self._get_icon(event.type)
        time_str = event.ts.strftime("%H:%M:%S")

        if event.type == EventType.EXPERIMENT_STARTED:
            self.start_time = event.ts
            print(f"\n{icon} Experiment Started: {event.data.get('name', 'unnamed')}")
            print("=" * 60)

        elif event.type == EventType.TRIAL_STARTED:
            self.trial_count += 1
            if self.verbose:
                print(f"[{time_str}] {icon} Trial {self.trial_count} started")

        elif event.type == EventType.TRIAL_COMPLETED:
            self.completed_count += 1
            score = event.data.get("score")
            if score and (not self.best_score or score > self.best_score):
                self.best_score = score

            score_str = f"{score:.4f}" if score else "N/A"
            print(
                f"[{time_str}] {icon} Trial completed "
                f"({self.completed_count}/{self.trial_count}) "
                f"Score: {score_str}"
            )
            if self.best_score:
                print(f"            Best so far: {self.best_score:.4f}")

        elif event.type == EventType.TRIAL_FAILED:
            self.completed_count += 1
            print(
                f"[{time_str}] ❌ Trial failed "
                f"({self.completed_count}/{self.trial_count}): "
                f"{event.data.get('error', 'Unknown error')}"
            )

        elif event.type == EventType.TRIAL_KILLED:
            self.completed_count += 1
            reason = event.data.get("reason", "Unknown reason")
            print(
                f"[{time_str}] ⛔ Trial killed "
                f"({self.completed_count}/{self.trial_count}) "
                f"Reason: {reason}"
            )
            if self.best_score:
                print(f"            Best so far: {self.best_score:.4f}")

        elif event.type == EventType.DECISION_RECORDED and self.verbose:
            actor = event.data.get("actor_id", "Unknown")
            rationale = event.data.get("rationale", "No rationale")
            print(f"[{time_str}] {icon} Decision by {actor}")
            print(f"            💭 {rationale}")

        elif event.type == EventType.EXPERIMENT_COMPLETED:
            duration = (
                (event.ts - self.start_time).total_seconds() if self.start_time else 0
            )
            print("=" * 60)
            print(f"{icon} Experiment Completed in {duration:.1f}s")
            print(f"   Trials: {self.completed_count}/{self.trial_count}")
            if self.best_score:
                print(f"   Best Score: {self.best_score:.4f}")

    def _get_icon(self, event_type: str) -> str:
        """Get icon for event type."""
        icons = {
            EventType.EXPERIMENT_STARTED: "🚀",
            EventType.EXPERIMENT_COMPLETED: "🏁",
            EventType.TRIAL_STARTED: "▶️",
            EventType.TRIAL_COMPLETED: "✅",
            EventType.TRIAL_FAILED: "❌",
            EventType.TRIAL_KILLED: "⛔",
            EventType.DECISION_RECORDED: "🤖",
        }
        return icons.get(event_type, "•")
