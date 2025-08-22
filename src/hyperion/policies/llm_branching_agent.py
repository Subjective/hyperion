"""LLM-based branching agent for intelligent hyperparameter exploration.

This agent combines beam search's systematic exploration with LLM intelligence
to create meaningful parameter branches based on trial results and patterns.
"""

import asyncio
import json
import logging
import re
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from hyperion.core.events import Event, EventType
from hyperion.core.state import ReadableState
from hyperion.framework.policy import Action, KillTrial, Policy, StartTrial
from hyperion.framework.search_space import Choice, Float, Int, sample_space

logger = logging.getLogger(__name__)


@dataclass
class TrialNode:
    """Node in the trial tree representing a trial and its relationships."""

    trial_id: str
    params: dict[str, Any]
    score: float | None
    metrics: dict[str, float]
    status: str
    depth: int
    parent_id: str | None = None
    children: list[str] = field(default_factory=lambda: list[str]())
    branching_rationale: str | None = None


@dataclass
class BranchingDecision:
    """Represents a branching decision made by the LLM."""

    parent_trial_id: str
    branch_count: int
    parameter_focus: list[str]  # Which parameters to vary
    variations: list[dict[str, Any]]  # Specific parameter sets
    rationale: str
    confidence: Literal["low", "medium", "high"]


class LLMBranchingAgent(Policy):
    """LLM-powered agent that creates intelligent parameter branches.

    This agent:
    1. Maintains a tree of trials with parent-child relationships
    2. Uses LLM to analyze patterns and decide branching strategies
    3. Creates multiple child trials from promising parents
    4. Can prune underperforming branches to focus resources
    5. Provides detailed rationale for all branching decisions
    """

    def __init__(
        self,
        *,
        space: dict[str, Any],
        experiment_id: str | None = None,
        llm: Callable[[str], str] | None = None,
        metric: str = "score",
        mode: Literal["min", "max"] = "max",
        max_depth: int = 5,
        beam_width: int = 3,  # Max trials to keep per depth
        branch_factor: int = 3,  # Max branches per parent
        prune_ratio: float = 0.5,  # Prune bottom 50% at each depth
        enable_pruning: bool = True,
        **kwargs: Any,
    ):
        """Initialize the LLM branching agent.

        Args:
            space: Search space specification (optional - LLM can infer parameters)
            experiment_id: ID of experiment to run trials for
            llm: Callable that takes a JSON prompt string and returns raw text (ideally JSON)
            metric: Metric to optimize
            mode: Optimization mode - "min" or "max"
            max_depth: Maximum depth of the trial tree
            beam_width: Maximum trials to keep active per depth
            branch_factor: Maximum branches to create per parent
            prune_ratio: Fraction of trials to prune at each depth
            enable_pruning: Whether to prune underperforming branches
        """
        self.space = space
        self.experiment_id = experiment_id
        if llm is None:
            raise ValueError(
                "LLMBranchingAgent requires an 'llm' callable (prompt: str) -> str."
            )
        self.llm: Callable[[str], str] = llm
        self.metric = metric
        self.mode = mode
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.branch_factor = branch_factor
        self.prune_ratio = prune_ratio
        self.enable_pruning = enable_pruning

        # Trial tree management
        self.trial_tree: dict[str, TrialNode] = {}
        self.trials_by_depth: defaultdict[int, list[str]] = defaultdict(list)
        self.frontier: dict[int, list[str]] = {}  # Top trials per depth

        # Decision tracking
        self.decisions_made = 0
        self.last_rationale = "Initial state - no decisions made yet"
        self.branching_history: deque[BranchingDecision] = deque(maxlen=20)

        # Event tracking
        self.recent_events: deque[Event] = deque(maxlen=100)
        self.pending_trials: set[str] = (
            set()
        )  # Trials we've started but not seen complete

    async def on_events(self, events: Iterable[Event]) -> None:
        """Process incoming events to update trial tree."""
        for event in events:
            self.recent_events.append(event)

            if event.type == EventType.TRIAL_STARTED:
                trial_id = event.data.get("trial_id")
                params = event.data.get("params", {})
                parent_id = event.data.get("parent_trial_id")

                if trial_id:
                    # Track depth based on parent
                    depth = 0
                    if parent_id and parent_id in self.trial_tree:
                        depth = self.trial_tree[parent_id].depth + 1

                    # Create node in tree
                    node = TrialNode(
                        trial_id=trial_id,
                        params=params,
                        score=None,
                        metrics={},
                        status="RUNNING",
                        depth=depth,
                        parent_id=parent_id,
                    )
                    self.trial_tree[trial_id] = node
                    self.trials_by_depth[depth].append(trial_id)

                    # Update parent's children list
                    if parent_id and parent_id in self.trial_tree:
                        self.trial_tree[parent_id].children.append(trial_id)

                    # Track as pending if we initiated it
                    if trial_id in self.pending_trials:
                        self.pending_trials.remove(trial_id)

            elif event.type == EventType.TRIAL_COMPLETED:
                trial_id = event.data.get("trial_id")
                score = event.data.get("score")
                metrics = event.data.get("metrics", {})

                if trial_id and trial_id in self.trial_tree:
                    node = self.trial_tree[trial_id]
                    node.score = score
                    node.metrics = metrics
                    node.status = "COMPLETED"

            elif event.type in (EventType.TRIAL_FAILED, EventType.TRIAL_KILLED):
                trial_id = event.data.get("trial_id")
                if trial_id and trial_id in self.trial_tree:
                    self.trial_tree[trial_id].status = (
                        "FAILED" if event.type == EventType.TRIAL_FAILED else "KILLED"
                    )

    async def decide(self, state: ReadableState) -> list[Action]:
        """Make branching decisions using LLM analysis."""
        if not self.experiment_id:
            return []

        free_slots = state.capacity_free()
        if free_slots <= 0:
            return []

        actions: list[Action] = []

        # Update frontier (top performers per depth)
        self._update_frontier()

        # Check if we need to bootstrap (no trials yet)
        if not self.trial_tree:
            return self._bootstrap_trials(free_slots)

        try:
            # Get LLM branching decisions
            decisions = await self._get_branching_decisions(state, free_slots)

            # Convert decisions to actions
            for decision in decisions:
                for params in decision.variations[:free_slots]:
                    trial_action = StartTrial(
                        experiment_id=self.experiment_id,
                        params=params,
                        parent_trial_id=decision.parent_trial_id,
                        tags={
                            "strategy": "llm_branching",
                            "parent": decision.parent_trial_id,
                            "confidence": decision.confidence,
                            "focus": ",".join(decision.parameter_focus),
                        },
                    )
                    actions.append(trial_action)
                    free_slots -= 1
                    if free_slots <= 0:
                        break

                # Store decision for history
                self.branching_history.append(decision)

                if free_slots <= 0:
                    break

            # Prune underperforming branches if enabled
            if self.enable_pruning:
                prune_actions = self._get_pruning_actions(state)
                actions.extend(prune_actions)

            self.decisions_made += 1

        except Exception as e:
            logger.error(f"LLM branching decision failed: {e}, falling back to random")
            # Fallback: sample randomly
            for _ in range(min(free_slots, 1)):
                params = sample_space(self.space)
                actions.append(
                    StartTrial(
                        experiment_id=self.experiment_id,
                        params=params,
                        tags={"strategy": "llm_branching", "fallback": "error"},
                    )
                )

        return actions

    def _bootstrap_trials(self, free_slots: int) -> list[Action]:
        """Create initial trials when no history exists."""
        actions: list[Action] = []
        num_initial = min(free_slots, self.beam_width)

        for _ in range(num_initial):
            params = sample_space(self.space)
            actions.append(
                StartTrial(
                    experiment_id=self.experiment_id,  # type: ignore[arg-type]
                    params=params,
                    tags={"strategy": "llm_branching", "bootstrap": True},
                )
            )

        self.last_rationale = f"Bootstrapping with {num_initial} random trials"
        return actions

    def _update_frontier(self) -> None:
        """Update the frontier of top trials per depth."""
        self.frontier.clear()

        for depth, trial_ids in self.trials_by_depth.items():
            # Get completed trials at this depth
            completed = [
                tid
                for tid in trial_ids
                if self.trial_tree[tid].status == "COMPLETED"
                and self.trial_tree[tid].score is not None
            ]

            if not completed:
                continue

            # Sort by score
            completed.sort(
                key=lambda tid: self._get_trial_score(self.trial_tree[tid]),
                reverse=(self.mode == "max"),
            )

            # Keep top beam_width trials
            self.frontier[depth] = completed[: self.beam_width]

    def _get_trial_score(self, node: TrialNode) -> float:
        """Get the score for a trial node."""
        if self.metric == "score":
            return float(node.score or float("-inf" if self.mode == "max" else "inf"))

        val = node.metrics.get(self.metric)
        if val is None:
            return float("-inf" if self.mode == "max" else "inf")
        return float(val)

    async def _get_branching_decisions(
        self, state: ReadableState, max_branches: int
    ) -> list[BranchingDecision]:
        """Query LLM for branching decisions."""
        # Build context for LLM
        context = self._build_branching_context(state, max_branches)

        try:
            # Query LLM
            response = await self._query_llm_for_branches(context)

            # Parse response into decisions
            decisions = self._parse_branching_response(response, max_branches)

            # Print decision summary
            self._print_decision_summary(decisions)

            return decisions

        except Exception as e:
            logger.error(f"Failed to get LLM branching decisions: {e}")
            # Fallback: branch from best frontier trial
            return self._fallback_branching(max_branches)

    def _build_branching_context(self, state: ReadableState, max_branches: int) -> str:
        """Build context string for LLM branching decisions."""
        # Get trial tree summary
        tree_summary = self._format_trial_tree()

        # Get performance trends
        trends = self._analyze_performance_trends()

        # Format search space
        space_desc = self._format_search_space()

        prompt = f"""You are an expert hyperparameter optimization agent using a branching search strategy.
Your task is to decide which trials to branch from and what parameter variations to explore.

OPTIMIZATION GOAL: {{"Maximize" if self.mode == "max" else "Minimize"}} the metric '{self.metric}'

SEARCH SPACE:
{space_desc}

TRIAL TREE STRUCTURE:
{tree_summary}

PERFORMANCE ANALYSIS:
{trends}

CURRENT FRONTIER (best trials per depth):
{self._format_frontier()}

BRANCHING CONSTRAINTS:
- Maximum depth: {self.max_depth}
- Available capacity: {max_branches} new trials
- Branch factor: up to {self.branch_factor} branches per parent

Based on the trial history, decide:
1. Which completed trials are most promising to branch from?
2. What parameter variations should each branch explore?
3. Why are these branches likely to improve performance?

Consider:
- Parameter sensitivity: Which parameters show the most impact?
- Unexplored regions: What combinations haven't been tried?
- Performance patterns: What correlations exist between parameters and scores?
- Exploitation vs exploration: Should we refine promising areas or explore new ones?

Respond with a JSON object containing an array of branching decisions:
{{
    "decisions": [
        {{
            "parent_trial_id": "trial_id_to_branch_from",
            "rationale": "Why this trial is worth branching from",
            "parameter_focus": ["param1", "param2"],
            "variations": [
                {{"param1": value1, "param2": value2, ...}},
                {{"param1": value3, "param2": value4, ...}}
            ],
            "confidence": "high"
        }}
    ],
    "overall_strategy": "Brief explanation of the branching strategy"
}}
"""  # noqa: E501
        return prompt

    def _format_trial_tree(self) -> str:
        """Format trial tree structure for LLM."""
        lines: list[str] = []
        for depth in sorted(self.trials_by_depth.keys()):
            trials = self.trials_by_depth[depth]
            completed = [t for t in trials if self.trial_tree[t].status == "COMPLETED"]

            if completed:
                lines.append(f"\nDepth {depth} ({len(completed)} completed):")
                for tid in completed[:5]:  # Show top 5
                    node = self.trial_tree[tid]
                    score_str = f"{node.score:.4f}" if node.score else "N/A"
                    param_str = ", ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in node.params.items()
                    )
                    lines.append(f"  - {tid}: score={score_str} | {param_str}")
                    if len(node.children) > 0:
                        lines.append(f"    → {len(node.children)} children")

        return "\n".join(lines) if lines else "No completed trials yet"

    def _analyze_performance_trends(self) -> str:
        """Analyze performance trends across the trial tree."""
        if not self.trial_tree:
            return "No trials to analyze"

        completed_trials = [
            node
            for node in self.trial_tree.values()
            if node.status == "COMPLETED" and node.score is not None
        ]

        if not completed_trials:
            return "No completed trials yet"

        # Find best overall trial
        best_trial = max(
            completed_trials,
            key=lambda n: self._get_trial_score(n) * (1 if self.mode == "max" else -1),
        )

        # Analyze parameter impact (simple correlation)
        param_impacts: list[str] = []
        for param in self.space:
            values_scores = [
                (node.params.get(param), self._get_trial_score(node))
                for node in completed_trials
                if param in node.params
            ]

            if len(values_scores) > 1:
                # Simple heuristic: parameter variance vs score variance
                param_impacts.append(
                    f"- {param}: {len(set(v for v, _ in values_scores))} unique values tested"
                )

        lines: list[str] = [
            f"Best trial: {best_trial.trial_id} with score {best_trial.score:.4f}",
            f"Total completed: {len(completed_trials)}",
            f"Average depth explored: {sum(n.depth for n in completed_trials) / len(completed_trials):.1f}",
            "\nParameter exploration:",
        ]
        lines.extend(param_impacts)

        return "\n".join(lines)

    def _format_frontier(self) -> str:
        """Format current frontier for LLM."""
        if not self.frontier:
            return "No frontier established yet"

        lines: list[str] = []
        for depth in sorted(self.frontier.keys()):
            trial_ids = self.frontier[depth]
            lines.append(f"Depth {depth}:")
            for tid in trial_ids:
                node = self.trial_tree[tid]
                score_str = f"{node.score:.4f}" if node.score else "N/A"
                lines.append(
                    f"  - {tid}: score={score_str}, {len(node.children)} children"
                )

        return "\n".join(lines)

    def _format_search_space(self) -> str:
        """Format search space for LLM understanding."""
        lines: list[str] = []
        for param, spec in self.space.items():
            if isinstance(spec, Float):
                desc = f"- {param}: float in [{spec.min}, {spec.max}]"
                if spec.log:
                    desc += " (log scale)"
            elif isinstance(spec, Int):
                desc = f"- {param}: integer in [{spec.min}, {spec.max}]"
                if spec.log:
                    desc += " (log scale)"
            elif isinstance(spec, Choice):
                desc = f"- {param}: choice from {spec.options}"
            else:
                desc = f"- {param}: {type(spec).__name__}"
            lines.append(desc)
        return "\n".join(lines)

    async def _query_llm_for_branches(self, context: str) -> dict[str, Any]:
        """Query LLM for branching decisions."""
        try:
            response_text = await asyncio.to_thread(self.llm, context)

            # Try to extract JSON
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            json_str = json_match.group(1) if json_match else response_text

            result = json.loads(json_str)
            return result

        except Exception as e:
            logger.warning(f"Failed to parse LLM branching response: {e}")
            # Re-raise to trigger fallback in _get_branching_decisions
            raise

    def _parse_branching_response(
        self, response: dict[str, Any], max_branches: int
    ) -> list[BranchingDecision]:
        """Parse LLM response into branching decisions."""
        decisions: list[BranchingDecision] = []

        # Store overall strategy
        self.last_rationale = response.get("overall_strategy", "No strategy provided")

        for decision_data in response.get("decisions", []):
            parent_id = decision_data.get("parent_trial_id")

            # Validate parent exists
            if not parent_id or parent_id not in self.trial_tree:
                # Try to find a good parent from frontier
                if self.frontier:
                    latest_depth = max(self.frontier.keys())
                    if latest_depth < self.max_depth and self.frontier[latest_depth]:
                        parent_id = self.frontier[latest_depth][0]
                    else:
                        continue
                else:
                    continue

            # Validate and adjust parameter variations
            variations: list[dict[str, Any]] = []
            for var in decision_data.get("variations", [])[: self.branch_factor]:
                validated = self._validate_params(var)
                if validated:
                    variations.append(validated)

            if not variations:
                # Generate variations if LLM didn't provide valid ones
                variations = self._generate_variations(
                    self.trial_tree[parent_id].params,
                    decision_data.get("parameter_focus", []),
                )

            decision = BranchingDecision(
                parent_trial_id=parent_id,
                branch_count=len(variations),
                parameter_focus=decision_data.get(
                    "parameter_focus", list(self.space.keys())
                ),
                variations=variations,
                rationale=decision_data.get("rationale", "No rationale provided"),
                confidence=decision_data.get("confidence", "medium"),  # type: ignore[arg-type]
            )
            decisions.append(decision)

            if sum(d.branch_count for d in decisions) >= max_branches:
                break

        return decisions

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and adjust parameters to match search space."""
        validated: dict[str, Any] = {}

        for param, spec in self.space.items():
            if param not in params:
                # Sample missing parameter
                validated[param] = sample_space({param: spec})[param]
                continue

            value = params[param]

            if isinstance(spec, Float):
                try:
                    value = float(value)
                    value = spec.clip(value)
                    validated[param] = value
                except (TypeError, ValueError):
                    validated[param] = sample_space({param: spec})[param]

            elif isinstance(spec, Int):
                try:
                    value = int(round(float(value)))
                    value = spec.clip(value)
                    validated[param] = value
                except (TypeError, ValueError):
                    validated[param] = sample_space({param: spec})[param]

            elif isinstance(spec, Choice):
                if value in spec.options:
                    validated[param] = value
                else:
                    validated[param] = sample_space({param: spec})[param]

            else:
                validated[param] = (
                    value if value is not None else sample_space({param: spec})[param]
                )

        return validated

    def _generate_variations(
        self, parent_params: dict[str, Any], focus_params: list[str]
    ) -> list[dict[str, Any]]:
        """Generate parameter variations from a parent configuration."""
        variations: list[dict[str, Any]] = []

        # If no focus specified, pick random parameters
        if not focus_params:
            focus_params = list(self.space.keys())[:2]  # Focus on 2 params

        for _ in range(self.branch_factor):
            new_params = dict(parent_params)

            # Vary the focused parameters
            for param in focus_params:
                if param in self.space:
                    spec = self.space[param]

                    if isinstance(spec, Float):
                        # Perturb by ±20% of range
                        perturbation = (
                            sample_space({param: spec})[param]
                            - parent_params.get(param, 0)
                        ) * 0.4
                        current = parent_params.get(
                            param, spec.min if spec.min is not None else 0.0
                        )
                        new_val = current + perturbation
                        new_params[param] = spec.clip(new_val)

                    elif isinstance(spec, Int):
                        # Step by ±1 or ±2
                        import random

                        step = random.choice([-2, -1, 1, 2])
                        current = parent_params.get(
                            param, spec.min if spec.min is not None else 0
                        )
                        new_val = current + step
                        new_params[param] = spec.clip(new_val)

                    else:
                        # Resample for Choice and others
                        new_params[param] = sample_space({param: spec})[param]

            variations.append(new_params)

        return variations

    def _fallback_branching(self, max_branches: int) -> list[BranchingDecision]:
        """Fallback branching strategy when LLM fails."""
        decisions: list[BranchingDecision] = []

        if self.frontier:
            # Branch from best trial in latest depth
            latest_depth = max(self.frontier.keys())
            if latest_depth < self.max_depth and self.frontier[latest_depth]:
                parent_id = self.frontier[latest_depth][0]
                parent_params = self.trial_tree[parent_id].params

                variations = self._generate_variations(parent_params, [])

                decision = BranchingDecision(
                    parent_trial_id=parent_id,
                    branch_count=min(len(variations), max_branches),
                    parameter_focus=list(self.space.keys()),
                    variations=variations[:max_branches],
                    rationale="Fallback: branching from best frontier trial",
                    confidence="low",
                )
                decisions.append(decision)

        return decisions

    def _get_pruning_actions(self, state: ReadableState) -> list[Action]:
        """Get pruning actions for underperforming trials."""
        actions: list[Action] = []

        # Only prune if we have established frontiers
        if len(self.frontier) < 2:
            return actions

        # Find running trials not in frontier
        running_trials = state.running_trials(self.experiment_id)
        frontier_ids: set[str] = set()
        for trial_ids in self.frontier.values():
            frontier_ids.update(trial_ids)

        # Identify candidates for pruning
        for trial in running_trials:
            if trial.trial_id not in frontier_ids and trial.trial_id in self.trial_tree:
                node = self.trial_tree[trial.trial_id]

                # Don't prune very recent trials or shallow ones
                if node.depth <= 1:
                    continue

                # Check if this trial's parent is also not in frontier (weak lineage)
                if node.parent_id and node.parent_id not in frontier_ids:
                    actions.append(
                        KillTrial(
                            trial_id=trial.trial_id,
                            reason=f"Pruning weak branch at depth {node.depth}",
                        )
                    )

        return actions

    def _print_decision_summary(self, decisions: list[BranchingDecision]) -> None:
        """Print a summary of branching decisions."""
        print("\n" + "=" * 80)
        print("🌳 \033[1;36mLLM Branching Agent Decision\033[0m")
        print("-" * 80)

        if self.last_rationale:
            print(f"📝 \033[1;33mStrategy:\033[0m {self.last_rationale}")

        if decisions:
            print("\n🔀 \033[1;32mBranching Decisions:\033[0m")
            for i, decision in enumerate(decisions, 1):
                print(f"\n  {i}. Branch from: {decision.parent_trial_id}")
                print(f"     📊 Confidence: {decision.confidence}")
                print(f"     🎯 Focus: {', '.join(decision.parameter_focus)}")
                print(f"     💡 Rationale: {decision.rationale}")
                print(f"     🌱 Creating {decision.branch_count} branches")
        else:
            print("\n  (No branching decisions)")

        # Show tree statistics
        completed = sum(
            1 for node in self.trial_tree.values() if node.status == "COMPLETED"
        )
        print("\n📈 \033[1;35mTree Statistics:\033[0m")
        print(f"   Total trials: {len(self.trial_tree)}")
        print(f"   Completed: {completed}")
        print(
            f"   Max depth reached: {max(self.trials_by_depth.keys()) if self.trials_by_depth else 0}"
        )
        print("=" * 80 + "\n")

    async def rationale(self) -> str | None:
        """Return the rationale for recent decisions."""
        recent_branches = len(self.branching_history)
        total_trials = len(self.trial_tree)

        return (
            f"LLM Branching Agent: {self.last_rationale} "
            f"[Decisions: {self.decisions_made}, Branches: {recent_branches}, "
            f"Trials: {total_trials}]"
        )
