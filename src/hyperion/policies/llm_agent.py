"""LLM-based agent policy for intelligent hyperparameter optimization."""

import asyncio
import json
import logging
import re
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from hyperion.core.events import Event, EventType
from hyperion.core.state import ReadableState, TrialView
from hyperion.framework.policy import Action, Policy, StartTrial
from hyperion.framework.search_space import Bool, Choice, Float, Int, sample_space

logger = logging.getLogger(__name__)


@dataclass
class TrialHistory:
    """Compact representation of a trial's history."""

    trial_id: str
    params: dict[str, Any]
    score: float | None
    metrics: dict[str, float]
    status: str
    parent_id: str | None = None


class LLMAgentPolicy(Policy):
    """LLM-powered optimization agent.

    This policy uses a Large Language Model to:
    - Analyze trial results and identify patterns
    - Make intelligent parameter suggestions
    - Provide detailed rationale for decisions
    - Balance exploration and exploitation
    """

    def __init__(
        self,
        *,
        space: dict[str, Any],
        experiment_id: str | None = None,
        llm: Callable[[str], str] | None = None,
        metric: str = "score",
        mode: Literal["min", "max"] = "max",
        max_history: int = 20,
        exploration_rate: float = 0.3,  # TODO: IMPLEMENT
        **kwargs: Any,
    ):
        """Initialize the LLM agent policy.

        Args:
            space: Search space specification (optional - LLM can infer parameters)
            experiment_id: ID of experiment to run trials for
            llm: Callable that takes a JSON prompt string and returns raw text (ideally JSON)
            metric: Metric to optimize
            mode: Optimization mode - "min" or "max"
            max_history: Maximum number of trials to keep in context
            exploration_rate: Probability of exploration vs exploitation
        """
        self.space = space
        self.experiment_id = experiment_id
        # Provider-agnostic LLM callable: (prompt_str) -> raw_text
        if llm is None:
            raise ValueError(
                "LLMAgentPolicy requires an 'llm' callable (prompt: str) -> str."
            )
        self.llm: Callable[[str], str] = llm
        self.metric = metric
        self.mode = mode
        self.max_history = max_history
        self.exploration_rate = exploration_rate

        # Track trial history
        self.trial_history: deque[TrialHistory] = deque(maxlen=max_history)
        self.decisions_made = 0
        self.last_rationale = "Initial state - no decisions made yet"

        # Cache for recent events
        self.recent_events: deque[Event] = deque(maxlen=50)

    async def on_events(self, events: Iterable[Event]) -> None:
        """Process incoming events to update internal state."""
        for event in events:
            self.recent_events.append(event)

            # Track trial completions with results
            if event.type == EventType.TRIAL_COMPLETED:
                trial_id = event.data.get("trial_id")
                score = event.data.get("score")
                event.data.get("metrics", {})

                # Find the trial in our tracking
                # Note: In a real implementation, we'd store params when we start trials
                logger.debug(f"Trial {trial_id} completed with score {score}")

    async def decide(self, state: ReadableState) -> list[Action]:
        """Make optimization decisions using LLM guidance.

        Args:
            state: Current system state

        Returns:
            List of actions to take
        """
        if not self.experiment_id:
            return []

        free_slots = state.capacity_free()
        if free_slots <= 0:
            return []

        actions: list[Action] = []

        # Gather current trial information
        all_trials = self._gather_trial_info(state)

        # Build context for LLM
        context = self._build_llm_context(all_trials, free_slots)

        try:
            # Query LLM for suggestions
            suggestions = await self._query_llm(context)

            # Parse suggestions into actions
            actions = self._parse_llm_suggestions(suggestions, free_slots)

            self.decisions_made += 1

        except Exception as e:
            logger.error(f"LLM decision failed: {e}, falling back to random sampling")
            # Fallback to random sampling
            for _ in range(min(free_slots, 1)):
                params = sample_space(self.space)
                actions.append(
                    StartTrial(
                        experiment_id=self.experiment_id,
                        params=params,
                        tags={"strategy": "llm_agent", "fallback": "random"},
                    )
                )
            self.last_rationale = f"Fallback to random sampling due to error: {e}"

        return actions

    def _gather_trial_info(self, state: ReadableState) -> list[TrialView]:
        """Gather information about all trials in the experiment."""
        if not self.experiment_id:
            return []

        # Get all trials for this experiment
        # Note: In the actual implementation, we'd need a method to get all trials
        # For now, we'll use best_trials as a proxy
        try:
            best_trials = state.best_trials(
                self.experiment_id,
                top_n=self.max_history,
                key=self.metric,
                mode=self.mode,  # type: ignore[arg-type]
            )
            return best_trials
        except Exception:
            return []

    def _build_llm_context(self, trials: list[TrialView], free_slots: int) -> str:
        """Build context string for the LLM."""
        # Format search space description
        space_desc = self._format_search_space()

        # Format trial history
        history_desc = self._format_trial_history(trials)

        # Build the prompt
        prompt = f"""You are an expert hyperparameter optimization agent. Your task is to suggest the next hyperparameters to try based on the search space and trial history.

OPTIMIZATION GOAL: {"Maximize" if self.mode == "max" else "Minimize"} the metric '{self.metric}'

SEARCH SPACE:
{space_desc}

TRIAL HISTORY (showing best trials):
{history_desc}

AVAILABLE CAPACITY: {free_slots} concurrent trials can be started

Based on the trial history and search space, suggest the next parameters to try. Consider:
1. What patterns do you see in successful trials?
2. Which parameter regions are underexplored?
3. Should we exploit promising areas or explore new regions?

Respond with a JSON object containing:
- "rationale": Your reasoning for the suggestions (2-3 sentences)
- "suggestions": Array of parameter sets to try (up to {min(free_slots, 3)} suggestions)
- "confidence": Your confidence level (low/medium/high)

Example response format:
{{
    "rationale": "The best trials have high learning rates around 0.01. We should explore nearby values while also testing different batch sizes.",
    "suggestions": [
        {{"lr": 0.008, "batch_size": 64}},
        {{"lr": 0.012, "batch_size": 32}}
    ],
    "confidence": "medium"
}}
"""  # noqa: E501
        return prompt

    def _format_search_space(self) -> str:
        """Format the search space for LLM understanding."""
        lines: list[str] = []
        for param, spec in self.space.items():
            if isinstance(spec, Float):
                desc = f"- {param}: float"
                if spec.min is not None and spec.max is not None:
                    desc += f" in [{spec.min}, {spec.max}]"
                elif spec.min is not None:
                    desc += f" >= {spec.min}"
                elif spec.max is not None:
                    desc += f" <= {spec.max}"
                if spec.log:
                    desc += " (log scale)"
            elif isinstance(spec, Int):
                desc = f"- {param}: integer"
                if spec.min is not None and spec.max is not None:
                    desc += f" in [{spec.min}, {spec.max}]"
                elif spec.min is not None:
                    desc += f" >= {spec.min}"
                elif spec.max is not None:
                    desc += f" <= {spec.max}"
                if spec.log:
                    desc += " (log scale)"
            elif isinstance(spec, Choice):
                desc = f"- {param}: choice from {spec.options}"
            elif isinstance(spec, Bool):
                desc = f"- {param}: boolean (true/false)"
            else:
                desc = f"- {param}: {type(spec).__name__}"
            lines.append(desc)
        return "\n".join(lines) if lines else "Empty search space"

    def _format_trial_history(self, trials: list[TrialView]) -> str:
        """Format trial history for LLM understanding."""
        if not trials:
            return "No trials completed yet."

        lines: list[str] = []
        for i, trial in enumerate(trials[:10], 1):  # Show top 10
            score_str = f"{trial.score:.4f}" if trial.score is not None else "N/A"
            param_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in trial.params.items()
            )
            lines.append(
                f"{i}. Score: {score_str} | Params: {param_str} | Status: {trial.status}"
            )

        return "\n".join(lines)

    async def _query_llm(self, context: str) -> dict[str, Any]:
        """Query the LLM for suggestions."""
        try:
            # Call provider-agnostic llm; run in thread to avoid blocking
            response_text = await asyncio.to_thread(self.llm, context)

            # Try to extract JSON from the response
            # Look for JSON between ``` markers or parse directly
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            json_str = json_match.group(1) if json_match else response_text

            result = json.loads(json_str)
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Return a default structure
            return {
                "rationale": "Failed to parse LLM response, using random exploration",
                "suggestions": [],
                "confidence": "low",
            }
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise

    def _parse_llm_suggestions(
        self, llm_response: dict[str, Any], max_actions: int
    ) -> list[Action]:
        """Parse LLM suggestions into concrete actions."""
        actions: list[Action] = []

        # Store the rationale
        self.last_rationale = llm_response.get("rationale", "No rationale provided")
        logger.debug(f"Rationale: {self.last_rationale}")

        # Pretty print for terminal
        print("\n" + "=" * 80)
        print("🤖 \033[1;36mLLM Agent Decision\033[0m")
        print("-" * 80)
        print(f"📝 \033[1;33mRationale:\033[0m {self.last_rationale}")

        suggestions = llm_response.get("suggestions", [])
        logger.debug(f"Suggestions: {suggestions}")

        print("\n💡 \033[1;32mSuggestions:\033[0m")
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
        else:
            print("   (No suggestions provided)")

        confidence = llm_response.get("confidence", "unknown")
        print(f"\n📊 \033[1;35mConfidence:\033[0m {confidence}")
        print("=" * 80 + "\n")
        if not suggestions:
            # If LLM didn't provide suggestions, fall back to random
            logger.warning("LLM provided no suggestions, using random sampling")
            params = sample_space(self.space)
            actions.append(
                StartTrial(
                    experiment_id=self.experiment_id,  # type: ignore[arg-type]
                    params=params,
                    tags={
                        "strategy": "llm_agent",
                        "confidence": llm_response.get("confidence", "low"),
                        "fallback": "no_suggestions",
                    },
                )
            )
            return actions

        # Convert suggestions to actions
        for i, suggestion in enumerate(suggestions[:max_actions]):
            if not isinstance(suggestion, dict):
                continue

            # Validate and adjust parameters to match search space
            params = self._validate_params(suggestion)  # type: ignore[arg-type]

            if self.experiment_id is not None:
                actions.append(
                    StartTrial(
                        experiment_id=self.experiment_id,
                        params=params,
                        tags={
                            "strategy": "llm_agent",
                            "confidence": llm_response.get("confidence", "medium"),
                            "suggestion_index": i,
                        },
                    )
                )

        # If no valid suggestions, add a random sample
        if not actions:
            params = sample_space(self.space)
            actions.append(
                StartTrial(
                    experiment_id=self.experiment_id,  # type: ignore[arg-type]
                    params=params,
                    tags={"strategy": "llm_agent", "fallback": "invalid_suggestions"},
                )
            )

        return actions

    def _validate_params(self, suggestion: dict[str, Any]) -> dict[str, Any]:
        """Validate and adjust suggested parameters to match search space constraints."""
        validated = {}

        for param, spec in self.space.items():
            if param not in suggestion:
                # If parameter missing, sample it
                validated[param] = sample_space(spec)
                continue

            value = suggestion[param]

            # Validate based on spec type
            if isinstance(spec, Float):
                try:
                    value = float(value)
                    validated[param] = spec.clip(value)  # Use the new clip method
                except (TypeError, ValueError):
                    validated[param] = sample_space(spec)

            elif isinstance(spec, Int):
                try:
                    value = int(round(float(value)))
                    validated[param] = spec.clip(value)  # Use the new clip method
                except (TypeError, ValueError):
                    validated[param] = sample_space(spec)

            elif isinstance(spec, Choice):
                if spec.validate(value):
                    validated[param] = value
                else:
                    # Try to find closest match or sample
                    validated[param] = sample_space(spec)

            elif isinstance(spec, Bool):
                # Convert various representations to boolean
                if isinstance(value, bool):
                    validated[param] = value
                elif isinstance(value, str):
                    validated[param] = value.lower() in ("true", "yes", "1")
                elif isinstance(value, int | float):
                    validated[param] = bool(value)
                else:
                    validated[param] = sample_space(spec)

            else:
                # For other types, use the suggested value or sample
                validated[param] = value if value is not None else sample_space(spec)

        return validated  # type: ignore[return]

    async def rationale(self) -> str | None:
        """Return the rationale for recent decisions."""
        return (
            f"LLM Agent: {self.last_rationale} [Decisions made: {self.decisions_made}]"
        )
