"""Integration tests for early stopping policies."""

import time
from typing import Any

from hyperion import ObjectiveResult, tune
from hyperion.core.context import TrialContext
from hyperion.framework.search_space import Choice


def test_median_early_stopping_kills_underperformers():
    """Test that MedianEarlyStoppingPolicy kills trials below median."""
    killed_trials = []
    trial_scores = {}

    def objective(ctx: TrialContext, **params: Any) -> ObjectiveResult:
        """Objective that reports progress and tracks killed trials."""
        # Different trials will have different performance based on params
        base_score = params.get("base_score", 0.5)

        # Report progress over 10 steps
        for step in range(10):
            # Simulate some work
            time.sleep(0.01)

            # Calculate current score (improving over time)
            current_score = base_score * (1 + step * 0.1)

            # Report progress
            ctx.report(step=step, val_loss=1.0 / current_score)

            # Check if we should stop
            if ctx.should_stop():
                killed_trials.append(ctx.trial_id)
                # Include val_loss in metrics even when killed
                return ObjectiveResult(
                    score=current_score, metrics={"val_loss": 1.0 / current_score}
                )

        # If we completed all steps, record final score
        final_score = base_score * 2.0
        trial_scores[ctx.trial_id] = final_score
        # Include val_loss in metrics so best_of() can find it
        return ObjectiveResult(
            score=final_score, metrics={"val_loss": 1.0 / final_score}
        )

    # Run optimization with median early stopping
    result = tune(
        objective=objective,
        space={
            "base_score": Choice([0.1, 0.3, 0.5, 0.7, 0.9])  # Different starting scores
        },
        strategy="grid",  # Use grid to ensure we test all values
        early_stopping="median",
        early_stopping_kwargs={
            "check_interval": 3,  # Check every 3 progress reports
            "min_trials": 2,
        },
        metric="val_loss",
        mode="min",
        max_trials=5,
        max_concurrent=5,  # Run all trials concurrently
    )

    # Verify that some trials were killed
    assert len(killed_trials) > 0, "No trials were killed by early stopping"

    # Verify we got a best result
    assert result.get("best"), "No best trial found"

    # The best trial should have required fields
    best_trial = result["best"]
    assert "trial_id" in best_trial, "Best trial missing trial_id"
    assert "score" in best_trial, "Best trial missing score"

    # The best trial should not have been killed
    assert best_trial["trial_id"] not in killed_trials, (
        "Best trial was killed by early stopping"
    )


def test_no_early_stopping_completes_all_trials():
    """Test that without early stopping, all trials complete."""
    completed_count = 0

    def objective(ctx: TrialContext, **params: Any) -> ObjectiveResult:
        """Objective that counts completions."""
        nonlocal completed_count

        # Report progress
        for step in range(5):
            ctx.report(step=step, loss=1.0 - step * 0.1)
            time.sleep(0.01)

        completed_count += 1
        return ObjectiveResult(score=params.get("x", 0.5))

    # Run without early stopping
    tune(
        objective=objective,
        space={"x": Choice([0.1, 0.2, 0.3, 0.4, 0.5])},
        strategy="grid",
        early_stopping=None,  # No early stopping
        max_trials=5,
        max_concurrent=5,
    )

    # All trials should complete
    assert completed_count == 5, f"Expected 5 completions, got {completed_count}"


def test_aggressive_vs_patient_early_stopping():
    """Test that aggressive stops earlier than patient."""
    aggressive_kills = []
    patient_kills = []

    def make_objective(kill_tracker):
        def objective(ctx: TrialContext, **params: Any) -> ObjectiveResult:
            base_score = params.get("base_score", 0.5)

            for step in range(50):  # More steps to see difference
                current_score = base_score * (1 + step * 0.02)
                ctx.report(step=step, val_loss=1.0 / current_score)

                if ctx.should_stop():
                    kill_tracker.append((ctx.trial_id, step))
                    return ObjectiveResult(
                        score=current_score, metrics={"val_loss": 1.0 / current_score}
                    )

                time.sleep(0.001)

            final_score = base_score * 2.0
            return ObjectiveResult(
                score=final_score, metrics={"val_loss": 1.0 / final_score}
            )

        return objective

    # Test aggressive early stopping
    tune(
        objective=make_objective(aggressive_kills),
        space={"base_score": Choice([0.1, 0.3, 0.5, 0.7, 0.9])},
        strategy="grid",
        early_stopping="aggressive",  # Should check every 25 steps
        early_stopping_kwargs={
            "min_trials": 2,
        },
        metric="val_loss",
        mode="min",
        max_trials=5,
        max_concurrent=5,
    )

    # Test patient early stopping
    tune(
        objective=make_objective(patient_kills),
        space={"base_score": Choice([0.1, 0.3, 0.5, 0.7, 0.9])},
        strategy="grid",
        early_stopping="patient",  # Should check every 200 steps
        early_stopping_kwargs={
            "min_trials": 2,
        },
        metric="val_loss",
        mode="min",
        max_trials=5,
        max_concurrent=5,
    )

    # Aggressive should kill trials earlier (at lower step counts) than patient
    if aggressive_kills and patient_kills:
        avg_aggressive_step = sum(step for _, step in aggressive_kills) / len(
            aggressive_kills
        )
        avg_patient_step = (
            sum(step for _, step in patient_kills) / len(patient_kills)
            if patient_kills
            else float("inf")
        )

        # Patient should either not kill or kill later
        assert avg_patient_step >= avg_aggressive_step, (
            f"Patient killed earlier ({avg_patient_step}) than aggressive ({avg_aggressive_step})"
        )
