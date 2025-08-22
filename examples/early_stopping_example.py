"""Example demonstrating early stopping in Hyperion.

This example shows how to use early stopping to terminate underperforming
trials early, saving computational resources for more promising candidates.
"""

import random
import time
from typing import Any

from hyperion import Float, ObjectiveResult, tune
from hyperion.core.context import TrialContext


def train_model_with_early_stopping(
    ctx: TrialContext, **params: Any
) -> ObjectiveResult:
    """Simulated model training with progress reporting.

    This objective simulates a training loop that:
    1. Reports progress at each epoch
    2. Can be stopped early if underperforming
    3. Returns final validation score
    """
    lr = params["lr"]
    momentum = params["momentum"]

    # Simulate that some hyperparameters are inherently better
    # (In reality, this would be determined by actual training)
    base_performance = (1.0 - abs(lr - 0.01)) * (1.0 - abs(momentum - 0.9))

    # Add some randomness to simulate training variance
    noise_factor = random.uniform(0.8, 1.2)
    base_performance *= noise_factor

    print(
        f"[Trial {ctx.trial_id[:8]}] Starting with lr={lr:.4f}, momentum={momentum:.2f}"
    )

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(20):
        # Simulate training for this epoch
        time.sleep(0.05)  # Simulate computation time

        # Calculate loss (decreasing over time for good configs)
        train_loss = (1.0 / base_performance) * (0.9**epoch) + random.uniform(0, 0.1)
        val_loss = train_loss * 1.1  # Validation loss slightly higher

        # Update best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Report progress to Hyperion
        ctx.report(
            step=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
        )

        # Check if we should stop early (trial killed by early stopping policy)
        if ctx.should_stop():
            print(f"[Trial {ctx.trial_id[:8]}] Stopped early at epoch {epoch}")
            # Return current performance - MUST include the metric being optimized
            return ObjectiveResult(
                score=1.0 / best_val_loss,  # Higher is better
                metrics={
                    "val_loss": best_val_loss,  # Critical: include the metric for best_of()
                    "stopped_epoch": float(epoch),
                },
            )

    # Completed all epochs
    print(f"[Trial {ctx.trial_id[:8]}] Completed all 20 epochs")
    final_score = 1.0 / best_val_loss

    # MUST include the metric being optimized in the final result
    return ObjectiveResult(
        score=final_score,
        metrics={
            "val_loss": best_val_loss,  # Critical: include the metric for best_of()
            "stopped_epoch": 20.0,
            "completed": 1.0,
        },
    )


def main():
    """Run hyperparameter optimization with different early stopping strategies."""

    print("=" * 60)
    print("Running hyperparameter optimization with early stopping")
    print("=" * 60)

    # Define search space
    space = {"lr": Float(0.0001, 0.1, log=True), "momentum": Float(0.5, 0.99)}

    # Example 1: Median early stopping
    print("\n1. Testing MEDIAN early stopping (balanced approach)")
    print("-" * 40)

    result = tune(
        objective=train_model_with_early_stopping,
        space=space,
        strategy="random",
        early_stopping="median",
        early_stopping_kwargs={
            "check_interval": 5,  # Check every 5 progress reports
            "min_trials": 3,  # Need at least 3 trials running to stop any
        },
        metric="val_loss",  # Metric to monitor for early stopping
        mode="min",  # Minimize validation loss
        max_trials=10,
        max_concurrent=4,
        show_progress=True,
    )

    print("\nBest result with median early stopping:")
    print(f"  Score: {result['best']['score']:.4f}")
    print(
        f"  Params: lr={result['best']['params']['lr']:.4f}, "
        f"momentum={result['best']['params']['momentum']:.2f}"
    )

    # Example 2: Aggressive early stopping
    print("\n2. Testing AGGRESSIVE early stopping (faster pruning)")
    print("-" * 40)

    result = tune(
        objective=train_model_with_early_stopping,
        space=space,
        strategy="random",
        early_stopping="aggressive",  # Checks more frequently (every 25 steps by default)
        early_stopping_kwargs={
            "min_trials": 2,
        },
        metric="val_loss",
        mode="min",
        max_trials=10,
        max_concurrent=4,
    )

    print("\nBest result with aggressive early stopping:")
    print(f"  Score: {result['best']['score']:.4f}")
    print(
        f"  Params: lr={result['best']['params']['lr']:.4f}, "
        f"momentum={result['best']['params']['momentum']:.2f}"
    )

    # Example 3: No early stopping (baseline)
    print("\n3. Testing NO early stopping (baseline)")
    print("-" * 40)

    result = tune(
        objective=train_model_with_early_stopping,
        space=space,
        strategy="random",
        early_stopping=None,  # No early stopping
        metric="val_loss",  # Still specify metric for best selection
        mode="min",
        max_trials=10,
        max_concurrent=4,
    )

    print("\nBest result without early stopping:")
    print(f"  Score: {result['best']['score']:.4f}")
    print(
        f"  Params: lr={result['best']['params']['lr']:.4f}, "
        f"momentum={result['best']['params']['momentum']:.2f}"
    )

    print("\n" + "=" * 60)
    print("Early stopping example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
