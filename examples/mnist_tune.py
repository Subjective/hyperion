"""Basic MNIST tuning example using the high-level tune() API.

This example tunes a tiny MLP on a small MNIST subset to keep runtime short.
"""

from __future__ import annotations

from typing import Any

import ollama
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from hyperion import Choice, Float, Int, ObjectiveResult, TrialContext, tune


class MLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def get_data() -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    # Use small subsets for speed
    train_subset = Subset(train_ds, range(0, 2_000))
    test_subset = Subset(test_ds, range(0, 1_000))

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return float(correct) / float(total)


def make_optimizer(name: str, params: Any, lr: float) -> optim.Optimizer:
    if name.lower() == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    if name.lower() == "adam":
        return optim.Adam(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def objective(
    ctx: TrialContext, *, lr: float, hidden: int, optimizer: str, epochs: int = 2
) -> ObjectiveResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data()

    model = MLP(hidden).to(device)
    opt = make_optimizer(optimizer, model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    model.train()
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            if (step + 1) % 50 == 0:
                val_acc = evaluate(model, test_loader, device)
                best_acc = max(best_acc, val_acc)
                ctx.report(
                    step=epoch * 10_000 + step,
                    loss=float(loss.item()),
                    val_acc=float(val_acc),
                )
                if ctx.should_stop():
                    break
        if ctx.should_stop():
            break

    # Final evaluation
    final_acc = evaluate(model, test_loader, device)
    best_acc = max(best_acc, final_acc)
    return ObjectiveResult(score=best_acc, metrics={"final_acc": best_acc})


def main() -> None:
    space = {
        "lr": Float(1e-4, 1e-1, log=True),
        "hidden": Int(64, 128),
        "optimizer": Choice(["sgd", "adam"]),
    }

    # Random Search - Classic baseline
    random_result = tune(
        objective=objective,
        space=space,
        strategy="random",
        metric="score",
        mode="max",
        max_trials=5,
        max_concurrent=1,
        show_progress=True,
        show_summary=True,
        storage="sqlite:///hyperion.db",
    )

    print("=== Random Search Tuning result ===")
    print(random_result)

    # Beam Search - Breadth-first tree exploration
    beam_result = tune(
        objective=objective,
        space=space,
        strategy="beam_search",
        strategy_kwargs={
            "K": 2,  # Keep top 2 trials per depth
            "width": 2,  # Generate 2 children per parent
            "max_depth": 3,  # Maximum depth of search tree
            "prune": False,  # Don't kill underperforming trials
        },
        metric="score",
        mode="max",
        max_trials=15,
        max_concurrent=2,
        show_progress=True,
        show_summary=True,
        storage="sqlite:///hyperion.db",
    )

    print("=== Beam Search Tuning result ===")
    print(beam_result)

    # LLM Agent - Intelligent sequential exploration

    # Here we provide an example provider-agnostic LLM callable using Ollama.
    # For other providers (e.g., OpenAI, Vertex AI, Bedrock), implement a similar
    # wrapper that accepts a JSON prompt string and returns RAW JSON text. Make sure
    # to remove any surrounding prose or code fences so the policy can parse it.
    def ollama_llm(prompt: str) -> str:
        resp = ollama.chat(
            model="gemma3:latest",  # any local Ollama model
            messages=[
                {"role": "system", "content": "Return RAW JSON only."},
                {"role": "user", "content": prompt},
            ],
            options={},
        )
        return resp.get("message", {}).get("content", "{}")

    agent_result = tune(
        objective,
        space,
        strategy="llm_agent",
        strategy_kwargs={
            "llm": ollama_llm,  # Provide any LLM via a simple callable (prompt: str) -> str
            "max_history": 10,  # Context window
        },
        metric="score",
        mode="max",
        max_trials=5,
        max_concurrent=1,
        show_progress=True,
        show_summary=True,
        storage="sqlite:///hyperion.db",
    )

    print("=== LLM Agent Tuning result ===")
    print(agent_result)

    # LLM Branching Agent - Intelligent tree-based exploration
    branching_agent_result = tune(
        objective=objective,
        space=space,
        strategy="llm_branching_agent",
        strategy_kwargs={
            "llm": ollama_llm,  # Provide any LLM via a simple callable (prompt: str) -> str
            "max_depth": 3,  # Maximum tree depth
            "beam_width": 2,  # Keep top 2 trials per depth
            "branch_factor": 2,  # Create up to 2 branches per parent
            "enable_pruning": True,  # Prune weak branches
        },
        metric="score",
        mode="max",
        max_trials=15,
        max_concurrent=2,
        show_progress=True,
        show_summary=True,
        storage="sqlite:///hyperion.db",
    )

    print("=== LLM Branching Agent Tuning result ===")
    print(branching_agent_result)

    # Bayesian Optimization - GP-guided intelligent search
    bayesian_result = tune(
        objective=objective,
        space=space,
        strategy="bayesian",
        strategy_kwargs={
            "n_initial": 5,  # Random exploration for first 5 trials
            "acquisition": "ei",  # Expected Improvement
            "xi": 0.01,  # Exploration parameter (higher = more exploration)
        },
        metric="score",
        mode="max",
        max_trials=15,
        max_concurrent=1,  # Bayesian optimization works best sequentially
        show_progress=True,
        show_summary=True,
        storage="sqlite:///hyperion.db",
    )

    print("=== Bayesian Optimization Tuning result ===")
    print(bayesian_result)

    # Population-Based Training - Evolutionary optimization
    pbt_result = tune(
        objective=objective,
        space=space,
        strategy="pbt",
        strategy_kwargs={
            "population_size": 4,  # 4 trials per generation
            "perturbation_factor": 1.2,  # Perturbation strength for exploration
        },
        metric="score",
        mode="max",
        max_trials=16,  # Will evolve through multiple generations
        max_concurrent=4,  # PBT needs parallel trials for population
        show_progress=True,
        show_summary=True,
        storage="sqlite:///hyperion.db",
    )

    print("=== Population-Based Training result ===")
    print(pbt_result)


if __name__ == "__main__":
    main()
