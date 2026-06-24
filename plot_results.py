"""Generate comparison plots from training history JSON files.

Functions:
    * :func:`plot_training_curves` — loss / accuracy over epochs.
    * :func:`plot_comparison_table` — bar chart comparing model metrics.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def load_history(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def plot_training_curves(
    histories: dict[str, list[dict]],
    save_path: str | Path | None = None,
) -> Figure:
    """Plot loss and accuracy curves for one or more runs.

    Args:
        histories: ``{"Asymmetric": [...], "Symmetric": [...]}``.
        save_path: If given, save the figure as an image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for name, hist in histories.items():
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_loss"] for h in hist], label=f"{name} train")
        if "val_loss" in hist[0]:
            axes[0].plot(
                epochs, [h["val_loss"] for h in hist],
                linestyle="--", label=f"{name} val",
            )

        axes[1].plot(epochs, [h["train_acc"] for h in hist], label=f"{name} train")
        axes[1].plot(
            epochs, [h["val_top1"] for h in hist],
            linestyle="--", label=f"{name} val",
        )

        axes[2].plot(epochs, [h["val_top5"] for h in hist], label=name)

    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[0].legend()
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Top-1 Accuracy")
    axes[1].legend()
    axes[2].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Top-5 Accuracy")
    axes[2].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_comparison_bar(
    results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Grouped bar chart comparing models on selected metrics.

    Args:
        results: ``{"Asymmetric": {"val_top1": 55.2, ...}, ...}``.
        metrics: Which metric keys to plot.  Defaults to top1/top5.
        save_path: Optional file path to save the figure.
    """
    if metrics is None:
        metrics = ["val_top1", "val_top5"]

    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(6, n_metrics * 3), 5))
    for i, name in enumerate(model_names):
        values = [results[name].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name)
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Comparison")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    """Print a Markdown-formatted comparison table to stdout."""
    metrics = sorted({k for v in results.values() for k in v})
    header = "| Method | " + " | ".join(m.replace("_", " ").title() for m in metrics) + " |"
    sep = "|--------|" + "|".join("--------" for _ in metrics) + "|"
    print(header)
    print(sep)
    for name, vals in results.items():
        row = f"| {name} | " + " | ".join(f"{vals.get(m, 0):.2f}" for m in metrics) + " |"
        print(row)
