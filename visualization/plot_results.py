"""Comparison plots and accuracy curves."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_comparison_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ("top1_accuracy", "top5_accuracy"),
    save_path: Optional[str] = None,
) -> None:
    """
    results: { "Symmetric": {"top1_accuracy": 0.5, ...}, "Asymmetric": {...} }
    """
    methods = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        vals = [results[method].get(m, 0) for m in metrics]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=method)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_title("Symmetric vs Asymmetric Cross-Modal Attention")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_curves(
    train_losses: List[float],
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "b-", label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="b")
    if val_accs:
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_accs, "g-", label="Val accuracy")
        ax2.set_ylabel("Accuracy", color="g")
        ax2.legend(loc="upper right")
    ax1.legend(loc="upper left")
    ax1.set_title("Training curve")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
