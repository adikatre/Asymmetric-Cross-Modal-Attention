"""Evaluation utilities: metrics computation and full-test-set evaluation.

The main entry points are:

* :func:`evaluate` — run a model over a dataloader and return metric dicts.
* :func:`evaluate_from_checkpoint` — load a checkpoint and evaluate it.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from data.preprocess import build_dataloaders
from models import AsymmetricVQAModel, SymmetricVQAModel
from training.config import TrainConfig


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module | None = None,
    device: torch.device | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    """Compute Top-1, Top-5 accuracy, and (optionally) loss on *loader*.

    Returns a dict with keys: ``val_top1``, ``val_top5``, and (if
    *criterion* is given) ``val_loss``.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, input_ids, attention_mask, answers in tqdm(loader, desc="  eval", leave=False):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        answers = answers.to(device)

        with autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(images, input_ids, attention_mask)
            if criterion is not None:
                loss = criterion(logits, answers)
                total_loss += loss.item() * answers.size(0)

        # Top-1
        correct_top1 += (logits.argmax(dim=1) == answers).sum().item()

        # Top-5
        _, top5_preds = logits.topk(5, dim=1)
        correct_top5 += (top5_preds == answers.unsqueeze(1)).any(dim=1).sum().item()

        total += answers.size(0)

    metrics: dict[str, float] = {
        "val_top1": correct_top1 / total * 100,
        "val_top5": correct_top5 / total * 100,
    }
    if criterion is not None:
        metrics["val_loss"] = total_loss / total
    return metrics


@torch.no_grad()
def compute_vqa_accuracy(
    model: nn.Module,
    loader,
    device: torch.device | None = None,
    use_amp: bool = False,
) -> float:
    """Compute the official VQA accuracy metric.

    VQA accuracy for a single prediction is::

        min(count_of_predicted_answer_in_10_human_answers / 3, 1)

    Since VQADataset uses ``multiple_choice_answer`` (the most common
    answer), this simplifies to top-1 accuracy for our setup.  A more
    exact implementation would require the full 10-answer annotations.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    correct = 0
    total = 0

    for images, input_ids, attention_mask, answers in tqdm(loader, desc="  vqa-acc", leave=False):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        answers = answers.to(device)

        with autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(images, input_ids, attention_mask)

        correct += (logits.argmax(dim=1) == answers).sum().item()
        total += answers.size(0)

    return correct / total * 100


def evaluate_from_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path = "data",
    batch_size: int = 64,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Load a saved checkpoint and evaluate on the validation set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg_dict = ckpt.get("config", {})
    model_type = cfg_dict.get("model_type", "asymmetric")
    num_answers = cfg_dict.get("num_answers", 1000)
    embed_dim = cfg_dict.get("embed_dim", 512)
    num_heads = cfg_dict.get("num_heads", 8)

    cls = AsymmetricVQAModel if model_type == "asymmetric" else SymmetricVQAModel
    model = cls(
        num_answers=num_answers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        freeze_encoders=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, val_loader, _, _ = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        top_k_answers=num_answers,
        max_samples=max_samples,
    )

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, val_loader, criterion, device)
    vqa_acc = compute_vqa_accuracy(model, val_loader, device)
    metrics["vqa_accuracy"] = vqa_acc
    return metrics


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to .pt checkpoint")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Save metrics to JSON")
    args = parser.parse_args()

    metrics = evaluate_from_checkpoint(
        args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
