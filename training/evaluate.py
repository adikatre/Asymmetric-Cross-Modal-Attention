"""Evaluation and metrics for VQA models."""

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """Top-1 accuracy on the loader."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            input_ids = batch[1]["input_ids"].to(device)
            attention_mask = batch[1]["attention_mask"].to(device)
            answers = batch[2]
            logits = model(images, input_ids, attention_mask)
            pred = logits.argmax(dim=1).cpu()
            correct += (pred == answers).sum().item()
            total += answers.size(0)
    return correct / total if total else 0.0


def top_k_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    k: int = 5,
    device: Optional[torch.device] = None,
) -> float:
    """Top-K accuracy: correct answer in top-K predictions."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            input_ids = batch[1]["input_ids"].to(device)
            attention_mask = batch[1]["attention_mask"].to(device)
            answers = batch[2]
            logits = model(images, input_ids, attention_mask)
            _, topk = logits.topk(k, dim=1)
            correct += (topk == answers.unsqueeze(1)).any(dim=1).sum().item()
            total += answers.size(0)
    return correct / total if total else 0.0


def compute_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Top-1 and Top-5 accuracy."""
    top1 = evaluate(model, loader, device)
    top5 = top_k_accuracy(model, loader, k=5, device=device)
    return {"top1_accuracy": top1, "top5_accuracy": top5}


def vqa_accuracy_per_answer(
    pred_indices: List[int],
    gt_indices: List[int],
    idx_to_answers: Optional[Dict[int, List[str]]] = None,
) -> float:
    """
    Official VQA accuracy: min(# humans who gave that answer / 3, 1).
    Simplified: when we have single ground-truth index, we use 1.0 if match else 0.0.
    For full VQA metric you need multiple human answers per question.
    """
    if len(pred_indices) != len(gt_indices):
        return 0.0
    return sum(1.0 for p, g in zip(pred_indices, gt_indices) if p == g) / len(pred_indices)
