"""Qualitative examples: show image, question, predicted answer, attention."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def run_inference_batch(
    model: torch.nn.Module,
    batch: List[Any],
    device: torch.device,
    idx_to_answer: Dict[int, str],
) -> List[Dict[str, str]]:
    """
    Run model on one batch and return list of {
        "question": str,
        "predicted": str,
        "ground_truth": str,
        "correct": bool,
    }.
    """
    images = batch[0].to(device)
    input_ids = batch[1]["input_ids"].to(device)
    attention_mask = batch[1]["attention_mask"].to(device)
    answers = batch[2]
    model.eval()
    with torch.no_grad():
        logits = model(images, input_ids, attention_mask)
    pred_indices = logits.argmax(dim=1).cpu().tolist()
    results = []
    for i in range(images.size(0)):
        pred_idx = pred_indices[i]
        gt_idx = answers[i].item()
        results.append({
            "predicted": idx_to_answer.get(pred_idx, "?"),
            "ground_truth": idx_to_answer.get(gt_idx, "?"),
            "correct": pred_idx == gt_idx,
        })
    return results
