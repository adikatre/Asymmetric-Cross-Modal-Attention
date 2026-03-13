"""Show side-by-side qualitative predictions for selected examples.

Generates a figure with:
    1. The input image
    2. The question
    3. Ground-truth and predicted answers (for each model)
    4. Attention heatmaps from the asymmetric model
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from visualization.attention_maps import (
    decode_tokens,
    denormalize,
    get_attention_weights,
)


@torch.no_grad()
def qualitative_grid(
    models: dict[str, nn.Module],
    dataset,
    idx_to_answer: dict[int, str],
    sample_indices: list[int] | None = None,
    n_samples: int = 6,
    device: torch.device | None = None,
    save_path: str | None = None,
) -> Figure:
    """Create a grid of qualitative examples comparing models.

    Each row shows one sample with:
      * Column 0: input image + question
      * Column 1..N: attention heatmap + predicted answer for each model

    Args:
        models:         ``{"Asymmetric": model1, "Symmetric": model2}``.
        dataset:        A :class:`VQADataset` instance.
        idx_to_answer:  Reverse vocabulary.
        sample_indices: Specific dataset indices to show.
        n_samples:      Number of random samples if *sample_indices* is None.
        device:         Target device.
        save_path:      Optional file path to save the figure.
    """
    if device is None:
        device = next(next(iter(models.values())).parameters()).device

    if sample_indices is None:
        sample_indices = torch.randperm(len(dataset))[:n_samples].tolist()

    model_names = list(models.keys())
    n_cols = 1 + len(model_names)  # image col + one col per model
    n_rows = len(sample_indices)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(sample_indices):
        image, input_ids, attention_mask, answer_idx = dataset[idx]
        question = dataset.samples[idx]["question"]
        gt_answer = idx_to_answer[answer_idx.item()]
        tokens = decode_tokens(input_ids)

        # Column 0: original image
        img_np = denormalize(image)
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"Q: {question}\nGT: {gt_answer}", fontsize=9)
        axes[row, 0].axis("off")

        # Remaining columns: one per model
        for col, name in enumerate(model_names, start=1):
            model = models[name]
            attn = get_attention_weights(model, image, input_ids, attention_mask, device)

            # Predicted answer
            img_t = image.unsqueeze(0).to(device)
            ids_t = input_ids.unsqueeze(0).to(device)
            mask_t = attention_mask.unsqueeze(0).to(device)
            logits, _ = model(img_t, ids_t, mask_t)
            pred_idx = logits.argmax(dim=1).item()
            pred_answer = idx_to_answer.get(pred_idx, "???")

            # txt→img attention heatmap
            attn_t2i = attn["txt_to_img"].squeeze(0).numpy()  # (N_txt, N_img)
            n_patches = attn_t2i.shape[1]
            grid_size = int(np.sqrt(n_patches - 1))
            combined = attn_t2i[:, 1:].mean(axis=0).reshape(grid_size, grid_size)
            combined_resized = np.array(
                Image.fromarray(combined).resize((224, 224), Image.BILINEAR)
            )

            axes[row, col].imshow(img_np)
            axes[row, col].imshow(combined_resized, alpha=0.5, cmap="jet")
            marker = "✓" if pred_answer == gt_answer else "✗"
            axes[row, col].set_title(
                f"{name}\nPred: {pred_answer} {marker}", fontsize=9,
            )
            axes[row, col].axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
