"""Attention heatmap visualization (overlay on image, weight plots)."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    title: str = "Attention",
    save_path: Optional[str] = None,
) -> None:
    """
    attention_weights: (num_patches_h, num_patches_w) or (num_patches,) for 1d.
    """
    if attention_weights.ndim == 1:
        n = int(np.sqrt(len(attention_weights)))
        if n * n != len(attention_weights):
            n = int(np.ceil(np.sqrt(len(attention_weights))))
            pad = n * n - len(attention_weights)
            attention_weights = np.pad(attention_weights, (0, pad), constant_values=0)
        attention_weights = attention_weights[: n * n].reshape(n, n)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def overlay_attention_on_image(
    image: np.ndarray,
    attention_weights: np.ndarray,
    patch_size: int = 16,
    title: str = "Attention overlay",
    save_path: Optional[str] = None,
    alpha: float = 0.5,
) -> None:
    """
    Overlay attention map on image. image: (H, W, 3) in [0,1] or uint8.
    attention_weights: (num_patches,) or (num_patches_h, num_patches_w).
    """
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    if attention_weights.ndim == 1:
        n = int(np.sqrt(len(attention_weights)))
        if n * n != len(attention_weights):
            n = int(np.ceil(np.sqrt(len(attention_weights))))
            pad = n * n - len(attention_weights)
            attention_weights = np.pad(attention_weights.astype(np.float32), (0, pad), constant_values=0)
        attention_weights = attention_weights[: n * n].reshape(n, n)
    # Resize attention to image size (repeat then crop/slice to match)
    h, w = image.shape[:2]
    ah, aw = attention_weights.shape
    # Simple repeat to get at least h x w
    repeat_h = max(1, (h + ah - 1) // ah)
    repeat_w = max(1, (w + aw - 1) // aw)
    attn_resized = np.repeat(np.repeat(attention_weights, repeat_h, axis=0), repeat_w, axis=1)
    attn_resized = attn_resized[:h, :w].astype(np.float32)
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(image)
    ax.imshow(attn_resized, cmap="jet", alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def extract_attention_weights_from_model(
    model: torch.nn.Module,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    block: str = "image_to_text",
) -> Optional[torch.Tensor]:
    """
    Hook or forward pass to get attention weights from a cross-attention block.
    Returns (B, num_heads, seq_q, seq_kv) or None if model doesn't expose weights.
    """
    # Optional: implement hook-based extraction for specific models
    return None
