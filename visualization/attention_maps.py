"""Generate attention heatmaps overlaid on images (Grad-CAM style).

Key functions:
    * :func:`get_attention_weights` — run a single sample and return weights.
    * :func:`plot_image_attention` — overlay text→image attention on the input.
    * :func:`plot_text_attention`  — bar chart of image→text attention per token.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from PIL import Image
from torchvision import transforms
from transformers import RobertaTokenizer

# ImageNet de-normalisation for display
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised ``(3, H, W)`` tensor back to a ``(H, W, 3)``
    uint8 NumPy array for display."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * _STD + _MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


@torch.no_grad()
def get_attention_weights(
    model: nn.Module,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Run a **single** sample through the model and return attention maps.

    Args:
        image:          ``(3, 224, 224)`` — single normalised image.
        input_ids:      ``(seq_len,)``
        attention_mask: ``(seq_len,)``

    Returns:
        dict with ``"img_to_txt"`` ``(1, N_img, N_txt)`` and
        ``"txt_to_img"`` ``(1, N_txt, N_img)``.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    img = image.unsqueeze(0).to(device)
    ids = input_ids.unsqueeze(0).to(device)
    mask = attention_mask.unsqueeze(0).to(device)

    _, attn = model(img, ids, mask)
    return {k: v.cpu() for k, v in attn.items()}


def plot_image_attention(
    image: torch.Tensor,
    attn_txt_to_img: torch.Tensor,
    question: str,
    tokens: list[str] | None = None,
    top_tokens: int = 4,
) -> Figure:
    """Overlay text→image attention on the original image.

    Shows one heatmap per top-attending text token, plus a combined map.

    Args:
        image:           ``(3, 224, 224)`` normalised tensor.
        attn_txt_to_img: ``(1, N_txt, N_img)`` attention weights.
        question:        The question string (for the title).
        tokens:          Decoded token strings; auto-decoded if ``None``.
        top_tokens:      Number of individual token maps to show.
    """
    img_np = denormalize(image)
    attn = attn_txt_to_img.squeeze(0).numpy()  # (N_txt, N_img)

    # ViT-B/16 has 196 patches (14×14) + 1 CLS
    n_patches = attn.shape[1]
    grid_size = int(np.sqrt(n_patches - 1))  # 14 for ViT-B/16
    # drop CLS column
    attn_spatial = attn[:, 1:]  # (N_txt, 196)

    # Combined attention: average over all text tokens
    combined = attn_spatial.mean(axis=0).reshape(grid_size, grid_size)
    combined_resized = np.array(
        Image.fromarray(combined).resize((224, 224), Image.BILINEAR)
    )

    # Identify top-attending tokens (by total attention mass)
    token_importance = attn_spatial.sum(axis=1)  # (N_txt,)
    top_idx = token_importance.argsort()[-top_tokens:][::-1]

    n_cols = min(top_tokens, len(top_idx)) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # Combined map
    axes[0].imshow(img_np)
    axes[0].imshow(combined_resized, alpha=0.5, cmap="jet")
    axes[0].set_title("Combined")
    axes[0].axis("off")

    # Per-token maps
    for i, idx in enumerate(top_idx):
        if i + 1 >= len(axes):
            break
        token_attn = attn_spatial[idx].reshape(grid_size, grid_size)
        token_resized = np.array(
            Image.fromarray(token_attn).resize((224, 224), Image.BILINEAR)
        )
        label = tokens[idx] if tokens else f"token {idx}"
        axes[i + 1].imshow(img_np)
        axes[i + 1].imshow(token_resized, alpha=0.5, cmap="jet")
        axes[i + 1].set_title(f'"{label}"')
        axes[i + 1].axis("off")

    fig.suptitle(question, fontsize=12)
    fig.tight_layout()
    return fig


def plot_text_attention(
    attn_img_to_txt: torch.Tensor,
    tokens: list[str],
    question: str,
) -> Figure:
    """Horizontal bar chart of image→text attention per token.

    Args:
        attn_img_to_txt: ``(1, N_img, N_txt)``
        tokens:          Decoded token strings.
        question:        Question string for the title.
    """
    attn = attn_img_to_txt.squeeze(0).numpy()  # (N_img, N_txt)
    # Average attention over all image patches
    token_weights = attn.mean(axis=0)  # (N_txt,)

    fig, ax = plt.subplots(figsize=(6, max(3, len(tokens) * 0.35)))
    y_pos = np.arange(len(tokens))
    ax.barh(y_pos, token_weights, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean attention weight")
    ax.set_title(f"Image → Text attention\n{question}")
    fig.tight_layout()
    return fig


def decode_tokens(input_ids: torch.Tensor) -> list[str]:
    """Decode token ids to readable strings using the RoBERTa tokenizer."""
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return [tokenizer.decode(tid) for tid in input_ids.tolist()]
