"""Image and text encoders for VQA: ViT-B/16 and RoBERTa-base (frozen)."""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel, AutoConfig


class ImageEncoder(nn.Module):
    """
    Image encoder using Vision Transformer (ViT-B/16).
    Produces patch embeddings (batch, num_patches, hidden_size) for cross-attention.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        freeze: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        # torchvision ViT: weights=ViT_B_16_Weights.IMAGENET1K_V1
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)
        self.backbone.heads = nn.Identity()  # remove classification head
        hidden_size = 768  # ViT-B hidden size
        self.proj = nn.Linear(hidden_size, embed_dim) if hidden_size != embed_dim else nn.Identity()
        self.embed_dim = embed_dim
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> patch embeddings then encoder; drop CLS token for cross-attn
        x = self.backbone._process_input(images)  # (B, num_patches, hidden)
        x = torch.cat(
            (self.backbone.class_token.expand(x.shape[0], -1, -1), x),
            dim=1,
        )
        x = x + self.backbone.encoder_pos_embedding
        x = self.backbone.encoder(x)  # (B, 1 + num_patches, hidden)
        patch_tokens = x[:, 1:, :]  # (B, 196, 768)
        return self.proj(patch_tokens)  # (B, 196, embed_dim)


class TextEncoder(nn.Module):
    """
    Text encoder using RoBERTa-base.
    Produces token embeddings (batch, seq_len, embed_dim) for cross-attention.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        freeze: bool = True,
        model_name: str = "roberta-base",
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for RoBERTa-base
        self.proj = nn.Linear(hidden_size, embed_dim) if hidden_size != embed_dim else nn.Identity()
        self.embed_dim = embed_dim
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # (B, seq_len) -> (B, seq_len, hidden_size)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        return self.proj(hidden)  # (B, seq_len, embed_dim)
