"""Symmetric cross-attention baseline: single shared cross-attention in both directions."""

import torch
import torch.nn as nn

from .encoders import ImageEncoder, TextEncoder
from .attention import CrossAttentionBlock


class SymmetricVQAModel(nn.Module):
    """
    Baseline: one shared cross-attention block used for both directions.
    Image and text both attend to each other with the same weights.
    """

    def __init__(
        self,
        num_answers: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, freeze=True)
        self.text_encoder = TextEncoder(embed_dim=embed_dim, freeze=True)
        # Single shared cross-attention (same block for both directions)
        self.cross_attn = CrossAttentionBlock(embed_dim, num_heads=num_heads, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_answers),
        )
        self.embed_dim = embed_dim

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a = self.image_encoder(images)  # (B, N, d)
        b = self.text_encoder(input_ids, attention_mask)  # (B, M, d)

        # Symmetric: same block for both directions
        a_attended = self.cross_attn(query=a, key_value=b, key_value_mask=attention_mask)
        b_attended = self.cross_attn(query=b, key_value=a, key_value_mask=None)

        a_pooled = a_attended.mean(dim=1)  # (B, d)
        b_pooled = b_attended.mean(dim=1)  # (B, d)
        z = torch.cat([a_pooled, b_pooled], dim=-1)  # (B, 2d)
        return self.classifier(z)
