import torch
import torch.nn as nn

from .encoders import ImageEncoder, TextEncoder
from .attention import SymmetricCrossModalFusion


class SymmetricVQAModel(nn.Module):
    """VQA model using **symmetric** (shared-weight) cross-modal attention.

    Architecturally identical to :class:`AsymmetricVQAModel` except the
    fusion layer uses a single :class:`SymmetricCrossModalFusion` block
    (same weights for both attention directions).  This serves as the
    baseline to isolate the effect of asymmetric attention.
    """

    def __init__(
        self,
        num_answers: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3,
        freeze_encoders: bool = True,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim, freeze=freeze_encoders)
        self.text_encoder = TextEncoder(embed_dim, freeze=freeze_encoders)
        self.fusion = SymmetricCrossModalFusion(embed_dim, num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_answers),
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Same signature as :class:`AsymmetricVQAModel`."""
        img_features = self.image_encoder(images)
        txt_features = self.text_encoder(input_ids, attention_mask)

        text_padding_mask = attention_mask == 0

        img_attended, txt_attended, attn_i2t, attn_t2i = self.fusion(
            img_features, txt_features, text_padding_mask,
        )

        img_pooled = img_attended.mean(dim=1)
        txt_pooled = txt_attended.mean(dim=1)

        z = torch.cat([img_pooled, txt_pooled], dim=-1)
        logits = self.classifier(z)

        return logits, {"img_to_txt": attn_i2t, "txt_to_img": attn_t2i}
