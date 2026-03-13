import torch
import torch.nn as nn

from .encoders import ImageEncoder, TextEncoder
from .attention import AsymmetricCrossModalFusion


class AsymmetricVQAModel(nn.Module):
    """Full VQA model using **asymmetric** cross-modal attention.

    Pipeline:
        1. Encode image  → patch embeddings  ``(B, 197, d)``
        2. Encode text   → token embeddings  ``(B, seq_len, d)``
        3. Asymmetric fusion (two independent cross-attention blocks)
        4. Mean-pool each attended stream → ``(B, d)`` each
        5. Concatenate → ``(B, 2d)``
        6. MLP classifier → ``(B, num_answers)``
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
        self.fusion = AsymmetricCrossModalFusion(embed_dim, num_heads)

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
        """
        Args:
            images:         ``(B, 3, 224, 224)``
            input_ids:      ``(B, seq_len)``
            attention_mask: ``(B, seq_len)``  (1 = real, 0 = pad)

        Returns:
            logits:       ``(B, num_answers)``
            attn_weights: dict with keys ``"img_to_txt"`` and ``"txt_to_img"``.
        """
        img_features = self.image_encoder(images)                    # (B, 197, d)
        txt_features = self.text_encoder(input_ids, attention_mask)  # (B, S, d)

        # MultiheadAttention expects True = ignore, but our mask has 1 = real.
        text_padding_mask = attention_mask == 0                      # (B, S)

        img_attended, txt_attended, attn_i2t, attn_t2i = self.fusion(
            img_features, txt_features, text_padding_mask,
        )

        # Mean-pool over sequence dimension
        img_pooled = img_attended.mean(dim=1)                        # (B, d)
        txt_pooled = txt_attended.mean(dim=1)                        # (B, d)

        z = torch.cat([img_pooled, txt_pooled], dim=-1)              # (B, 2d)
        logits = self.classifier(z)                                  # (B, A)

        return logits, {"img_to_txt": attn_i2t, "txt_to_img": attn_t2i}
