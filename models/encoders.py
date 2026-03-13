import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import RobertaModel


class ImageEncoder(nn.Module):
    """ViT-B/16 image encoder that produces spatial patch embeddings.

    Loads a pretrained ViT-B/16 and extracts the patch token sequence
    (including the CLS token) before the classification head, then
    projects from the ViT hidden dimension (768) to ``embed_dim``.

    Output shape: ``(batch, 197, embed_dim)`` — 196 spatial patches + 1 CLS token.
    """

    VIT_HIDDEN_DIM = 768

    def __init__(self, embed_dim: int = 512, freeze: bool = True):
        super().__init__()
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder = vit.encoder

        self.projection = nn.Linear(self.VIT_HIDDEN_DIM, embed_dim)

        if freeze:
            for param in self.conv_proj.parameters():
                param.requires_grad = False
            self.class_token.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: ``(B, 3, 224, 224)`` normalised image batch.

        Returns:
            Patch embeddings ``(B, 197, embed_dim)``.
        """
        # Patch embedding: (B, hidden_dim, H', W') -> (B, N, hidden_dim)
        x = self.conv_proj(images)                          # (B, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)                    # (B, 196, 768)

        # Prepend CLS token
        cls = self.class_token.expand(x.shape[0], -1, -1)   # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)                      # (B, 197, 768)

        # Transformer encoder
        x = self.encoder(x)                                  # (B, 197, 768)

        # Project to common embedding dimension
        x = self.projection(x)                               # (B, 197, embed_dim)
        return x


class TextEncoder(nn.Module):
    """RoBERTa-base text encoder that produces per-token embeddings.

    Loads a pretrained ``roberta-base`` model and projects the last
    hidden state from 768 to ``embed_dim``.

    Output shape: ``(batch, seq_len, embed_dim)``.
    """

    ROBERTA_HIDDEN_DIM = 768

    def __init__(self, embed_dim: int = 512, freeze: bool = True):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.projection = nn.Linear(self.ROBERTA_HIDDEN_DIM, embed_dim)

        if freeze:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      ``(B, seq_len)`` token ids from the RoBERTa tokenizer.
            attention_mask: ``(B, seq_len)`` mask (1 = real token, 0 = padding).

        Returns:
            Token embeddings ``(B, seq_len, embed_dim)``.
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state                   # (B, seq_len, 768)
        hidden = self.projection(hidden)                     # (B, seq_len, embed_dim)
        return hidden
