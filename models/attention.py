import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block: queries from one modality attend to
    keys/values from the other modality.

    Includes multi-head attention, layer normalisation, residual
    connections, and a position-wise feed-forward network.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query:            ``(B, Nq, d)`` — sequence that *asks*.
            key_value:        ``(B, Nkv, d)`` — sequence that *answers*.
            key_padding_mask: ``(B, Nkv)`` bool mask where ``True`` means
                              the position is padding and should be ignored.

        Returns:
            output:       ``(B, Nq, d)`` — cross-attended representation.
            attn_weights: ``(B, Nq, Nkv)`` — attention weight matrix.
        """
        # Pre-norm cross-attention with residual
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attended, attn_weights = self.cross_attn(
            q, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        query = query + attended

        # Pre-norm feed-forward with residual
        query = query + self.ff(self.norm_ff(query))
        return query, attn_weights


class AsymmetricCrossModalFusion(nn.Module):
    """Two **separate** cross-attention blocks with independent weights:

    * Block 1 — image patches attend to text tokens  (a ← b)
    * Block 2 — text tokens attend to image patches  (b ← a)

    Because the blocks have different learned parameters, the model can
    capture directional, asymmetric interactions between the modalities.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_img_to_txt = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_attn_txt_to_img = CrossAttentionBlock(embed_dim, num_heads, dropout)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features:    ``(B, N_img, d)``
            text_features:     ``(B, N_txt, d)``
            text_padding_mask: ``(B, N_txt)`` bool mask (True = padding).

        Returns:
            img_attended:    ``(B, N_img, d)`` — image informed by text.
            txt_attended:    ``(B, N_txt, d)`` — text informed by image.
            attn_img_to_txt: ``(B, N_img, N_txt)`` — which text tokens each
                             image patch attended to.
            attn_txt_to_img: ``(B, N_txt, N_img)`` — which image patches each
                             text token attended to.
        """
        img_attended, attn_img_to_txt = self.cross_attn_img_to_txt(
            query=image_features,
            key_value=text_features,
            key_padding_mask=text_padding_mask,
        )
        txt_attended, attn_txt_to_img = self.cross_attn_txt_to_img(
            query=text_features,
            key_value=image_features,
            # Image patches are never padded, so no mask needed.
        )
        return img_attended, txt_attended, attn_img_to_txt, attn_txt_to_img


class SymmetricCrossModalFusion(nn.Module):
    """A **single shared** cross-attention block used in both directions.

    The same weights process both (image → text) and (text → image),
    so the model cannot learn direction-specific patterns.  This serves
    as the baseline to compare against the asymmetric variant.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.shared_cross_attn = CrossAttentionBlock(embed_dim, num_heads, dropout)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same signature and return format as
        :class:`AsymmetricCrossModalFusion` so the two are drop-in
        replaceable."""
        img_attended, attn_img_to_txt = self.shared_cross_attn(
            query=image_features,
            key_value=text_features,
            key_padding_mask=text_padding_mask,
        )
        txt_attended, attn_txt_to_img = self.shared_cross_attn(
            query=text_features,
            key_value=image_features,
        )
        return img_attended, txt_attended, attn_img_to_txt, attn_txt_to_img
