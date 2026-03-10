"""Cross-attention blocks for asymmetric cross-modal fusion."""

import torch
import torch.nn as nn
import math


class CrossAttentionBlock(nn.Module):
    """
    Multi-head cross-attention: Q from one modality, K and V from the other.
    Includes layer norm and residual connection.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        query: (B, seq_q, embed_dim) from modality A
        key_value: (B, seq_kv, embed_dim) from modality B
        key_value_mask: (B, seq_kv) optional; 0 for padding
        Returns: (B, seq_q, embed_dim)
        """
        B, seq_q, _ = query.shape
        _, seq_kv, _ = key_value.shape

        q = self.q_proj(query).view(B, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, heads, seq_q, seq_kv)
        if key_value_mask is not None:
            # mask: (B, seq_kv) -> (B, 1, 1, seq_kv)
            scores = scores.masked_fill(key_value_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, seq_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, seq_q, self.embed_dim)
        out = self.out_proj(out)
        return self.norm(query + self.dropout(out))


class AsymmetricCrossModalFusion(nn.Module):
    """
    Two separate cross-attention blocks:
    - Block 1: Q=a (image), K,V=b (text) -> a_attended (image attends to text)
    - Block 2: Q=b (text), K,V=a (image) -> b_attended (text attends to image)
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_a_to_b = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_attn_b_to_a = CrossAttentionBlock(embed_dim, num_heads, dropout)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        b_mask: torch.Tensor | None = None,
        a_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        a: (B, N, embed_dim) image patch embeddings
        b: (B, M, embed_dim) text token embeddings
        b_mask: (B, M) attention mask for text (1 = valid)
        a_mask: (B, N) optional mask for image (usually None)
        Returns: (a_attended, b_attended) each same shape as input
        """
        a_attended = self.cross_attn_a_to_b(query=a, key_value=b, key_value_mask=b_mask)
        b_attended = self.cross_attn_b_to_a(query=b, key_value=a, key_value_mask=a_mask)
        return a_attended, b_attended
