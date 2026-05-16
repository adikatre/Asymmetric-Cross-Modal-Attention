# models.py - shared across 04_pretrain_coco_itc_itm.ipynb and 04_train_clip_upgrade_v2.ipynb.
#
# Both notebooks materialize this exact content via a %%writefile cell so the
# definitions stay in sync. Edit ONE canonical source (this file in the repo
# scratchpad / both notebook cells) and the materialization is reproducible.
#
# The asymmetric cross-modal fusion is the research artifact: each block runs
# image-queries-attend-to-text FIRST, then text-queries-attend-to-the-image-side-
# output. That ordering is preserved across every block in the stack and across
# both notebooks.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from transformers import RobertaModel


class CLIPImageEncoder(nn.Module):
    """open_clip ViT-{B-16, L-14} @ configurable image_size, LAION-2B pretrained.
    Returns (B, N_tokens, embed_dim) where N_tokens = (image_size/patch)**2 + 1.

    Manual forward (not visual.output_tokens=True) because that flag drops the
    CLS token; we keep CLS + patches so the fusion sees the full sequence.
    """

    def __init__(self, embed_dim, image_size, model_name, pretrained,
                 freeze=False, grad_checkpointing=False):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, force_image_size=image_size,
        )
        self.visual = clip_model.visual
        del clip_model
        self.visual.proj = None

        if grad_checkpointing:
            self.visual.set_grad_checkpointing(True)

        if freeze:
            for p in self.visual.parameters():
                p.requires_grad = False

        # Read width dynamically so this works for both ViT-B (768) and ViT-L (1024).
        clip_width = self.visual.transformer.width
        self.projection = nn.Linear(clip_width, embed_dim)

        patch = self.visual.patch_size if isinstance(self.visual.patch_size, int) else self.visual.patch_size[0]
        expected_tokens = (image_size // patch) ** 2 + 1
        actual_tokens = self.visual.positional_embedding.shape[0]
        assert actual_tokens == expected_tokens, (
            f"position embedding interpolation failed: got {actual_tokens} tokens, "
            f"want {expected_tokens} (image_size={image_size}, patch={patch})"
        )
        print(f"[CLIPImageEncoder] {model_name} / {pretrained} @ {image_size} -> "
              f"{actual_tokens} tokens, width={clip_width}, grad_ckpt={grad_checkpointing}")

    def forward(self, images):
        v = self.visual
        x = v.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls = v.class_embedding.to(x.dtype).view(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.patch_dropout(x)
        x = v.ln_pre(x)

        batch_first = getattr(v.transformer, "batch_first", True)
        if not batch_first:
            x = x.permute(1, 0, 2)
        x = v.transformer(x)
        if not batch_first:
            x = x.permute(1, 0, 2)

        x = v.ln_post(x)
        return self.projection(x)


class TextEncoder(nn.Module):
    """RoBERTa text encoder. Reads hidden dim from config (works for both
    roberta-base @ 768 and roberta-large @ 1024). Bypasses the freshly-initialized
    pooler.dense. Calls gradient_checkpointing_enable() if requested.
    """
    POOLING = "last_hidden_state"

    def __init__(self, embed_dim, model_name="roberta-large",
                 freeze=False, grad_checkpointing=False):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden = self.roberta.config.hidden_size
        self.projection = nn.Linear(hidden, embed_dim)

        if grad_checkpointing:
            self.roberta.gradient_checkpointing_enable()
        if freeze:
            for p in self.roberta.parameters():
                p.requires_grad = False

        assert hasattr(self.roberta, "pooler"), \
            "expected RoBERTa to expose a pooler (which we intentionally bypass)"
        print(f"[TextEncoder] {model_name} pooling={self.POOLING} "
              f"hidden_dim={hidden} grad_ckpt={grad_checkpointing}")

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.projection(out.last_hidden_state)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block: queries from one modality attend to KVs
    from another, then a residual FFN."""

    def __init__(self, embed_dim, num_heads, dropout=0.1, attn_dropout=None):
        super().__init__()
        if attn_dropout is None:
            attn_dropout = dropout
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, key_value, key_padding_mask=None):
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attended, attn_weights = self.cross_attn(
            q, kv, kv, key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True)
        query = query + attended
        query = query + self.ff(self.norm_ff(query))
        return query, attn_weights


class AsymmetricCrossModalFusion(nn.Module):
    """Stack of `depth` asymmetric fusion blocks. Each block:
        img_attended = cross_attn_img_to_txt(query=img, kv=txt)
        txt_attended = cross_attn_txt_to_img(query=txt, kv=img_attended)
    Each block has its own learned weights -- no weight sharing across blocks.
    The asymmetric image-first ordering is the research contribution; do not flip.
    """

    def __init__(self, embed_dim, num_heads, depth=1, dropout=0.1, attn_dropout=None):
        super().__init__()
        self.depth = depth
        self.blocks_img_to_txt = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.blocks_txt_to_img = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout, attn_dropout)
            for _ in range(depth)
        ])

    def forward(self, image_features, text_features, text_padding_mask=None):
        img, txt = image_features, text_features
        last_attn_i2t = last_attn_t2i = None
        for blk_i2t, blk_t2i in zip(self.blocks_img_to_txt, self.blocks_txt_to_img):
            img_attended, last_attn_i2t = blk_i2t(
                query=img, key_value=txt,
                key_padding_mask=text_padding_mask)
            txt_attended, last_attn_t2i = blk_t2i(
                query=txt, key_value=img_attended)
            img, txt = img_attended, txt_attended
        return img, txt, last_attn_i2t, last_attn_t2i


class AttentionPool(nn.Module):
    """Learnable-query attention pool: maps a token sequence to a single vector.
    Supports key_padding_mask for variable-length text streams.
    """

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))


class AsymmetricVQAModelE2E(nn.Module):
    """End-to-end asymmetric cross-modal model used by BOTH pretraining and finetuning.

    Always constructs encoders + fusion + attention pools.

    If `num_answers` is not None: constructs the VQA classifier.
    If `pretrain_heads=True`: constructs ITC projection MLPs + ITM head + logit_scale.

    Use `pretrain_heads=True, num_answers=None` for pretraining.
    Use `pretrain_heads=False, num_answers=3129` for finetuning.
    `load_state_dict(strict=False)` from a pretrain checkpoint tolerates the
    missing classifier keys and unmatched pretrain-head keys.
    """

    def __init__(self,
                 num_answers,
                 embed_dim=768, num_heads=12, fusion_depth=4,
                 dropout=0.3, attn_dropout=0.2, cls_dropout=0.5,
                 image_size=336,
                 clip_model_name="ViT-L-14", clip_pretrained="laion2b_s32b_b82k",
                 text_model_name="roberta-large",
                 vision_grad_checkpointing=True,
                 text_grad_checkpointing=True,
                 pretrain_heads=False,
                 pool_num_heads=8):
        super().__init__()
        self.image_encoder = CLIPImageEncoder(
            embed_dim=embed_dim, image_size=image_size,
            model_name=clip_model_name, pretrained=clip_pretrained,
            freeze=False, grad_checkpointing=vision_grad_checkpointing,
        )
        self.text_encoder = TextEncoder(
            embed_dim=embed_dim, model_name=text_model_name,
            freeze=False, grad_checkpointing=text_grad_checkpointing,
        )
        self.fusion = AsymmetricCrossModalFusion(
            embed_dim, num_heads, depth=fusion_depth,
            dropout=dropout, attn_dropout=attn_dropout,
        )
        self.pool_img = AttentionPool(embed_dim, num_heads=pool_num_heads)
        self.pool_txt = AttentionPool(embed_dim, num_heads=pool_num_heads)

        self.classifier = None
        if num_answers is not None:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(cls_dropout),
                nn.Linear(embed_dim, num_answers),
            )

        self.pretrain_heads = pretrain_heads
        if pretrain_heads:
            self.itc_proj_img = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(),
                nn.Linear(embed_dim, 256),
            )
            self.itc_proj_txt = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(),
                nn.Linear(embed_dim, 256),
            )
            self.itm_head = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim), nn.GELU(),
                nn.Linear(embed_dim, 2),
            )
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode(self, images, input_ids, attention_mask):
        """Encoders + fusion + pooling -> (img_vec, txt_vec) each (B, embed_dim).
        Shared path used by both VQA forward and pretraining forward.
        """
        img_tokens = self.image_encoder(images)
        txt_tokens = self.text_encoder(input_ids, attention_mask)
        text_pad_mask = attention_mask == 0
        img_att, txt_att, _, _ = self.fusion(img_tokens, txt_tokens, text_pad_mask)
        img_vec = self.pool_img(img_att)
        txt_vec = self.pool_txt(txt_att, key_padding_mask=text_pad_mask)
        return img_vec, txt_vec

    def forward(self, images, input_ids, attention_mask):
        """VQA forward. Surfaces fusion attention weights for downstream viz."""
        img_tokens = self.image_encoder(images)
        txt_tokens = self.text_encoder(input_ids, attention_mask)
        text_pad_mask = attention_mask == 0
        img_att, txt_att, attn_i2t, attn_t2i = self.fusion(
            img_tokens, txt_tokens, text_pad_mask)
        img_vec = self.pool_img(img_att)
        txt_vec = self.pool_txt(txt_att, key_padding_mask=text_pad_mask)
        z = torch.cat([img_vec, txt_vec], dim=-1)
        assert self.classifier is not None, "VQA forward requires num_answers != None"
        logits = self.classifier(z)
        return logits, {"img_to_txt": attn_i2t, "txt_to_img": attn_t2i}

    def forward_pretrain(self, images, input_ids, attention_mask):
        """Returns {img_vec, txt_vec, img_emb, txt_emb, logit_scale}.
        img_emb / txt_emb are L2-normalized 256-d projections for ITC.
        """
        assert self.pretrain_heads, "forward_pretrain requires pretrain_heads=True"
        img_vec, txt_vec = self.encode(images, input_ids, attention_mask)
        img_proj = self.itc_proj_img(img_vec)
        txt_proj = self.itc_proj_txt(txt_vec)
        img_emb = F.normalize(img_proj, dim=-1)
        txt_emb = F.normalize(txt_proj, dim=-1)
        return {
            "img_vec": img_vec, "txt_vec": txt_vec,
            "img_emb": img_emb, "txt_emb": txt_emb,
            "logit_scale": self.logit_scale,
        }
