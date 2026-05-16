# MEMO_v2 — Asymmetric Cross-Modal VQA, "v2" upgrade

_Fill in the bracketed values after the runs complete. Keep this memo at one page — the goal is honest accounting, not a writeup._

## What changed vs `03_train_clip_upgrade.ipynb`

The control (`03_*`) was CLIP ViT-B/16 @ 384 + RoBERTa-base + 1 asymmetric fusion block, ~64% val VQA acc. `v2` ships two notebooks:

1. [notebooks/04_pretrain_coco_itc_itm.ipynb](notebooks/04_pretrain_coco_itc_itm.ipynb) — cross-modal pretraining on COCO 2014 captions (ITC + ITM with hard-negative mining), producing `pretrain_final.pt`.
2. [notebooks/04_train_clip_upgrade_v2.ipynb](notebooks/04_train_clip_upgrade_v2.ipynb) — VQA finetuning from the pretrained init.

Architectural deltas (same in both notebooks, so the pretrained weights are drop-in):

- Vision: ViT-L/14 @ 336 (LAION-2B). Token count preserved at 24×24+1=577.
- Text: RoBERTa-large (hidden 1024). Pooler still bypassed.
- Fusion: **depth-4 asymmetric stack**. Each block runs img→txt cross-attn FIRST, then txt→img-attended. No weight sharing across blocks. `EMBED_DIM=768`, `NUM_HEADS=12`.
- Pooling: learnable-query attention pool per stream (replaces CLS-of-each concatenation).

Optimization deltas:

- `bfloat16` autocast (L4 supports it natively); GradScaler removed.
- LLRD on both encoders, `decay=0.95`.
- `RandomHorizontalFlip` dropped from train transform (contradicts the `RandAugmentNoGeom` rationale).
- `MAX_SAMPLES=None`, single shared tokenizer, `NUM_WORKERS=2`, `persistent_workers=True`, `prefetch_factor=4`.
- Auto-resume cell rewritten with a defensive `_epoch\d+$` regex anchored on the new run name.

## What worked

- [Fill in after runs: which deltas moved the needle, e.g. "pretraining drove +X pts vs LAION-2B-only init", "attention pool +Y pts", "LLRD let bottom layers train without hurting top layers"]

## What didn't

- [Fill in after runs: e.g. "ITC R@1 plateaued at Z% — we didn't get the full alignment we wanted before VQA finetune", or "ViT-L gave smaller gains than expected because RoBERTa-large overfit small VQA train splits", or "fallback path X was needed because Y"]
- Memory budget actuals: [fill peak GB per stage from the `[Mem]` lines in the notebooks].
- Phase 1 fallbacks taken (in order, from the plan's ladder: bs 8→4, FUSION_DEPTH 4→3, roberta-large→base, EMBED_DIM 768→640): [list which were needed, or "none"].

## Did pretraining help?

| Init                                | Final val_vqa_acc (EMA) |
|-------------------------------------|---:|
| `03_*` baseline (B/16 + base + d1)  | [fill — from `results/metrics/asymmetric_clip384_s42_history.json`] |
| `v2` from LAION-2B (pt0)            | [fill — only if you ran the pt0 path, or from the gate-failed branch] |
| `v2` from `pretrain_final.pt` (pt1) | [fill — primary headline number] |

Phase 2 gate result (from `results/metrics/pretrain_history.json` last epoch):
- ITC R@1 = [%] (gate ≥ 50%)
- ITM acc = [%] (gate ≥ 75%)
- Checkpoint used for finetuning init: `pretrain_final.pt` / fell back to LAION-2B.

## Final number

**Best EMA val_vqa_acc**: [%] at epoch [N], run name `asymmetric_clip_l14_336_robertaL_d4_pt{0|1}_s42`.

Per-type breakdown (from `results/figures/per_type_v2.png`):
- Yes/No: [%]
- Number: [%]
- Other: [%]

## Recommended next move

The realistic ceiling for this architecture family (LAION-2B encoders + COCO captions pretraining + asymmetric fusion + VQA finetune) is ~71–74%. The professor's 80% target requires a **pretrained vision-language model** as the starting point — BLIP, ALBEF, or BLIP-2 — which is a different research direction. Concretely:

1. Swap the LAION-2B CLIP encoder + RoBERTa-large + COCO ITC/ITM pretraining for the encoders + alignment learned by BLIP (or BLIP-2's Q-Former) directly. This collapses 6 hours of pretraining + tens of millions of params into a one-line `from_pretrained()` call, and brings in supervision from ~129M image-text pairs we can't realistically train on locally.
2. Keep the asymmetric depth-4 fusion + attention-pool head on top — the research artifact survives the encoder swap.
3. Re-run the finetuning notebook with `CLIP_MODEL` / `TEXT_MODEL` constants pointed at the BLIP encoders.

The bookkeeping in this notebook (per-type gates, EMA, LLRD, history schema, test predictions export) is already aligned with that next step — only the encoder constructors change.
