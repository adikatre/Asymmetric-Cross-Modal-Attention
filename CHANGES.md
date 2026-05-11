# CHANGES — `02_..._unfrozen_5.ipynb` → `03_train_clip_upgrade.ipynb`

Direct A/B with the 64.27% baseline. Same answer vocab (3129), same train/val splits, same soft-target construction, same VQA accuracy metric (`min(1, count/3)` approximation). Only the items below are different.

**Expected outcome at convergence: 67–69% val VQA accuracy** with a per-question-type breakdown (yes/no, number, other) printed each epoch. If we land outside that range, the per-type table tells us whether the remaining gap is vision-side (color/spatial questions) or language-side (counts, reasoning).

## Change table

Each change is *independently* revertible so you can A/B which lever drove the gain.

| # | Change | Where | Why | Revert |
|---|--------|-------|-----|--------|
| 1 | **Vision encoder: CLIP ViT-B/16 @ 384** (open_clip + `laion2b_s34b_b88k`) replaces torchvision `vit_b_16(IMAGENET1K_V1)` @ 224. Position embeddings interpolated to 24×24+1 = **577 tokens** via `open_clip.create_model_and_transforms(..., force_image_size=384)`. The final shared-space `visual.proj` is set to `None` so we keep raw transformer outputs. | `CLIPImageEncoder` cell | LAION-2B CLIP features outperform ImageNet ViT on most multimodal downstream tasks. The 6× larger patch grid (576 vs 196) gives the fusion module much finer spatial granularity. | Restore the baseline `ImageEncoder` class (torchvision `vit_b_16`), set `IMAGE_SIZE=224`, switch normalization to ImageNet (0.485/0.456/0.406, 0.229/0.224/0.225). All other changes are encoder-agnostic. |
| 2 | **New H5 image cache at 384×384** (`vqa_images_384.h5`), built directly from raw JPEGs — *not* upsampled from the 256-px baseline cache. ~50 GB on disk (123k × 384²×3 uint8). The build cell has a three-way fast path: local → Drive (`/content/drive/MyDrive/VQA_cache/vqa_images_384.h5`) → build from scratch. | H5 build cell, `VQADataset` | Bicubic upsampling 256→384 would silently throw away most of the resolution gain. Building from originals at 384 preserves fidelity. ~30 min one-time on Colab L4. | Point `VQADataset(h5_path=...)` back at `vqa_images.h5` and add `transforms.Resize(IMAGE_SIZE, interpolation=BICUBIC)` to the front of both train and val transforms. |
| 3 | **Explicit pooler bypass + diagnostic** in `TextEncoder`. The baseline already used `last_hidden_state` (full sequence) implicitly. We now declare `POOLING = "last_hidden_state"`, assert RoBERTa has a pooler that we are intentionally not calling, and print a one-line diagnostic at init. | `TextEncoder` cell | The pooler's `dense` layer is **freshly initialized** when `roberta-base` is loaded (HuggingFace warns about this). Routing `pooler_output` into fusion would feed random weights into the gradient path. Making the choice explicit prevents a future regression where someone changes `last_hidden_state` to `pooler_output`. | Drop the `POOLING` class attribute, the assert, and the diagnostic print. Behavior is unchanged either way (the baseline already used `last_hidden_state`). |
| 4 | **EMA of model weights (decay 0.9998)**. `ModelEMA` tracks every float tensor in `state_dict()` (params + float buffers). Live weights and EMA weights are both evaluated each epoch; best checkpoint is chosen by EMA val accuracy. Two checkpoints saved per "best" event: `*_best_live.pt` and `*_best_ema.pt`. | `ModelEMA` cell, `run_training` | EMA typically buys 0.3–0.5 points over the live-weights curve at no training-time cost, and it stabilizes the validation signal so the best-checkpoint choice is less noisy. | Set `USE_EMA = False` in the config cell. `run_training` and `evaluate_with_types` both no-op the EMA path in that case. |
| 5 | **Light VQA-safe augmentation** on the train transform only: `RandomHorizontalFlip(0.5)` + `RandAugmentNoGeom(n=2, m=9)` + `ColorJitter(0.1, 0.1, 0.1, hue=0)` + `RandomErasing(p=0.25, scale=(0.02, 0.10))`. `RandAugmentNoGeom` is a tiny subclass that drops `TranslateX/Y`, `ShearX/Y`, and `Rotate` from the op space. Val transform is clean (ToTensor + Normalize). | Transforms cell | The baseline overfits (train 68.97 vs val 64.27); regularization should help. Geometric ops are excluded because VQA labels frequently depend on spatial relations ("what is on the left", "how many people standing") — those ops would silently break supervision. | Replace `get_train_transform()` body with just `ToTensor() + Normalize(CLIP_MEAN, CLIP_STD)` (matches val). |
| 6 | **Per-question-type val each epoch** using VQA-v2's official `answer_type` field from annotations (`"yes/no" \| "number" \| "other"`). `VQADataset` returns an extra `answer_type_idx` per sample; `evaluate_with_types` aggregates per-type and the run loop prints + logs all three numbers per epoch. | `VQADataset`, `evaluate_with_types`, `run_training`, history JSON | Localizes where the remaining gap is — vision-side vs language-side. Pulled from `ann["answer_type"]` instead of regex over question strings (more reliable and matches the official VQA-v2 evaluation taxonomy). | Drop `answer_type_idx` from the dataset return tuple, switch eval to the baseline's single-number `evaluate()`, and adjust the train loop to no longer destructure the 5-tuple. |
| 7 | **Modern AMP API** + **vision gradient checkpointing flag**. `from torch.amp import GradScaler, autocast` (no more `torch.cuda.amp` deprecation warnings); `GradScaler("cuda")`, `autocast("cuda")`. `USE_GRAD_CHECKPOINT` config flag toggles `self.visual.set_grad_checkpointing(True)` on the open_clip ViT. Default off (turn on if you OOM at `BATCH_SIZE=16`). | Imports, `_amp_ctx`, `train_one_epoch_amp`, `CLIPImageEncoder`, config | Future-proof against torch deprecations; explicit memory escape hatch for the 384 resolution bump. | Revert imports to `from torch.cuda.amp import GradScaler, autocast` and drop the `"cuda"` device arg; set `USE_GRAD_CHECKPOINT = False` and remove the `grad_checkpointing` plumbing in `CLIPImageEncoder`. |
| 8 | **Training schedule**: `EPOCHS=12` (was 18), `WARMUP_FRAC=0.05` (was 1 epoch / ~5.5%), `BATCH_SIZE=16` × `GRAD_ACCUM_STEPS=4` = effective batch 64 (was 24, no accum). Cosine schedule unchanged. AdamW with same differential LRs (1e-5 encoders, 1e-4 fusion+head), same WD exclusion list. Added `GRAD_CLIP_NORM=1.0` for AMP safety. | Config, `run_training`, `train_one_epoch_amp` | Better features + EMA should converge faster (12 epochs is enough); 5% warmup is calculated from total optimizer steps rather than a hard epoch count (cleaner with grad accum); effective batch 64 is the standard for VQA fine-tunes and gives smoother loss curves at 384. | Set `EPOCHS=18`, `WARMUP_FRAC` to `1/EPOCHS`, `BATCH_SIZE=24`, `GRAD_ACCUM_STEPS=1`, and `GRAD_CLIP_NORM=None`. |

## Memory & throughput

- **Per-step memory**: `BATCH_SIZE=16` × 384 with CLIP-B fully unfrozen + RoBERTa fully unfrozen + fusion + AMP → expected ~14–17 GB on L4 (well under 24 GB). The first optimizer step prints `[Mem] peak after first optimizer step: X.XX GB` so you can verify before committing.
- **OOM fallback** (do this if peak > 22 GB): set `BATCH_SIZE=8`, `GRAD_ACCUM_STEPS=8` (still effective batch 64), and `USE_GRAD_CHECKPOINT=True`. The checkpointing roughly halves activation memory on the vision tower at the cost of ~20–30% slower training.
- **Disk**: the new `vqa_images_384.h5` is ~50 GB. Colab L4 instances have >100 GB ephemeral disk so this fits, but after the first build consider copying it to Drive at `/content/drive/MyDrive/VQA_cache/vqa_images_384.h5` so subsequent runs fast-path the copy.
- **DataLoader**: `num_workers=4`, `pin_memory=True`, `persistent_workers=True`. H5 handles are opened lazily *per worker* inside `__getitem__` (never shared) — same pattern as the baseline.

## Inline correctness guards

The notebook is designed to fail loudly on first run if any swap is wrong:

- `CLIPImageEncoder.__init__` asserts (a) `visual.width == 768` and (b) position embedding has exactly 577 entries (24×24+1) after `force_image_size=384` — catches a silent PE-interpolation failure.
- `TextEncoder.__init__` asserts the RoBERTa pooler exists (so we know we're choosing to bypass a real layer, not papering over a load failure) and prints `[TextEncoder] pooling=last_hidden_state ...` once.
- The sanity-check cell instantiates the full model on CPU, runs a dummy forward, and asserts `vision == (2, 577, 512)`, `text == (2, 20, 512)`, `logits == (2, 3129)`.
- `ModelEMA.update` asserts the shadow moved on the first call (catches "all params are integers" / "wrong state_dict" bugs).
- The first optimizer step prints peak GPU memory.

## What's intentionally not changed

- Answer vocabulary (3129 classes, identical normalization).
- Train/val splits (full VQA-v2 train2014 / val2014).
- Soft-target construction (`count / 10`, BCE-with-logits).
- VQA accuracy metric (`min(1, count/3)` approximation in `evaluate_with_types`).
- `CrossAttentionBlock` / `AsymmetricCrossModalFusion` / classifier head — verbatim from baseline cell 28+30. The asymmetric-fusion research contribution is preserved exactly.
- Optimizer (AdamW), param-group split, weight-decay exclusion list.
- Differential LR (1e-5 encoders / 1e-4 fusion+head).
- Cosine schedule shape (only the warmup-step *count* changed).
- Visualization / attention / qualitative cells from the baseline were **omitted** from v1 — the goal here is a clean accuracy A/B. Add them back from the baseline notebook once we have a number.

## Out of scope (for next iteration if this one works)

- CLIP ViT-L/14 (larger backbone).
- Mixup / label smoothing / ensembling (avoid confounding the encoder-swap signal).
- Warm-starting fusion+classifier from `asymmetric_s42_best.pt` (would muddy the A/B; encoder weights aren't transferable anyway).
- Longer max question length than 20.
