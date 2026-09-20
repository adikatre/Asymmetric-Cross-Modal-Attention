# Experiment artifacts

This directory contains the curated, checkpoint-free records produced by the two
canonical notebooks. The directory names describe whether CLIP ViT-B/32 and
RoBERTa-base were frozen or fully fine-tuned.

## What to inspect

For each encoder setting, start with:

1. `*_final_training_report.json` for the aggregate three-seed, full-validation
   comparison.
2. `*_campaign.json` and per-seed `*_summary.json` files for run-level provenance.
3. `*_config.json` and `*_preflight.json` for the configuration fingerprint and
   runtime environment.
4. `figures/` for training curves, checkpoint-selected diagnostics, and qualitative
   visualizations.
5. The executed notebook for the complete saved cell-output record.

## Result boundaries

- Headline validation results use all 214,354 validation questions and paired seeds
  7, 42, and 123. They are recorded in the final training reports.
- Validation `.npz` files retain the per-question arrays used for paired comparisons
  and conditional image-bootstrap intervals.
- Ablation and question-type figures use selected checkpoints and the fixed
  10,000-question validation subset unless their accompanying record says otherwise.
- `evalai_submissions/` retains metadata and manually recorded EvalAI scores for the
  selected test2015 exports. These are individual selected checkpoints, not
  three-seed test averages.

## Deliberate exclusions

Model checkpoints (`.pt`, `.pth`, and `.ckpt`) are not committed. The four test2015
prediction JSON files are also excluded because they add roughly 82 MB under
`results/*/predictions/` and are duplicated under `evalai_submissions/`. Their
metadata records the model selection, prediction count, file hash, and original
filename.
