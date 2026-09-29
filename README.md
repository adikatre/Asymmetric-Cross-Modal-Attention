# Direction-Specific Cross-Modal Attention for VQA v2.0

Code and experiment artifacts for comparing independent directional cross-attention
with a parameter-matched symmetric control on VQA v2. Encoders are 
ViT-B/32 and RoBERTa-base in two settings with frozen and fully fine-tuned
encoders. For CLEVR and Flikr30k, the ViT-B/16 vision encoder is used. 

## Key Files

| Setting | Source notebook | Executed experiment record | Primary report |
| --- | --- | --- | --- |
| Frozen encoders | [Frozen notebook](notebooks/train_evaluate_visualize_colab_frozen_5_gcp.ipynb) | [Frozen executed notebook](outputs/results/frozen_encoder/train_evaluate_visualize_colab_frozen_5_gcp.executed.ipynb) | [Frozen final report (json)](outputs/results/frozen_encoder/metrics/clipB32_224_frozen_top3000_directional_parallel_v1_8d4983bd4bfe_final_training_report.json) |
| Fully fine-tuned encoders | [Unfrozen notebook](notebooks/train_evaluate_visualize_colab_unfrozen_5_gcp_s7.ipynb) | [Unfrozen executed notebook](outputs/results/unfrozen_encoder/train_evaluate_visualize_colab_unfrozen_5_gcp_s7.continuation.executed.ipynb) | [Unfrozen final report (json)](outputs/results/unfrozen_encoder/metrics/clipB32_224_fullft_top3000_directional_parallel_v1_ccb164279e25_final_training_report.json) |

## VQA v2.0 Results

| Encoder setting | Symmetric-PM | Asymmetric | Paired gain |
| --- | ---: | ---: | ---: |
| Frozen | 59.29 ± 0.18 | 59.77 ± 0.20 | +0.48 ± 0.19 pp |
| Fully fine-tuned | 65.30 ± 0.09 | 65.53 ± 0.07 | +0.23 ± 0.16 pp |

Selected-checkpoint EvalAI test results, modality-blind ablations, attention
visualizations, and question-type analyses are under [`outputs/`](outputs/).
EvalAI files represent the highest-performing checkpoint for each model.

## Reproducing the experiments

The notebooks were run on a single NVIDIA A100-40GB. Full reproduction requires
the VQA v2 images, questions, and annotations; datasets, pretrained model weights, and
training checkpoints are not stored in this repository.

VQA v2.0 dataset: https://visualqa.org/
EvalAI result submission: https://eval.ai/web/challenges/challenge-page/830/overview

Arrange the VQA files as follows:

```text
data/
├── answers/
│   ├── v2_mscoco_train2014_annotations.json
│   └── v2_mscoco_val2014_annotations.json
├── images/
│   ├── train2014/
│   ├── val2014/
│   └── test2015/                 # optional; required only for test export
└── questions/
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    ├── v2_OpenEnded_mscoco_val2014_questions.json
    └── v2_OpenEnded_mscoco_test2015_questions.json  # optional
```

Before a full run, review the configuration cell and run the built-in preflight
benchmark. The frozen notebook caches encoder hidden states once and trains only the
projection, fusion, and classifier layers. The unfrozen notebook trains both encoders
end to end. Each campaign trains the asymmetric and symmetric parameter-matched
models with the same recipe for seeds 7, 42, and 123.

## Repository layout

```text
notebooks/           Two source notebooks
outputs/             Curated executed notebooks, metrics, figures, and metadata
archive/notebooks/   Old notebooks & iterations
docs/                Various Documentation and Reports
```

## Artifact guide

[`outputs/README.md`](outputs/README.md) documents the committed artifacts and the
boundary between full-validation evidence, fixed-subset diagnostics, and selected
EvalAI exports. In summary:

- `outputs/results/frozen_encoder/` contains the frozen campaign record.
- `outputs/results/unfrozen_encoder/` contains the fully fine-tuned campaign record.
- `metrics/` contains configuration fingerprints, histories, per-seed summaries,
  validation predictions, and aggregate reports.
- `figures/` contains training curves and diagnostic visualizations.
- EvalAI score records are retained.

