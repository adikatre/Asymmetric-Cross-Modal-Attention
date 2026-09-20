# Direction-Specific Cross-Modal Attention for VQA

Code and experiment artifacts for comparing independent directional cross-attention
with a parameter-matched symmetric control on VQA v2. The experiments use CLIP
ViT-B/32 and RoBERTa-base in two settings: frozen encoders and fully fine-tuned
encoders.

The comparison isolates whether image-to-text and text-to-image interactions should
learn separate weights. Both variants have the same number of learned fusion
parameters. They are not compute-matched: the asymmetric model makes two attention
calls, while the symmetric parameter-matched control makes four.

## Reviewer entry points

These are the two canonical implementation notebooks used for the reported study.

| Setting | Source notebook | Executed experiment record | Primary report |
| --- | --- | --- | --- |
| Frozen encoders | [Frozen notebook](notebooks/train_evaluate_visualize_colab_frozen_5_gcp.ipynb) | [Frozen executed notebook](outputs/results/frozen_encoder/train_evaluate_visualize_colab_frozen_5_gcp.executed.ipynb) | [Frozen final report](outputs/results/frozen_encoder/metrics/clipB32_224_frozen_top3000_directional_parallel_v1_8d4983bd4bfe_final_training_report.json) |
| Fully fine-tuned encoders | [Unfrozen notebook](notebooks/train_evaluate_visualize_colab_unfrozen_5_gcp_s7.ipynb) | [Unfrozen executed notebook](outputs/results/unfrozen_encoder/train_evaluate_visualize_colab_unfrozen_5_gcp_s7.continuation.executed.ipynb) | [Unfrozen final report](outputs/results/unfrozen_encoder/metrics/clipB32_224_fullft_top3000_directional_parallel_v1_ccb164279e25_final_training_report.json) |

The executed notebooks are large because they retain cell output. If GitHub cannot
render one inline, download it or inspect the smaller reports and figures linked
below.

## Main validation results

Scores use the official VQA leave-one-annotator-out metric over all 214,354 VQA v2
validation questions. Each value is the mean and sample standard deviation across
paired seeds 7, 42, and 123.

| Encoder setting | Symmetric-PM | Asymmetric | Paired gain |
| --- | ---: | ---: | ---: |
| Frozen | 59.29 ± 0.18 | **59.77 ± 0.20** | **+0.48 ± 0.19 pp** |
| Fully fine-tuned | 65.30 ± 0.09 | **65.53 ± 0.07** | **+0.23 ± 0.16 pp** |

These are modest validation gains. Per-seed results and conditional image-bootstrap
intervals are recorded in the two final reports above. The fixed 10,000-question
subset is used only for checkpoint selection and diagnostic analyses; its scores are
not the headline validation results.

Selected-checkpoint EvalAI test results, modality-blind ablations, attention
visualizations, and question-type analyses are retained under [`outputs/`](outputs/).
EvalAI files represent one selected checkpoint per model, not three-seed test means.

## Reproducing the experiments

The notebooks were run on a single NVIDIA A100-SXM4-40GB. Full reproduction requires
the VQA v2 images, questions, and annotations; datasets, pretrained model weights, and
training checkpoints are not stored in this repository.

Create a local environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
jupyter lab
```

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

The notebooks discover the repository root automatically. Paths can also be set
explicitly for a headless or cloud run:

```bash
export VQA_PROJECT_ROOT=/path/to/Asymmetric-Cross-Modal-Attention
export VQA_DATA_DIR=/path/to/vqa/data
export VQA_OUTPUT_DIR=/path/to/run/output
```

Before a full run, review the configuration cell and run the built-in preflight
benchmark. The frozen notebook caches encoder hidden states once and trains only the
projection, fusion, and classifier layers. The unfrozen notebook trains both encoders
end to end. Each campaign trains the asymmetric and symmetric parameter-matched
models with the same recipe for seeds 7, 42, and 123.

## Artifact guide

[`outputs/README.md`](outputs/README.md) documents the committed artifacts and the
boundary between full-validation evidence, fixed-subset diagnostics, and selected
EvalAI exports. In summary:

- `outputs/results/frozen_encoder/` contains the frozen campaign record.
- `outputs/results/unfrozen_encoder/` contains the fully fine-tuned campaign record.
- `metrics/` contains configuration fingerprints, histories, per-seed summaries,
  validation predictions, and aggregate reports.
- `figures/` contains training curves and diagnostic visualizations.
- No `.pt`, `.pth`, or `.ckpt` model checkpoints are committed.
- Bulky test prediction JSON files are excluded; their validation metadata and saved
  EvalAI score records are retained.

## Repository layout

```text
notebooks/           Two canonical source notebooks
outputs/             Curated executed notebooks, metrics, figures, and metadata
tests/               Offline checks for scoring, parameter parity, and campaign logic
scripts/             GCP launch helpers and plotting utilities
archive/notebooks/   Historical development notebooks (not recommended entry points)
docs/                Project documentation and reports
```

Run the offline notebook checks with:

```bash
python3 -m unittest discover -s tests -v
```

The tests do not download VQA data or pretrained weights. They statically parse the
notebook and exercise the scoring, masking, parameter-parity, initialization,
checkpoint-resume, and export-validation logic with synthetic fixtures.

## Historical notebooks

Earlier classifier, generative, and data-preparation notebooks are retained under
[`archive/notebooks/`](archive/notebooks/) for provenance. They are not the
implementation entry points for the reported frozen and fully fine-tuned campaigns.
