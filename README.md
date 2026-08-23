# Asymmetric Cross-Modal Attention for VQA

This repository explores asymmetric and symmetric cross-modal attention for
Visual Question Answering (VQA). The experiments compare directional
image-to-text and text-to-image fusion using frozen and fine-tuned visual and
language encoders.

The project is notebook-first. The active notebooks are preserved as the
original experiment records, including their saved outputs and Colab-specific
paths.

## Active notebooks

| Notebook | Purpose |
| --- | --- |
| [`train_evaluate_visualize_colab_frozen_a100.ipynb`](notebooks/train_evaluate_visualize_colab_frozen_a100.ipynb) | Frozen-encoder classifier training and evaluation on an A100 runtime. |
| [`train_evaluate_visualize_colab_unfrozen_5_gcp_s7.ipynb`](notebooks/train_evaluate_visualize_colab_unfrozen_5_gcp_s7.ipynb) | Fine-tuned classifier experiment for the completed seed-7 run. |
| [`train_generative_frozen_asymmetric.ipynb`](notebooks/train_generative_frozen_asymmetric.ipynb) | Frozen-encoder generative VQA with asymmetric fusion. |
| [`train_generative_frozen_symmetric.ipynb`](notebooks/train_generative_frozen_symmetric.ipynb) | Frozen-encoder generative VQA with symmetric fusion. |

These notebooks have not been merged or rewritten. They expect VQA data under
`/content/data` and write checkpoints, metrics, figures, and predictions under
`/content/results`. Some cells also copy data to or from Google Drive. Review
the configuration and storage cells before running a notebook in a new Colab
session.

## Environment

For a local Jupyter environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Model weights and VQA data are not included. The archived preparation notebook
documents the original data setup:
[`prep_data.ipynb`](archive/notebooks/data/prep_data.ipynb).

## Repository layout

```text
archive/notebooks/   Historical notebooks retained without modification
data/                Local VQA datasets (ignored by Git)
docs/                Proposal, implementation review, and report PDFs
notebooks/           Active experiment notebooks
outputs/             Local checkpoints and experiment outputs (ignored by Git)
scripts/             Reusable plotting utilities
```

Historical notebooks are kept for reproducibility but are not the recommended
entry points:

- Data preparation: [`prep_data.ipynb`](archive/notebooks/data/prep_data.ipynb)
- Frozen classifiers: [`train_evaluate_visualize_colab_frozen.ipynb`](archive/notebooks/classifier/frozen/train_evaluate_visualize_colab_frozen.ipynb) and [`train_evaluate_visualize_colab_frozen_4.ipynb`](archive/notebooks/classifier/frozen/train_evaluate_visualize_colab_frozen_4.ipynb)
- Unfrozen classifiers: [`train_evaluate_visualize_colab_unfrozen_5.ipynb`](archive/notebooks/classifier/unfrozen/train_evaluate_visualize_colab_unfrozen_5.ipynb), [`train_evaluate_visualize_colab_unfrozen_5_gcp_s123.ipynb`](archive/notebooks/classifier/unfrozen/train_evaluate_visualize_colab_unfrozen_5_gcp_s123.ipynb), and [`train_evaluate_visualize_colab_unfrozen_5_s123_buns.ipynb`](archive/notebooks/classifier/unfrozen/train_evaluate_visualize_colab_unfrozen_5_s123_buns.ipynb)

## Documentation

- [Original project proposal](docs/project_proposal.md)
- [Implementation review](docs/implementation_review.md)
- [Frozen-encoder report](docs/reports/frozen_encoders.pdf)
- [Unfrozen-encoder report](docs/reports/unfrozen_encoders.pdf)

## Results and plotting

Historical local results have been consolidated under `outputs/legacy/`. The
entire `outputs/` directory is intentionally ignored so checkpoints,
predictions, metrics, and generated figures are not committed.

Reusable plotting functions are available in
[`scripts/plot_results.py`](scripts/plot_results.py). They accept training
history JSON files containing epoch-level loss and accuracy metrics.
