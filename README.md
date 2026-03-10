# Asymmetric Cross-Modal Attention

Implementation of **Asymmetric Cross-Modal Attention** for Visual Question Answering (VQA), following the project proposal in `Detailed_Project_Proposal.md`. The model uses two separate cross-attention blocks (image→text and text→image) and is compared against a symmetric baseline that shares a single cross-attention block.

## Architecture

- **Image encoder**: ViT-B/16 (torchvision, frozen)
- **Text encoder**: RoBERTa-base (HuggingFace, frozen)
- **Fusion**: Asymmetric = two cross-attention blocks; Symmetric = one shared block
- **Classifier**: MLP over concatenated pooled representations → top-K answers

## Setup

```bash
pip install -r requirements.txt
```

## Data (VQA v2.0)

1. Create directory layout:

   ```
   data/
   ├── images/
   │   ├── train2014/   # COCO train images
   │   └── val2014/     # COCO val images
   ├── questions/
   │   ├── train_questions.json
   │   └── val_questions.json
   └── answers/
       ├── train_annotations.json  (or train_answers.json)
       └── val_annotations.json    (or val_answers.json)
   ```

2. Download from [VQA v2.0](https://visualqa.org/download.html) and [MS COCO](https://cocodataset.org/) (see `data/download_data.py` for URLs).

3. For quick iteration, use a subset (e.g. 1K pairs) via `--subset 1000`.

## Training

```bash
# Asymmetric model (default), small subset
python run_train.py --model asymmetric --data_dir data --subset 1000 --epochs 5

# Symmetric baseline
python run_train.py --model symmetric --data_dir data --subset 1000

# Full training (no subset)
python run_train.py --model asymmetric --data_dir data --epochs 20 --batch_size 64
```

Checkpoints are saved under `results/checkpoints/`.

## Project layout

```
asymmetric-cross-modal-attention/
├── README.md
├── requirements.txt
├── run_train.py
├── data/
│   ├── dataset.py      # VQADataset
│   ├── preprocess.py   # answer vocab, merge Q&A
│   └── download_data.py
├── models/
│   ├── encoders.py     # ImageEncoder (ViT), TextEncoder (RoBERTa)
│   ├── attention.py    # CrossAttentionBlock, AsymmetricCrossModalFusion
│   ├── baselines.py    # SymmetricVQAModel
│   └── asymmetric_model.py
├── training/
│   ├── config.py
│   ├── train.py
│   └── evaluate.py
├── visualization/
│   ├── attention_maps.py
│   ├── plot_results.py
│   └── qualitative_examples.py
├── notebooks/
│   └── 01_data_exploration.ipynb
└── results/
    ├── checkpoints/
    ├── figures/
    └── metrics/
```

## References

- Project proposal: `Detailed_Project_Proposal.md`
- VQA: https://visualqa.org/
- Encoders: ViT-B/16 (torchvision), RoBERTa (HuggingFace)
