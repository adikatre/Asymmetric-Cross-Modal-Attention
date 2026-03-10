# Pull request: Implement asymmetric cross-modal attention for VQA

Implements the full codebase from `Detailed_Project_Proposal.md`.

## Summary

- **Data pipeline**: `VQADataset`, preprocess (top-K answer vocab, merge Q&A), `download_data.py`
- **Models**: `ImageEncoder` (ViT-B/16), `TextEncoder` (RoBERTa-base), `CrossAttentionBlock`, `AsymmetricCrossModalFusion`, `SymmetricVQAModel`, `AsymmetricVQAModel`
- **Training**: config, train loop, evaluate (top-1/top-5)
- **Visualization**: attention heatmaps, comparison plots, qualitative examples
- **Entrypoint**: `run_train.py` for symmetric/asymmetric training
- **Notebook**: `01_data_exploration.ipynb`
- **README** and **requirements.txt** per proposal

## How to run

```bash
pip install -r requirements.txt
python run_train.py --model asymmetric --data_dir data --subset 1000 --epochs 5
```

Data layout: `data/{images/,questions/,answers/}` with VQA v2.0 JSON and COCO images (see README).

## Create the PR

Branch `feature/implement-asymmetric-vqa` has been pushed. Open:

**https://github.com/adikatre/Asymmetric-Cross-Modal-Attention/pull/new/feature/implement-asymmetric-vqa**

Use the title and body above (or copy from this file).
