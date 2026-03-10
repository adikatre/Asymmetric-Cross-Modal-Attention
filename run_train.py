"""
Train symmetric baseline or asymmetric VQA model.

Usage:
  python run_train.py --model asymmetric --data_dir data --subset 1000
  python run_train.py --model symmetric --data_dir data
"""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from data import VQADataset
from data.preprocess import build_answer_vocab
from models import AsymmetricVQAModel, SymmetricVQAModel
from training.config import DataConfig, ModelConfig, TrainConfig
from training.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["symmetric", "asymmetric"], default="asymmetric")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--subset", type=int, default=None, help="Subset size for debugging")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    parser.add_argument("--num_answers", type=int, default=1000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    questions_train = data_dir / "questions" / "train_questions.json"
    answers_train = data_dir / "answers" / "train_annotations.json"
    questions_val = data_dir / "questions" / "val_questions.json"
    answers_val = data_dir / "answers" / "val_annotations.json"
    image_dir = data_dir / "images"

    # Fallback names for VQA v2 files
    if not answers_train.exists():
        answers_train = data_dir / "answers" / "train_answers.json"
    if not answers_val.exists():
        answers_val = data_dir / "answers" / "val_answers.json"

    if not questions_train.exists() or not answers_train.exists():
        print("Data not found. Create data/questions/ and data/answers/ with VQA v2 JSON files.")
        print("See data/download_data.py and Detailed_Project_Proposal.md.")
        return 1

    answer_to_idx, _ = build_answer_vocab(str(answers_train), top_k=args.num_answers)
    num_answers = len(answer_to_idx)

    train_ds = VQADataset(
        image_dir=str(image_dir),
        questions_file=str(questions_train),
        answers_file=str(answers_train),
        answer_to_idx=answer_to_idx,
        top_k_answers=args.num_answers,
        subset_size=args.subset,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = None
    if questions_val.exists() and answers_val.exists():
        val_ds = VQADataset(
            image_dir=str(image_dir),
            questions_file=str(questions_val),
            answers_file=str(answers_val),
            answer_to_idx=answer_to_idx,
            subset_size=min(args.subset or 5000, 5000),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

    model_config = ModelConfig()
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.model == "symmetric":
        model = SymmetricVQAModel(
            num_answers=num_answers,
            embed_dim=model_config.embed_dim,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
        )
    else:
        model = AsymmetricVQAModel(
            num_answers=num_answers,
            embed_dim=model_config.embed_dim,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
        )

    train_model(
        model,
        train_loader,
        val_loader=val_loader,
        config=train_config,
        model_name=args.model,
    )
    return 0


if __name__ == "__main__":
    exit(main())
