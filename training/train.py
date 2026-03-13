"""Training loop for symmetric and asymmetric VQA models.

Usage (from project root)::

    # Quick dev run (1 000 samples, 5 epochs)
    python -m training.train --preset dev

    # Full asymmetric training
    python -m training.train --model-type asymmetric --epochs 20

    # Full symmetric baseline
    python -m training.train --model-type symmetric --epochs 20 --seed 42
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.preprocess import build_dataloaders
from models import AsymmetricVQAModel, SymmetricVQAModel
from training.config import DEV_CONFIG, FULL_CONFIG, INITIAL_CONFIG, TrainConfig
from training.evaluate import evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: TrainConfig, device: torch.device) -> nn.Module:
    cls = AsymmetricVQAModel if cfg.model_type == "asymmetric" else SymmetricVQAModel
    model = cls(
        num_answers=cfg.num_answers,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        freeze_encoders=cfg.freeze_encoders,
    )
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, input_ids, attention_mask, answers in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        answers = answers.to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(images, input_ids, attention_mask)
            loss = criterion(logits, answers)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * answers.size(0)
        correct += (logits.argmax(dim=1) == answers).sum().item()
        total += answers.size(0)

    return {
        "train_loss": total_loss / total,
        "train_acc": correct / total * 100,
    }


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────
    print("Building dataloaders …")
    train_loader, val_loader, answer_to_idx, idx_to_answer = build_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        top_k_answers=cfg.num_answers,
        max_question_len=cfg.max_question_len,
        max_samples=cfg.max_samples,
        num_workers=cfg.num_workers,
    )
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")

    # Save answer vocab alongside checkpoints
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = cfg.checkpoint_dir / f"{cfg.run_name}_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(answer_to_idx, f)

    # ── Model / optimiser ────────────────────────────────────────────
    print(f"Building {cfg.model_type} model …")
    model = build_model(cfg, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    # ── Training loop ────────────────────────────────────────────────
    history: list[dict] = []
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{cfg.epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp,
        )

        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        elapsed = time.time() - t0

        epoch_metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "elapsed_s": round(elapsed, 1),
        }
        history.append(epoch_metrics)

        print(
            f"  loss {train_metrics['train_loss']:.4f} | "
            f"train_acc {train_metrics['train_acc']:.2f}% | "
            f"val_acc {val_metrics['val_top1']:.2f}% | "
            f"val_top5 {val_metrics['val_top5']:.2f}% | "
            f"{elapsed:.0f}s"
        )

        # Checkpoint
        if epoch % cfg.save_every == 0:
            ckpt_path = cfg.checkpoint_dir / f"{cfg.run_name}_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(cfg),
                "metrics": epoch_metrics,
            }, ckpt_path)

        if val_metrics["val_top1"] > best_val_acc:
            best_val_acc = val_metrics["val_top1"]
            best_path = cfg.checkpoint_dir / f"{cfg.run_name}_best.pt"
            torch.save({"model_state_dict": model.state_dict(), **epoch_metrics}, best_path)
            print(f"  ★ New best val_top1: {best_val_acc:.2f}%  →  {best_path}")

    # ── Save history ─────────────────────────────────────────────────
    history_path = cfg.metrics_dir / f"{cfg.run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {history_path}")


# ── CLI ──────────────────────────────────────────────────────────────

def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a VQA model")
    parser.add_argument("--model-type", choices=["asymmetric", "symmetric"], default="asymmetric")
    parser.add_argument("--preset", choices=["dev", "initial", "full"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    presets = {"dev": DEV_CONFIG, "initial": INITIAL_CONFIG, "full": FULL_CONFIG}
    cfg = presets.get(args.preset, TrainConfig())

    cfg.model_type = args.model_type
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.data_dir is not None:
        cfg.data_dir = Path(args.data_dir)
    if args.no_amp:
        cfg.use_amp = False
    if args.run_name is not None:
        cfg.run_name = args.run_name
    else:
        cfg.__post_init__()  # regenerate default run_name

    return cfg


if __name__ == "__main__":
    train(_parse_args())
