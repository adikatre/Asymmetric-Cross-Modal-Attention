"""Training loop for symmetric and asymmetric VQA models."""

import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.config import DataConfig, ModelConfig, TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        images = batch[0].to(device)
        input_ids = batch[1]["input_ids"].to(device)
        attention_mask = batch[1]["attention_mask"].to(device)
        answers = batch[2].to(device)

        optimizer.zero_grad()
        logits = model(images, input_ids, attention_mask)
        loss = criterion(logits, answers)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n if n else 0.0


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[TrainConfig] = None,
    checkpoint_dir: Optional[str] = None,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Full training loop. Saves checkpoints each epoch if checkpoint_dir is set.
    Returns dict with train_losses, val_accs (if val_loader provided), and best_accuracy.
    """
    if config is None:
        config = TrainConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = Path(checkpoint_dir or config.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_accs = []
    best_acc = 0.0

    for epoch in range(config.epochs):
        loss = train_epoch(model, train_loader, optimizer, device, criterion)
        train_losses.append(loss)

        if val_loader is not None:
            from training.evaluate import evaluate
            acc = evaluate(model, val_loader, device)
            val_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
            print(f"Epoch {epoch + 1}/{config.epochs}  train_loss={loss:.4f}  val_acc={acc:.2%}")
        else:
            print(f"Epoch {epoch + 1}/{config.epochs}  train_loss={loss:.4f}")

        if config.save_every_epoch and checkpoint_path:
            p = checkpoint_path / f"{model_name}_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": loss,
                "config": config,
            }, p)
    return {
        "train_losses": train_losses,
        "val_accs": val_accs,
        "best_accuracy": best_acc,
    }
