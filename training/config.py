"""Training and model hyperparameter configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    image_dir: str = "data/images"
    train_questions: str = "data/questions/train_questions.json"
    train_answers: str = "data/answers/train_annotations.json"
    val_questions: str = "data/questions/val_questions.json"
    val_answers: str = "data/answers/val_annotations.json"
    top_k_answers: int = 1000
    max_question_length: int = 64
    image_size: int = 224
    tokenizer_name: str = "roberta-base"
    subset_size: Optional[int] = None  # e.g. 1000 for debug, None for full


@dataclass
class ModelConfig:
    embed_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.3


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 20
    num_workers: int = 4
    save_every_epoch: bool = True
    checkpoint_dir: str = "results/checkpoints"
    seed: int = 42


def get_config(
    subset_size: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    """Return (DataConfig, ModelConfig, TrainConfig)."""
    data = DataConfig(subset_size=subset_size)
    model = ModelConfig()
    train = TrainConfig()
    if checkpoint_dir:
        train.checkpoint_dir = checkpoint_dir
    return data, model, train
