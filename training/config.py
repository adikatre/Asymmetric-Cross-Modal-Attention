"""Centralised hyperparameter configuration for all training runs."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # ── Model ────────────────────────────────────────────────────────
    model_type: str = "asymmetric"          # "asymmetric" or "symmetric"
    num_answers: int = 1000
    embed_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.3
    freeze_encoders: bool = True

    # ── Data ─────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    max_question_len: int = 20
    max_samples: int | None = None          # None = use full dataset

    # ── Training ─────────────────────────────────────────────────────
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 20
    num_workers: int = 4
    seed: int = 42

    # ── Checkpointing / logging ──────────────────────────────────────
    checkpoint_dir: Path = Path("results/checkpoints")
    metrics_dir: Path = Path("results/metrics")
    save_every: int = 1                     # save a checkpoint every N epochs
    run_name: str = ""                      # auto-filled if empty

    # ── Mixed precision ──────────────────────────────────────────────
    use_amp: bool = True

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.metrics_dir = Path(self.metrics_dir)
        self.data_dir = Path(self.data_dir)
        if not self.run_name:
            self.run_name = f"{self.model_type}_s{self.seed}"


# ── Preset configs ───────────────────────────────────────────────────

DEV_CONFIG = TrainConfig(
    max_samples=1_000,
    epochs=5,
    batch_size=16,
    run_name="dev",
)

INITIAL_CONFIG = TrainConfig(
    max_samples=50_000,
    epochs=10,
)

FULL_CONFIG = TrainConfig(
    max_samples=None,
    epochs=20,
)
