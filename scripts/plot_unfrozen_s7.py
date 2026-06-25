"""Generate training curves for the unfrozen_s7 asymmetric vs symmetric runs.

Uses plot_results.plot_training_curves from the original visualization script.
That function expects ``val_top1`` / ``val_top5`` keys, so we remap this run's
``val_vqa_acc`` / ``val_vqa_acc_top1000`` onto them before plotting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plot_results import load_history, plot_training_curves  # noqa: E402

RESULTS = Path("RESULTS/unfrozen_s7/metrics")
RUNS = {
    "Asymmetric": RESULTS / "clipL336_vqaScore_top3000_s42_asymmetric_history.json",
    "Symmetric": RESULTS / "clipL336_vqaScore_top3000_s42_symmetric_history.json",
}
OUT = Path("RESULTS/unfrozen_s7/training_curves.png")


def remap(hist):
    """Map this run's VQA accuracy keys onto the names plot_results expects."""
    for h in hist:
        h["val_top1"] = h["val_vqa_acc"]
        h["val_top5"] = h["val_vqa_acc_top1000"]
    return hist


def main():
    histories = {name: remap(load_history(p)) for name, p in RUNS.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plot_training_curves(histories, save_path=OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
