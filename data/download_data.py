"""Download and organise the VQA v2.0 dataset.

Run once to populate the ``data/`` directory with questions, annotations,
and (optionally) MS-COCO images.  Image downloads are gated behind
``--download-images`` because they are ~19 GB total.

Usage (from project root)::

    python -m data.download_data --data-dir data
    python -m data.download_data --data-dir data --download-images
"""

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

# ── VQA v2.0 download URLs ──────────────────────────────────────────
QUESTIONS = {
    "train": "https://s3.amazonaws.com/cvqa/v2_Questions_Train_mscoco.zip",
    "val": "https://s3.amazonaws.com/cvqa/v2_Questions_Val_mscoco.zip",
}
ANNOTATIONS = {
    "train": "https://s3.amazonaws.com/cvqa/v2_Annotations_Train_mscoco.zip",
    "val": "https://s3.amazonaws.com/cvqa/v2_Annotations_Val_mscoco.zip",
}
IMAGES = {
    "train": "http://images.cocodataset.org/zips/train2014.zip",
    "val": "http://images.cocodataset.org/zips/val2014.zip",
}


class _DownloadProgressBar(tqdm):
    """tqdm wrapper for ``urllib`` reporthook."""

    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress bar, skipping if it exists."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract *zip_path* into *dest_dir*, then delete the zip."""
    print(f"  Extracting {zip_path.name} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()


def _download_and_extract(urls: dict[str, str], dest_dir: Path) -> None:
    for split, url in urls.items():
        filename = url.rsplit("/", 1)[-1]
        zip_path = dest_dir / filename
        _download(url, zip_path)
        if zip_path.exists():
            _extract_zip(zip_path, dest_dir)


def _organise(data_dir: Path) -> None:
    """Move extracted files into the canonical directory structure.

    Expected layout after this function::

        data/
        ├── questions/
        │   ├── v2_OpenEnded_mscoco_train2014_questions.json
        │   └── v2_OpenEnded_mscoco_val2014_questions.json
        ├── answers/
        │   ├── v2_mscoco_train2014_annotations.json
        │   └── v2_mscoco_val2014_annotations.json
        └── images/
            ├── train2014/
            └── val2014/
    """
    q_dir = data_dir / "questions"
    a_dir = data_dir / "answers"
    i_dir = data_dir / "images"
    for d in (q_dir, a_dir, i_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Move question JSON files
    for f in data_dir.glob("v2_OpenEnded_mscoco_*_questions.json"):
        f.rename(q_dir / f.name)

    # Move annotation JSON files
    for f in data_dir.glob("v2_mscoco_*_annotations.json"):
        f.rename(a_dir / f.name)

    # Move image directories
    for split in ("train2014", "val2014"):
        src = data_dir / split
        if src.exists():
            src.rename(i_dir / split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VQA v2.0 dataset")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--download-images", action="store_true",
        help="Also download MS-COCO images (~19 GB). Omit for questions/answers only.",
    )
    args = parser.parse_args()
    data_dir: Path = args.data_dir

    print("=== Downloading VQA v2.0 Questions ===")
    _download_and_extract(QUESTIONS, data_dir)

    print("\n=== Downloading VQA v2.0 Annotations ===")
    _download_and_extract(ANNOTATIONS, data_dir)

    if args.download_images:
        print("\n=== Downloading MS-COCO Images (this will take a while) ===")
        _download_and_extract(IMAGES, data_dir)

    print("\n=== Organising files ===")
    _organise(data_dir)
    print("Done!  Dataset ready at:", data_dir.resolve())


if __name__ == "__main__":
    main()
