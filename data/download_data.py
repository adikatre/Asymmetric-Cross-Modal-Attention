"""
Script to download and organize VQA v2.0 data.
See https://visualqa.org/download.html for official links.

Usage:
  python -m data.download_data --data_dir ./data --subset train val

For Colab: download manually or use gdown/wget for the JSON and image zip files.
"""

import argparse
import os
from pathlib import Path


# Official VQA v2.0 URLs (as of project proposal)
VQA_V2_ANNOTATIONS = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
VQA_V2_QUESTIONS = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
# COCO images: train2014, val2014 from MS COCO dataset
COCO_TRAIN = "http://images.cocodataset.org/zips/train2014.zip"
COCO_VAL = "http://images.cocodataset.org/zips/val2014.zip"


def main():
    parser = argparse.ArgumentParser(description="Download VQA v2.0 data")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    parser.add_argument(
        "--subset",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="Which splits to prepare",
    )
    parser.add_argument("--skip_download", action="store_true", help="Only create dir structure")
    args = parser.parse_args()

    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    (root / "questions").mkdir(exist_ok=True)
    (root / "answers").mkdir(exist_ok=True)

    print("VQA v2.0 data directory structure created at:", root.resolve())
    print()
    print("To download the full dataset manually:")
    print("  1. Questions: ", VQA_V2_QUESTIONS)
    print("  2. Annotations: ", VQA_V2_ANNOTATIONS)
    print("  3. COCO train images: ", COCO_TRAIN)
    print("  4. COCO val images: ", COCO_VAL)
    print()
    print("Extract so that:")
    print("  data/questions/ contains train_questions.json, val_questions.json")
    print("  data/answers/ contains train_answers.json (or annotations), val_answers.json")
    print("  data/images/train2014/ and data/images/val2014/ contain COCO images")
    print()
    if not args.skip_download:
        print("Automatic download not implemented (large files). Use the URLs above or Kaggle/Colab.")
    return 0


if __name__ == "__main__":
    exit(main())
