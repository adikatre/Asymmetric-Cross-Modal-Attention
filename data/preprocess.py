"""VQA v2.0 dataset and preprocessing utilities.

The main export is :class:`VQADataset`, a PyTorch ``Dataset`` that yields
``(image, input_ids, attention_mask, answer_idx)`` tuples ready for training.

Helper functions are provided for:
* building the top-K answer vocabulary,
* creating standard image transforms, and
* building train / val dataloaders in one call.
"""

import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import RobertaTokenizer


# ── Public helpers ───────────────────────────────────────────────────

def build_answer_vocab(
    annotations_file: str | Path,
    top_k: int = 1000,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build an answer vocabulary from the *top_k* most frequent answers.

    Returns:
        answer_to_idx: mapping from answer string to class index.
        idx_to_answer: inverse mapping (index → string).
    """
    with open(annotations_file) as f:
        annotations = json.load(f)["annotations"]

    counter: Counter[str] = Counter()
    for ann in annotations:
        counter[ann["multiple_choice_answer"]] += 1

    most_common = [ans for ans, _ in counter.most_common(top_k)]
    answer_to_idx = {ans: idx for idx, ans in enumerate(most_common)}
    idx_to_answer = {idx: ans for ans, idx in answer_to_idx.items()}
    return answer_to_idx, idx_to_answer


def get_image_transform(split: str = "train") -> transforms.Compose:
    """Standard image transform for ViT-B/16 (224 × 224, ImageNet norm)."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ── Dataset ──────────────────────────────────────────────────────────

class VQADataset(Dataset):
    """PyTorch dataset for VQA v2.0.

    Each item is a tuple of::

        (image_tensor, input_ids, attention_mask, answer_idx)

    where ``answer_idx`` is the index into the top-K answer vocabulary.

    Args:
        questions_file:  Path to the VQA questions JSON.
        annotations_file: Path to the VQA annotations JSON.
        image_dir:       Directory containing COCO images (e.g. ``data/images/train2014``).
        answer_to_idx:   Pre-built answer vocabulary (from :func:`build_answer_vocab`).
                         If ``None``, one is built from *annotations_file*.
        top_k_answers:   Number of answer classes (only used when building vocab).
        max_question_len: Maximum token length for questions.
        transform:       Image transform pipeline.  Defaults to val-style if not given.
        max_samples:     If set, only use the first *max_samples* valid pairs
                         (useful for quick debugging runs).
    """

    def __init__(
        self,
        questions_file: str | Path,
        annotations_file: str | Path,
        image_dir: str | Path,
        answer_to_idx: dict[str, int] | None = None,
        top_k_answers: int = 1000,
        max_question_len: int = 20,
        transform: transforms.Compose | None = None,
        max_samples: int | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.max_question_len = max_question_len
        self.transform = transform or get_image_transform("val")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # Build or reuse answer vocabulary
        if answer_to_idx is None:
            self.answer_to_idx, self.idx_to_answer = build_answer_vocab(
                annotations_file, top_k_answers,
            )
        else:
            self.answer_to_idx = answer_to_idx
            self.idx_to_answer = {v: k for k, v in answer_to_idx.items()}

        # Load questions and annotations
        with open(questions_file) as f:
            questions_data = json.load(f)["questions"]
        with open(annotations_file) as f:
            annotations_data = json.load(f)["annotations"]

        # Index annotations by question_id for fast look-up
        ann_by_qid = {ann["question_id"]: ann for ann in annotations_data}

        # Keep only samples whose answer is in the top-K vocab
        self.samples: list[dict] = []
        for q in questions_data:
            qid = q["question_id"]
            ann = ann_by_qid.get(qid)
            if ann is None:
                continue
            answer = ann["multiple_choice_answer"]
            if answer not in self.answer_to_idx:
                continue
            self.samples.append({
                "question": q["question"],
                "image_id": q["image_id"],
                "answer_idx": self.answer_to_idx[answer],
            })
            if max_samples is not None and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def _image_path(self, image_id: int) -> Path:
        """Resolve the COCO image filename from an image id."""
        # COCO filenames: COCO_{split}_{id:012d}.jpg
        # Determine the split from the image_dir name.
        split = self.image_dir.name  # e.g. "train2014"
        return self.image_dir / f"COCO_{split}_{image_id:012d}.jpg"

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Image
        img_path = self._image_path(sample["image_id"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Question
        encoding = self.tokenizer(
            sample["question"],
            max_length=self.max_question_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)           # (seq_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (seq_len,)

        # Answer
        answer_idx = torch.tensor(sample["answer_idx"], dtype=torch.long)

        return image, input_ids, attention_mask, answer_idx


# ── Convenience builder ──────────────────────────────────────────────

def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    top_k_answers: int = 1000,
    max_question_len: int = 20,
    max_samples: int | None = None,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, dict[str, int], dict[int, str]]:
    """Build train and validation dataloaders in one call.

    Returns:
        train_loader, val_loader, answer_to_idx, idx_to_answer
    """
    data_dir = Path(data_dir)

    train_ann = data_dir / "answers" / "v2_mscoco_train2014_annotations.json"
    val_ann = data_dir / "answers" / "v2_mscoco_val2014_annotations.json"
    train_q = data_dir / "questions" / "v2_OpenEnded_mscoco_train2014_questions.json"
    val_q = data_dir / "questions" / "v2_OpenEnded_mscoco_val2014_questions.json"
    train_img = data_dir / "images" / "train2014"
    val_img = data_dir / "images" / "val2014"

    # Build vocab from training annotations
    answer_to_idx, idx_to_answer = build_answer_vocab(train_ann, top_k_answers)

    train_ds = VQADataset(
        questions_file=train_q,
        annotations_file=train_ann,
        image_dir=train_img,
        answer_to_idx=answer_to_idx,
        max_question_len=max_question_len,
        transform=get_image_transform("train"),
        max_samples=max_samples,
    )
    val_ds = VQADataset(
        questions_file=val_q,
        annotations_file=val_ann,
        image_dir=val_img,
        answer_to_idx=answer_to_idx,
        max_question_len=max_question_len,
        transform=get_image_transform("val"),
        max_samples=max_samples,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, answer_to_idx, idx_to_answer
