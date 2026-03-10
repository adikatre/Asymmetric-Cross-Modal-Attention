"""PyTorch Dataset for VQA v2.0."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .preprocess import build_answer_vocab, merge_questions_answers


class VQADataset(Dataset):
    """
    VQA v2.0 Dataset: loads image, question, and answer index.
    Uses top-K most frequent answers as classes.
    """

    def __init__(
        self,
        image_dir: str,
        questions_file: str,
        answers_file: str,
        answer_to_idx: Optional[Dict[str, int]] = None,
        top_k_answers: int = 1000,
        max_question_length: int = 64,
        image_size: int = 224,
        transform=None,
        tokenizer_name: str = "roberta-base",
        subset_size: Optional[int] = None,
    ):
        """
        Args:
            image_dir: Root directory containing train2014/ and/or val2014/
            questions_file: Path to questions JSON
            answers_file: Path to answers/annotations JSON
            answer_to_idx: Pre-built answer vocab; if None, built from answers_file
            top_k_answers: Number of answer classes (used only if answer_to_idx is None)
            max_question_length: Max token length for questions
            image_size: Spatial size for resizing images
            transform: Optional torchvision transform (if None, default resize + normalize)
            tokenizer_name: HuggingFace tokenizer for questions
            subset_size: If set, use only first N samples (for debugging)
        """
        self.image_dir = Path(image_dir)
        self.max_question_length = max_question_length
        self.image_size = image_size
        self.transform = transform

        if answer_to_idx is None:
            answer_to_idx, _ = build_answer_vocab(answers_file, top_k=top_k_answers)
        self.answer_to_idx = answer_to_idx
        self.num_answers = len(answer_to_idx)

        with open(questions_file, "r") as f:
            q_data = json.load(f)
        questions = q_data.get("questions", q_data) if isinstance(q_data, dict) else q_data
        answers_data = json.load(open(answers_file))
        self.samples = merge_questions_answers(questions, answers_data, answer_to_idx)

        if subset_size is not None:
            self.samples = self.samples[:subset_size]

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Resolve image paths: support both flat and train2014/val2014 structure
        self.image_paths: Dict[int, str] = {}
        for s in self.samples:
            iid = s["image_id"]
            if iid in self.image_paths:
                continue
            for sub in ("train2014", "val2014", ""):
                d = self.image_dir / sub if sub else self.image_dir
                for fmt in (f"COCO_{{}}_{iid:012d}.jpg", f"{iid}.jpg", f"{iid:012d}.jpg"):
                    p = d / fmt
                    if p.exists():
                        self.image_paths[iid] = str(p)
                        break
                if iid in self.image_paths:
                    break
            if iid not in self.image_paths:
                self.image_paths[iid] = str(self.image_dir / "train2014" / f"COCO_train2014_{iid:012d}.jpg")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
        sample = self.samples[idx]
        image_id = sample["image_id"]
        image_path = self.image_paths.get(image_id)
        if not image_path or not Path(image_path).exists():
            # Return a dummy image if path missing (for runs without full data)
            image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))
        else:
            image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image_tensor = transform(image)

        question = sample["question"]
        encoded = self.tokenizer(
            question,
            padding="max_length",
            max_length=self.max_question_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        answer_idx = sample["answer_idx"]
        return image_tensor, {"input_ids": input_ids, "attention_mask": attention_mask}, answer_idx
