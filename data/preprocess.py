"""Data preprocessing utilities for VQA v2.0."""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_questions(questions_path: str) -> List[dict]:
    """Load VQA questions from JSON file."""
    with open(questions_path, "r") as f:
        data = json.load(f)
    return data.get("questions", data) if isinstance(data, dict) else data


def load_answers(answers_path: str) -> List[dict]:
    """Load VQA answers from JSON file."""
    with open(answers_path, "r") as f:
        data = json.load(f)
    return data.get("annotations", data) if isinstance(data, dict) else data


def build_answer_vocab(
    answers_path: str,
    top_k: int = 1000,
    min_count: int = 1,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build vocabulary from answer annotations (top-K most frequent answers).
    Returns (answer_to_idx, idx_to_answer).
    """
    answers_data = load_answers(answers_path)
    counter: Counter = Counter()
    for ann in answers_data:
        # VQA v2 format: list of dicts with 'answer' key
        ans_list = ann.get("answers", [])
        if isinstance(ans_list, list):
            for a in ans_list:
                ans_text = a.get("answer", a) if isinstance(a, dict) else str(a)
                counter[ans_text.strip().lower()] += 1
        else:
            counter[str(ans_list).strip().lower()] += 1

    # Top-K by frequency
    most_common = [a for a, _ in counter.most_common(top_k) if counter[a] >= min_count]
    answer_to_idx: Dict[str, int] = {a: i for i, a in enumerate(most_common)}
    idx_to_answer: Dict[int, str] = {i: a for a, i in answer_to_idx.items()}
    return answer_to_idx, idx_to_answer


def get_top_k_answers(
    answers_path: str,
    k: int = 1000,
) -> List[str]:
    """Return list of top-K answer strings in order of frequency."""
    answer_to_idx, _ = build_answer_vocab(answers_path, top_k=k)
    return list(answer_to_idx.keys())


def merge_questions_answers(
    questions: List[dict],
    answers: List[dict],
    answer_to_idx: Dict[str, int],
) -> List[dict]:
    """
    Merge questions and answers by question_id; filter to samples with valid (top-K) answers.
    """
    # Index answers by question_id
    qid_to_answers: Dict[int, list] = {}
    for ann in answers:
        qid = ann.get("question_id", ann.get("question_id"))
        ans_list = ann.get("answers", [])
        if isinstance(ans_list, list):
            texts = [
                (a.get("answer", a) if isinstance(a, dict) else str(a)).strip().lower()
                for a in ans_list
            ]
        else:
            texts = [str(ans_list).strip().lower()]
        qid_to_answers[qid] = texts

    merged = []
    for q in questions:
        qid = q.get("question_id", q.get("id"))
        image_id = q.get("image_id", q.get("image_id"))
        question = q.get("question", q.get("question_text", ""))
        ans_list = qid_to_answers.get(qid, [])
        # Majority vote or first that is in vocab
        best_answer = None
        best_count = 0
        for a in ans_list:
            if a in answer_to_idx:
                count = ans_list.count(a)
                if count > best_count:
                    best_count = count
                    best_answer = a
        if best_answer is not None:
            merged.append({
                "question_id": qid,
                "image_id": image_id,
                "question": question,
                "answer": best_answer,
                "answer_idx": answer_to_idx[best_answer],
            })
    return merged


def create_image_id_to_path_map(
    image_dir: Path,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Dict[int, str]:
    """Build mapping from VQA image_id to file path. Assumes COCO naming: COCO_*_<image_id>.jpg."""
    image_id_to_path: Dict[int, str] = {}
    for ext in extensions:
        for p in image_dir.rglob(f"*{ext}"):
            try:
                # COCO: COCO_train2014_000000XXXXXX.jpg
                stem = p.stem
                if "_" in stem:
                    parts = stem.split("_")
                    num_part = parts[-1]
                    image_id_to_path[int(num_part)] = str(p)
            except (ValueError, IndexError):
                continue
    return image_id_to_path
