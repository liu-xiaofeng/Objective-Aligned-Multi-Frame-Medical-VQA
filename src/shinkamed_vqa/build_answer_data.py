from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .dataset_utils import dump_json, dump_jsonl, load_json, load_manifest, resolve_example_frame_paths
except ImportError:
    from dataset_utils import dump_json, dump_jsonl, load_json, load_manifest, resolve_example_frame_paths


ANSWER_PROMPT_PREFIX = (
    "You are answering a multi-image medical VQA question directly. "
    "Return JSON only with keys answer_letter, confidence, rationale, option_scores. "
    "answer_letter must be one capital letter that matches the best option.\n"
)


def _extract_brief_reasoning(text: str, max_words: int = 40) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)[0].strip()
    words = sentence.split()
    if len(words) > max_words:
        sentence = " ".join(words[:max_words]).rstrip(",;:") + "..."
    return sentence


def _smooth_option_scores(correct_answer: str, options: list[str], peak: float = 0.85) -> list[float]:
    num_options = max(len(options), 1)
    scores = [(1.0 - peak) / max(num_options - 1, 1)] * num_options
    answer_letter = str(correct_answer or "").strip().upper()[:1]
    if "A" <= answer_letter <= chr(ord("A") + num_options - 1):
        scores[ord(answer_letter) - ord("A")] = peak
    return [round(float(score), 4) for score in scores]


def _minimal_rationale(example: dict[str, Any]) -> str:
    organ = str(example.get("organ", "") or "").strip()
    modality = str(example.get("modality", "") or "").strip()
    brief = _extract_brief_reasoning(str(example.get("reasoning_chain", "") or ""))
    if organ and modality:
        prefix = f"The correct option is best supported by the multi-image {modality} evidence from the {organ}."
    else:
        prefix = "The correct option is best supported by the multi-image medical evidence."
    return f"{prefix} {brief}".strip()


def _instruction_text(example: dict[str, Any]) -> str:
    return (
        ANSWER_PROMPT_PREFIX
        + f"Question: {example.get('question', '')}\n"
        + f"Options: {list(example.get('options', []))}"
    )


def build_answer_data(
    *,
    manifest_path: Path,
    split_ids_path: Path,
    output_train_path: Path,
    output_val_path: Path,
    output_summary_path: Path,
    max_frames: int,
    seed: int,
    val_size: int,
) -> dict[str, Any]:
    records = load_manifest(manifest_path)
    split_ids = load_json(split_ids_path)
    reserved_eval_ids = set(split_ids.get("search_dev", []))
    reserved_eval_ids.update(split_ids.get("calibration_val", []))
    reserved_eval_ids.update(split_ids.get("report_test", []))
    train_pool = [row for row in records if row.get("question_id") not in reserved_eval_ids]
    rng = random.Random(seed)
    rng.shuffle(train_pool)

    unique_rows: list[dict[str, Any]] = []
    repeated_train_rows: list[dict[str, Any]] = []
    repeat_hist = Counter()

    for example in train_pool:
        frame_paths = resolve_example_frame_paths(example)[:max_frames]
        target = {
            "answer_letter": str(example.get("correct_answer", "A")).strip().upper()[:1] or "A",
            "confidence": 0.85,
            "rationale": _minimal_rationale(example),
            "option_scores": _smooth_option_scores(example.get("correct_answer", "A"), list(example.get("options", []))),
        }
        row = {
            "question_id": example["question_id"],
            "system": example.get("system"),
            "organ": example.get("organ"),
            "keyword": example.get("keyword"),
            "modality": example.get("modality"),
            "question": example.get("question"),
            "options": list(example.get("options", [])),
            "correct_answer": example.get("correct_answer"),
            "reasoning_chain": example.get("reasoning_chain"),
            "frame_paths": frame_paths,
            "instruction_text": _instruction_text(example),
            "target_text": json.dumps(target, ensure_ascii=False, sort_keys=True),
        }
        unique_rows.append(row)
        repeat_hist["1"] += 1
        repeated_train_rows.append(dict(row))

    val_size = min(max(64, int(val_size)), max(64, len(unique_rows) // 5))
    val_rows = unique_rows[:val_size]
    val_ids = {row["question_id"] for row in val_rows}
    train_rows = [row for row in repeated_train_rows if row["question_id"] not in val_ids]

    dump_jsonl(train_rows, output_train_path)
    dump_jsonl(val_rows, output_val_path)
    summary = {
        "mode": "answer_only",
        "pool_examples": len(train_pool),
        "unique_examples": len(unique_rows),
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "repeat_histogram": dict(repeat_hist),
    }
    dump_json(summary, output_summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--split-ids-path", required=True)
    parser.add_argument("--output-train-path", required=True)
    parser.add_argument("--output-val-path", required=True)
    parser.add_argument("--output-summary-path", required=True)
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-size", type=int, default=128)
    args = parser.parse_args()
    summary = build_answer_data(
        manifest_path=Path(args.manifest_path),
        split_ids_path=Path(args.split_ids_path),
        output_train_path=Path(args.output_train_path),
        output_val_path=Path(args.output_val_path),
        output_summary_path=Path(args.output_summary_path),
        max_frames=args.max_frames,
        seed=args.seed,
        val_size=args.val_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
