from __future__ import annotations

import json
import os
import pickle
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from sklearn.model_selection import train_test_split


def task_root() -> Path:
    return Path(__file__).resolve().parent


def workspace_root() -> Path:
    env_root = os.environ.get("SHINKAMED_VQA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return task_root().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_paths() -> dict[str, Path]:
    root = workspace_root()
    return {
        "raw": ensure_dir(root / "data" / "raw"),
        "processed": ensure_dir(root / "data" / "processed"),
        "results": ensure_dir(root / "results"),
    }


def load_runtime_config(config_path: str | Path) -> dict[str, Any]:
    resolved_config_path = Path(config_path).expanduser().resolve()
    with open(resolved_config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(resolved_config_path)
    cfg["_config_dir"] = str(resolved_config_path.parent)
    return cfg


def resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots: list[Path] = []
    if base_dir is not None:
        search_roots.append(Path(base_dir).expanduser().resolve())
    search_roots.append(workspace_root())

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (search_roots[0] / candidate).resolve()


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def dump_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def dump_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_answer_letter(raw: Any, num_options: int | None = None) -> str:
    def _is_valid(letter: str) -> bool:
        if len(letter) != 1 or not ("A" <= letter <= "Z"):
            return False
        if num_options is None:
            return True
        return 0 <= (ord(letter) - ord("A")) < int(num_options)

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            index = int(raw)
        except Exception:
            index = -1
        if num_options is None:
            if 0 <= index < 26:
                return chr(ord("A") + index)
        elif 0 <= index < int(num_options):
            return chr(ord("A") + index)

    text = str(raw or "").strip().upper()
    if not text:
        return ""
    if _is_valid(text):
        return text

    patterns = [
        r"\b(?:OPTION|ANSWER|CHOICE)\s*[:\-]?\s*\(?([A-Z])\)?\b",
        r"^\(?([A-Z])\)?(?:[\.\:\)])?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip().upper()
            if _is_valid(candidate):
                return candidate

    for candidate in re.findall(r"\b([A-Z])\b", text):
        candidate = candidate.strip().upper()
        if _is_valid(candidate):
            return candidate
    return ""


def answer_letter_to_index(letter: str) -> int:
    normalized = normalize_answer_letter(letter)
    if not normalized:
        return -1
    return ord(normalized) - ord("A")


def answer_index_to_letter(index: int) -> str:
    return chr(ord("A") + index)


def resolve_answer_choice(raw_index: Any, raw_letter: Any, num_options: int) -> tuple[int, str]:
    try:
        if raw_index is not None and str(raw_index).strip() != "":
            index = int(raw_index)
        else:
            index = -1
    except Exception:
        index = -1
    if 0 <= index < num_options:
        return index, answer_index_to_letter(index)

    letter = normalize_answer_letter(raw_letter, num_options=num_options)
    index = answer_letter_to_index(letter)
    if 0 <= index < num_options:
        return index, letter
    return -1, ""


def make_stratify_label(record: dict[str, Any]) -> str:
    frame_count = len(record.get("frame_relpaths", []))
    return "|".join(
        [
            str(record.get("system", "unknown")),
            str(record.get("modality", "unknown")),
            str(frame_count),
        ]
    )


def _make_safe_labels(labels: list[str]) -> list[str]:
    counts = Counter(labels)
    return [label if counts[label] >= 2 else "__rare__" for label in labels]


def build_fixed_splits(
    records: list[dict[str, Any]],
    *,
    seed: int,
    mini_debug: int,
    search_dev: int,
    calibration_val: int,
    report_test: int,
) -> dict[str, list[str]]:
    ids = [record["question_id"] for record in records]
    labels = [make_stratify_label(record) for record in records]
    index = list(range(len(records)))

    if len(records) < mini_debug + search_dev + calibration_val + report_test:
        raise ValueError("Not enough MedFrameQA records for the requested split sizes.")

    safe_labels = _make_safe_labels(labels)
    search_pool_size = mini_debug + search_dev + calibration_val
    search_idx, report_idx = train_test_split(
        index,
        train_size=search_pool_size,
        test_size=report_test,
        random_state=seed,
        stratify=safe_labels,
    )

    rng = random.Random(seed + 1)
    shuffled_search = list(search_idx)
    rng.shuffle(shuffled_search)
    mini_idx = shuffled_search[:mini_debug]
    rest_idx = shuffled_search[mini_debug:]
    rest_labels = _make_safe_labels([labels[i] for i in rest_idx])
    search_dev_idx, calibration_idx = train_test_split(
        rest_idx,
        train_size=search_dev,
        test_size=calibration_val,
        random_state=seed + 2,
        stratify=rest_labels,
    )

    return {
        "mini_debug": [ids[i] for i in sorted(mini_idx)],
        "search_dev": [ids[i] for i in sorted(search_dev_idx)],
        "calibration_val": [ids[i] for i in sorted(calibration_idx)],
        "report_test": [ids[i] for i in sorted(report_idx)],
    }


def manifest_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["question_id"]: record for record in records}


def load_split_examples(manifest_path: str | Path, split_ids_path: str | Path, split_name: str) -> list[dict[str, Any]]:
    records = load_manifest(manifest_path)
    split_ids = load_json(split_ids_path)
    selected_ids = split_ids[split_name]
    by_id = manifest_index(records)
    return [by_id[qid] for qid in selected_ids]


def resolve_frame_path(path_str: str, relpath: str | None = None) -> str:
    path = Path(path_str)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        root = workspace_root()
        candidates.extend([
            root / path,
            root / "data" / "raw" / "medframeqa_snapshot" / path,
        ])
    if relpath:
        rel = Path(relpath)
        root = workspace_root()
        candidates.extend([
            root / rel,
            root / "data" / "raw" / "medframeqa_snapshot" / rel,
        ])
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate.resolve())
    tried = [str(candidate) for candidate in candidates]
    raise FileNotFoundError(
        f"Could not resolve frame path for '{path_str}'"
        + (f" (relpath='{relpath}')" if relpath else "")
        + f". Tried: {tried}"
    )


def resolve_example_frame_paths(example: dict[str, Any]) -> list[str]:
    frame_paths = list(example.get("frame_paths", []))
    frame_relpaths = list(example.get("frame_relpaths", []))
    resolved: list[str] = []
    for idx, frame_path in enumerate(frame_paths):
        relpath = frame_relpaths[idx] if idx < len(frame_relpaths) else None
        resolved.append(resolve_frame_path(frame_path, relpath=relpath))
    return resolved
