from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
import pyarrow.parquet as pq

try:
    from .dataset_utils import (
        answer_letter_to_index,
        build_fixed_splits,
        default_paths,
        dump_json,
        dump_jsonl,
    )
except ImportError:
    from dataset_utils import (
        answer_letter_to_index,
        build_fixed_splits,
        default_paths,
        dump_json,
        dump_jsonl,
    )


def extract_images_from_snapshot(snapshot_root: str | Path) -> dict[str, int]:
    snapshot_root = Path(snapshot_root)
    data_dir = snapshot_root / "data"
    written = 0
    skipped = 0
    for parquet_path in sorted(data_dir.glob("*.parquet")):
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            image_urls = list(row.get("image_url") or [])
            for idx, relpath in enumerate(image_urls, start=1):
                image_struct = row.get(f"image_{idx}")
                if not image_struct or not image_struct.get("bytes"):
                    skipped += 1
                    continue
                output_path = snapshot_root / relpath
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not output_path.exists():
                    output_path.write_bytes(image_struct["bytes"])
                    written += 1
    return {"written": written, "skipped": skipped}


def build_manifest(snapshot_root: str | None) -> list[dict]:
    dataset = load_dataset("SuhaoYu1020/MedFrameQA")["test"]
    dataset = dataset.remove_columns(["image_1", "image_2", "image_3", "image_4", "image_5"])

    records = []
    for item in dataset:
        relpaths = list(item["image_url"])
        if snapshot_root:
            frame_paths = [str((Path(snapshot_root) / rel).resolve()) for rel in relpaths]
        else:
            frame_paths = relpaths
        records.append(
            {
                "question_id": item["question_id"],
                "system": item["system"],
                "organ": item["organ"],
                "keyword": item["keyword"],
                "modality": item["modality"],
                "video_id": item["video_id"],
                "question": item["question"],
                "options": list(item["options"]),
                "correct_answer": item["correct_answer"],
                "correct_answer_index": answer_letter_to_index(item["correct_answer"]),
                "reasoning_chain": item["reasoning_chain"],
                "frame_relpaths": relpaths,
                "frame_paths": frame_paths,
            }
        )
    return records


def maybe_snapshot_dataset(skip_snapshot: bool) -> str | None:
    if skip_snapshot:
        return None
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return None
    paths = default_paths()
    snapshot_dir = paths["raw"] / "medframeqa_snapshot"
    return snapshot_download(
        repo_id="SuhaoYu1020/MedFrameQA",
        repo_type="dataset",
        local_dir=snapshot_dir,
        local_dir_use_symlinks=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_snapshot", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    paths = default_paths()
    snapshot_root = maybe_snapshot_dataset(skip_snapshot=args.skip_snapshot)
    extraction_stats = None
    if snapshot_root:
        extraction_stats = extract_images_from_snapshot(snapshot_root)
    records = build_manifest(snapshot_root=snapshot_root)

    manifest_path = paths["processed"] / "medframeqa_manifest.jsonl"
    split_path = paths["processed"] / "medframeqa_split_ids.json"
    summary_path = paths["processed"] / "medframeqa_summary.json"

    dump_jsonl(records, manifest_path)
    split_ids = build_fixed_splits(
        records,
        seed=args.seed,
        mini_debug=32,
        search_dev=256,
        calibration_val=256,
        report_test=1024,
    )
    dump_json(split_ids, split_path)
    dump_json(
        {
            "num_records": len(records),
            "snapshot_root": snapshot_root,
            "extraction_stats": extraction_stats,
            "manifest_path": str(manifest_path),
            "split_path": str(split_path),
        },
        summary_path,
    )
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote split ids to {split_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
