from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from .build_answer_data import build_answer_data
    from .dataset_utils import dump_json, load_runtime_config, resolve_path
    from .posthoc_eval import run_posthoc_eval
    from .train_lora import train_localization_lora
except ImportError:
    from build_answer_data import build_answer_data
    from dataset_utils import dump_json, load_runtime_config, resolve_path
    from posthoc_eval import run_posthoc_eval
    from train_lora import train_localization_lora


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def run_direct_pipeline(
    *,
    runtime_config: Path,
    train_template: Path,
    results_root: Path,
    seed: int,
    max_frames: int,
    val_size: int,
    target_strategy: str,
    lora_r: int,
    lora_alpha: int,
    num_train_epochs: float,
    learning_rate: float,
    gradient_accumulation_steps: int,
    init_adapter_path: str | None,
    calibration_examples: int,
    report_examples: int,
) -> dict[str, Any]:
    results_root = resolve_path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    runtime_cfg = load_runtime_config(resolve_path(runtime_config))
    config_dir = runtime_cfg.get("_config_dir")
    manifest_path = resolve_path(runtime_cfg["data"]["manifest_path"], base_dir=config_dir)
    split_ids_path = resolve_path(runtime_cfg["data"]["split_ids_path"], base_dir=config_dir)

    train_jsonl = results_root / "train.jsonl"
    val_jsonl = results_root / "val.jsonl"
    data_summary_path = results_root / "data_summary.json"
    adapter_dir = results_root / "adapter"
    eval_root = results_root / "posthoc_eval"
    runtime_eval_cfg_path = results_root / "runtime_eval.yaml"

    data_summary = build_answer_data(
        manifest_path=manifest_path,
        split_ids_path=split_ids_path,
        output_train_path=train_jsonl,
        output_val_path=val_jsonl,
        output_summary_path=data_summary_path,
        max_frames=int(max_frames),
        seed=int(seed),
        val_size=int(val_size),
    )

    train_template_path = resolve_path(train_template)
    train_cfg = yaml.safe_load(train_template_path.read_text())
    backend_cfg = runtime_cfg["task_backend"]
    train_cfg["model"]["family"] = str(backend_cfg.get("family", train_cfg["model"].get("family", "medgemma")))
    train_cfg["model"]["model_name"] = str(backend_cfg.get("model_name", train_cfg["model"].get("model_name")))
    train_cfg["model"]["trust_remote_code"] = bool(backend_cfg.get("trust_remote_code", train_cfg["model"].get("trust_remote_code", False)))
    train_cfg["model"]["local_files_only"] = bool(backend_cfg.get("local_files_only", train_cfg["model"].get("local_files_only", False)))
    train_cfg["data"]["train_path"] = str(train_jsonl)
    train_cfg["data"]["val_path"] = str(val_jsonl)
    train_cfg["data"]["max_frames"] = int(max_frames)
    train_cfg["model"]["max_frames"] = int(max_frames)
    train_cfg["training"]["output_dir"] = str(adapter_dir)
    train_cfg["lora"]["target_strategy"] = str(target_strategy)
    train_cfg["lora"]["r"] = int(lora_r)
    train_cfg["lora"]["alpha"] = int(lora_alpha)
    train_cfg["training"]["num_train_epochs"] = float(num_train_epochs)
    train_cfg["training"]["learning_rate"] = float(learning_rate)
    train_cfg["training"]["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
    train_cfg["training"]["gradient_checkpointing"] = True
    train_cfg["training"]["init_adapter_path"] = str(resolve_path(init_adapter_path)) if init_adapter_path else None
    train_cfg_path = results_root / "train_config.yaml"
    _write_yaml(train_cfg_path, train_cfg)

    train_summary = train_localization_lora(str(train_cfg_path))

    runtime_cfg.pop("_config_path", None)
    runtime_cfg.pop("_config_dir", None)
    runtime_eval_cfg = json.loads(json.dumps(runtime_cfg))
    runtime_eval_cfg["task_backend"]["max_frames"] = int(max_frames)
    _write_yaml(runtime_eval_cfg_path, runtime_eval_cfg)

    posthoc_summary = run_posthoc_eval(
        runtime_config=runtime_eval_cfg_path,
        results_root=eval_root,
        adapter_dirs=[adapter_dir],
        calibration_examples=int(calibration_examples),
        report_examples=int(report_examples),
    )

    rows = posthoc_summary.get("rows", [])
    by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], {})[row["split_name"]] = row
    base_rows = by_variant["vanilla_direct_base"]
    adapter_variant = next(key for key in by_variant if key != "vanilla_direct_base")
    adapter_rows = by_variant[adapter_variant]

    summary = {
        "recipe": "answer_only",
        "data_summary": data_summary,
        "train_summary": train_summary,
        "init_adapter_path": str(resolve_path(init_adapter_path)) if init_adapter_path else None,
        "posthoc_summary_path": str((eval_root / "posthoc_eval_summary.json").resolve()),
        "adapter_variant": adapter_variant,
        "deltas": {
            "calibration_accuracy": adapter_rows["calibration_val"]["accuracy"] - base_rows["calibration_val"]["accuracy"],
            "report_accuracy": adapter_rows["report_test"]["accuracy"] - base_rows["report_test"]["accuracy"],
        },
    }
    dump_json(summary, results_root / "pipeline_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--train-template", required=True)
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--target-strategy", required=True)
    parser.add_argument("--lora-r", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument("--num-train-epochs", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--init-adapter-path")
    parser.add_argument("--calibration-examples", type=int, default=256)
    parser.add_argument("--report-examples", type=int, default=1024)
    args = parser.parse_args()

    summary = run_direct_pipeline(
        runtime_config=Path(args.runtime_config),
        train_template=Path(args.train_template),
        results_root=Path(args.results_root),
        seed=int(args.seed),
        max_frames=int(args.max_frames),
        val_size=int(args.val_size),
        target_strategy=str(args.target_strategy),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        num_train_epochs=float(args.num_train_epochs),
        learning_rate=float(args.learning_rate),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        init_adapter_path=args.init_adapter_path,
        calibration_examples=int(args.calibration_examples),
        report_examples=int(args.report_examples),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
