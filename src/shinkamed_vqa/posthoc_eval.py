from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

try:
    from .dataset_utils import load_runtime_config, resolve_path, task_root
    from .evaluate import evaluate_program
except ImportError:
    from dataset_utils import load_runtime_config, resolve_path, task_root
    from evaluate import evaluate_program


TASK_ROOT = task_root()


def _variant_name_for_adapter(adapter_path: Path) -> str:
    if adapter_path.name.startswith("checkpoint-") and adapter_path.parent.name == "adapter":
        return f"{adapter_path.parent.parent.name}_{adapter_path.name}"
    if adapter_path.name == "adapter":
        return adapter_path.parent.name
    return adapter_path.stem


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _plain_eval_config(base_cfg: dict[str, Any], adapter_mode: str, adapter_path: str | None) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["task_backend"]["adapter_mode"] = adapter_mode
    cfg["task_backend"]["adapter_path"] = adapter_path
    return cfg


def run_posthoc_eval(
    *,
    runtime_config: Path,
    results_root: Path,
    adapter_dirs: list[Path],
    calibration_examples: int,
    report_examples: int,
) -> dict[str, Any]:
    runtime_template = load_runtime_config(resolve_path(runtime_config))
    runtime_template.pop("_config_path", None)
    runtime_template.pop("_config_dir", None)
    results_root = resolve_path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, dict[str, Any]]] = [
        ("vanilla_direct_base", _plain_eval_config(runtime_template, "base", None))
    ]
    for adapter_dir in adapter_dirs:
        adapter_path = resolve_path(adapter_dir)
        variants.append(
            (
                f"vanilla_direct_{_variant_name_for_adapter(adapter_path)}",
                _plain_eval_config(runtime_template, "localization_lora", str(adapter_path)),
            )
        )

    split_plan = [
        ("calibration_val", calibration_examples),
        ("report_test", report_examples),
    ]

    summary: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_config": str(resolve_path(runtime_config)),
        "adapter_dirs": [str(resolve_path(p)) for p in adapter_dirs],
        "rows": [],
        "deltas": {},
    }

    with tempfile.TemporaryDirectory(prefix="public_posthoc_eval_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for variant_name, cfg in variants:
            cfg_path = tmpdir_path / f"{variant_name}.yaml"
            for split_name, max_examples in split_plan:
                cfg["evaluation"]["max_examples"] = int(max_examples)
                _write_yaml(cfg_path, cfg)
                run_dir = results_root / variant_name / split_name
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                metrics = evaluate_program(
                    program_path=TASK_ROOT / "programs" / "vanilla_direct.py",
                    results_dir=run_dir,
                    config_path=cfg_path,
                    split_name=split_name,
                )
                public = metrics.get("public", {})
                summary["rows"].append(
                    {
                        "variant": variant_name,
                        "split_name": split_name,
                        "accuracy": float(public.get("accuracy", 0.0) or 0.0),
                        "ece": float(public.get("ece", 0.0) or 0.0),
                        "avg_runtime_sec": float(public.get("avg_runtime_sec", 0.0) or 0.0),
                        "avg_tokens": float(public.get("avg_tokens", 0.0) or 0.0),
                    }
                )

    by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary["rows"]:
        by_variant.setdefault(row["variant"], {})[row["split_name"]] = row

    base_rows = by_variant["vanilla_direct_base"]
    for variant_name, split_rows in by_variant.items():
        if variant_name == "vanilla_direct_base":
            continue
        summary["deltas"][variant_name] = {
            split_name: split_rows[split_name]["accuracy"] - base_rows[split_name]["accuracy"]
            for split_name, _ in split_plan
        }

    (results_root / "posthoc_eval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--adapter-dirs", nargs="+", required=True)
    parser.add_argument("--calibration-examples", type=int, default=256)
    parser.add_argument("--report-examples", type=int, default=1024)
    args = parser.parse_args()
    summary = run_posthoc_eval(
        runtime_config=Path(args.runtime_config),
        results_root=Path(args.results_root),
        adapter_dirs=[Path(path) for path in args.adapter_dirs],
        calibration_examples=int(args.calibration_examples),
        report_examples=int(args.report_examples),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
