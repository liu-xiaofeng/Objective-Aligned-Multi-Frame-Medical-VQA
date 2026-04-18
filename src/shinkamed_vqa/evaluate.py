from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

try:
    from .dataset_utils import default_paths, dump_pickle, load_runtime_config, load_split_examples, resolve_path
    from .metrics import aggregate_prediction_metrics, validate_prediction
except ImportError:
    from dataset_utils import default_paths, dump_pickle, load_runtime_config, load_split_examples, resolve_path
    from metrics import aggregate_prediction_metrics, validate_prediction


def _load_run_experiment(program_path: Path):
    spec = importlib.util.spec_from_file_location("public_eval_program", str(program_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load program: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_experiment = getattr(module, "run_experiment", None)
    if run_experiment is None:
        raise RuntimeError(f"`run_experiment` not found in {program_path}")
    return run_experiment


def build_examples(config: dict[str, Any], split_name: str, limit: int | None = None) -> list[dict[str, Any]]:
    paths = default_paths()
    processed_dir = paths["processed"]
    config_dir = config.get("_config_dir")
    manifest_path = resolve_path(
        config.get("data", {}).get("manifest_path", processed_dir / "medframeqa_manifest.jsonl"),
        base_dir=config_dir,
    )
    split_ids_path = resolve_path(
        config.get("data", {}).get("split_ids_path", processed_dir / "medframeqa_split_ids.json"),
        base_dir=config_dir,
    )
    examples = load_split_examples(manifest_path, split_ids_path, split_name)
    if limit is None:
        limit = int(config.get("evaluation", {}).get("max_examples", len(examples)))
    return examples[:limit]


def _write_failure_artifacts(results_dir: Path, error: str, objective_mode: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "correct.json").write_text(json.dumps({"correct": False, "error": error}, indent=2))
    (results_dir / "metrics.json").write_text(
        json.dumps(
            {
                "combined_score": 0.0,
                "public": {
                    "accuracy": 0.0,
                    "ece": 1.0,
                    "selective_accuracy@80": 0.0,
                    "verifier_agreement": 0.0,
                    "avg_runtime_sec": 0.0,
                    "avg_tokens": 0.0,
                },
                "private": {
                    "num_examples": 0,
                    "failure_feedback": error,
                    "objective_mode": objective_mode,
                    "diagnostics": {},
                },
                "text_feedback": error,
            },
            indent=2,
        )
    )
    dump_pickle({"rows": []}, results_dir / "extra.pkl")


def evaluate_program(*, program_path: Path, results_dir: Path, config_path: Path, split_name: str) -> dict[str, Any]:
    try:
        config = load_runtime_config(config_path)
        examples = build_examples(config, split_name)
        run_experiment = _load_run_experiment(program_path)
        predictions: list[dict[str, Any]] = []
        for example in examples:
            prediction = run_experiment(example=example, runtime_cfg=config)
            valid, error = validate_prediction(prediction)
            if not valid:
                raise RuntimeError(error or "Prediction validation failed")
            predictions.append(prediction)
        metrics = aggregate_prediction_metrics(
            predictions,
            coverage=float(config.get("evaluation", {}).get("coverage", 0.8)),
            objective=config.get("evaluation", {}).get("objective"),
        )
        extra_data = metrics.pop("extra_data", {"rows": []})
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "correct.json").write_text(json.dumps({"correct": True, "error": None}, indent=2))
        (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        dump_pickle(extra_data, results_dir / "extra.pkl")
        return metrics
    except Exception as exc:
        objective_mode = "accuracy_first"
        _write_failure_artifacts(results_dir, str(exc), objective_mode)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-path", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-name", required=True)
    args = parser.parse_args()
    metrics = evaluate_program(
        program_path=resolve_path(args.program_path),
        results_dir=resolve_path(args.results_dir),
        config_path=resolve_path(args.config),
        split_name=args.split_name,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
