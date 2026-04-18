from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Callable

try:
    from .dataset_utils import dump_json
    from .metrics import aggregate_prediction_metrics
except ImportError:
    from dataset_utils import dump_json
    from metrics import aggregate_prediction_metrics


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _logit(prob: float) -> float:
    prob = min(max(prob, 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _load_rows(extra_path: Path) -> list[dict[str, Any]]:
    payload = pickle.loads(extra_path.read_bytes())
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        raise RuntimeError(f"Malformed extra.pkl rows in {extra_path}")
    return rows


def _rows_to_predictions(rows: list[dict[str, Any]], transform: Callable[[float], float]) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    for row in rows:
        trace = {
            "verifier_used": bool(row.get("verifier_used", False)),
            "support_score": float(row.get("support_score", 0.0) or 0.0),
            "contradiction_score": float(row.get("contradiction_score", 0.0) or 0.0),
            "conflict_signal": float(row.get("conflict_signal", 0.0) or 0.0),
            "direct_decomposed_disagreement": bool(row.get("direct_decomposed_disagreement", False)),
            "pairwise_override_used": bool(row.get("pairwise_override_used", False)),
            "leave_one_out_instability": float(row.get("leave_one_out_instability", 0.0) or 0.0),
            "reasoning_triggered": bool(row.get("reasoning_triggered", False)),
            "localization_triggered": bool(row.get("localization_triggered", False)),
            "localization_override_used": bool(row.get("localization_override_used", False)),
            "pairwise_triggered": bool(row.get("pairwise_triggered", False)),
            "counterfactual_triggered": bool(row.get("counterfactual_triggered", False)),
            "trigger_path": row.get("trigger_path", "unknown"),
            "selected_frame_indices": row.get("selected_frame_indices", []),
            "num_frames": int(row.get("num_frames", 0) or 0),
            "runtime_sec": float(row.get("avg_runtime_sec", 0.0) or 0.0),
            "token_count": float(row.get("avg_tokens", 0.0) or 0.0),
            "verifier_note": row.get("verifier_note", ""),
            "option_margin": float(row.get("option_margin", 0.0) or 0.0),
            "selected_candidate_rank": int(row.get("selected_candidate_rank", 0) or 0),
        }
        preds.append(
            {
                "question_id": row.get("question_id"),
                "gold_answer": row.get("gold_answer"),
                "answer_letter": row.get("answer_letter"),
                "confidence": transform(float(row.get("confidence", 0.0) or 0.0)),
                "verifier_passed": bool(row.get("verifier_passed", False)),
                "trace": trace,
            }
        )
    return preds


def _nll(rows: list[dict[str, Any]], transform: Callable[[float], float]) -> float:
    total = 0.0
    for row in rows:
        p = _clip01(transform(float(row.get("confidence", 0.0) or 0.0)))
        y = 1.0 if bool(row.get("correct", False)) else 0.0
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / max(len(rows), 1)


def _brier(rows: list[dict[str, Any]], transform: Callable[[float], float]) -> float:
    total = 0.0
    for row in rows:
        p = _clip01(transform(float(row.get("confidence", 0.0) or 0.0)))
        y = 1.0 if bool(row.get("correct", False)) else 0.0
        total += (p - y) ** 2
    return total / max(len(rows), 1)


def _fit_temperature(rows: list[dict[str, Any]]) -> tuple[float, Callable[[float], float]]:
    best_t = 1.0
    best_loss = float("inf")
    for raw_t in range(5, 61):
        temp = raw_t / 10.0
        transform = lambda c, t=temp: _sigmoid(_logit(_clip01(c)) / t)
        loss = _nll(rows, transform)
        if loss < best_loss:
            best_loss = loss
            best_t = temp
    return best_t, (lambda c, t=best_t: _sigmoid(_logit(_clip01(c)) / t))


def _fit_histogram(rows: list[dict[str, Any]], bins: int) -> tuple[list[float], Callable[[float], float]]:
    totals = [0] * bins
    correct = [0.0] * bins
    for row in rows:
        conf = _clip01(float(row.get("confidence", 0.0) or 0.0))
        idx = min(bins - 1, int(conf * bins))
        totals[idx] += 1
        correct[idx] += 1.0 if bool(row.get("correct", False)) else 0.0
    mapping = []
    fallback = sum(correct) / max(sum(totals), 1)
    for n, c in zip(totals, correct):
        mapping.append((c / n) if n else fallback)
    return mapping, (lambda c, m=mapping, b=bins: _clip01(m[min(b - 1, int(_clip01(c) * b))]))


def _method_transforms(cal_rows: list[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], Callable[[float], float]]]:
    out: dict[str, tuple[dict[str, Any], Callable[[float], float]]] = {
        "identity": ({}, lambda c: _clip01(c)),
    }
    temp, tf = _fit_temperature(cal_rows)
    out["temperature_nll"] = ({"temperature": temp}, tf)
    map5, hf5 = _fit_histogram(cal_rows, 5)
    out["histogram5"] = ({"bin_mapping": map5}, hf5)
    return out


def run_posthoc_calibration(*, results_root: Path, output_root: Path) -> dict[str, Any]:
    summary = json.loads((results_root / "pipeline_summary.json").read_text())
    posthoc_root = results_root / "posthoc_eval"
    adapter_variant = str(summary["adapter_variant"])
    split_names = ["calibration_val", "report_test"]
    split_rows = {
        split_name: _load_rows(posthoc_root / adapter_variant / split_name / "extra.pkl") for split_name in split_names
    }
    transforms = _method_transforms(split_rows["calibration_val"])

    method_summaries: dict[str, Any] = {}
    for method_name, (fit_info, transform) in transforms.items():
        split_metrics: dict[str, Any] = {}
        for split_name, rows in split_rows.items():
            preds = _rows_to_predictions(rows, transform)
            metrics = aggregate_prediction_metrics(preds, coverage=0.8, objective={"mode": "accuracy_first"})
            split_metrics[split_name] = metrics
        method_summaries[method_name] = {
            "fit": fit_info,
            "calibration_val": split_metrics["calibration_val"]["public"],
            "report_test": split_metrics["report_test"]["public"],
            "calibration_combined": split_metrics["calibration_val"]["combined_score"],
            "report_combined": split_metrics["report_test"]["combined_score"],
        }

    best_by_calibration_ece = min(
        method_summaries.items(),
        key=lambda item: item[1]["calibration_val"]["ece"],
    )[0]
    best_by_report_combined = max(
        method_summaries.items(),
        key=lambda item: item[1]["report_combined"],
    )[0]

    result = {
        "results_root": str(results_root),
        "adapter_variant": adapter_variant,
        "methods": method_summaries,
        "best_by_calibration_ece": best_by_calibration_ece,
        "best_by_report_combined": best_by_report_combined,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    dump_json(result, output_root / "posthoc_calibration_summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    summary = run_posthoc_calibration(
        results_root=Path(args.results_root).resolve(),
        output_root=Path(args.output_root).resolve(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
