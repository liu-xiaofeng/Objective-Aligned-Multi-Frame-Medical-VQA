from __future__ import annotations

import math
from collections import Counter
from typing import Any


FAILURE_LABELS = [
    "missed intermediate frame",
    "cross-frame conflict",
    "anatomical localization flip",
    "overconfident wrong answer",
]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _objective_weights(mode: str) -> dict[str, float]:
    normalized = str(mode or "legacy_joint").strip().lower()
    if normalized == "legacy_joint":
        return {
            "accuracy": 0.60,
            "verifier_agreement": 0.15,
            "ece_complement": 0.15,
            "selective_accuracy": 0.10,
        }
    if normalized == "accuracy_first":
        return {
            "accuracy": 0.78,
            "verifier_agreement": 0.04,
            "ece_complement": 0.08,
            "selective_accuracy": 0.10,
        }
    raise ValueError(f"Unsupported evaluation.objective.mode: {mode}")


def compute_ece(correct_flags: list[int], confidences: list[float], bins: int = 10) -> float:
    if not correct_flags:
        return 1.0
    bin_totals = [0] * bins
    bin_correct = [0.0] * bins
    bin_conf = [0.0] * bins
    for is_correct, conf in zip(correct_flags, confidences):
        conf = _clip01(conf)
        idx = min(bins - 1, int(conf * bins))
        bin_totals[idx] += 1
        bin_correct[idx] += float(is_correct)
        bin_conf[idx] += conf
    total = float(len(correct_flags))
    ece = 0.0
    for n, c_sum, conf_sum in zip(bin_totals, bin_correct, bin_conf):
        if n == 0:
            continue
        acc = c_sum / n
        mean_conf = conf_sum / n
        ece += abs(acc - mean_conf) * (n / total)
    return float(ece)


def selective_accuracy_at_coverage(
    correct_flags: list[int],
    confidences: list[float],
    coverage: float = 0.8,
) -> float:
    if not correct_flags:
        return 0.0
    paired = sorted(zip(confidences, correct_flags), reverse=True)
    keep = max(1, math.ceil(len(paired) * coverage))
    selected = paired[:keep]
    return float(sum(flag for _, flag in selected) / keep)


def validate_prediction(prediction: Any) -> tuple[bool, str | None]:
    if not isinstance(prediction, dict):
        return False, "prediction is not a dict"
    required = {
        "answer_letter",
        "confidence",
        "frame_findings",
        "fusion_reasoning",
        "option_scores",
        "verifier_passed",
        "trace",
    }
    missing = required - set(prediction.keys())
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    answer_letter = str(prediction["answer_letter"]).strip().upper()
    if len(answer_letter) != 1 or not ("A" <= answer_letter <= "Z"):
        return False, "answer_letter is invalid"
    try:
        confidence = float(prediction["confidence"])
    except Exception:
        return False, "confidence is not numeric"
    if confidence < 0.0 or confidence > 1.0:
        return False, "confidence out of range"
    if not isinstance(prediction["frame_findings"], list) or not prediction["frame_findings"]:
        return False, "frame_findings missing or empty"
    if not isinstance(prediction["option_scores"], list) or not prediction["option_scores"]:
        return False, "option_scores missing or empty"
    if not isinstance(prediction["verifier_passed"], bool):
        return False, "verifier_passed must be bool"
    if not isinstance(prediction["trace"], dict):
        return False, "trace must be a dict"
    return True, None


def classify_failure(record: dict[str, Any]) -> str | None:
    if (not record["correct"]) and record["confidence"] >= 0.75:
        return "overconfident wrong answer"
    selected = record.get("selected_frame_indices", [])
    total_frames = record.get("num_frames", 0)
    if total_frames >= 3 and len(selected) < total_frames:
        return "missed intermediate frame"
    if record["verifier_used"] and (
        (
            record["reasoning_triggered"]
            or record["localization_triggered"]
            or record["pairwise_triggered"]
            or record["counterfactual_triggered"]
        )
        and (
            record["conflict_signal"] >= 0.45
            or record["direct_decomposed_disagreement"]
            or record["leave_one_out_instability"] >= 0.34
        )
    ):
        return "cross-frame conflict"
    if record["correct"]:
        return None
    return "anatomical localization flip"


def build_feedback(records: list[dict[str, Any]]) -> str:
    counter = Counter(label for label in (classify_failure(r) for r in records) if label)
    if not counter:
        return "Top failure modes: none observed; preserve current scaffold and focus on stronger accuracy."
    parts = [f"{label}={count}" for label, count in counter.most_common(4)]
    return "Top failure modes: " + ", ".join(parts)


def aggregate_prediction_metrics(
    predictions: list[dict[str, Any]],
    *,
    coverage: float = 0.8,
    objective: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        trace = pred.get("trace", {})
        gold = pred.get("gold_answer", "")
        correct = pred["answer_letter"].strip().upper() == str(gold).strip().upper()
        verifier_used = bool(trace.get("verifier_used", True))
        contradiction_score = _clip01(trace.get("contradiction_score", 0.0) or 0.0)
        leave_one_out_instability = _clip01(trace.get("leave_one_out_instability", 0.0) or 0.0)
        conflict_signal = _clip01(
            trace.get(
                "conflict_signal",
                max(contradiction_score, leave_one_out_instability),
            )
            or 0.0
        )
        row = {
            "question_id": pred.get("question_id"),
            "gold_answer": gold,
            "answer_letter": pred.get("answer_letter"),
            "correct": correct,
            "confidence": _clip01(pred["confidence"]),
            "verifier_used": verifier_used,
            "verifier_passed": bool(pred["verifier_passed"]),
            "support_score": _clip01(trace.get("support_score", 0.0) or 0.0),
            "contradiction_score": contradiction_score,
            "conflict_signal": conflict_signal,
            "direct_decomposed_disagreement": bool(trace.get("direct_decomposed_disagreement", False)),
            "pairwise_override_used": bool(trace.get("pairwise_override_used", False)),
            "leave_one_out_instability": leave_one_out_instability,
            "reasoning_triggered": bool(trace.get("reasoning_triggered", False)),
            "localization_triggered": bool(trace.get("localization_triggered", False)),
            "localization_override_used": bool(trace.get("localization_override_used", False)),
            "pairwise_triggered": bool(trace.get("pairwise_triggered", False)),
            "counterfactual_triggered": bool(trace.get("counterfactual_triggered", False)),
            "trigger_path": str(trace.get("trigger_path", "unknown")),
            "selected_frame_indices": trace.get("selected_frame_indices", []),
            "num_frames": int(trace.get("num_frames", 0) or 0),
            "avg_runtime_sec": float(trace.get("runtime_sec", 0.0) or 0.0),
            "avg_tokens": float(trace.get("token_count", 0.0) or 0.0),
            "verifier_note": trace.get("verifier_note", ""),
            "option_margin": float(trace.get("option_margin", 0.0) or 0.0),
            "selected_candidate_rank": int(trace.get("selected_candidate_rank", 0) or 0),
        }
        row["failure_mode"] = classify_failure(row)
        rows.append(row)

    correct_flags = [int(r["correct"]) for r in rows]
    confidences = [r["confidence"] for r in rows]
    verifier = [int(r["verifier_passed"]) for r in rows if r["verifier_used"]]
    accuracy = float(sum(correct_flags) / len(correct_flags)) if correct_flags else 0.0
    ece = compute_ece(correct_flags, confidences)
    selective = selective_accuracy_at_coverage(correct_flags, confidences, coverage=coverage)
    verifier_agreement = float(sum(verifier) / len(verifier)) if verifier else 0.0
    avg_runtime = float(sum(r["avg_runtime_sec"] for r in rows) / len(rows)) if rows else 0.0
    avg_tokens = float(sum(r["avg_tokens"] for r in rows) / len(rows)) if rows else 0.0

    objective_cfg = objective or {}
    objective_mode = str(objective_cfg.get("mode", "legacy_joint"))
    weights = _objective_weights(objective_mode)
    combined_score = (
        weights["accuracy"] * accuracy
        + weights["verifier_agreement"] * verifier_agreement
        + weights["ece_complement"] * (1.0 - ece)
        + weights["selective_accuracy"] * selective
    )

    diagnostics = {
        "cross_frame_conflict_rate": _mean(
            [1.0 if r["failure_mode"] == "cross-frame conflict" else 0.0 for r in rows]
        ),
        "mean_conflict_signal": _mean([r["conflict_signal"] for r in rows]),
        "mean_leave_one_out_instability": _mean([r["leave_one_out_instability"] for r in rows]),
        "direct_decomposed_disagreement_rate": _mean(
            [1.0 if r["direct_decomposed_disagreement"] else 0.0 for r in rows]
        ),
        "pairwise_override_rate": _mean([1.0 if r["pairwise_override_used"] else 0.0 for r in rows]),
        "reasoning_trigger_rate": _mean([1.0 if r["reasoning_triggered"] else 0.0 for r in rows]),
        "localization_trigger_rate": _mean([1.0 if r["localization_triggered"] else 0.0 for r in rows]),
        "localization_override_rate": _mean(
            [1.0 if r["localization_override_used"] else 0.0 for r in rows]
        ),
        "pairwise_trigger_rate": _mean([1.0 if r["pairwise_triggered"] else 0.0 for r in rows]),
        "counterfactual_trigger_rate": _mean(
            [1.0 if r["counterfactual_triggered"] else 0.0 for r in rows]
        ),
    }
    public = {
        "accuracy": accuracy,
        "ece": ece,
        "selective_accuracy@80": selective,
        "verifier_agreement": verifier_agreement,
        "avg_runtime_sec": avg_runtime,
        "avg_tokens": avg_tokens,
    }
    private = {
        "num_examples": len(rows),
        "failure_feedback": build_feedback(rows),
        "objective_mode": objective_mode,
        "diagnostics": diagnostics,
    }
    return {
        "combined_score": combined_score,
        "public": public,
        "private": private,
        "extra_data": {"rows": rows},
        "text_feedback": private["failure_feedback"],
    }
