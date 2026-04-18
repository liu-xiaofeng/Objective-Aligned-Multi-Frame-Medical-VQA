from __future__ import annotations

import time
from typing import Any

try:
    from backend import backend_from_runtime
    from dataset_utils import answer_index_to_letter, resolve_answer_choice, resolve_example_frame_paths
except ImportError:
    from ..backend import backend_from_runtime
    from ..dataset_utils import answer_index_to_letter, resolve_answer_choice, resolve_example_frame_paths


def _normalize_scores(scores: list[float], num_options: int) -> list[float]:
    clipped = [max(0.0, min(1.0, float(score))) for score in scores[:num_options]]
    if len(clipped) < num_options:
        clipped.extend([0.0] * (num_options - len(clipped)))
    total = sum(clipped)
    if total <= 0:
        return [1.0 / num_options] * num_options
    return [score / total for score in clipped]


def _placeholder_findings(frame_paths: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": idx,
            "frame_path": frame_path,
            "finding": "vanilla direct baseline did not decompose frame evidence",
            "anatomy": "unknown",
            "supports_question": True,
            "confidence": 0.25,
        }
        for idx, frame_path in enumerate(frame_paths)
    ]


def run_experiment(example: dict[str, Any], runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    backend = backend_from_runtime(runtime_cfg)
    frame_paths = resolve_example_frame_paths(example)
    direct = backend.answer_direct(example, frame_paths)

    answer_index, _ = resolve_answer_choice(
        direct.get("answer_index"),
        direct.get("answer_letter", ""),
        len(example["options"]),
    )
    if not (0 <= answer_index < len(example["options"])):
        answer_index = 0
    option_scores = _normalize_scores(direct.get("option_scores", []), len(example["options"]))
    ranked_scores = sorted(option_scores, reverse=True)
    option_margin = ranked_scores[0] - (ranked_scores[1] if len(ranked_scores) > 1 else 0.0)

    return {
        "answer_letter": answer_index_to_letter(answer_index),
        "confidence": max(0.0, min(1.0, float(direct.get("confidence", max(option_scores))))),
        "frame_findings": _placeholder_findings(frame_paths),
        "fusion_reasoning": str(direct.get("rationale", "vanilla direct answer")),
        "option_scores": option_scores,
        "verifier_passed": False,
        "trace": {
            "version": "vanilla_direct_v0.1",
            "selected_frame_indices": list(range(len(frame_paths))),
            "num_frames": len(frame_paths),
            "runtime_sec": time.time() - start,
            "token_count": int(getattr(backend, "total_tokens", 0)),
            "verifier_used": False,
            "verifier_note": "vanilla direct baseline did not run a verifier",
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "conflict_signal": 0.0,
            "direct_decomposed_disagreement": False,
            "pairwise_override_used": False,
            "leave_one_out_instability": 0.0,
            "option_margin": float(option_margin),
            "selected_candidate_rank": 0,
        },
        "gold_answer": example["correct_answer"],
        "question_id": example["question_id"],
    }
