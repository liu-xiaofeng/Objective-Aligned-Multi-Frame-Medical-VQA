from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from dataset_utils import answer_index_to_letter, answer_letter_to_index, resolve_answer_choice
except ImportError:
    from .dataset_utils import answer_index_to_letter, answer_letter_to_index, resolve_answer_choice

try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except Exception:
    AutoModelForImageTextToText = None
    AutoProcessor = None

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


_HF_MODEL_CACHE: dict[tuple[str, str, str, str, str, bool, bool, bool], tuple[Any, Any, Any]] = {}


def _coerce_probability_like(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().lower()
    if not text:
        return 0.0
    aliases = {
        "high": 0.85,
        "very high": 0.95,
        "medium": 0.55,
        "moderate": 0.55,
        "low": 0.25,
        "very low": 0.10,
        "yes": 1.0,
        "no": 0.0,
        "true": 1.0,
        "false": 0.0,
    }
    if text in aliases:
        return aliases[text]
    if text.endswith("%"):
        try:
            return float(text[:-1].strip()) / 100.0
        except Exception:
            return 0.0
    try:
        return float(text)
    except Exception:
        return 0.0


def _clip01(value: Any) -> float:
    return max(0.0, min(1.0, _coerce_probability_like(value)))


def _safe_json_loads(text: str, default: Any) -> Any:
    text = (text or "").strip()
    if not text:
        return default
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    return default


def _data_uri(path: str | Path) -> str:
    path = Path(path)
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    content = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{content}"


def _response_total_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and isinstance(usage, dict):
        total_tokens = usage.get("total_tokens")
    return int(total_tokens or 0)


def detect_ollama_base_url() -> str:
    try:
        proc = subprocess.run(
            ["ollama", "host"],
            check=True,
            capture_output=True,
            text=True,
        )
        host = proc.stdout.strip()
        if not host:
            raise RuntimeError("empty ollama host output")
        if host.startswith("http://") or host.startswith("https://"):
            return host.rstrip("/") + "/v1"
        return f"http://{host}/v1"
    except Exception as exc:
        raise RuntimeError(
            "Failed to auto-detect Ollama host. "
            "Make sure `module load ollama` is active and `ollama serve` has been started."
        ) from exc


def _sanitize_frame_findings(
    findings: Any,
    frame_paths: list[str],
    default_anatomy: str,
) -> list[dict[str, Any]]:
    if not isinstance(findings, list):
        findings = []
    sanitized: list[dict[str, Any]] = []
    for idx, frame_path in enumerate(frame_paths):
        item = findings[idx] if idx < len(findings) and isinstance(findings[idx], dict) else {}
        sanitized.append(
            {
                "frame_index": idx,
                "frame_path": frame_path,
                "finding": str(item.get("finding", f"fallback finding {idx}")).strip(),
                "anatomy": str(item.get("anatomy", default_anatomy or "unknown")).strip(),
                "supports_question": bool(item.get("supports_question", True)),
                "confidence": _clip01(item.get("confidence", 0.4)),
            }
        )
    return sanitized


def _sanitize_option_scores(scores: Any, num_options: int) -> list[float]:
    def _coerce_score_value(value: Any) -> float | None:
        if isinstance(value, dict):
            for key in ("score", "probability", "confidence", "value", "weight"):
                if key in value:
                    return _coerce_score_value(value[key])
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.endswith("%"):
                try:
                    return float(stripped[:-1]) / 100.0
                except Exception:
                    return None
            lowered = stripped.lower()
            if lowered in {"high", "very high", "medium", "moderate", "low", "very low"}:
                return _coerce_probability_like(lowered)
            try:
                return float(stripped)
            except Exception:
                return None
        try:
            return float(value)
        except Exception:
            return None

    normalized: list[float | None] = [None] * num_options
    ordered_values: list[float] = []

    if isinstance(scores, dict):
        if "option_scores" in scores:
            return _sanitize_option_scores(scores["option_scores"], num_options)
        for key, value in scores.items():
            index = answer_letter_to_index(str(key).strip().upper())
            coerced = _coerce_score_value(value)
            if 0 <= index < num_options and coerced is not None:
                normalized[index] = coerced

    elif isinstance(scores, list):
        for position, item in enumerate(scores):
            if isinstance(item, dict):
                option_key = (
                    item.get("option_letter")
                    or item.get("answer_letter")
                    or item.get("letter")
                    or item.get("option")
                )
                coerced = _coerce_score_value(item)
                if option_key is not None:
                    index = answer_letter_to_index(str(option_key).strip().upper())
                    if 0 <= index < num_options and coerced is not None:
                        normalized[index] = coerced
                        continue
                if coerced is not None and position < num_options:
                    ordered_values.append(coerced)
                continue

            coerced = _coerce_score_value(item)
            if coerced is not None:
                ordered_values.append(coerced)

    fill_index = 0
    for value in ordered_values:
        while fill_index < num_options and normalized[fill_index] is not None:
            fill_index += 1
        if fill_index >= num_options:
            break
        normalized[fill_index] = value
        fill_index += 1

    clipped = [_clip01(score if score is not None else 0.0) for score in normalized[:num_options]]
    if len(clipped) < num_options:
        clipped.extend([0.0] * (num_options - len(clipped)))
    if sum(clipped) <= 0:
        return [1.0 / num_options] * num_options
    return clipped


def _resolve_torch_dtype(dtype_name: str | None) -> Any:
    if torch is None:
        return None
    name = str(dtype_name or "auto").strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _default_device_map(device_map: str | None) -> str:
    return str(device_map or "auto")


def _load_pil_images(frame_paths: list[str]) -> list[Any]:
    if Image is None:
        raise RuntimeError(
            "Pillow is required for the HF MedGemma backend. Install `pillow` in the active environment."
        )
    images: list[Any] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _apply_hf_chat_template(
    processor: Any,
    family: str,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tokenize: bool,
    return_dict: bool = True,
    return_tensors: str | None = "pt",
) -> Any:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": tokenize,
    }
    if return_dict:
        kwargs["return_dict"] = True
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors

    normalized_family = str(family or "medgemma").strip().lower()
    if normalized_family in {"qwen3_5", "qwen2_5_vl", "qwen_vl"}:
        try:
            return processor.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            pass
    return processor.apply_chat_template(messages, **kwargs)


def _build_hf_text_image_inputs(
    processor: Any,
    family: str,
    *,
    text: str,
    images: list[Any],
    return_tensors: str = "pt",
) -> Any:
    normalized_family = str(family or "medgemma").strip().lower()
    if normalized_family in {"qwen3_5", "qwen2_5_vl", "qwen_vl"}:
        try:
            return processor(
                text=text,
                images=images,
                return_tensors=return_tensors,
                padding=True,
            )
        except TypeError:
            pass
    return processor(
        text=text,
        images=images,
        return_tensors=return_tensors,
        padding=True,
    )


def _load_hf_vlm_bundle(cfg: "BackendConfig") -> tuple[Any, Any, Any]:
    if torch is None or AutoProcessor is None or AutoModelForImageTextToText is None:
        raise RuntimeError(
            "HF VLM backend requires `torch`, `transformers`, and `accelerate` in the active environment."
        )

    if not cfg.model_name:
        raise ValueError("HF VLM backend requires model_name.")

    cache_key = (
        str(cfg.family or "medgemma"),
        cfg.model_name,
        str(cfg.dtype or "auto"),
        _default_device_map(cfg.device_map),
        str(cfg.adapter_mode or "base"),
        str(cfg.adapter_path or ""),
        bool(cfg.load_in_4bit),
        bool(cfg.trust_remote_code),
        bool(cfg.local_files_only),
    )
    cached = _HF_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    torch_dtype = _resolve_torch_dtype(cfg.dtype)
    model_kwargs: dict[str, Any] = {
        "device_map": _default_device_map(cfg.device_map),
        "trust_remote_code": bool(cfg.trust_remote_code),
        "local_files_only": bool(cfg.local_files_only),
    }
    if bool(cfg.load_in_4bit):
        if BitsAndBytesConfig is None:
            raise RuntimeError("`bitsandbytes` and a recent `transformers` install are required for load_in_4bit.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    try:
        processor = AutoProcessor.from_pretrained(
            cfg.model_name,
            trust_remote_code=bool(cfg.trust_remote_code),
            local_files_only=bool(cfg.local_files_only),
        )
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name,
            **model_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HF VLM model `{cfg.model_name}`. "
            "If the repo is gated, request access and make sure `HF_TOKEN` is set or `huggingface-cli login` has been run."
        ) from exc

    if str(cfg.adapter_mode or "base") == "localization_lora":
        if not cfg.adapter_path:
            raise RuntimeError("adapter_mode=localization_lora requires adapter_path.")
        if PeftModel is None:
            raise RuntimeError("`peft` is required to load localization_lora adapters.")
        try:
            model = PeftModel.from_pretrained(
                model,
                cfg.adapter_path,
                is_trainable=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LoRA adapter from `{cfg.adapter_path}` for model `{cfg.model_name}`."
            ) from exc

    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = "cpu"

    bundle = (processor, model, device)
    _HF_MODEL_CACHE[cache_key] = bundle
    return bundle


@dataclass
class BackendConfig:
    backend_type: str
    model_name: str | None = None
    family: str = "medgemma"
    base_url: str | None = None
    api_key: str | None = None
    max_frames: int = 5
    temperature: float = 0.0
    max_tokens: int = 768
    mock_mode: str = "hash"
    dtype: str = "auto"
    device_map: str = "auto"
    adapter_mode: str = "base"
    adapter_path: str | None = None
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    local_files_only: bool = False


class BaseTaskBackend:
    def __init__(self) -> None:
        self.total_tokens = 0

    def describe_frames(self, example: dict[str, Any], frame_paths: list[str]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def fuse_reasoning(
        self,
        example: dict[str, Any],
        frame_findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def score_options(
        self,
        example: dict[str, Any],
        frame_paths: list[str],
        frame_findings: list[dict[str, Any]],
        fusion: dict[str, Any],
    ) -> list[float]:
        raise NotImplementedError

    def verify_answer(
        self,
        example: dict[str, Any],
        frame_paths: list[str],
        frame_findings: list[dict[str, Any]],
        fusion: dict[str, Any],
        predicted_index: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def answer_direct(self, example: dict[str, Any], frame_paths: list[str]) -> dict[str, Any]:
        raise NotImplementedError


class MockVisionBackend(BaseTaskBackend):
    def __init__(self, cfg: BackendConfig):
        super().__init__()
        self.cfg = cfg

    def _rng(self, key: str) -> random.Random:
        seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
        return random.Random(seed)

    def describe_frames(self, example: dict[str, Any], frame_paths: list[str]) -> list[dict[str, Any]]:
        findings = []
        for idx, path in enumerate(frame_paths):
            findings.append(
                {
                    "frame_index": idx,
                    "frame_path": path,
                    "finding": f"mock finding for {example['organ']} frame {idx}",
                    "anatomy": example["organ"],
                    "supports_question": True,
                    "confidence": 0.55 + 0.05 * (idx % 3),
                }
            )
        return findings

    def fuse_reasoning(
        self,
        example: dict[str, Any],
        frame_findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "summary": (
                f"Mock fusion for {example['question_id']} using "
                f"{len(frame_findings)} frames in {example['modality']}."
            ),
            "key_evidence": [f["finding"] for f in frame_findings[:2]],
            "differential_clues": [frame_findings[0]["finding"]] if frame_findings else [],
        }

    def score_options(
        self,
        example: dict[str, Any],
        frame_paths: list[str],
        frame_findings: list[dict[str, Any]],
        fusion: dict[str, Any],
    ) -> list[float]:
        if self.cfg.mock_mode == "gold":
            scores = [0.05] * len(example["options"])
            gold_idx = answer_letter_to_index(example["correct_answer"])
            if 0 <= gold_idx < len(scores):
                scores[gold_idx] = 0.95
            return scores

        rng = self._rng(example["question_id"])
        return [0.1 + 0.8 * rng.random() for _ in example["options"]]

    def verify_answer(
        self,
        example: dict[str, Any],
        frame_paths: list[str],
        frame_findings: list[dict[str, Any]],
        fusion: dict[str, Any],
        predicted_index: int,
    ) -> dict[str, Any]:
        gold_idx = answer_letter_to_index(example["correct_answer"])
        if self.cfg.mock_mode == "gold":
            passed = predicted_index == gold_idx
            return {
                "verifier_passed": passed,
                "rationale": "mock verifier (gold)",
                "support_score": 0.95 if passed else 0.15,
                "contradiction_score": 0.05 if passed else 0.8,
            }
        key = f"{example['question_id']}|{predicted_index}"
        rng = self._rng(key)
        decision = rng.random() > 0.25
        return {
            "verifier_passed": decision,
            "rationale": "mock verifier (hash)",
            "support_score": 0.75 if decision else 0.35,
            "contradiction_score": 0.15 if decision else 0.6,
        }

    def answer_direct(self, example: dict[str, Any], frame_paths: list[str]) -> dict[str, Any]:
        scores = _sanitize_option_scores(
            self.score_options(example, frame_paths, [], {}),
            len(example["options"]),
        )
        answer_index = max(range(len(scores)), key=lambda idx: scores[idx])
        return {
            "answer_index": answer_index,
            "answer_letter": answer_index_to_letter(answer_index),
            "confidence": max(scores),
            "rationale": "mock direct answer",
            "option_scores": scores,
        }


class OpenAIVisionBackend(BaseTaskBackend):
    def __init__(self, cfg: BackendConfig):
        super().__init__()
        if not cfg.model_name:
            raise ValueError("OpenAI vision backend requires model_name.")
        if not cfg.base_url or str(cfg.base_url).lower() in {"auto", "ollama", "auto_ollama"}:
            cfg.base_url = detect_ollama_base_url()
        self.cfg = cfg
        self.client = OpenAI(
            api_key=cfg.api_key or os.getenv("LOCAL_OPENAI_API_KEY", "local"),
            base_url=cfg.base_url,
        )

    def _query_json(self, prompt: str, frame_paths: list[str], default: Any) -> Any:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame_path in frame_paths[: self.cfg.max_frames]:
            content.append({"type": "image_url", "image_url": {"url": _data_uri(frame_path)}})
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        self.total_tokens += _response_total_tokens(response)
        text = response.choices[0].message.content or ""
        return _safe_json_loads(text, default)

    def describe_frames(self, example: dict[str, Any], frame_paths: list[str]) -> list[dict[str, Any]]:
        default = _sanitize_frame_findings([], frame_paths, example.get("organ", "unknown"))
        prompt = (
            "You are a medical multi-image VQA assistant. "
            "Return JSON only as a list with one object per image. "
            "Each object must contain frame_index, finding, anatomy, supports_question, confidence. "
            "Use concise clinically relevant findings only.\n"
            f"System: {example.get('system', 'unknown')}\n"
            f"Organ: {example.get('organ', 'unknown')}\n"
            f"Modality: {example.get('modality', 'unknown')}\n"
            f"Question: {example['question']}\n"
            f"Options: {example['options']}"
        )
        findings = self._query_json(prompt, frame_paths, default)
        return _sanitize_frame_findings(findings, frame_paths, example.get("organ", "unknown"))

    def fuse_reasoning(
        self,
        example: dict[str, Any],
        frame_findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = (
            "Return JSON only with keys summary, key_evidence, differential_clues. "
            "Fuse the frame findings into a single multi-frame explanation and mention what distinguishes the best option.\n"
            f"Question: {example['question']}\n"
            f"Options: {example['options']}\n"
            f"FrameFindings: {json.dumps(frame_findings)}"
        )
        default = {"summary": "fallback fusion", "key_evidence": [], "differential_clues": []}
        response = self._query_json(prompt, [], default)
        if not isinstance(response, dict):
            return default
        return {
            "summary": str(response.get("summary", default["summary"])),
            "key_evidence": list(response.get("key_evidence", [])),
            "differential_clues": list(response.get("differential_clues", [])),
        }

    def score_options(
        self,
        example: dict[str, Any],
        frame_paths: list[str],
        frame_findings: list[dict[str, Any]],
        fusion: dict[str, Any],
    ) -> list[float]:
        prompt = (
            "Return JSON only. Prefer an object with key option_scores whose value is a list of floats in [0,1], "
            "one per answer option. Score each option by how well it is supported by the images, frame findings, "
            "and multi-frame fusion. Be especially careful with anatomical localization and temporal ordering.\n"
            f"Question: {example['question']}\n"
            f"Options: {example['options']}\n"
            f"FrameFindings: {json.dumps(frame_findings)}\n"
            f"Fusion: {json.dumps(fusion)}"
        )
        default = {"option_scores": [1.0 / len(example["options"])] * len(example["options"])}
        response = self._query_json(prompt, frame_paths, default)
        scores = response.get("option_scores") if isinstance(response, dict) else response
        return _sanitize_option_scores(scores, len(example["options"]))

    def verify_answer(
        self,
        example: dict[str, Any],
        frame_paths: list[str],
        frame_findings: list[dict[str, Any]],
        fusion: dict[str, Any],
        predicted_index: int,
    ) -> dict[str, Any]:
        prompt = (
            "Return JSON only with keys verifier_passed, rationale, support_score, contradiction_score. "
            "Judge whether the proposed option is truly supported by the images and the multi-frame evidence. "
            "If the option depends on the wrong anatomy, wrong frame relation, or weak evidence, fail it.\n"
            f"Question: {example['question']}\n"
            f"Options: {example['options']}\n"
            f"PredictedIndex: {predicted_index}\n"
            f"PredictedOption: {example['options'][predicted_index]}\n"
            f"FrameFindings: {json.dumps(frame_findings)}\n"
            f"Fusion: {json.dumps(fusion)}"
        )
        default = {
            "verifier_passed": False,
            "rationale": "fallback verifier",
            "support_score": 0.2,
            "contradiction_score": 0.7,
        }
        response = self._query_json(prompt, frame_paths, default)
        if not isinstance(response, dict):
            return default
        return {
            "verifier_passed": bool(response.get("verifier_passed", False)),
            "rationale": str(response.get("rationale", "")),
            "support_score": _clip01(response.get("support_score", 0.2)),
            "contradiction_score": _clip01(response.get("contradiction_score", 0.7)),
        }

    def answer_direct(self, example: dict[str, Any], frame_paths: list[str]) -> dict[str, Any]:
        prompt = (
            "You are answering a multi-image medical VQA question directly. "
            "Return JSON only with keys answer_letter, confidence, rationale, option_scores. "
            "answer_letter must be one capital letter that matches the best option.\n"
            f"Question: {example['question']}\n"
            f"Options: {example['options']}"
        )
        default = {
            "answer_letter": "A",
            "confidence": 0.25,
            "rationale": "fallback direct answer",
            "option_scores": [1.0 / len(example["options"])] * len(example["options"]),
        }
        response = self._query_json(prompt, frame_paths, default)
        if not isinstance(response, dict):
            response = default
        scores = _sanitize_option_scores(response.get("option_scores"), len(example["options"]))
        answer_index, answer_letter = resolve_answer_choice(
            response.get("answer_index"),
            response.get("answer_letter", ""),
            len(example["options"]),
        )
        if not (0 <= answer_index < len(example["options"])):
            answer_index = max(range(len(scores)), key=lambda idx: scores[idx])
            answer_letter = answer_index_to_letter(answer_index)
        return {
            "answer_index": answer_index,
            "answer_letter": answer_letter,
            "confidence": _clip01(response.get("confidence", max(scores))),
            "rationale": str(response.get("rationale", "")),
            "option_scores": scores,
        }


class HFVLMBackend(BaseTaskBackend):
    def __init__(self, cfg: BackendConfig):
        super().__init__()
        self.cfg = cfg
        self.processor, self.model, self.device = _load_hf_vlm_bundle(cfg)

    def _generate_json(self, prompt: str, frame_paths: list[str], default: Any) -> Any:
        if torch is None:
            raise RuntimeError("torch is required for the HF VLM backend")

        images = _load_pil_images(frame_paths[: self.cfg.max_frames])
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = _apply_hf_chat_template(
            self.processor,
            self.cfg.family,
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        input_len = int(inputs["input_ids"].shape[-1])

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.cfg.max_tokens),
            "do_sample": bool(self.cfg.temperature and self.cfg.temperature > 0),
        }
        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = float(self.cfg.temperature)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        self.total_tokens += int(generated_ids.shape[-1])
        text = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )[0]
        return _safe_json_loads(text, default)

    def describe_frames(self, example: dict[str, Any], frame_paths: list[str]) -> list[dict[str, Any]]:
        default = _sanitize_frame_findings([], frame_paths, example.get("organ", "unknown"))
        prompt = (
            "Return JSON only as a list with one object per image. "
            "Each object must contain frame_index, finding, anatomy, supports_question, confidence. "
            "Use concise clinically relevant findings and explicit anatomy/pathology phrases.\n"
            f"System: {example.get('system', 'unknown')}\n"
            f"Organ: {example.get('organ', 'unknown')}\n"
            f"Modality: {example.get('modality', 'unknown')}\n"
            f"Question: {example['question']}\n"
            f"Options: {example['options']}"
        )
        findings = self._generate_json(prompt, frame_paths, default)
        return _sanitize_frame_findings(findings, frame_paths, example.get("organ", "unknown"))

    def fuse_reasoning(
        self,
        example: dict[str, Any],
        frame_findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = (
            "Return JSON only with keys summary, key_evidence, differential_clues. "
            "Fuse the frame findings into a single multi-frame explanation and mention the strongest clues.\n"
            f"Question: {example['question']}\n"
            f"Options: {example['options']}\n"
            f"FrameFindings: {json.dumps(frame_findings)}"
        )
        default = {"summary": "fallback fusion", "key_evidence": [], "differential_clues": []}
        response = self._generate_json(prompt, [], default)
        if not isinstance(response, dict):
            return default
        return {
            "summary": str(response.get("summary", default["summary"])),
            "key_evidence": list(response.get("key_evidence", [])),
            "differential_clues": list(response.get("differential_clues", [])),
        }

    def score_options(
        self,
        example: dict[str, Any],
        frame_paths: list[str],
        frame_findings: list[dict[str, Any]],
        fusion: dict[str, Any],
    ) -> list[float]:
        prompt = (
            "Return JSON only. Prefer an object with key option_scores whose value is a list of floats in [0,1], "
            "one per answer option. Score support from the images, the frame findings, and the fused reasoning.\n"
            f"Question: {example['question']}\n"
            f"Options: {example['options']}\n"
            f"FrameFindings: {json.dumps(frame_findings)}\n"
            f"Fusion: {json.dumps(fusion)}"
        )
        default = {"option_scores": [1.0 / len(example["options"])] * len(example["options"])}
        response = self._generate_json(prompt, frame_paths, default)
        scores = response.get("option_scores") if isinstance(response, dict) else response
        return _sanitize_option_scores(scores, len(example["options"]))

    def verify_answer(
        self,
        example: dict[str, Any],
        frame_paths: list[str],
        frame_findings: list[dict[str, Any]],
        fusion: dict[str, Any],
        predicted_index: int,
    ) -> dict[str, Any]:
        prompt = (
            "Return JSON only with keys verifier_passed, rationale, support_score, contradiction_score. "
            "Verify whether the proposed option is truly supported by the full multi-image evidence.\n"
            f"Question: {example['question']}\n"
            f"Options: {example['options']}\n"
            f"PredictedIndex: {predicted_index}\n"
            f"PredictedOption: {example['options'][predicted_index]}\n"
            f"FrameFindings: {json.dumps(frame_findings)}\n"
            f"Fusion: {json.dumps(fusion)}"
        )
        default = {
            "verifier_passed": False,
            "rationale": "fallback verifier",
            "support_score": 0.2,
            "contradiction_score": 0.7,
        }
        response = self._generate_json(prompt, frame_paths, default)
        if not isinstance(response, dict):
            return default
        return {
            "verifier_passed": bool(response.get("verifier_passed", False)),
            "rationale": str(response.get("rationale", "")),
            "support_score": _clip01(response.get("support_score", 0.2)),
            "contradiction_score": _clip01(response.get("contradiction_score", 0.7)),
        }

    def answer_direct(self, example: dict[str, Any], frame_paths: list[str]) -> dict[str, Any]:
        prompt = (
            "Return JSON only with keys answer_letter, confidence, rationale, option_scores. "
            "Choose the best answer option from the medical images.\n"
            f"Question: {example['question']}\n"
            f"Options: {example['options']}"
        )
        default = {
            "answer_letter": "A",
            "confidence": 0.25,
            "rationale": "fallback direct answer",
            "option_scores": [1.0 / len(example["options"])] * len(example["options"]),
        }
        response = self._generate_json(prompt, frame_paths, default)
        if not isinstance(response, dict):
            response = default
        scores = _sanitize_option_scores(response.get("option_scores"), len(example["options"]))
        answer_index, answer_letter = resolve_answer_choice(
            response.get("answer_index"),
            response.get("answer_letter", ""),
            len(example["options"]),
        )
        if not (0 <= answer_index < len(example["options"])):
            answer_index = max(range(len(scores)), key=lambda idx: scores[idx])
            answer_letter = answer_index_to_letter(answer_index)
        return {
            "answer_index": answer_index,
            "answer_letter": answer_letter,
            "confidence": _clip01(response.get("confidence", max(scores))),
            "rationale": str(response.get("rationale", "")),
            "option_scores": scores,
        }


class HFMedGemmaBackend(HFVLMBackend):
    pass


def backend_from_runtime(runtime_cfg: dict[str, Any]) -> BaseTaskBackend:
    backend_cfg = runtime_cfg.get("task_backend", {})
    cfg = BackendConfig(
        backend_type=backend_cfg.get("type", "mock"),
        model_name=backend_cfg.get("model_name"),
        family=str(backend_cfg.get("family", "medgemma")),
        base_url=backend_cfg.get("base_url"),
        api_key=backend_cfg.get("api_key"),
        max_frames=int(backend_cfg.get("max_frames", 5)),
        temperature=float(backend_cfg.get("temperature", 0.0)),
        max_tokens=int(backend_cfg.get("max_tokens", 768)),
        mock_mode=str(backend_cfg.get("mock_mode", "hash")),
        dtype=str(backend_cfg.get("dtype", "auto")),
        device_map=str(backend_cfg.get("device_map", "auto")),
        adapter_mode=str(backend_cfg.get("adapter_mode", "base")),
        adapter_path=backend_cfg.get("adapter_path"),
        load_in_4bit=bool(backend_cfg.get("load_in_4bit", False)),
        trust_remote_code=bool(backend_cfg.get("trust_remote_code", False)),
        local_files_only=bool(backend_cfg.get("local_files_only", False)),
    )
    if cfg.backend_type == "mock":
        return MockVisionBackend(cfg)
    if cfg.backend_type == "openai_vision":
        return OpenAIVisionBackend(cfg)
    if cfg.backend_type == "hf_vlm":
        return HFVLMBackend(cfg)
    if cfg.backend_type == "hf_medgemma":
        cfg.family = "medgemma"
        return HFMedGemmaBackend(cfg)
    raise ValueError(f"Unknown task backend type: {cfg.backend_type}")
