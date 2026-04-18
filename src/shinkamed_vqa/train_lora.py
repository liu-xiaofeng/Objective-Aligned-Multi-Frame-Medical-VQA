from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except Exception as exc:  # pragma: no cover - runtime guarded
    raise RuntimeError("train_lora.py requires `peft` in the active environment.") from exc

try:
    from backend import _apply_hf_chat_template, _build_hf_text_image_inputs, _load_pil_images
    from dataset_utils import dump_json, load_runtime_config
except ImportError:
    from .backend import _apply_hf_chat_template, _build_hf_text_image_inputs, _load_pil_images
    from .dataset_utils import dump_json, load_runtime_config


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _bf16_ok() -> bool:
    return torch.cuda.is_available() and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())


def _resolve_target_modules(model: nn.Module, family: str) -> list[str]:
    hints_by_family = {
        "medgemma": ["projector", "connector", "vision_proj", "mm_projector", "multi_modal"],
        "qwen3_5": ["projector", "connector", "vision_proj", "mm_projector", "multi_modal", "merger", "visual"],
        "qwen2_5_vl": ["projector", "connector", "vision_proj", "mm_projector", "multi_modal", "merger", "visual"],
        "qwen_vl": ["projector", "connector", "vision_proj", "mm_projector", "multi_modal", "merger", "visual"],
    }
    hints = hints_by_family.get(str(family).strip().lower(), hints_by_family["medgemma"])
    found: list[str] = []
    candidate_names: list[str] = []
    for name, module in model.named_modules():
        lower = name.lower()
        if any(hint in lower for hint in hints):
            candidate_names.append(name)
        if isinstance(module, nn.Linear) and any(hint in lower for hint in hints):
            found.append(name)
    found = sorted(set(found))
    if found:
        return found
    preview = ", ".join(candidate_names[:20]) if candidate_names else "none"
    raise RuntimeError(
        f"Could not resolve bridge target modules for family `{family}`. Candidate module names: {preview}"
    )


def _extract_layer_index(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    match = re.search(r"\.h\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    return None


def _resolve_decoder_target_modules(
    model: nn.Module,
    *,
    suffixes: tuple[str, ...],
    last_n: int,
    include_vision: bool = True,
) -> list[str]:
    linear_names: list[tuple[int, str]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lower = name.lower()
        is_vision = "vision_tower" in lower or "vision_model" in lower or ".visual." in lower
        if not include_vision and is_vision:
            continue
        layer_idx = _extract_layer_index(name)
        if layer_idx is None:
            continue
        if name.endswith(suffixes):
            linear_names.append((layer_idx, name))
    if not linear_names:
        raise RuntimeError(f"Could not find decoder linear modules matching suffixes={suffixes}.")
    max_layer = max(idx for idx, _ in linear_names)
    min_layer = max(0, max_layer - max(int(last_n), 1) + 1)
    selected = [name for idx, name in linear_names if idx >= min_layer]
    if not selected:
        raise RuntimeError(
            f"Could not select decoder target modules for last_n={last_n} and suffixes={suffixes}."
        )
    return sorted(set(selected))


def _resolve_vision_target_modules(model: nn.Module, *, suffixes: tuple[str, ...], last_n: int) -> list[str]:
    linear_names: list[tuple[int, str]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lower = name.lower()
        if not ("vision_tower" in lower or "vision_model" in lower or ".visual." in lower):
            continue
        layer_idx = _extract_layer_index(name)
        if layer_idx is None:
            continue
        if name.endswith(suffixes):
            linear_names.append((layer_idx, name))
    if not linear_names:
        raise RuntimeError(f"Could not find vision linear modules matching suffixes={suffixes}.")
    max_layer = max(idx for idx, _ in linear_names)
    min_layer = max(0, max_layer - max(int(last_n), 1) + 1)
    selected = [name for idx, name in linear_names if idx >= min_layer]
    if not selected:
        raise RuntimeError(
            f"Could not select vision target modules for last_n={last_n} and suffixes={suffixes}."
        )
    return sorted(set(selected))


def _resolve_target_modules_from_strategy(model: nn.Module, family: str, strategy: str) -> list[str]:
    normalized = str(strategy or "bridge_linear").strip().lower()
    if normalized in {"bridge", "bridge_linear"}:
        return _resolve_target_modules(model, family)
    if normalized == "decoder_qv_last2":
        return _resolve_decoder_target_modules(model, suffixes=("q_proj", "v_proj"), last_n=2)
    if normalized == "decoder_qv_last4":
        return _resolve_decoder_target_modules(model, suffixes=("q_proj", "v_proj"), last_n=4)
    if normalized == "decoder_qv_last8":
        return _resolve_decoder_target_modules(model, suffixes=("q_proj", "v_proj"), last_n=8)
    if normalized == "decoder_qvo_last2":
        return _resolve_decoder_target_modules(model, suffixes=("q_proj", "v_proj", "o_proj"), last_n=2)
    if normalized == "decoder_qvo_last4":
        return _resolve_decoder_target_modules(model, suffixes=("q_proj", "v_proj", "o_proj"), last_n=4)
    if normalized == "decoder_qvo_last8":
        return _resolve_decoder_target_modules(model, suffixes=("q_proj", "v_proj", "o_proj"), last_n=8)
    if normalized == "decoder_qvo_last8_textonly":
        return _resolve_decoder_target_modules(
            model,
            suffixes=("q_proj", "v_proj", "o_proj"),
            last_n=8,
            include_vision=False,
        )
    if normalized == "vision_qv_last1":
        return _resolve_vision_target_modules(model, suffixes=("q_proj", "v_proj"), last_n=1)
    if normalized == "decoder_qvo_last8_plus_vision_qv_last1":
        modules = _resolve_decoder_target_modules(
            model,
            suffixes=("q_proj", "v_proj", "o_proj"),
            last_n=8,
            include_vision=False,
        )
        modules.extend(_resolve_vision_target_modules(model, suffixes=("q_proj", "v_proj"), last_n=1))
        return sorted(set(modules))
    raise RuntimeError(f"Unsupported LoRA target strategy: {strategy}")


def _load_model_and_processor(cfg: dict[str, Any]) -> tuple[Any, Any]:
    model_name = str(cfg["model_name"])
    trust_remote_code = bool(cfg.get("trust_remote_code", False))
    local_files_only = bool(cfg.get("local_files_only", False))
    torch_dtype = torch.bfloat16 if _bf16_ok() else torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )
    return processor, model


def _make_messages(sample: dict[str, Any], images: list[Any], target_text: str | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": sample["instruction_text"]})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if target_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        )
    return messages


@dataclass
class LocalizationCollator:
    processor: Any
    family: str
    max_frames: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(features) != 1:
            raise RuntimeError(
                "Localization LoRA training currently supports per-device batch size 1 only."
            )
        sample = features[0]
        images = _load_pil_images(sample["frame_paths"][: self.max_frames])
        user_messages = _make_messages(sample, images, target_text=None)
        full_messages = _make_messages(sample, images, target_text=sample["target_text"])

        user_text = _apply_hf_chat_template(
            self.processor,
            self.family,
            user_messages,
            add_generation_prompt=True,
            tokenize=False,
            return_dict=False,
            return_tensors=None,
        )
        full_text = _apply_hf_chat_template(
            self.processor,
            self.family,
            full_messages,
            add_generation_prompt=False,
            tokenize=False,
            return_dict=False,
            return_tensors=None,
        )
        user_inputs = _build_hf_text_image_inputs(
            self.processor,
            self.family,
            text=user_text,
            images=images,
            return_tensors="pt",
        )
        full_inputs = _build_hf_text_image_inputs(
            self.processor,
            self.family,
            text=full_text,
            images=images,
            return_tensors="pt",
        )
        labels = full_inputs["input_ids"].clone()
        user_len = int(user_inputs["input_ids"].shape[-1])
        labels[..., :user_len] = -100
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == int(pad_id)] = -100
        batch = {key: value for key, value in full_inputs.items()}
        batch["labels"] = labels
        return batch


def train_localization_lora(config_path: str) -> dict[str, Any]:
    cfg = load_runtime_config(config_path)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_rows = _load_jsonl(Path(data_cfg["train_path"]))
    val_rows = _load_jsonl(Path(data_cfg["val_path"]))
    output_dir = Path(cfg["training"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, model = _load_model_and_processor(model_cfg)
    target_strategy = str(cfg.get("lora", {}).get("target_strategy", "bridge_linear"))
    init_adapter_path = cfg.get("training", {}).get("init_adapter_path")
    if init_adapter_path:
        model = PeftModel.from_pretrained(
            model,
            str(Path(init_adapter_path).resolve()),
            is_trainable=True,
        )
        target_modules = list(getattr(model.peft_config.get("default", None), "target_modules", []) or [])
    else:
        target_modules = _resolve_target_modules_from_strategy(
            model,
            str(model_cfg.get("family", "medgemma")),
            target_strategy,
        )
        lora_cfg = LoraConfig(
            r=int(cfg["lora"].get("r", 16)),
            lora_alpha=int(cfg["lora"].get("alpha", 32)),
            lora_dropout=float(cfg["lora"].get("dropout", 0.05)),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
    if bool(cfg["training"].get("gradient_checkpointing", True)):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model.train()

    collator = LocalizationCollator(
        processor=processor,
        family=str(model_cfg.get("family", "medgemma")),
        max_frames=int(cfg.get("data", {}).get("max_frames", model_cfg.get("max_frames", 5))),
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=float(cfg["training"].get("num_train_epochs", 1.0)),
        learning_rate=float(cfg["training"].get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(cfg["training"].get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg["training"].get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg["training"].get("gradient_accumulation_steps", 8)),
        warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.03)),
        logging_steps=int(cfg["training"].get("logging_steps", 10)),
        eval_steps=int(cfg["training"].get("eval_steps", 50)),
        save_steps=int(cfg["training"].get("save_steps", 50)),
        save_total_limit=int(cfg["training"].get("save_total_limit", 2)),
        bf16=bool(cfg["training"].get("bf16", _bf16_ok())),
        fp16=bool(cfg["training"].get("fp16", False)),
        remove_unused_columns=False,
        report_to=[],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=val_rows,
        data_collator=collator,
    )
    train_result = trainer.train()
    if trainer.is_world_process_zero():
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))

    summary = {
        "output_dir": str(output_dir),
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "target_strategy": target_strategy,
        "target_modules": target_modules,
        "init_adapter_path": str(Path(init_adapter_path).resolve()) if init_adapter_path else None,
        "train_runtime": float(train_result.metrics.get("train_runtime", 0.0) or 0.0),
        "train_loss": float(train_result.metrics.get("train_loss", 0.0) or 0.0),
        "global_step": int(train_result.metrics.get("global_step", 0) or 0),
    }
    if trainer.is_world_process_zero():
        dump_json(summary, output_dir / "train_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = train_localization_lora(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
