

It keeps only the runnable Python path for MedFrameQA data preparation, direct-answer LoRA training, evaluation, and post-hoc calibration.

## What is included

- `src/shinkamed_vqa/`: package code
- `scripts/`: thin CLI entry points
- `configs/`: minimal runtime and training templates
- `data/processed/`: bundled split metadata
- `results/`: output root for experiments

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Set the repo root explicitly so all scripts resolve paths the same way:

```bash
export SHINKAMED_VQA_ROOT=$PWD
```

## Prepare MedFrameQA

This downloads the Hugging Face dataset metadata, optionally snapshots image bytes, writes a local manifest, and creates the public split file.

```bash
python scripts/prepare_medframeqa.py
```

Outputs:

- `data/processed/medframeqa_manifest.jsonl`
- `data/processed/medframeqa_split_ids.json`
- `data/processed/medframeqa_summary.json`

If you only want metadata and do not want to snapshot image bytes locally:

```bash
python scripts/prepare_medframeqa.py --skip_snapshot
```

## Quick Debug Evaluation

Use the mock backend first to verify the repo wiring:

```bash
python scripts/evaluate_program.py \
  --program-path src/shinkamed_vqa/programs/vanilla_direct.py \
  --results-dir results/debug_eval \
  --config configs/runtime_local_debug.yaml \
  --split-name mini_debug
```

## Direct-Answer LoRA Pipeline

This is the main public training path. It builds answer-only supervision, trains a LoRA adapter, and runs base-vs-adapter evaluation on `calibration_val` and `report_test`.

MedGemma example:

```bash
python scripts/run_direct_answer_pipeline.py \
  --results-root results/direct_answer_medgemma \
  --runtime-config configs/runtime_medgemma_eval.yaml \
  --train-template configs/train_lora_medgemma.yaml \
  --target-strategy decoder_qvo_last8_textonly \
  --lora-r 16 \
  --lora-alpha 32 \
  --num-train-epochs 1.0 \
  --learning-rate 2e-4 \
  --gradient-accumulation-steps 8
```

Qwen example:

```bash
python scripts/run_direct_answer_pipeline.py \
  --results-root results/direct_answer_qwen \
  --runtime-config configs/runtime_qwen_eval.yaml \
  --train-template configs/train_lora_qwen.yaml \
  --target-strategy decoder_qvo_last8_textonly \
  --lora-r 16 \
  --lora-alpha 32 \
  --num-train-epochs 1.0 \
  --learning-rate 2e-4 \
  --gradient-accumulation-steps 8
```

Main artifacts:

- `results/.../train.jsonl`
- `results/.../val.jsonl`
- `results/.../adapter/`
- `results/.../posthoc_eval/posthoc_eval_summary.json`
- `results/.../pipeline_summary.json`

## Standalone Evaluation

Evaluate the bundled vanilla direct program on a chosen split:

```bash
python scripts/evaluate_program.py \
  --program-path src/shinkamed_vqa/programs/vanilla_direct.py \
  --results-dir results/vanilla_report_test \
  --config configs/runtime_medgemma_eval.yaml \
  --split-name report_test
```

## Post-Hoc Evaluation Only

If you already trained an adapter and only want the base-vs-adapter comparison:

```bash
python scripts/run_posthoc_eval.py \
  --runtime-config configs/runtime_medgemma_eval.yaml \
  --results-root results/posthoc_eval_only \
  --adapter-dirs results/direct_answer_medgemma/adapter
```

## Post-Hoc Calibration

After the direct pipeline finishes, run confidence calibration on the adapter outputs:

```bash
python scripts/run_posthoc_calibration.py \
  --results-root results/direct_answer_medgemma \
  --output-root results/direct_answer_medgemma/posthoc_calibration
```

This writes `posthoc_calibration_summary.json` with identity, temperature, and histogram calibration comparisons.

## Notes

- The public-only repo intentionally omits the older controller-search, policy-editing, and paper-bundle code.
- The evaluation path here is sequential and does not depend on `ShinkaEvolve`.
- `backend.py` still supports OpenAI-compatible and Hugging Face backends, but the default public configs focus on local Hugging Face VLM inference.
