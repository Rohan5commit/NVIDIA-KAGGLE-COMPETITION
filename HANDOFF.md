# HANDOFF

## Purpose

This document records the current state of the Kaggle competition work so another developer or agent can continue without rebuilding context from scratch.

## Objective

- Competition: `nvidia-nemotron-model-reasoning-challenge`
- Deliverable: a high-accuracy Nemotron reasoning LoRA adapter and submission package
- Canonical repo: `Rohan5commit/NVIDIA-KAGGLE-COMPETITION`

## Storage rule

Use GitHub plus Kaggle only.

Do not create local task workspaces or keep project artifacts locally.

## Current Kaggle snapshot

Captured at `2026-04-09 20:26:41 +0800`

- Kernel: `rohansan1/nemotron-reasoning-lora-trainer`
- Active version: `9`
- Status: `RUNNING`
- Machine shape: `NvidiaRtxPro6000`

## Important operational constraint

Kaggle runtime network access could not resolve `github.com` during kernel execution.

Observed in live kernel log:
- `fatal: unable to access 'https://github.com/Rohan5commit/NVIDIA-KAGGLE-COMPETITION.git/': Could not resolve host: github.com`

Implication:
- do not rely on runtime `git clone` for hotfix delivery
- the launcher must continue embedding patched GitHub file contents directly into the Kaggle kernel source

## What is confirmed to work

- The Kaggle kernel is pinned to `NvidiaRtxPro6000`.
- Stage 1 SFT completed successfully in `v7`.
- The repo has been consolidated into one canonical repo.
- Progress telemetry code now exists in the repo:
  - `progress.py`
  - `training/kaggle_kernel_entry.py`
  - `training/stage1_sft.py`
  - `training/stage2_grpo.py`
- The Kaggle launcher has been republished to inject updated repo files at runtime from an embedded patch payload.

## Failure chronology

### v1 and v2

- early launcher and runtime bootstrap failures
- mount layout assumptions were wrong for the current Kaggle environment

### v3 and v4

- stage 1 training started
- failed in Nemotron MoE
- root issue: `index_add_` dtype mismatch

### v5 and v6

- progressed past the MoE dtype issue
- failed in the PEFT LoRA bitsandbytes path
- root issue: `fused_dropout not implemented for Byte`

### v7

- stage 1 SFT completed successfully
- training summary was written under the stage 1 outputs
- then evaluation failed in `eval/local_eval.py`
- root issue: `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`

### v8

- intended to add progress reporting and the eval fallback fix
- failed immediately after launch
- root issue: `training/kaggle_kernel_entry.py` imported `progress.py` from the repo root without first inserting the repo root onto `sys.path`
- exact failure: `ModuleNotFoundError: No module named 'progress'`

### v9

- includes the `eval/local_eval.py` fallback fix for the `v7` CUDA kernel issue
- includes the `training/kaggle_kernel_entry.py` repo-root import fix for the `v8` failure
- includes runtime progress telemetry
- currently `RUNNING`

## Files that matter most for the current run

- `common.py`
- `progress.py`
- `training/stage1_sft.py`
- `training/stage2_grpo.py`
- `training/kaggle_kernel_entry.py`
- `eval/local_eval.py`
- `submission/package_lora.py`

## How the current Kaggle launcher works

1. Wait for Kaggle input mounts.
2. Materialize the repo from the bundled Kaggle runtime archive.
3. Attempt GitHub sync, but tolerate failure.
4. Overwrite critical repo files in the working tree from a patch payload embedded in the kernel source.
5. Force runtime-safe config overrides:
   - `attn_implementation: eager`
   - `loftq_init: false`
6. Run `training/kaggle_kernel_entry.py` in `stage1_fast` mode.

## Immediate next actions

1. Keep monitoring `v9`.
2. Once session outputs become visible, confirm these files are being emitted:
   - `nemotron-run-progress.json`
   - `nemotron-run-progress-events.jsonl`
   - `nemotron-kernel-run.log`
3. If `v9` fails, inspect the latest kernel log before changing anything else.
4. If `v9` reaches the eval stage successfully, continue to packaging and leaderboard submission work.
5. Continue storing all continuity notes in this repo or on Kaggle only.
