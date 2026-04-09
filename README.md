# NVIDIA KAGGLE COMPETITION

This is the single canonical repo for the NVIDIA Nemotron Model Reasoning Challenge work.

## Repo status

- Canonical repo: `Rohan5commit/NVIDIA-KAGGLE-COMPETITION`
- This is the only repo that should be used for this competition effort
- Storage rule: use GitHub plus Kaggle only, not local task workspaces

## Goal

Build a high-accuracy LoRA adapter and submission pipeline for the Kaggle NVIDIA Nemotron Model Reasoning Challenge using Nemotron-3-Nano-30B.

## Live Kaggle snapshot

Snapshot taken at `2026-04-09 20:26:41 +0800`

- Kernel: `rohansan1/nemotron-reasoning-lora-trainer`
- Active kernel version: `9`
- Status at snapshot time: `RUNNING`
- GPU shape: `NvidiaRtxPro6000`
- Competition: `nvidia-nemotron-model-reasoning-challenge`
- Kaggle URL: https://www.kaggle.com/code/rohansan1/nemotron-reasoning-lora-trainer

## What has worked

- Runtime asset discovery was fixed for Kaggle's current `/kaggle/input` mount layout.
- The kernel is explicitly pinned to `NvidiaRtxPro6000`.
- Stage 1 SFT completed successfully in `v7`.
- Repo consolidation is complete and future work now continues from this repo only.
- Explicit progress telemetry was added to the pipeline for future status checks.
- The Kaggle launcher now embeds selected GitHub repo files directly into the kernel source before execution so the run can pick up hotfixes without relying on local files.

## What failed

- `v1` and `v2`: launcher and runtime bootstrap failures.
- `v3` and `v4`: Nemotron MoE `index_add_` dtype mismatch.
- `v5` and `v6`: PEFT LoRA bitsandbytes path failed with `fused_dropout not implemented for Byte`.
- `v7`: Stage 1 finished, then evaluation failed in `eval/local_eval.py` with `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`.
- `v8`: the new progress instrumentation launch failed immediately because `training/kaggle_kernel_entry.py` could not import repo-root `progress.py`.

## Current run intent

`v9` is the first run that includes both of these fixes:

- the eval fallback patch in `eval/local_eval.py` for the Nemotron CUDA kernel issue seen in `v7`
- the repo-root import path fix in `training/kaggle_kernel_entry.py` after the `v8` failure

## Important runtime constraint

Kaggle was not able to resolve `github.com` during live kernel execution. Because of that, runtime `git clone` should be treated as unreliable. The current launcher strategy is:

1. materialize the bundled runtime archive from Kaggle inputs
2. overwrite critical repo files at runtime from a patch payload embedded in the kernel source
3. keep using Kaggle-hosted assets for the rest of the pipeline

## Key files

- `common.py`
- `progress.py`
- `training/stage1_sft.py`
- `training/stage2_grpo.py`
- `training/kaggle_kernel_entry.py`
- `eval/local_eval.py`
- `submission/package_lora.py`
- `training/train_config.yaml`
- `notebooks/solution_writeup.ipynb`

## Operating rule

Do not create local task workspaces or store project artifacts locally for this effort. Keep task code, notes, and state in GitHub and Kaggle only.
