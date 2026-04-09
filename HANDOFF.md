# HANDOFF

## Purpose

This document records what has happened so far so another developer or agent can continue without reconstructing the full history.

## Objective

Target competition:
- nvidia-nemotron-model-reasoning-challenge

Target deliverable:
- a high-accuracy LoRA adapter and submission package for Nemotron reasoning evaluation on Kaggle

## Storage constraint

As of 2026-04-09, project storage for this effort is GitHub plus Kaggle only.

Do not create local task workspaces or keep project notes or artifacts locally.

## Repo consolidation result

Repo consolidation is complete.

- Final repo: Rohan5commit/NVIDIA-KAGGLE-COMPETITION
- The temporary duplicate repo was deleted
- There is now one repo for this competition effort

This repo is the one future agents and developers should use.

## Active Kaggle snapshot

Captured at 2026-04-09 14:48:04 +08

- kernel: rohansan1/nemotron-reasoning-lora-trainer
- version: 7
- status: RUNNING
- machine shape: NvidiaRtxPro6000

The active v7 run was not interrupted during repo consolidation.

## What was done successfully

### Kaggle recovery and relaunch

- An emergency backup dataset was created to preserve state during stop and restart handling.
- The Kaggle kernel was recreated and republished after earlier failed versions.
- The launcher was updated to handle delayed Kaggle mounts and the newer /kaggle/input layout.
- The kernel was pinned to NvidiaRtxPro6000 and that machine shape was later verified via the Kaggle API.

### Pipeline availability

- The competition repo codebase was functional enough to reach the stage-1 SFT path.
- The pipeline includes data scripts, stage-1 SFT, stage-2 GRPO, evaluation, notebook, and adapter packaging scripts.
- The code, config, notebook, processed dataset files, and summary files are present in this repo.
- Repo consolidation completed so future work can continue from one repo only.

## What failed

### Versions v1 and v2

- early launcher and runtime failures
- missing assumptions around input mount layout
- repo sync and clone fallback issues

### Versions v3 and v4

- stage-1 training started but failed in Nemotron MoE
- core error family: dtype mismatch in index_add

### Versions v5 and v6

- progressed past the earlier MoE issue
- failed in the PEFT LoRA bitsandbytes forward path
- core error: fused_dropout not implemented for Byte

## Why exact percent done was missing

Kaggle's normal kernel status API only exposed coarse states like RUNNING and ERROR for this script kernel. It did not provide live train-step counters. That forced status reporting to use forecasted progress instead of measured progress.

This should be fixed on the next run by writing explicit progress state from the training pipeline itself.

## Required next actions

1. Wait for v7 to finish or fail.
2. Capture v7 outputs from Kaggle.
3. Store useful outputs in canonical storage only.
4. Update any remaining Kaggle launcher references to the final repo name NVIDIA-KAGGLE-COMPETITION.
5. Add exact progress reporting before launching the next Kaggle version.
6. Continue work in this repo only.
