# HANDOFF

## Purpose

This document records what has happened so far so another developer or agent can continue without reconstructing the full history.

## Objective

Target competition:
- nvidia-nemotron-model-reasoning-challenge

Target deliverable:
- a high-accuracy LoRA adapter and submission package for Nemotron reasoning evaluation on Kaggle

## Constraint change

As of 2026-04-09, project storage for this effort is GitHub plus Kaggle only.

Do not create local task workspaces or keep project notes or artifacts locally.

## Repos

- Canonical continuation repo: Rohan5commit/NVIDIA-KAGGLE-COMPETITION
- Legacy repo with the original code history: Rohan5commit/nemotron-reasoning-lora

Current fact:
- there are currently two private repos related to this competition on the account
- the canonical repo going forward is NVIDIA-KAGGLE-COMPETITION
- the legacy repo should be treated as migration source and historical context until the active Kaggle run is safely concluded

## Active Kaggle snapshot

Captured at 2026-04-09 14:48:04 +08

- kernel: rohansan1/nemotron-reasoning-lora-trainer
- version: 7
- status: RUNNING
- machine shape: NvidiaRtxPro6000

## What was done successfully

### Kaggle recovery and relaunch

- An emergency backup dataset was created to preserve state during stop and restart handling.
- The Kaggle kernel was recreated and republished after earlier failed versions.
- The launcher was updated to handle delayed Kaggle mounts and the newer /kaggle/input layout.
- The kernel was pinned to NvidiaRtxPro6000 and that machine shape was later verified via the Kaggle API.

### Pipeline availability

- The competition repo codebase was functional enough to reach the stage-1 SFT path.
- The pipeline includes data scripts, stage-1 SFT, stage-2 GRPO, evaluation, notebook, and adapter packaging scripts.
- The current code, config, notebook, and summary set has now been migrated into the canonical repo.

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

## Source migration result

Migrated into this canonical repo:
- pipeline source files
- config files
- notebook
- packaging code
- evaluation code
- runtime launcher files
- artifact summary JSON files

Not yet migrated into this canonical repo:
- data/processed/*.jsonl large processed data files from the legacy repo

Reason:
- remote-only migration path
- keep the canonical repo focused on executable source and handoff context first

## Required next actions

1. Wait for v7 to finish or fail.
2. Capture v7 outputs from Kaggle.
3. Store useful outputs in canonical storage only.
4. Update any remaining Kaggle launcher references from nemotron-reasoning-lora to NVIDIA-KAGGLE-COMPETITION.
5. Add exact progress reporting before launching the next Kaggle version.
6. Once continuity is safe, archive or retire the legacy repo so there is only one active repo for this competition.
