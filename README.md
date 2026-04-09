# NVIDIA KAGGLE COMPETITION

This is the canonical continuation repo for the NVIDIA Nemotron Model Reasoning Challenge work.

## Canonical repo status

- Canonical repo for all future work: Rohan5commit/NVIDIA-KAGGLE-COMPETITION
- Legacy repo that already existed: Rohan5commit/nemotron-reasoning-lora
- There are currently two private competition-related repos on the account.
- Until the active Kaggle v7 run finishes, treat nemotron-reasoning-lora as legacy and read-only history.
- All future work should go here.

## Goal

Build a high-accuracy LoRA adapter and submission pipeline for the Kaggle NVIDIA Nemotron Model Reasoning Challenge using the Nemotron-3-Nano-30B model family.

## Live Kaggle snapshot

Snapshot taken at 2026-04-09 14:48:04 +08

- Kernel: rohansan1/nemotron-reasoning-lora-trainer
- Kernel version: 7
- Status at snapshot time: RUNNING
- GPU shape: NvidiaRtxPro6000
- Competition: nvidia-nemotron-model-reasoning-challenge
- Kaggle URL: https://www.kaggle.com/code/rohansan1/nemotron-reasoning-lora-trainer

## What is in this repo

This repo now contains the migrated source, config, notebook, runtime launcher files, evaluation code, packaging code, and artifact summaries from the legacy repo.

Included paths now cover:

- common.py
- data/*.py
- training/*.py
- eval/local_eval.py
- submission/package_lora.py
- notebooks/solution_writeup.ipynb
- training/train_config.yaml
- artifacts/*.json

## What is not yet in this repo

The large processed dataset files from the legacy repo were intentionally not mirrored during the remote-only migration:

- data/processed/curated_full.jsonl
- data/processed/synthetic_seed_candidates.jsonl
- data/processed/train_grpo.jsonl
- data/processed/train_sft.jsonl
- data/processed/validation.jsonl

Those can be regenerated from code or moved from Kaggle outputs and datasets after the active run finishes.

## What has worked so far

- The private Kaggle kernel was recreated and republished through multiple versions.
- The machine shape was explicitly pinned and later verified as NvidiaRtxPro6000.
- Runtime asset discovery was improved for the newer Kaggle input mount layout.
- Repo archive fallback worked when GitHub access inside Kaggle failed.
- The pipeline advanced far enough to reach stage-1 training instead of failing during early bootstrap.
- Compatibility fixes were identified for Nemotron plus PEFT plus bitsandbytes runtime issues.
- The current v7 run passed the earlier failure window and was still running at the snapshot time above.

## What has not worked so far

- v1 and v2 failed during earlier launcher and runtime setup.
- One earlier run landed on the wrong accelerator class before the machine shape was pinned correctly.
- v3 and v4 reached stage-1 but failed inside Nemotron MoE with an index_add dtype mismatch.
- v5 and v6 progressed further but failed in the PEFT LoRA bitsandbytes path with fused_dropout not implemented for Byte.
- Kaggle script kernels did not expose reliable live train-step counters through the normal status endpoint, so exact percent done was not available from the active run.

## Next actions after the active run ends

1. Capture the final outputs of v7 from Kaggle.
2. Commit or export useful artifacts into this repo or Kaggle from this repo, not to local disk.
3. Repoint remaining launcher or repo references so the next run uses NVIDIA-KAGGLE-COMPETITION instead of the legacy repo.
4. Add explicit run progress telemetry for the next Kaggle version.
5. Decide whether to archive the old nemotron-reasoning-lora repo after v7 is no longer needed.

## Storage rule

Do not create local task workspaces or keep project artifacts on the local machine for this effort. Future code, notes, and handoff files must live in this repo or on Kaggle.
