# AGENTS

## Repo role

This is the single canonical repo for the NVIDIA Nemotron Kaggle competition work.

Do not use local task directories for this project. Keep project files in this repo or on Kaggle only.

## Current repo state

- Canonical repo: Rohan5commit/NVIDIA-KAGGLE-COMPETITION
- Repo consolidation completed on 2026-04-09
- The temporary duplicate repo was deleted
- There is now one repo for this competition effort

## Current live run

Snapshot taken at 2026-04-09 14:48:04 +08

- Kernel: rohansan1/nemotron-reasoning-lora-trainer
- Version: 7
- Status at snapshot time: RUNNING
- GPU: NvidiaRtxPro6000

## Non-negotiable rules

- Do not interrupt the active Kaggle run unless explicitly instructed.
- Do not create local project workspaces for this task.
- Do not store handoff notes locally.
- Do not assume percent done from Kaggle unless it is backed by explicit progress artifacts.
- On the next Kaggle run, add persistent progress reporting so status checks can read exact step counts.
- Keep future work in this repo only.

## Kaggle resources

- Competition: nvidia-nemotron-model-reasoning-challenge
- Active kernel: rohansan1/nemotron-reasoning-lora-trainer
- Runtime asset datasets referenced by the kernel metadata:
  - rohansan1/nemotron-runtime-assets
  - rohansan1/nemotron-runtime-repo
- Emergency backup dataset previously created during recovery:
  - rohansan1/nemotron-emergency-backup-20260408-235729

## Known version history

- v1 to v2: early launcher and runtime failures
- v3 to v4: stage-1 reached, then Nemotron MoE index_add dtype mismatch
- v5 to v6: later PEFT and bitsandbytes LoRA dropout failure with Byte input
- v7: current run after launcher-side fixes; still running at the snapshot time above

## First actions for the next agent

1. Check whether v7 finished, failed, or produced useful outputs.
2. Pull the final Kaggle outputs and store them from this repo path, not from a local workspace.
3. Update any remaining launcher or clone URLs that still point to the pre-rename repo name.
4. Add exact progress telemetry before the next Kaggle kernel version is pushed.
5. Continue using this repo only.
