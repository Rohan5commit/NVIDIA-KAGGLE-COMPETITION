# AGENTS

## Scope

Single canonical repo for this Kaggle effort:

- `Rohan5commit/NVIDIA-KAGGLE-COMPETITION`

Do not create parallel competition repos.

## Active Kaggle target

- Kernel: `rohansan1/nemotron-reasoning-lora-trainer`
- Competition: `nvidia-nemotron-model-reasoning-challenge`
- GPU shape: `NvidiaRtxPro6000`

## Latest known state

- Kernel version: `v10`
- Session: `CANCEL_ACKNOWLEDGED`
- Start block: weekly Kaggle GPU quota exhausted (`30.00h`)

## Rules

- Keep project updates in this repo and Kaggle.
- Do not interrupt a running session unless explicitly instructed.
- Report `% done` and ETA from actual progress artifacts when available:
  - `artifacts/remote/run_progress.json`
  - `artifacts/remote/run_progress_events.jsonl`

## Automation now available

- `kaggle/control_kernel.py`
  - `status`: live state + progress + ETA
  - `ensure-running`: auto-start session if idle
- `.github/workflows/kaggle-kernel-autoresume.yml`
  - every 15 minutes
  - requires `KAGGLE_API_TOKEN` GitHub secret

## First actions for the next agent

1. Ensure secret `KAGGLE_API_TOKEN` exists in repo settings.
2. Run workflow `Kaggle Kernel Auto Resume`.
3. Confirm kernel transitions to `RUNNING`.
4. Monitor run with:
   - `python kaggle/control_kernel.py status --kernel rohansan1/nemotron-reasoning-lora-trainer`
