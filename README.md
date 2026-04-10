# NVIDIA KAGGLE COMPETITION

Canonical repository for the NVIDIA Nemotron Model Reasoning Challenge work.

## Canonical identifiers

- GitHub repo: `Rohan5commit/NVIDIA-KAGGLE-COMPETITION`
- Kaggle kernel: `rohansan1/nemotron-reasoning-lora-trainer`
- Kaggle competition: `nvidia-nemotron-model-reasoning-challenge`

## Live status snapshot (2026-04-10)

- Latest kernel version: `v10`
- Kernel session state: `CANCEL_ACKNOWLEDGED`
- Current blocker: `Maximum weekly GPU quota of 30.00 hours reached.`
- Attempted restart paths:
  - `save_kernel` with relaunch token: blocked by quota
  - `create_kernel_session`: blocked by quota (`HTTP 429 RESOURCE_EXHAUSTED`)

## What is already implemented

- Data scripts:
  - `data/download_datasets.py`
  - `data/filter_and_curate.py`
  - `data/generate_synthetic.py`
- Training scripts:
  - `training/stage1_sft.py`
  - `training/stage2_grpo.py`
  - `training/train_config.yaml`
- Eval and packaging:
  - `eval/local_eval.py`
  - `submission/package_lora.py`
- Runtime launcher and telemetry:
  - `kaggle/runtime_kernel/run_pipeline_kernel.py`
  - `training/kaggle_kernel_entry.py`
  - `progress.py`

## Recent fixes (before quota block)

- Added 15-minute checkpoint saves in Stage 1 and Stage 2.
- Added automatic resume from latest checkpoint.
- Kept GRPO offloading disabled (`offload_optimizer=false`, `offload_reference_model=false`).
- Fixed runtime repo URL in launcher to:
  - `https://github.com/Rohan5commit/NVIDIA-KAGGLE-COMPETITION.git`
- Added automated kernel relaunch workflow:
  - `.github/workflows/kaggle-kernel-autoresume.yml`
  - `kaggle/control_kernel.py`

## Auto-resume setup

GitHub Action expects secret:

- `KAGGLE_API_TOKEN`

When set, the workflow runs every 15 minutes and attempts:

1. Read kernel status + progress snapshot
2. Start a new kernel session if not running
3. Exit cleanly when blocked only by weekly quota

## Outputs and progress

The kernel writes structured progress to:

- `artifacts/remote/run_progress.json`
- `artifacts/remote/run_progress_events.jsonl`

The status helper can read these and compute `% done` and ETA:

```bash
python kaggle/control_kernel.py status --kernel rohansan1/nemotron-reasoning-lora-trainer
```

## Operational rule

Keep all competition code, status notes, and automation in this repo and Kaggle.
