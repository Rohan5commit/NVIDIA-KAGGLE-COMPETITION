# HANDOFF

## Objective

Train and submit a Nemotron reasoning LoRA for Kaggle competition:

- `nvidia-nemotron-model-reasoning-challenge`
- Kernel: `rohansan1/nemotron-reasoning-lora-trainer`
- Canonical repo: `Rohan5commit/NVIDIA-KAGGLE-COMPETITION`

## Current status (2026-04-10)

- Latest kernel version: `10`
- Session status: `CANCEL_ACKNOWLEDGED`
- Hard blocker: Kaggle weekly GPU quota exhausted
  - API error: `Maximum weekly GPU quota of 30.00 hours reached.`
  - Seen from both:
    - `SaveKernel`
    - `CreateKernelSession`

## What was fixed in this cycle

1. Added robust progress and checkpoint behavior:
   - Stage 1 + Stage 2 save every 15 minutes
   - Auto-resume from last checkpoint
2. Fixed launcher repo URL to canonical repo:
   - `kaggle/runtime_kernel/run_pipeline_kernel.py`
3. Added non-interactive kernel control utility:
   - `kaggle/control_kernel.py`
   - Supports:
     - `status` (includes `% done` and ETA from progress artifacts)
     - `ensure-running` (starts kernel when not running)
4. Added GitHub auto-relaunch workflow:
   - `.github/workflows/kaggle-kernel-autoresume.yml`
   - Runs every 15 minutes
   - Uses secret `KAGGLE_API_TOKEN`

## Immediate next steps

1. Set repository secret:
   - `KAGGLE_API_TOKEN` (Kaggle access token)
2. Trigger workflow manually:
   - `Kaggle Kernel Auto Resume`
3. Let schedule continue retries until quota reset.
4. Once status is `RUNNING`, monitor:
   - `python kaggle/control_kernel.py status --kernel rohansan1/nemotron-reasoning-lora-trainer`


## OCI resume source (2026-04-24)

The canonical off-GitHub backup for Kaggle Stage 1 is now OCI Object Storage.

- Region: `ap-singapore-1`
- Namespace: `ax9iu0ga9n79`
- Bucket: `nemotron-kaggle-v9-backup`
- Object: `kaggle-v9-stage1-minimal.tar`
- Checksum object: `kaggle-v9-stage1-minimal.tar.sha256`
- Manifest: `artifacts/oci_backup_manifest.json`

When resuming Kaggle later, pull from OCI first. Do not look for the deleted GitHub release.

Helpful scripts:

- `scripts/oci_session_bootstrap.sh`
- `scripts/oci_kaggle_stage1_pull.sh`

## Notes on API behavior

- Cancel endpoint now requires `kernel_session_id`.
- Session status endpoint does not provide `kernel_session_id`.
- Kernel output endpoint can include very large logs; do not serialize the entire response unless needed.

## Files touched this cycle

- `.github/workflows/kaggle-kernel-autoresume.yml`
- `kaggle/control_kernel.py`
- `kaggle/runtime_kernel/run_pipeline_kernel.py`
- `README.md`
- `AGENTS.md`
