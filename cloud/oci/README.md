# OCI backup and login notes

This repo's canonical off-GitHub backup for the Kaggle Nemotron run is now OCI Object Storage.

## Hard guardrail

- Stay within OCI Always Free only.
- Do **not** create paid compute, paid block volumes, paid databases, or paid load balancers.
- Use Object Storage only when total stored data remains below the user's requested `20 GB` safety ceiling.

## Canonical backup location

- Region: `ap-singapore-1`
- Namespace: `ax9iu0ga9n79`
- Bucket: `nemotron-kaggle-v9-backup`
- Objects:
  - `kaggle-v9-stage1-minimal.tar`
  - `kaggle-v9-stage1-minimal.tar.sha256`

See `artifacts/oci_backup_manifest.json` for exact sizes and checksums.

## Standard agent workflow

1. Authenticate to OCI with `scripts/oci_session_bootstrap.sh`.
2. Download the Kaggle Stage 1 backup with `scripts/oci_kaggle_stage1_pull.sh`.
3. Use that OCI backup, not a GitHub release, when resuming Kaggle work.
4. Remove temp session files after the task is complete.

## Important note

The old GitHub release copy for this backup was intentionally deleted after OCI verification.
