#!/usr/bin/env bash
set -euo pipefail

OCI_BIN="${OCI_BIN:-$HOME/.oci-venv/bin/oci}"
SESSION_CONFIG="${OCI_SESSION_CONFIG:-/tmp/oci-session-bootstrap/session_config}"
PROFILE="${OCI_SESSION_PROFILE_NAME:-TEMP_SESSION}"
NAMESPACE="${OCI_NAMESPACE:-ax9iu0ga9n79}"
BUCKET="${OCI_BUCKET:-nemotron-kaggle-v9-backup}"
DEST_DIR="${1:-$PWD/oci-stage1-backup}"
TAR_NAME="kaggle-v9-stage1-minimal.tar"
SUM_NAME="kaggle-v9-stage1-minimal.tar.sha256"

mkdir -p "$DEST_DIR"

common_args=(--auth security_token --config-file "$SESSION_CONFIG" --profile "$PROFILE")
export OCI_CLI_SUPPRESS_FILE_PERMISSIONS_WARNING=True
export SUPPRESS_LABEL_WARNING=True

"$OCI_BIN" "${common_args[@]}" os object get --namespace-name "$NAMESPACE" --bucket-name "$BUCKET" --name "$SUM_NAME" --file "$DEST_DIR/$SUM_NAME"
"$OCI_BIN" "${common_args[@]}" os object get --namespace-name "$NAMESPACE" --bucket-name "$BUCKET" --name "$TAR_NAME" --file "$DEST_DIR/$TAR_NAME"

python3 - "$DEST_DIR/$SUM_NAME" "$DEST_DIR/$TAR_NAME" <<'PY'
import hashlib, sys
sum_path, tar_path = sys.argv[1], sys.argv[2]
expected = open(sum_path).read().strip().split()[0]
h = hashlib.sha256()
with open(tar_path, 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
actual = h.hexdigest()
if actual != expected:
    raise SystemExit(f"sha256 mismatch: expected {expected} got {actual}")
print(f"sha256 ok: {actual}")
PY

echo "Downloaded backup to: $DEST_DIR"
