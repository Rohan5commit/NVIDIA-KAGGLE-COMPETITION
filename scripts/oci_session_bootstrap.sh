#!/usr/bin/env bash
set -euo pipefail

OCI_BIN="${OCI_BIN:-$HOME/.oci-venv/bin/oci}"
SOURCE_CONFIG="${OCI_CONFIG_SOURCE:-$HOME/.oci/config}"
SOURCE_PROFILE="${OCI_SOURCE_PROFILE:-DEFAULT}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-/tmp/oci-session-bootstrap}"
TARGET_PROFILE="${OCI_SESSION_PROFILE:-TEMP}"
SESSION_ROOT="${OCI_SESSION_ROOT:-$HOME/.oci/sessions}"

if [[ ! -x "$OCI_BIN" ]]; then
  echo "Missing OCI CLI at $OCI_BIN" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_CONFIG" ]]; then
  echo "Missing OCI config at $SOURCE_CONFIG" >&2
  exit 1
fi

mkdir -p "$BOOTSTRAP_DIR" "$SESSION_ROOT"

readarray -t CFG_VALUES < <(python3 - "$SOURCE_CONFIG" "$SOURCE_PROFILE" <<'PY'
import configparser, sys
cfg = configparser.ConfigParser()
cfg.read(sys.argv[1])
profile = sys.argv[2]
if profile not in cfg:
    raise SystemExit(f"Profile not found: {profile}")
section = cfg[profile]
for key in ("user", "tenancy", "region"):
    print(section.get(key, ""))
PY
)
USER_OCID="${CFG_VALUES[0]}"
TENANCY_OCID="${CFG_VALUES[1]}"
REGION="${CFG_VALUES[2]}"

if [[ -z "$USER_OCID" || -z "$TENANCY_OCID" || -z "$REGION" ]]; then
  echo "Could not read user/tenancy/region from $SOURCE_CONFIG [$SOURCE_PROFILE]" >&2
  exit 1
fi

TEMP_KEY="$BOOTSTRAP_DIR/oci_api_key"
TEMP_PUB_SSH="$BOOTSTRAP_DIR/oci_api_key.pub"
TEMP_PUB_PEM="$BOOTSTRAP_DIR/oci_api_key_pub.pem"
BOOTSTRAP_CONFIG="$BOOTSTRAP_DIR/config"
SESSION_CONFIG="$BOOTSTRAP_DIR/session_config"

rm -f "$TEMP_KEY" "$TEMP_PUB_SSH" "$TEMP_PUB_PEM" "$BOOTSTRAP_CONFIG" "$SESSION_CONFIG"
ssh-keygen -t rsa -b 2048 -m PEM -N '' -f "$TEMP_KEY" >/dev/null
openssl rsa -in "$TEMP_KEY" -pubout -out "$TEMP_PUB_PEM" >/dev/null 2>&1
TEMP_FINGERPRINT="$(openssl rsa -pubout -outform DER -in "$TEMP_KEY" 2>/dev/null | openssl md5 -c | awk '{print $2}')"

cat > "$BOOTSTRAP_CONFIG" <<CFG
[$TARGET_PROFILE]
user=$USER_OCID
tenancy=$TENANCY_OCID
region=$REGION
fingerprint=$TEMP_FINGERPRINT
key_file=$TEMP_KEY
CFG
chmod 600 "$BOOTSTRAP_CONFIG"

OCI_CLI_SUPPRESS_FILE_PERMISSIONS_WARNING=True SUPPRESS_LABEL_WARNING=True \
  "$OCI_BIN" \
  --config-file "$BOOTSTRAP_CONFIG" \
  --profile "$TARGET_PROFILE" \
  session authenticate \
  --region "$REGION" \
  --profile-name "$TARGET_PROFILE" \
  --config-location "$BOOTSTRAP_CONFIG" \
  --token-location "$SESSION_ROOT" \
  --public-key-file-path "$TEMP_PUB_PEM"

SESSION_KEY="$SESSION_ROOT/$TARGET_PROFILE/oci_api_key.pem"
SESSION_TOKEN="$SESSION_ROOT/$TARGET_PROFILE/token"
if [[ ! -f "$SESSION_KEY" || ! -f "$SESSION_TOKEN" ]]; then
  echo "Session auth finished without expected token/key files." >&2
  exit 1
fi
SESSION_FINGERPRINT="$(openssl rsa -pubout -outform DER -in "$SESSION_KEY" 2>/dev/null | openssl md5 -c | awk '{print $2}')"

cat > "$SESSION_CONFIG" <<CFG
[${TARGET_PROFILE}_SESSION]
user=$USER_OCID
tenancy=$TENANCY_OCID
region=$REGION
fingerprint=$SESSION_FINGERPRINT
key_file=$SESSION_KEY
security_token_file=$SESSION_TOKEN
CFG
chmod 600 "$SESSION_CONFIG"

cat <<OUT
OCI session ready.
Session config: $SESSION_CONFIG
Use it like this:
OCI_CLI_SUPPRESS_FILE_PERMISSIONS_WARNING=True SUPPRESS_LABEL_WARNING=True \
  "$OCI_BIN" --auth security_token --config-file "$SESSION_CONFIG" --profile "${TARGET_PROFILE}_SESSION" os ns get --raw-output
OUT
