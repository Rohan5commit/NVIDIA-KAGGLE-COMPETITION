from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    kaggle_repo = Path("/kaggle/working/nemotron-reasoning-lora")
    return kaggle_repo if kaggle_repo.exists() else Path.cwd()


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import discover_offline_wheel_dirs, load_config
from common import discover_optional_source_dirs

ARTIFACTS_DIR = ROOT / "artifacts" / "remote"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = ARTIFACTS_DIR / "bootstrap.log"
STATUS_PATH = ARTIFACTS_DIR / "bootstrap_status.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install offline runtime dependencies for the Kaggle notebook.")
    parser.add_argument("--sync", action="store_true", help="Run the install command in the foreground.")
    return parser.parse_args()


def discover_wheel_files(config: dict[str, object]) -> list[str]:
    wheel_dirs = [str(path) for path in discover_offline_wheel_dirs(config)]
    wheel_patterns = [
        "bitsandbytes-*.whl",
        "trl-*.whl",
        "flash_attn-*.whl",
        "causal_conv1d-*.whl",
        "mamba_ssm-*.whl",
        "vllm-*.whl",
    ]
    wheel_files: list[str] = []
    for wheel_dir in wheel_dirs:
        path = Path(wheel_dir)
        if not path.exists():
            continue
        for pattern in wheel_patterns:
            matches = sorted(path.glob(pattern))
            if matches:
                wheel_files.append(str(matches[-1]))
    return sorted(dict.fromkeys(wheel_files))


def main() -> None:
    args = parse_args()
    config = load_config(ROOT / "training" / "train_config.yaml")
    wheel_dirs = [str(path) for path in discover_offline_wheel_dirs(config)]
    source_dirs = [str(path) for path in discover_optional_source_dirs(config)]
    wheel_files = discover_wheel_files(config)
    source_export = ""
    python_paths = os.pathsep.join(source_dirs)
    if python_paths:
        source_export = f"export PYTHONPATH=\"{python_paths}:$PYTHONPATH\" && "
    install_command = "true"
    if wheel_files:
        install_command = "python -m pip install --no-deps " + " ".join(f'"{wheel}"' for wheel in wheel_files)
    command = [
        "bash",
        "-lc",
        (
            f"cd {ROOT} && "
            f"{source_export}"
            f"{install_command}"
        ),
    ]
    if args.sync:
        completed = subprocess.run(command, cwd=str(ROOT), env=os.environ.copy(), capture_output=True, text=True)
        LOG_PATH.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        status = {
            "returncode": completed.returncode,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "log_path": str(LOG_PATH),
            "root": str(ROOT),
            "wheel_dirs": wheel_dirs,
            "source_dirs": source_dirs,
            "wheel_files": wheel_files,
        }
        STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        completed.check_returncode()
        print(json.dumps(status, indent=2))
        return
    with LOG_PATH.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n=== bootstrap started {datetime.now(timezone.utc).isoformat()} ===\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    status = {
        "pid": process.pid,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "log_path": str(LOG_PATH),
        "root": str(ROOT),
        "wheel_dirs": wheel_dirs,
        "source_dirs": source_dirs,
        "wheel_files": wheel_files,
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
