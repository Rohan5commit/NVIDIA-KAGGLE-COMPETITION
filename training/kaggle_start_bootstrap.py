from __future__ import annotations

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
ARTIFACTS_DIR = ROOT / "artifacts" / "remote"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = ARTIFACTS_DIR / "bootstrap.log"
STATUS_PATH = ARTIFACTS_DIR / "bootstrap_status.json"


def main() -> None:
    command = [
        "bash",
        "-lc",
        (
            f"cd {ROOT} && "
            "python -m pip install --upgrade pip && "
            "python -m pip install "
            "\"accelerate>=1.5.0\" "
            "\"bitsandbytes>=0.45.0\" "
            "\"datasets>=3.3.0\" "
            "\"flash-attn>=2.7.0\" "
            "\"huggingface_hub>=0.30.0\" "
            "\"matplotlib>=3.9.0\" "
            "\"numpy>=2.0.0\" "
            "\"packaging>=24.2\" "
            "\"pandas>=2.2.0\" "
            "\"peft>=0.17.0\" "
            "\"pyyaml>=6.0.2\" "
            "\"safetensors>=0.5.3\" "
            "\"scikit-learn>=1.6.0\" "
            "\"seaborn>=0.13.2\" "
            "\"sympy>=1.13.3\" "
            "\"tenacity>=9.0.0\" "
            "\"transformers>=4.57.0\" "
            "\"trl>=0.23.0\" "
            "\"vllm>=0.8.5\""
        ),
    ]
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
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
