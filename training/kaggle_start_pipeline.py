from __future__ import annotations

import json
import os
import subprocess
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
LOG_PATH = ARTIFACTS_DIR / "pipeline.log"
STATUS_PATH = ARTIFACTS_DIR / "pipeline_status.json"


def main() -> None:
    commands = [
        "python data/download_datasets.py --streaming",
        "python data/filter_and_curate.py",
        "python data/generate_synthetic.py",
        "python training/stage1_sft.py",
        "python eval/local_eval.py --stage1-dir outputs/stage1_sft",
        "python training/stage2_grpo.py",
        "python eval/local_eval.py --stage1-dir outputs/stage1_sft --stage2-dir outputs/stage2_grpo",
        "python submission/package_lora.py --adapter-dir outputs/stage2_grpo",
    ]
    shell_script = " && ".join(commands)
    command = ["bash", "-lc", f"cd {ROOT} && set -euo pipefail && {shell_script}"]
    with LOG_PATH.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n=== pipeline started {datetime.now(timezone.utc).isoformat()} ===\n")
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
        "commands": commands,
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
