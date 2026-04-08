from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
import subprocess


ASSET_DATASET_ROOT = Path("/kaggle/input/nemotron-runtime-assets")
DEFAULT_REPO_ARCHIVE = ASSET_DATASET_ROOT / "nemotron-reasoning-lora.tar.gz"
DEFAULT_WORKING_REPO = Path("/kaggle/working/nemotron-reasoning-lora")
LOG_PATH = Path("/kaggle/working/nemotron-kernel-run.log")
STATUS_PATH = Path("/kaggle/working/nemotron-kernel-status.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle entrypoint that unpacks runtime assets and runs the training pipeline.")
    parser.add_argument("--asset-root", default=str(ASSET_DATASET_ROOT))
    parser.add_argument("--repo-archive", default=str(DEFAULT_REPO_ARCHIVE))
    parser.add_argument("--working-repo", default=str(DEFAULT_WORKING_REPO))
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-grpo", action="store_true")
    return parser.parse_args()


def ensure_repo(repo_archive: Path, working_repo: Path) -> None:
    if working_repo.exists():
        shutil.rmtree(working_repo)
    working_repo.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(repo_archive, "r:gz") as archive:
        archive.extractall(working_repo.parent)


def main() -> None:
    args = parse_args()
    asset_root = Path(args.asset_root)
    repo_archive = Path(args.repo_archive)
    working_repo = Path(args.working_repo)
    if not asset_root.exists():
        raise FileNotFoundError(f"Asset dataset root not found: {asset_root}")
    if not repo_archive.exists():
        raise FileNotFoundError(f"Repo archive not found: {repo_archive}")

    ensure_repo(repo_archive, working_repo)
    env = os.environ.copy()
    env["NEMOTRON_OFFLINE_WHEEL_DIRS"] = str(asset_root / "offline_wheels")

    commands = [
        ["python", "training/kaggle_probe.py"],
        ["python", "training/kaggle_start_bootstrap.py", "--sync"],
        ["python", "training/stage1_sft.py"],
        ["python", "eval/local_eval.py", "--stage1-dir", "outputs/stage1_sft"],
    ]
    if not args.skip_synthetic:
        commands.insert(2, ["python", "data/generate_synthetic.py"])
    if not args.skip_grpo:
        commands.extend(
            [
                ["python", "training/stage2_grpo.py"],
                ["python", "eval/local_eval.py", "--stage1-dir", "outputs/stage1_sft", "--stage2-dir", "outputs/stage2_grpo"],
                ["python", "submission/package_lora.py", "--adapter-dir", "outputs/stage2_grpo"],
            ]
        )

    started_at = datetime.now(timezone.utc).isoformat()
    with LOG_PATH.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"Kernel entry started at {started_at}\n")
        for command in commands:
            completed = subprocess.run(command, cwd=str(working_repo), env=env, text=True, capture_output=True)
            log_handle.write(f"\n$ {' '.join(command)}\n")
            if completed.stdout:
                log_handle.write(completed.stdout)
            if completed.stderr:
                log_handle.write(completed.stderr)
            log_handle.flush()
            completed.check_returncode()

    status = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_root": str(asset_root),
        "repo_archive": str(repo_archive),
        "working_repo": str(working_repo),
        "log_path": str(LOG_PATH),
        "commands": commands,
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
