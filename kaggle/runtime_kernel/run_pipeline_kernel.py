from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path


ASSET_ROOT = Path("/kaggle/input/nemotron-runtime-assets")
REPO_ARCHIVE = ASSET_ROOT / "nemotron-reasoning-lora.tar.gz"
WORKING_REPO = Path("/kaggle/working/nemotron-reasoning-lora")


def main() -> None:
    if not REPO_ARCHIVE.exists():
        raise FileNotFoundError(f"Missing repo archive: {REPO_ARCHIVE}")

    if WORKING_REPO.exists():
        shutil.rmtree(WORKING_REPO)
    WORKING_REPO.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(REPO_ARCHIVE, "r:gz") as archive:
        archive.extractall(WORKING_REPO.parent)

    working_wheel_dir = Path("/kaggle/working/offline_wheels")
    if working_wheel_dir.exists():
        shutil.rmtree(working_wheel_dir)
    wheel_dir = ASSET_ROOT / "offline_wheels"
    wheel_zip = ASSET_ROOT / "offline_wheels.zip"
    wheel_tar = ASSET_ROOT / "offline_wheels.tar"
    if wheel_dir.exists():
        shutil.copytree(wheel_dir, working_wheel_dir)
    elif wheel_zip.exists():
        with zipfile.ZipFile(wheel_zip) as archive:
            archive.extractall(working_wheel_dir)
    elif wheel_tar.exists():
        with tarfile.open(wheel_tar) as archive:
            archive.extractall(working_wheel_dir)
    else:
        raise FileNotFoundError(f"Missing offline wheel bundle under {ASSET_ROOT}")

    resolved_wheel_dir = working_wheel_dir
    nested_wheel_dir = working_wheel_dir / "offline_wheels"
    if nested_wheel_dir.exists():
        resolved_wheel_dir = nested_wheel_dir

    env = os.environ.copy()
    env["NEMOTRON_OFFLINE_WHEEL_DIRS"] = str(resolved_wheel_dir)
    subprocess.run(["python", "training/kaggle_kernel_entry.py"], cwd=str(WORKING_REPO), env=env, check=True)


if __name__ == "__main__":
    main()
