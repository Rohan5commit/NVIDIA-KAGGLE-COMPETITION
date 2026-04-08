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


def locate_repo_source() -> tuple[str, Path]:
    if REPO_ARCHIVE.exists():
        return "archive", REPO_ARCHIVE
    markers = sorted(ASSET_ROOT.rglob("training/train_config.yaml"))
    if markers:
        return "directory", markers[0].parent.parent
    raise FileNotFoundError(f"Missing repo source under {ASSET_ROOT}")


def materialize_repo() -> None:
    repo_source_type, repo_source = locate_repo_source()
    if WORKING_REPO.exists():
        shutil.rmtree(WORKING_REPO)
    WORKING_REPO.parent.mkdir(parents=True, exist_ok=True)
    if repo_source_type == "archive":
        with tarfile.open(repo_source, "r:gz") as archive:
            archive.extractall(WORKING_REPO.parent)
        return
    shutil.copytree(repo_source, WORKING_REPO)


def materialize_wheels() -> Path:
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
        recursive_wheels = sorted(ASSET_ROOT.rglob("*.whl"))
        if not recursive_wheels:
            raise FileNotFoundError(f"Missing offline wheel bundle under {ASSET_ROOT}")
        working_wheel_dir.mkdir(parents=True, exist_ok=True)
        for wheel_path in recursive_wheels:
            shutil.copy2(wheel_path, working_wheel_dir / wheel_path.name)

    nested_wheel_dir = working_wheel_dir / "offline_wheels"
    if nested_wheel_dir.exists():
        return nested_wheel_dir
    return working_wheel_dir


def main() -> None:
    materialize_repo()
    resolved_wheel_dir = materialize_wheels()

    env = os.environ.copy()
    env["NEMOTRON_OFFLINE_WHEEL_DIRS"] = str(resolved_wheel_dir)
    subprocess.run(["python", "training/kaggle_kernel_entry.py"], cwd=str(WORKING_REPO), env=env, check=True)


if __name__ == "__main__":
    main()
