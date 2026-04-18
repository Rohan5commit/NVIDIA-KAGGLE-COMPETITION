from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path


REPO_ARCHIVE_NAME = "nemotron-reasoning-lora.tar.gz"
WORKING_REPO = Path("/kaggle/working/nemotron-reasoning-lora")
PREFERRED_DATASET_NAMES = ("nemotron-runtime-repo", "nemotron-runtime-assets")
REPO_ASSET_CANDIDATES = [
    Path("/kaggle/input/nemotron-runtime-repo"),
    Path("/kaggle/input/nemotron-runtime-assets"),
]
HEARTBEAT_PATH = Path("/kaggle/working/gpu_probe_heartbeat.json")


def emit(stage: str, **extra: object) -> None:
    payload = {"stage": stage, "ts": time.time()}
    payload.update(extra)
    HEARTBEAT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def discover_input_roots() -> list[Path]:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return []
    try:
        return sorted(path for path in input_root.iterdir() if path.is_dir())
    except Exception:
        return []


def discover_named_dataset_paths() -> list[Path]:
    dataset_root = Path("/kaggle/input/datasets")
    if not dataset_root.exists():
        return []
    discovered: list[Path] = []
    for owner_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for dataset_name in PREFERRED_DATASET_NAMES:
            candidate = owner_dir / dataset_name
            if candidate.exists():
                discovered.append(candidate)
    return discovered


def all_repo_asset_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for path in [*REPO_ASSET_CANDIDATES, *discover_named_dataset_paths(), *discover_input_roots()]:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(path)
    return candidates


def locate_repo_source() -> tuple[str, Path]:
    for asset_root in all_repo_asset_candidates():
        marker_path = asset_root / "training" / "train_config.yaml"
        if marker_path.exists():
            return "directory", asset_root
        markers = sorted(asset_root.rglob("training/train_config.yaml"))
        if markers:
            return "directory", markers[0].parent.parent
    for asset_root in all_repo_asset_candidates():
        archive_path = asset_root / REPO_ARCHIVE_NAME
        if archive_path.exists():
            return "archive", archive_path
    raise FileNotFoundError("repo source not found")


def locate_runtime_asset_root() -> Path | None:
    candidates = all_repo_asset_candidates()
    for root in candidates:
        if (root / "offline_wheels").exists() or (root / "offline_wheels.zip").exists() or (root / "offline_wheels.tar").exists():
            return root
    for root in candidates:
        if (root / REPO_ARCHIVE_NAME).exists():
            return root
    for root in candidates:
        for pattern in ("offline_wheels", "offline_wheels.zip", "offline_wheels.tar", REPO_ARCHIVE_NAME):
            match = next(iter(root.rglob(pattern)), None)
            if match is not None:
                return match if match.is_dir() else match.parent
    return None


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


def materialize_wheels(asset_root: Path | None) -> Path | None:
    if asset_root is None:
        return None
    working_wheel_dir = Path("/kaggle/working/offline_wheels")
    if working_wheel_dir.exists():
        shutil.rmtree(working_wheel_dir)
    wheel_dir = asset_root / "offline_wheels"
    wheel_zip = asset_root / "offline_wheels.zip"
    wheel_tar = asset_root / "offline_wheels.tar"
    if wheel_dir.exists():
        return wheel_dir
    if wheel_zip.exists():
        with zipfile.ZipFile(wheel_zip) as archive:
            archive.extractall(working_wheel_dir)
    elif wheel_tar.exists():
        with tarfile.open(wheel_tar) as archive:
            archive.extractall(working_wheel_dir)
    else:
        return None
    nested_wheel_dir = working_wheel_dir / "offline_wheels"
    return nested_wheel_dir if nested_wheel_dir.exists() else working_wheel_dir


def run_command(command: list[str], env: dict[str, str]) -> None:
    emit("command_start", command=command)
    subprocess.run(command, cwd=str(WORKING_REPO), env=env, check=True)
    emit("command_done", command=command)


def main() -> None:
    emit("probe_started")
    materialize_repo()
    asset_root = locate_runtime_asset_root()
    wheel_dir = materialize_wheels(asset_root)
    emit(
        "assets_ready",
        asset_root=str(asset_root) if asset_root else None,
        wheel_dir=str(wheel_dir) if wheel_dir else None,
        repo_exists=WORKING_REPO.exists(),
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if asset_root is not None:
        env["NEMOTRON_ASSET_ROOT"] = str(asset_root)
    if wheel_dir is not None:
        env["NEMOTRON_OFFLINE_WHEEL_DIRS"] = str(wheel_dir)

    run_command(["python", "training/kaggle_probe.py"], env)
    run_command(["python", "training/kaggle_start_bootstrap.py", "--sync"], env)
    run_command(["python", "training/kaggle_gpu_smoke.py"], env)
    emit("probe_completed")


if __name__ == "__main__":
    main()
