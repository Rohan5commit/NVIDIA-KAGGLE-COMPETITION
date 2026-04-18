from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path
import json


WHEEL_ASSET_ROOT = Path("/kaggle/input/nemotron-runtime-assets")
REPO_ASSET_CANDIDATES = [
    Path("/kaggle/input/nemotron-runtime-repo"),
    WHEEL_ASSET_ROOT,
]
REPO_ARCHIVE_NAME = "nemotron-reasoning-lora.tar.gz"
WORKING_REPO = Path("/kaggle/working/nemotron-reasoning-lora")
ASSET_SNAPSHOT_PATH = Path("/kaggle/working/asset_snapshot.json")
LAUNCHER_ERROR_PATH = Path("/kaggle/working/launcher_error.txt")
LAUNCHER_HEARTBEAT_PATH = Path("/kaggle/working/launcher_heartbeat.json")
LAUNCHER_HEARTBEAT_EVENTS_PATH = Path("/kaggle/working/launcher_heartbeat_events.jsonl")


def emit_launcher_heartbeat(stage: str, **extra: object) -> None:
    payload = {
        "stage": stage,
        "ts": time.time(),
        "cwd": os.getcwd(),
    }
    payload.update(extra)
    LAUNCHER_HEARTBEAT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with LAUNCHER_HEARTBEAT_EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def discover_input_roots() -> list[Path]:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return []
    try:
        return sorted(path for path in input_root.iterdir() if path.is_dir())
    except Exception:
        return []


def all_repo_asset_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for path in [*REPO_ASSET_CANDIDATES, *discover_input_roots()]:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(path)
    return candidates


def wait_for_input_mounts(timeout_seconds: int = 180, poll_seconds: int = 6) -> None:
    deadline = time.time() + timeout_seconds
    logged_roots = False
    while time.time() < deadline:
        roots = discover_input_roots()
        if roots and not logged_roots:
            print(f"[info] /kaggle/input roots detected: {', '.join(path.name for path in roots[:20])}")
            logged_roots = True
        candidates = all_repo_asset_candidates()
        for candidate in candidates:
            if (candidate / REPO_ARCHIVE_NAME).exists():
                return
            if (candidate / "training" / "train_config.yaml").exists():
                return
        for root in roots:
            if next(iter(root.rglob("training/train_config.yaml")), None) is not None:
                return
            if next(iter(root.rglob(REPO_ARCHIVE_NAME)), None) is not None:
                return
        print("[warn] Repo assets not mounted yet; waiting...")
        time.sleep(poll_seconds)


def locate_runtime_asset_root() -> Path | None:
    candidates = all_repo_asset_candidates()
    for root in candidates:
        if (root / "offline_wheels").exists() or (root / "offline_wheels.zip").exists() or (root / "offline_wheels.tar").exists():
            return root
    for root in candidates:
        if (root / REPO_ARCHIVE_NAME).exists():
            return root
    for root in candidates:
        for match in root.rglob("offline_wheels"):
            if match.is_dir():
                return match.parent
        for match in root.rglob("offline_wheels.zip"):
            if match.is_file():
                return match.parent
        for match in root.rglob("offline_wheels.tar"):
            if match.is_file():
                return match.parent
    for root in candidates:
        for match in root.rglob(REPO_ARCHIVE_NAME):
            if match.is_file():
                return match.parent
    return None


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
    roots = ", ".join(str(path) for path in all_repo_asset_candidates())
    raise FileNotFoundError(f"Missing repo source under any of: {roots}")


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


def github_archive_url(repo_url: str, ref: str = "main") -> str | None:
    trimmed = repo_url.rstrip("/")
    if trimmed.endswith(".git"):
        trimmed = trimmed[:-4]
    prefix = "https://github.com/"
    if not trimmed.startswith(prefix):
        return None
    owner_repo = trimmed[len(prefix) :]
    if owner_repo.count("/") != 1:
        return None
    owner, repo = owner_repo.split("/", 1)
    return f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/{ref}"


def download_latest_repo_archive(repo_url: str, destination: Path) -> bool:
    archive_url = github_archive_url(repo_url)
    if archive_url is None:
        return False
    temp_archive = destination.parent / "nemotron-reasoning-lora-latest.tar.gz"
    extracted_root = None
    try:
        urllib.request.urlretrieve(archive_url, temp_archive)
        with tarfile.open(temp_archive, "r:gz") as archive:
            root_members = [
                member.name.split("/", 1)[0]
                for member in archive.getmembers()
                if member.name and not member.name.startswith("./")
            ]
            archive.extractall(destination.parent)
        if not root_members:
            return False
        extracted_root = destination.parent / root_members[0]
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(extracted_root), str(destination))
        return True
    except Exception as error:
        print(f"[warn] Unable to download latest repo archive from {archive_url}: {error}")
        return False
    finally:
        if temp_archive.exists():
            temp_archive.unlink()
        if extracted_root is not None and extracted_root.exists():
            shutil.rmtree(extracted_root, ignore_errors=True)


def maybe_sync_latest_repo() -> None:
    if os.environ.get("NEMOTRON_SYNC_LATEST_REPO", "").strip().lower() not in {"1", "true", "yes"}:
        print("[info] Skipping live GitHub sync; using mounted runtime assets.")
        return
    repo_url = os.environ.get("NEMOTRON_GITHUB_REPO", "https://github.com/Rohan5commit/NVIDIA-KAGGLE-COMPETITION.git")
    temp_clone = WORKING_REPO.parent / "nemotron-reasoning-lora-latest"
    if temp_clone.exists():
        shutil.rmtree(temp_clone)
    try:
        completed = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(temp_clone)],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        print(f"[info] Synced latest repo from {repo_url}: {completed.stdout.strip()}")
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        stdout = (error.stdout or "").strip()
        details = stderr or stdout or str(error)
        print(f"[warn] git clone failed for {repo_url}: {details}")
        if temp_clone.exists():
            shutil.rmtree(temp_clone)
        if not download_latest_repo_archive(repo_url, temp_clone):
            return
        print(f"[info] Synced latest repo archive from {repo_url}")
    except Exception as error:
        print(f"[warn] Unable to sync latest repo from {repo_url}, using bundled archive: {error}")
        if temp_clone.exists():
            shutil.rmtree(temp_clone)
        return
    marker = temp_clone / "training" / "train_config.yaml"
    if not marker.exists():
        print("[warn] Synced repo is missing training/train_config.yaml, keeping bundled archive.")
        shutil.rmtree(temp_clone, ignore_errors=True)
        return
    if WORKING_REPO.exists():
        shutil.rmtree(WORKING_REPO)
    shutil.move(str(temp_clone), str(WORKING_REPO))


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
    elif wheel_zip.exists():
        with zipfile.ZipFile(wheel_zip) as archive:
            archive.extractall(working_wheel_dir)
    elif wheel_tar.exists():
        with tarfile.open(wheel_tar) as archive:
            archive.extractall(working_wheel_dir)
    else:
        recursive_wheels = sorted(asset_root.rglob("*.whl"))
        if not recursive_wheels:
            return None
        working_wheel_dir.mkdir(parents=True, exist_ok=True)
        for wheel_path in recursive_wheels:
            shutil.copy2(wheel_path, working_wheel_dir / wheel_path.name)

    nested_wheel_dir = working_wheel_dir / "offline_wheels"
    if nested_wheel_dir.exists():
        return nested_wheel_dir
    return working_wheel_dir


def dump_asset_snapshot() -> None:
    snapshots: list[dict[str, object]] = []
    for asset_root in all_repo_asset_candidates():
        files: list[dict[str, object]] = []
        if asset_root.exists():
            for path in sorted(asset_root.rglob("*"))[:400]:
                record = {
                    "path": str(path.relative_to(asset_root)),
                    "is_dir": path.is_dir(),
                }
                if path.is_file():
                    record["size"] = path.stat().st_size
                files.append(record)
        snapshots.append(
            {
                "asset_root": str(asset_root),
                "exists": asset_root.exists(),
                "entries": files,
            }
        )
    payload = {"asset_roots": snapshots}
    ASSET_SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


emit_launcher_heartbeat("module_loaded")


def apply_runtime_config_overrides() -> None:
    config_path = WORKING_REPO / "training" / "train_config.yaml"
    if not config_path.exists():
        return
    config_text = config_path.read_text(encoding="utf-8")
    # Current Kaggle Nemotron runtime only initializes with eager attention in Transformers.
    config_text = config_text.replace("attn_implementation: flash_attention_2", "attn_implementation: eager")
    config_text = config_text.replace("attn_implementation: sdpa", "attn_implementation: eager")
    # LoftQ requires Hub-style model identifiers; local Kaggle model mounts break initialization.
    config_text = config_text.replace("loftq_init: true", "loftq_init: false")
    config_path.write_text(config_text, encoding="utf-8")

    stage1_path = WORKING_REPO / "training" / "stage1_sft.py"
    if stage1_path.exists():
        stage1_text = stage1_path.read_text(encoding="utf-8")
        # trl>=1.0 expects `processing_class` in SFTTrainer instead of `tokenizer`
        # and no longer accepts `dataset_text_field` as an SFTTrainer constructor arg.
        stage1_text = stage1_text.replace("        tokenizer=tokenizer,\n", "        processing_class=tokenizer,\n")
        stage1_text = stage1_text.replace('        dataset_text_field="text",\n', "")
        stage1_path.write_text(stage1_text, encoding="utf-8")


def main() -> None:
    emit_launcher_heartbeat("main_started")
    dump_asset_snapshot()
    emit_launcher_heartbeat("asset_snapshot_written", path=str(ASSET_SNAPSHOT_PATH))
    try:
        wait_for_input_mounts()
        emit_launcher_heartbeat("input_mounts_ready")
        materialize_repo()
        emit_launcher_heartbeat("repo_materialized", working_repo_exists=WORKING_REPO.exists())
        maybe_sync_latest_repo()
        emit_launcher_heartbeat("repo_sync_attempted", working_repo_exists=WORKING_REPO.exists())
        runtime_asset_root = locate_runtime_asset_root()
        emit_launcher_heartbeat("runtime_asset_root_resolved", asset_root=str(runtime_asset_root) if runtime_asset_root else None)
        resolved_wheel_dir = materialize_wheels(runtime_asset_root)
        emit_launcher_heartbeat("runtime_assets_resolved", wheel_dir=str(resolved_wheel_dir) if resolved_wheel_dir is not None else None)
        apply_runtime_config_overrides()
        emit_launcher_heartbeat("runtime_overrides_applied")

        env = os.environ.copy()
        if runtime_asset_root is not None:
            env["NEMOTRON_ASSET_ROOT"] = str(runtime_asset_root)
        if resolved_wheel_dir is not None:
            env["NEMOTRON_OFFLINE_WHEEL_DIRS"] = str(resolved_wheel_dir)
        env["PYTHONUNBUFFERED"] = "1"
        pipeline_mode = env.get("NEMOTRON_KERNEL_MODE", "stage1_fast").strip().lower()
        command = ["python", "training/kaggle_kernel_entry.py", "--skip-synthetic"]
        if pipeline_mode != "full":
            command.append("--skip-grpo")
            print(f"[info] Kernel mode={pipeline_mode}: running stage1/eval/package only.")
        else:
            print(f"[info] Kernel mode={pipeline_mode}: running full stage1+stage2 pipeline.")
        emit_launcher_heartbeat("pipeline_subprocess_starting", command=command, pipeline_mode=pipeline_mode)
        subprocess.run(
            command,
            cwd=str(WORKING_REPO),
            env=env,
            check=True,
        )
        emit_launcher_heartbeat("pipeline_subprocess_completed")
    except Exception:
        emit_launcher_heartbeat("launcher_failed", error=traceback.format_exc())
        LAUNCHER_ERROR_PATH.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
