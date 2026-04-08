from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import traceback
import zipfile
from pathlib import Path
import json


ASSET_ROOT = Path("/kaggle/input/nemotron-runtime-assets")
REPO_ARCHIVE = ASSET_ROOT / "nemotron-reasoning-lora.tar.gz"
WORKING_REPO = Path("/kaggle/working/nemotron-reasoning-lora")
ASSET_SNAPSHOT_PATH = Path("/kaggle/working/asset_snapshot.json")
LAUNCHER_ERROR_PATH = Path("/kaggle/working/launcher_error.txt")


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


def maybe_sync_latest_repo() -> None:
    repo_url = os.environ.get("NEMOTRON_GITHUB_REPO", "https://github.com/Rohan5commit/nemotron-reasoning-lora.git")
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


def materialize_wheels() -> Path | None:
    working_wheel_dir = Path("/kaggle/working/offline_wheels")
    if working_wheel_dir.exists():
        shutil.rmtree(working_wheel_dir)
    wheel_dir = ASSET_ROOT / "offline_wheels"
    wheel_zip = ASSET_ROOT / "offline_wheels.zip"
    wheel_tar = ASSET_ROOT / "offline_wheels.tar"
    if wheel_dir.exists():
        return wheel_dir
    elif wheel_zip.exists():
        with zipfile.ZipFile(wheel_zip) as archive:
            archive.extractall(working_wheel_dir)
    elif wheel_tar.exists():
        with tarfile.open(wheel_tar) as archive:
            archive.extractall(working_wheel_dir)
    else:
        recursive_wheels = sorted(ASSET_ROOT.rglob("*.whl"))
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
    files: list[dict[str, object]] = []
    if ASSET_ROOT.exists():
        for path in sorted(ASSET_ROOT.rglob("*"))[:400]:
            record = {
                "path": str(path.relative_to(ASSET_ROOT)),
                "is_dir": path.is_dir(),
            }
            if path.is_file():
                record["size"] = path.stat().st_size
            files.append(record)
    payload = {
        "asset_root": str(ASSET_ROOT),
        "exists": ASSET_ROOT.exists(),
        "entries": files,
    }
    ASSET_SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    dump_asset_snapshot()
    try:
        materialize_repo()
        maybe_sync_latest_repo()
        resolved_wheel_dir = materialize_wheels()
        apply_runtime_config_overrides()

        env = os.environ.copy()
        if resolved_wheel_dir is not None:
            env["NEMOTRON_OFFLINE_WHEEL_DIRS"] = str(resolved_wheel_dir)
        subprocess.run(
            ["python", "training/kaggle_kernel_entry.py", "--skip-synthetic"],
            cwd=str(WORKING_REPO),
            env=env,
            check=True,
        )
    except Exception:
        LAUNCHER_ERROR_PATH.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
