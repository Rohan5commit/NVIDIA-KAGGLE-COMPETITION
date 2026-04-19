from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from progress import ProgressReporter


ASSET_DATASET_ROOT = Path("/kaggle/input/nemotron-runtime-assets")
DEFAULT_REPO_ARCHIVE_NAME = "nemotron-reasoning-lora.tar.gz"
DEFAULT_REPO_ARCHIVE = ASSET_DATASET_ROOT / DEFAULT_REPO_ARCHIVE_NAME
DEFAULT_WORKING_REPO = Path("/kaggle/working/nemotron-reasoning-lora")
LOG_PATH = Path("/kaggle/working/nemotron-kernel-run.log")
STATUS_PATH = Path("/kaggle/working/nemotron-kernel-status.json")
PROGRESS_PATH = Path("/kaggle/working/nemotron-run-progress.json")
PROGRESS_EVENTS_PATH = Path("/kaggle/working/nemotron-run-progress-events.jsonl")
DEFAULT_KERNEL_EVAL_MAX_SAMPLES = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle entrypoint that unpacks runtime assets and runs the training pipeline.")
    parser.add_argument("--asset-root", default=str(ASSET_DATASET_ROOT))
    parser.add_argument("--repo-archive", default=str(DEFAULT_REPO_ARCHIVE))
    parser.add_argument("--working-repo", default=str(DEFAULT_WORKING_REPO))
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-grpo", action="store_true")
    return parser.parse_args()


def discover_input_roots() -> list[Path]:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return []
    try:
        return sorted(path for path in input_root.iterdir() if path.is_dir())
    except Exception:
        return []


def resolve_asset_root(requested_root: Path) -> Path:
    candidates = [requested_root]
    env_root = os.environ.get("NEMOTRON_ASSET_ROOT", "").strip()
    if env_root:
        candidates.insert(0, Path(env_root))
    candidates.extend(discover_input_roots())
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        if (candidate / DEFAULT_REPO_ARCHIVE_NAME).exists():
            return candidate
        if (candidate / "offline_wheels").exists():
            return candidate
        archive_match = next(iter(candidate.rglob(DEFAULT_REPO_ARCHIVE_NAME)), None)
        if archive_match is not None:
            return archive_match.parent
    raise FileNotFoundError(f"Asset dataset root not found: {requested_root}")


def resolve_repo_archive(asset_root: Path, requested_archive: Path) -> Path:
    if requested_archive.exists():
        return requested_archive
    candidate = asset_root / DEFAULT_REPO_ARCHIVE_NAME
    if candidate.exists():
        return candidate
    archive_match = next(iter(asset_root.rglob(DEFAULT_REPO_ARCHIVE_NAME)), None)
    if archive_match is not None:
        return archive_match
    return requested_archive


def ensure_repo(repo_archive: Path, working_repo: Path) -> None:
    if working_repo.exists():
        shutil.rmtree(working_repo)
    working_repo.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(repo_archive, "r:gz") as archive:
        archive.extractall(working_repo.parent)


def command_name(command: list[str]) -> str:
    return " ".join(command)


def parse_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        print(f"[warn] Invalid integer for {name}={raw_value!r}; using default {default}.")
        return default


def run_and_tee(
    command: list[str],
    working_repo: Path,
    env: dict[str, str],
    log_handle,
    *,
    reporter: ProgressReporter | None = None,
    active_phase: str | None = None,
    heartbeat_interval_seconds: float = 15.0,
) -> None:
    process = subprocess.Popen(
        command,
        cwd=str(working_repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    last_output_at = time.time()
    while True:
        ready, _, _ = select.select([process.stdout], [], [], heartbeat_interval_seconds)
        if ready:
            line = process.stdout.readline()
            if line:
                last_output_at = time.time()
                sys.stdout.write(line)
                sys.stdout.flush()
                log_handle.write(line)
                log_handle.flush()
                continue
        if process.poll() is not None:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_handle.write(line)
            log_handle.flush()
            break
        if reporter is not None:
            reporter.update(
                status="running",
                message="command_heartbeat",
                phase=active_phase,
                append_event=True,
                extra={
                    "active_command": command,
                    "heartbeat_interval_seconds": heartbeat_interval_seconds,
                    "seconds_since_last_output": round(time.time() - last_output_at, 2),
                },
            )
    process.wait()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def main() -> None:
    args = parse_args()
    asset_root = resolve_asset_root(Path(args.asset_root))
    repo_archive = resolve_repo_archive(asset_root, Path(args.repo_archive))
    working_repo = Path(args.working_repo)
    if repo_archive.exists():
        ensure_repo(repo_archive, working_repo)
    elif not working_repo.exists():
        raise FileNotFoundError(f"Repo archive not found and working repo missing: {repo_archive}")
    env = os.environ.copy()
    env.setdefault("NEMOTRON_ASSET_ROOT", str(asset_root))
    env.setdefault("NEMOTRON_OFFLINE_WHEEL_DIRS", str(asset_root / "offline_wheels"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["NEMOTRON_PROGRESS_PATH"] = str(PROGRESS_PATH)
    env["NEMOTRON_PROGRESS_EVENTS_PATH"] = str(PROGRESS_EVENTS_PATH)
    eval_max_samples = parse_int_env("NEMOTRON_EVAL_MAX_SAMPLES", DEFAULT_KERNEL_EVAL_MAX_SAMPLES)

    commands = [
        ["python", "training/kaggle_probe.py"],
        ["python", "training/kaggle_start_bootstrap.py", "--sync"],
        ["python", "training/stage1_sft.py"],
        ["python", "eval/local_eval.py", "--stage1-dir", "outputs/stage1_sft", "--max-samples", str(eval_max_samples)],
    ]
    if not args.skip_synthetic:
        commands.insert(2, ["python", "data/generate_synthetic.py"])
    if not args.skip_grpo:
        commands.extend(
            [
                ["python", "training/stage2_grpo.py"],
                [
                    "python",
                    "eval/local_eval.py",
                    "--stage1-dir",
                    "outputs/stage1_sft",
                    "--stage2-dir",
                    "outputs/stage2_grpo",
                    "--max-samples",
                    str(eval_max_samples),
                ],
                ["python", "submission/package_lora.py", "--adapter-dir", "outputs/stage2_grpo"],
            ]
        )
    else:
        commands.append(["python", "submission/package_lora.py", "--adapter-dir", "outputs/stage1_sft"])

    started_at = datetime.now(timezone.utc).isoformat()
    reporter = ProgressReporter("kernel_entry")
    reporter.update(
        status="running",
        message="kernel_entry_started",
        phase_percent=0.0,
        overall_percent=0.0,
        append_event=True,
        extra={
            "started_at_utc": started_at,
            "commands": [command_name(command) for command in commands],
            "working_repo": str(working_repo),
        },
    )
    with LOG_PATH.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"Kernel entry started at {started_at}\n")
        log_handle.flush()
        try:
            for index, command in enumerate(commands, start=1):
                command_env = env.copy()
                command_env["NEMOTRON_PROGRESS_COMMAND_INDEX"] = str(index)
                command_env["NEMOTRON_PROGRESS_COMMAND_COUNT"] = str(len(commands))
                command_env["NEMOTRON_PROGRESS_COMMAND_NAME"] = command_name(command)
                reporter.update(
                    status="running",
                    message="command_started",
                    phase=Path(command[1]).stem if len(command) > 1 else command[0],
                    phase_percent=0.0,
                    append_event=True,
                    extra={
                        "active_command": command,
                        "active_command_index": index,
                        "commands_total": len(commands),
                    },
                )
                log_handle.write(f"\n$ {' '.join(command)}\n")
                log_handle.flush()
                run_and_tee(
                    command,
                    working_repo,
                    command_env,
                    log_handle,
                    reporter=reporter,
                    active_phase=Path(command[1]).stem if len(command) > 1 else command[0],
                )
                reporter.update(
                    status="running",
                    message="command_finished",
                    phase=Path(command[1]).stem if len(command) > 1 else command[0],
                    phase_percent=100.0,
                    append_event=True,
                    extra={
                        "active_command": command,
                        "active_command_index": index,
                        "commands_total": len(commands),
                        "commands_completed": index,
                    },
                )
        except Exception as error:
            reporter.update(
                status="failed",
                message="command_failed",
                phase=Path(command[1]).stem if len(command) > 1 else command[0],
                append_event=True,
                extra={
                    "active_command": command,
                    "active_command_index": index,
                    "commands_total": len(commands),
                    "error": str(error),
                },
            )
            raise

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
    reporter.update(
        status="completed",
        message="kernel_entry_finished",
        phase="complete",
        phase_percent=100.0,
        overall_percent=100.0,
        append_event=True,
        extra={"finished_at_utc": status["finished_at_utc"]},
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
