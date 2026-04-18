from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


HEARTBEAT_PATH = Path("/kaggle/working/probe_heartbeat.json")
LOG_PATH = Path("/kaggle/working/probe_log.txt")


def write_payload(payload: dict[str, object]) -> None:
    HEARTBEAT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if stderr:
        return f"{stdout}\nSTDERR:\n{stderr}".strip()
    return stdout


def main() -> None:
    payload = {
        "stage": "probe_started",
        "ts": time.time(),
        "cwd": os.getcwd(),
        "input_exists": Path("/kaggle/input").exists(),
        "working_exists": Path("/kaggle/working").exists(),
    }
    write_payload(payload)

    input_roots = []
    input_root = Path("/kaggle/input")
    if input_root.exists():
        input_roots = sorted(path.name for path in input_root.iterdir() if path.is_dir())

    disk = {}
    for mount in ["/", "/kaggle/working", "/kaggle/input"]:
        try:
            usage = shutil.disk_usage(mount)
        except Exception:
            continue
        disk[mount] = {
            "total_gib": round(usage.total / (1024**3), 2),
            "free_gib": round(usage.free / (1024**3), 2),
        }

    payload = {
        "stage": "probe_collected",
        "ts": time.time(),
        "input_roots": input_roots[:30],
        "disk": disk,
        "nvidia_smi": command_output(["nvidia-smi"]),
        "python_version": command_output(["python", "--version"]),
    }
    write_payload(payload)
    print(json.dumps(payload, indent=2), flush=True)

    time.sleep(20)
    write_payload({"stage": "probe_completed", "ts": time.time()})


if __name__ == "__main__":
    main()
