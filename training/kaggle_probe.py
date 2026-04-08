from __future__ import annotations

import json
import os
import shutil
import socket
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    kaggle_repo = Path("/kaggle/working/nemotron-reasoning-lora")
    ROOT = kaggle_repo if kaggle_repo.exists() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import ensure_dir, save_json


PACKAGE_NAMES = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "vllm",
    "unsloth",
    "flash_attn",
    "huggingface_hub",
    "requests",
]

MODEL_IDS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
    "nvidia/OpenMathReasoning",
]

URLS = [
    "https://huggingface.co",
    "https://pypi.org",
    "https://www.google.com",
]


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def gpu_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}

    info: dict[str, Any] = {"available": bool(torch.cuda.is_available())}
    if not info["available"]:
        return info
    info["device_count"] = torch.cuda.device_count()
    info["device_name"] = torch.cuda.get_device_name(0)
    info["capability"] = list(torch.cuda.get_device_capability(0))
    properties = torch.cuda.get_device_properties(0)
    info["total_memory_mib"] = int(properties.total_memory / (1024 * 1024))
    return info


def disk_info() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mount_point in ["/", "/kaggle/working", "/kaggle/input", "/dev/shm"]:
        try:
            usage = shutil.disk_usage(mount_point)
        except FileNotFoundError:
            continue
        output[mount_point] = {
            "total_gib": round(usage.total / (1024**3), 2),
            "used_gib": round(usage.used / (1024**3), 2),
            "free_gib": round(usage.free / (1024**3), 2),
        }
    return output


def internet_checks() -> dict[str, Any]:
    try:
        import requests
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}

    results: dict[str, Any] = {}
    for url in URLS:
        try:
            response = requests.get(url, timeout=10)
            results[url] = {"ok": True, "status": response.status_code}
        except Exception as error:
            results[url] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return results


def hub_checks() -> dict[str, Any]:
    try:
        import requests
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}

    results: dict[str, Any] = {}
    for repo_id in MODEL_IDS:
        try:
            response = requests.get(f"https://huggingface.co/api/models/{repo_id}", timeout=(5, 15))
            results[repo_id] = {
                "ok": response.ok,
                "status": response.status_code,
            }
            if response.ok:
                payload = response.json()
                results[repo_id]["sha"] = payload.get("sha")
                results[repo_id]["private"] = payload.get("private")
            else:
                results[repo_id]["error"] = response.text[:200]
        except Exception as error:
            results[repo_id] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return results


def kaggle_inputs() -> list[str]:
    base = Path("/kaggle/input")
    if not base.exists():
        return []
    return sorted(path.name for path in base.iterdir())


def main() -> None:
    output = {
        "cwd": os.getcwd(),
        "python": sys.version,
        "hostname": socket.gethostname(),
        "env_subset": {
            "KAGGLE_KERNEL_RUN_TYPE": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
            "KAGGLE_URL_BASE": os.environ.get("KAGGLE_URL_BASE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "gpu": gpu_info(),
        "packages": package_versions(),
        "disk": disk_info(),
        "internet": internet_checks(),
        "huggingface": hub_checks(),
        "kaggle_inputs": kaggle_inputs(),
        "repo_exists": Path("/kaggle/working/nemotron-reasoning-lora").exists(),
        "repo_archive_exists": Path("/kaggle/working/nemotron-reasoning-lora.tar.gz").exists(),
    }
    artifacts_dir = ensure_dir(ROOT / "artifacts" / "remote")
    save_json(artifacts_dir / "kaggle_probe.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
