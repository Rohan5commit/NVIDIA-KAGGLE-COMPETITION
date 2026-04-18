from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any


if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    kaggle_repo = Path("/kaggle/working/nemotron-reasoning-lora")
    ROOT = kaggle_repo if kaggle_repo.exists() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    bootstrap_optional_python_paths,
    discover_local_model_path,
    ensure_dir,
    load_config,
    save_json,
)


def gpu_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}

    if not torch.cuda.is_available():
        return {"available": False}

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    properties = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_mib": int(properties.total_memory / (1024 * 1024)),
        "free_memory_mib": int(free_bytes / (1024 * 1024)),
    }


def tokenizer_probe(model_path: Path | None) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}

    source = str(model_path) if model_path else "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=True,
            local_files_only=bool(model_path),
        )
        tokens = tokenizer.encode("2 + 2 =", add_special_tokens=True)
        return {
            "ok": True,
            "source": source,
            "token_count": len(tokens),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    except Exception as error:
        return {
            "ok": False,
            "source": source,
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> None:
    config = load_config(ROOT / "training" / "train_config.yaml")
    added_paths = bootstrap_optional_python_paths(config)
    model_path = discover_local_model_path(config)
    payload = {
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "added_python_paths": added_paths,
        "env_subset": {
            "NEMOTRON_ASSET_ROOT": os.environ.get("NEMOTRON_ASSET_ROOT"),
            "NEMOTRON_OFFLINE_WHEEL_DIRS": os.environ.get("NEMOTRON_OFFLINE_WHEEL_DIRS"),
        },
        "gpu": gpu_info(),
        "model_path": str(model_path) if model_path else None,
        "model_path_exists": bool(model_path and model_path.exists()),
        "tokenizer_probe": tokenizer_probe(model_path),
    }

    artifacts_dir = ensure_dir(ROOT / "artifacts" / "remote")
    save_json(artifacts_dir / "kaggle_gpu_smoke.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
