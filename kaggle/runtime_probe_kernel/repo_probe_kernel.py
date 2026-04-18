from __future__ import annotations

import json
from pathlib import Path


REPO_ARCHIVE_NAME = "nemotron-reasoning-lora.tar.gz"
PREFERRED_DATASET_NAMES = ("nemotron-runtime-repo", "nemotron-runtime-assets")
REPO_ASSET_CANDIDATES = [
    Path("/kaggle/input/nemotron-runtime-repo"),
    Path("/kaggle/input/nemotron-runtime-assets"),
]


def discover_input_roots() -> list[Path]:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return []
    return sorted(path for path in input_root.iterdir() if path.is_dir())


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


def main() -> None:
    repo_source_type, repo_source = locate_repo_source()
    runtime_asset_root = locate_runtime_asset_root()
    payload = {
        "input_roots": [str(path) for path in discover_input_roots()],
        "named_dataset_paths": [str(path) for path in discover_named_dataset_paths()],
        "candidates": [str(path) for path in all_repo_asset_candidates()],
        "repo_source_type": repo_source_type,
        "repo_source": str(repo_source),
        "runtime_asset_root": str(runtime_asset_root) if runtime_asset_root else None,
    }
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
