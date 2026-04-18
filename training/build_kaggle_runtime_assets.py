from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_DIR = ROOT / "artifacts" / "kaggle_runtime_assets"
DEFAULT_WHEEL_SOURCE_DIR = Path("/tmp/linux_wheels")

REQUIRED_WHEEL_PATTERNS = [
    "bitsandbytes-*.whl",
    "trl-*.whl",
    "flash_attn-*.whl",
    "causal_conv1d-*.whl",
    "mamba_ssm-*.whl",
]
OPTIONAL_WHEEL_PATTERNS = [
    "vllm-*.whl",
]
REPO_ARCHIVE_NAME = "nemotron-reasoning-lora.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage a Kaggle runtime asset bundle with the repo archive and wheels.")
    parser.add_argument("--output-dir", default=str(DEFAULT_STAGE_DIR))
    parser.add_argument("--wheel-source-dir", default=str(DEFAULT_WHEEL_SOURCE_DIR))
    parser.add_argument("--include-vllm", action="store_true")
    return parser.parse_args()


def copy_latest_matching(pattern: str, source_dir: Path, target_dir: Path) -> str:
    matches = sorted(source_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Required wheel not found for pattern {pattern} in {source_dir}")
    wheel_path = matches[-1]
    destination = target_dir / wheel_path.name
    shutil.copy2(wheel_path, destination)
    return destination.name


def build_repo_archive(repo_root: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    exclude_parts = {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        ".DS_Store",
    }
    excluded_prefixes = (
        "artifacts/kaggle_runtime_assets/",
        "artifacts/kaggle_runtime_assets_stage/",
        "artifacts/kaggle_runtime_repo_stage/",
        "artifacts/remote/",
        "data/raw/",
        "outputs/",
        "submission/submission.zip",
    )
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(repo_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(repo_root)
            relative_text = relative.as_posix()
            if any(part in exclude_parts for part in relative.parts):
                continue
            if any(relative_text.startswith(prefix) for prefix in excluded_prefixes):
                continue
            archive.add(path, arcname=Path("nemotron-reasoning-lora") / relative)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    wheel_source_dir = Path(args.wheel_source_dir).resolve()
    wheel_target_dir = output_dir / "offline_wheels"
    output_dir.mkdir(parents=True, exist_ok=True)
    wheel_target_dir.mkdir(parents=True, exist_ok=True)

    staged_wheels: list[str] = []
    for pattern in REQUIRED_WHEEL_PATTERNS:
        staged_wheels.append(copy_latest_matching(pattern, wheel_source_dir, wheel_target_dir))
    if args.include_vllm:
        for pattern in OPTIONAL_WHEEL_PATTERNS:
            try:
                staged_wheels.append(copy_latest_matching(pattern, wheel_source_dir, wheel_target_dir))
            except FileNotFoundError:
                continue

    archive_path = output_dir / REPO_ARCHIVE_NAME
    build_repo_archive(ROOT, archive_path)

    manifest = {
        "repo_archive": archive_path.name,
        "wheel_source_dir": str(wheel_source_dir),
        "staged_wheels": staged_wheels,
        "wheel_count": len(staged_wheels),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
