from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PROGRESS_DIR = ROOT / "artifacts" / "remote"
DEFAULT_PROGRESS_PATH = DEFAULT_PROGRESS_DIR / "run_progress.json"
DEFAULT_PROGRESS_EVENTS_PATH = DEFAULT_PROGRESS_DIR / "run_progress_events.jsonl"

PROGRESS_PATH_ENV = "NEMOTRON_PROGRESS_PATH"
PROGRESS_EVENTS_PATH_ENV = "NEMOTRON_PROGRESS_EVENTS_PATH"
COMMAND_INDEX_ENV = "NEMOTRON_PROGRESS_COMMAND_INDEX"
COMMAND_COUNT_ENV = "NEMOTRON_PROGRESS_COMMAND_COUNT"
COMMAND_NAME_ENV = "NEMOTRON_PROGRESS_COMMAND_NAME"
KERNEL_MODE_ENV = "NEMOTRON_KERNEL_MODE"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with NamedTemporaryFile("w", encoding="utf-8", dir=str(target.parent), delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    temp_path.replace(target)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class ProgressReporter:
    def __init__(self, phase: str | None = None):
        self.phase = phase
        self.progress_path = Path(os.environ.get(PROGRESS_PATH_ENV, str(DEFAULT_PROGRESS_PATH)))
        self.events_path = Path(os.environ.get(PROGRESS_EVENTS_PATH_ENV, str(DEFAULT_PROGRESS_EVENTS_PATH)))
        self.command_index = self._read_int(COMMAND_INDEX_ENV)
        self.command_count = self._read_int(COMMAND_COUNT_ENV)
        self.command_name = os.environ.get(COMMAND_NAME_ENV, "")
        self.kernel_mode = os.environ.get(KERNEL_MODE_ENV, "").strip()

    @staticmethod
    def _read_int(name: str) -> int | None:
        raw = os.environ.get(name)
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _compute_overall_percent(self, phase_percent: float | None, overall_percent: float | None) -> float | None:
        if overall_percent is not None:
            return max(0.0, min(100.0, float(overall_percent)))
        if self.command_index is None or self.command_count in {None, 0}:
            return phase_percent
        local_fraction = 0.0 if phase_percent is None else max(0.0, min(100.0, float(phase_percent))) / 100.0
        return round(((self.command_index - 1) + local_fraction) / self.command_count * 100.0, 4)

    def load(self) -> dict[str, Any]:
        if not self.progress_path.exists():
            return {}
        try:
            with self.progress_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def update(
        self,
        *,
        status: str,
        message: str | None = None,
        phase: str | None = None,
        phase_percent: float | None = None,
        overall_percent: float | None = None,
        current_step: int | None = None,
        total_steps: int | None = None,
        epoch: float | None = None,
        append_event: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.load()
        active_phase = phase or self.phase or payload.get("phase") or ""
        payload.update(
            {
                "schema_version": 1,
                "updated_at_utc": utc_now(),
                "status": status,
                "phase": active_phase,
                "message": message or payload.get("message") or "",
                "kernel_mode": self.kernel_mode or payload.get("kernel_mode") or "",
                "command_name": self.command_name or payload.get("command_name") or "",
                "command_index": self.command_index,
                "command_count": self.command_count,
            }
        )
        if phase_percent is not None:
            payload["phase_percent"] = round(max(0.0, min(100.0, float(phase_percent))), 4)
        elif "phase_percent" not in payload:
            payload["phase_percent"] = None
        payload["overall_percent"] = self._compute_overall_percent(payload.get("phase_percent"), overall_percent)
        if current_step is not None:
            payload["current_step"] = int(current_step)
        if total_steps is not None:
            payload["total_steps"] = int(total_steps)
        if epoch is not None:
            payload["epoch"] = round(float(epoch), 6)
        if extra:
            payload.update(extra)
        atomic_write_json(self.progress_path, payload)
        event_payload = dict(payload)
        event_payload["event_at_utc"] = utc_now()
        if append_event:
            append_jsonl(self.events_path, event_payload)
        summary = {
            "status": payload.get("status"),
            "phase": payload.get("phase"),
            "message": payload.get("message"),
            "overall_percent": payload.get("overall_percent"),
            "phase_percent": payload.get("phase_percent"),
            "current_step": payload.get("current_step"),
            "total_steps": payload.get("total_steps"),
            "command_index": payload.get("command_index"),
            "command_count": payload.get("command_count"),
            "command_name": payload.get("command_name"),
        }
        print(f"[progress] {json.dumps(summary, sort_keys=True)}", flush=True)
        return payload
