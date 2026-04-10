#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError

from kagglesdk.kernels.services.kernels_api_service import ApiGetKernelRequest
from kagglesdk.kernels.types.kernels_api_service import (
    ApiCreateKernelSessionRequest,
    ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
)


RUNNING_STATES = {"RUNNING", "QUEUED", "PENDING", "STARTING", "INITIALIZING"}
DEFAULT_KERNEL_REF = "rohansan1/nemotron-reasoning-lora-trainer"
DEFAULT_MACHINE_SHAPE = "NvidiaRtxPro6000"


@dataclass(frozen=True)
class KernelRef:
    username: str
    slug: str

    @property
    def full(self) -> str:
        return f"{self.username}/{self.slug}"


def parse_kernel_ref(value: str) -> KernelRef:
    if "/" not in value:
        raise ValueError(f"Kernel ref must be <username>/<slug>, got: {value!r}")
    user_name, kernel_slug = value.split("/", 1)
    if not user_name or not kernel_slug:
        raise ValueError(f"Kernel ref must be <username>/<slug>, got: {value!r}")
    return KernelRef(user_name, kernel_slug)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso8601(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def enum_name(value: Any) -> str:
    text = str(value or "")
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def is_running_state(value: Any) -> bool:
    return enum_name(value) in RUNNING_STATES


def kaggle_api() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    return api


def safe_request_json(url: str, timeout: int = 30) -> dict[str, Any] | None:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload
        return None
    except Exception:
        return None


def safe_request_text(url: str, timeout: int = 30) -> str:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception:
        return ""


def estimate_eta_from_events(events_text: str, latest_percent: float | None) -> dict[str, Any]:
    empty_result = {
        "eta_seconds": None,
        "eta_utc": None,
        "progress_rate_percent_per_hour": None,
        "latest_event_percent": None,
        "effective_latest_percent": latest_percent,
    }
    points: list[tuple[datetime, float]] = []
    for line in events_text.splitlines()[-500:]:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        timestamp = parse_iso8601(payload.get("event_at_utc") or payload.get("updated_at_utc"))
        percent = payload.get("overall_percent")
        if timestamp is None or percent is None:
            continue
        try:
            progress = float(percent)
        except (TypeError, ValueError):
            continue
        points.append((timestamp, progress))
    if not points:
        return empty_result

    latest_time, latest_value = points[-1]
    effective_latest = latest_percent
    if effective_latest is None:
        effective_latest = latest_value
    else:
        effective_latest = max(float(effective_latest), latest_value)

    if len(points) < 2:
        return {
            **empty_result,
            "latest_event_percent": round(latest_value, 4),
            "effective_latest_percent": round(float(effective_latest), 4),
        }

    previous_time, previous_value = points[0]
    for point_time, point_value in reversed(points[:-1]):
        if (latest_time - point_time).total_seconds() >= 300:
            previous_time, previous_value = point_time, point_value
            break

    delta_seconds = (latest_time - previous_time).total_seconds()
    delta_percent = latest_value - previous_value
    if delta_seconds <= 0 or delta_percent <= 0:
        return {
            **empty_result,
            "latest_event_percent": round(latest_value, 4),
            "effective_latest_percent": round(float(effective_latest), 4),
        }

    rate_per_second = delta_percent / delta_seconds
    remaining_percent = max(0.0, 100.0 - float(effective_latest))
    eta_seconds = int(round(remaining_percent / rate_per_second)) if rate_per_second > 0 else None
    eta_utc = None
    if eta_seconds is not None:
        eta_utc = datetime.now(timezone.utc).timestamp() + eta_seconds
        eta_utc = datetime.fromtimestamp(eta_utc, tz=timezone.utc).isoformat()
    return {
        "eta_seconds": eta_seconds,
        "eta_utc": eta_utc,
        "progress_rate_percent_per_hour": round(rate_per_second * 3600.0, 4),
        "latest_event_percent": round(latest_value, 4),
        "effective_latest_percent": round(float(effective_latest), 4),
    }


def estimate_eta_from_log(log_text: str, latest_percent: float | None) -> dict[str, Any]:
    result = {"eta_seconds": None, "eta_utc": None, "progress_rate_percent_per_hour": None}
    if latest_percent is None or not log_text:
        return result
    points: list[tuple[float, float]] = []
    time_pattern = re.compile(r'"time":\s*([0-9]+(?:\.[0-9]+)?)')
    percent_patterns = [
        re.compile(r'overall_percent\\":\s*([0-9]+(?:\.[0-9]+)?)'),
        re.compile(r'"overall_percent":\s*([0-9]+(?:\.[0-9]+)?)'),
    ]
    for line in log_text.splitlines()[-12000:]:
        if "overall_percent" not in line or '"time":' not in line:
            continue
        time_match = time_pattern.search(line)
        if not time_match:
            continue
        percent_match = None
        for pattern in percent_patterns:
            percent_match = pattern.search(line)
            if percent_match:
                break
        if not percent_match:
            continue
        try:
            t_value = float(time_match.group(1))
            p_value = float(percent_match.group(1))
        except (TypeError, ValueError):
            continue
        points.append((t_value, p_value))

    if len(points) < 2:
        return result
    latest_time, latest_value = points[-1]
    prev_time, prev_value = points[0]
    for candidate_time, candidate_value in reversed(points[:-1]):
        if latest_time - candidate_time >= 300:
            prev_time, prev_value = candidate_time, candidate_value
            break
    delta_seconds = latest_time - prev_time
    delta_percent = latest_value - prev_value
    if delta_seconds <= 0 or delta_percent <= 0:
        return result
    rate_per_second = delta_percent / delta_seconds
    remaining_percent = max(0.0, 100.0 - float(latest_percent))
    if rate_per_second <= 0:
        return result
    eta_seconds = int(round(remaining_percent / rate_per_second))
    eta_utc = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() + eta_seconds, tz=timezone.utc).isoformat()
    return {
        "eta_seconds": eta_seconds,
        "eta_utc": eta_utc,
        "progress_rate_percent_per_hour": round(rate_per_second * 3600.0, 4),
    }


def latest_progress_from_log(log_text: str) -> float | None:
    if not log_text:
        return None
    numeric_patterns = [
        r'overall_percent\\":\s*([0-9]+(?:\.[0-9]+)?)',
        r'"overall_percent":\s*([0-9]+(?:\.[0-9]+)?)',
    ]
    for pattern in numeric_patterns:
        matches = re.findall(pattern, log_text)
        if matches:
            try:
                return float(matches[-1])
            except (TypeError, ValueError):
                continue
    return None


def collect_progress_snapshot(api: KaggleApi, kernel_ref: KernelRef) -> dict[str, Any]:
    request = ApiListKernelSessionOutputRequest()
    request.user_name = kernel_ref.username
    request.kernel_slug = kernel_ref.slug
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)

    files = []
    for item in list(getattr(response, "files", []) or []):
        file_name = getattr(item, "file_name", "") or ""
        url = getattr(item, "url", "") or ""
        if not file_name or not url:
            continue
        files.append({"file_name": file_name, "url": url})

    progress_entry = next((f for f in files if f["file_name"].endswith("run_progress.json")), None)
    events_entry = next((f for f in files if f["file_name"].endswith("run_progress_events.jsonl")), None)

    progress_payload = safe_request_json(progress_entry["url"]) if progress_entry else None
    events_text = safe_request_text(events_entry["url"]) if events_entry else ""
    log_size = len(getattr(response, "log", "") or "")

    overall_percent = None
    if isinstance(progress_payload, dict):
        try:
            overall_percent = float(progress_payload.get("overall_percent"))
        except (TypeError, ValueError):
            overall_percent = None

    log_text = getattr(response, "log", "") or ""
    eta = estimate_eta_from_events(events_text, overall_percent)
    log_percent = latest_progress_from_log(log_text)
    overall_percent_estimate = eta.get("effective_latest_percent")
    if log_percent is not None:
        if overall_percent_estimate is None:
            overall_percent_estimate = round(log_percent, 4)
        else:
            overall_percent_estimate = round(max(float(overall_percent_estimate), float(log_percent)), 4)
    log_eta = estimate_eta_from_log(log_text, overall_percent_estimate)
    if eta.get("eta_seconds") is None and log_eta.get("eta_seconds") is not None:
        eta["eta_seconds"] = log_eta["eta_seconds"]
        eta["eta_utc"] = log_eta["eta_utc"]
        eta["progress_rate_percent_per_hour"] = log_eta["progress_rate_percent_per_hour"]
    return {
        "updated_at_utc": now_utc(),
        "files_count": len(files),
        "log_size_bytes": log_size,
        "progress_file": progress_entry["file_name"] if progress_entry else None,
        "events_file": events_entry["file_name"] if events_entry else None,
        "progress": progress_payload,
        "log_percent_estimate": log_percent,
        "overall_percent_estimate": overall_percent_estimate,
        "eta": eta,
    }


def kernel_status(api: KaggleApi, kernel_ref: KernelRef) -> dict[str, Any]:
    status_request = ApiGetKernelSessionStatusRequest()
    status_request.user_name = kernel_ref.username
    status_request.kernel_slug = kernel_ref.slug
    with api.build_kaggle_client() as client:
        status_response = client.kernels.kernels_api_client.get_kernel_session_status(status_request)
    return {
        "status": enum_name(getattr(status_response, "status", None)),
        "failure_message": getattr(status_response, "failure_message", "") or "",
    }


def create_kernel_session(api: KaggleApi, kernel_ref: KernelRef, machine_shape: str | None) -> dict[str, Any]:
    get_request = ApiGetKernelRequest()
    get_request.user_name = kernel_ref.username
    get_request.kernel_slug = kernel_ref.slug
    with api.build_kaggle_client() as client:
        kernel = client.kernels.kernels_api_client.get_kernel(get_request)

    metadata = kernel.metadata
    create_request = ApiCreateKernelSessionRequest()
    create_request.slug = kernel_ref.full
    create_request.language = metadata.language
    create_request.kernel_type = metadata.kernel_type
    create_request.enable_internet = bool(getattr(metadata, "enable_internet", True))
    docker_image = getattr(metadata, "docker_image", None)
    if docker_image:
        create_request.docker_image = docker_image
    create_request.machine_shape = machine_shape or getattr(metadata, "machine_shape", None) or DEFAULT_MACHINE_SHAPE

    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.create_kernel_session(create_request)
    return {"response": response.to_dict(), "machine_shape": create_request.machine_shape}


def write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def command_status(api: KaggleApi, kernel_ref: KernelRef, json_path: str | None) -> int:
    payload = {
        "kernel": kernel_ref.full,
        "status": kernel_status(api, kernel_ref),
        "snapshot": collect_progress_snapshot(api, kernel_ref),
    }
    write_json(json_path, payload)
    print(json.dumps(payload, indent=2))
    return 0


def command_ensure_running(
    api: KaggleApi,
    kernel_ref: KernelRef,
    machine_shape: str | None,
    json_path: str | None,
    fail_on_quota: bool,
) -> int:
    before = kernel_status(api, kernel_ref)
    payload: dict[str, Any] = {
        "kernel": kernel_ref.full,
        "checked_at_utc": now_utc(),
        "status_before": before,
        "machine_shape": machine_shape or DEFAULT_MACHINE_SHAPE,
        "action": None,
        "error": None,
    }

    if is_running_state(before["status"]):
        payload["action"] = "already_running"
        payload["status_after"] = before
        payload["snapshot"] = collect_progress_snapshot(api, kernel_ref)
        write_json(json_path, payload)
        print(json.dumps(payload, indent=2))
        return 0

    try:
        launch_result = create_kernel_session(api, kernel_ref, machine_shape)
        payload["action"] = "started"
        payload["launch_result"] = launch_result
    except HTTPError as error:
        response_text = ""
        status_code = None
        if error.response is not None:
            status_code = error.response.status_code
            response_text = error.response.text or ""
        payload["error"] = {
            "type": type(error).__name__,
            "status_code": status_code,
            "message": str(error),
            "response_text": response_text,
        }
        lower_text = response_text.lower()
        if status_code == 429 and "weekly gpu quota" in lower_text:
            payload["action"] = "quota_blocked"
            payload["status_after"] = kernel_status(api, kernel_ref)
            payload["snapshot"] = collect_progress_snapshot(api, kernel_ref)
            write_json(json_path, payload)
            print(json.dumps(payload, indent=2))
            return 2 if fail_on_quota else 0
        payload["action"] = "start_failed"
        write_json(json_path, payload)
        print(json.dumps(payload, indent=2))
        return 1

    payload["status_after"] = kernel_status(api, kernel_ref)
    payload["snapshot"] = collect_progress_snapshot(api, kernel_ref)
    write_json(json_path, payload)
    print(json.dumps(payload, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control and monitor Kaggle kernel sessions.")
    parser.add_argument(
        "command",
        choices=["status", "ensure-running"],
        help="status: fetch session status/progress; ensure-running: start session if idle.",
    )
    parser.add_argument("--kernel", default=DEFAULT_KERNEL_REF, help="Kernel in <username>/<slug> format.")
    parser.add_argument("--machine-shape", default=DEFAULT_MACHINE_SHAPE)
    parser.add_argument("--json-out", default=None, help="Optional path to write JSON output.")
    parser.add_argument(
        "--fail-on-quota",
        action="store_true",
        help="Return non-zero when weekly GPU quota blocks startup.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kernel_ref = parse_kernel_ref(args.kernel)
    api = kaggle_api()
    if args.command == "status":
        return command_status(api, kernel_ref, args.json_out)
    return command_ensure_running(
        api,
        kernel_ref,
        machine_shape=args.machine_shape,
        json_path=args.json_out,
        fail_on_quota=bool(args.fail_on_quota),
    )


if __name__ == "__main__":
    raise SystemExit(main())
