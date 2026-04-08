from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    DatasetExample,
    append_jsonl,
    ensure_dir,
    extract_boxed_answer,
    load_config,
    normalize_final_answer,
    pick_first,
    save_json,
)


QUESTION_FIELDS = [
    "problem",
    "question",
    "prompt",
    "instruction",
    "query",
    "input",
]
REASONING_FIELDS = [
    "solution",
    "reasoning_trace",
    "reasoning",
    "cot",
    "response",
    "completion",
    "output",
    "generated_solution",
    "selected_solution",
    "assistant",
    "answer",
]
ANSWER_FIELDS = [
    "answer",
    "final_answer",
    "ground_truth",
    "expected_answer",
    "label",
]
SUBJECT_FIELDS = ["type", "subject", "category", "topic"]
LEVEL_FIELDS = ["level", "difficulty", "difficulty_level"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize Hugging Face math reasoning datasets.")
    parser.add_argument("--config", default="training/train_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode when supported.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing raw files.")
    parser.add_argument("--cap-override", type=int, default=None, help="Override the configured per-source cap for a faster smoke run.")
    return parser.parse_args()


def split_gsm8k_answer(raw_answer: str) -> tuple[str, str]:
    text = raw_answer or ""
    if "####" in text:
        reasoning, answer = text.rsplit("####", 1)
        return reasoning.strip(), normalize_final_answer(answer)
    return text.strip(), extract_boxed_answer(text) or normalize_final_answer(text)


def stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")).strip())
            elif isinstance(block, str):
                parts.append(block.strip())
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def conversation_to_question_and_answer(example: dict[str, Any]) -> tuple[str, str, str]:
    for field in ("messages", "conversation", "conversations", "chat"):
        value = example.get(field)
        if not isinstance(value, list):
            continue
        user_parts: list[str] = []
        assistant_parts: list[str] = []
        for message in value:
            if not isinstance(message, dict):
                continue
            role = message.get("role") or message.get("from")
            content = stringify_message_content(message.get("content") or message.get("value") or message.get("text"))
            if role in {"user", "human"} and content:
                user_parts.append(content)
            elif role in {"assistant", "gpt"} and content:
                assistant_parts.append(content)
        question = "\n".join(user_parts).strip()
        assistant = "\n".join(assistant_parts).strip()
        return question, assistant, extract_boxed_answer(assistant) or ""
    return "", "", ""


def normalize_example(source_name: str, source_id: str, split_name: str, example: dict[str, Any]) -> DatasetExample | None:
    question = str(pick_first(example, QUESTION_FIELDS, default="") or "").strip()
    reasoning = ""
    answer = ""
    subject = pick_first(example, SUBJECT_FIELDS)
    level = pick_first(example, LEVEL_FIELDS)
    difficulty_hint = str(example.get("difficulty_hint") or example.get("source") or "").strip() or None

    if source_name == "gsm8k":
        question = str(example.get("question", "")).strip()
        reasoning, answer = split_gsm8k_answer(str(example.get("answer", "")))
    else:
        reasoning = str(pick_first(example, REASONING_FIELDS, default="") or "").strip()
        answer = normalize_final_answer(str(pick_first(example, ANSWER_FIELDS, default="") or ""))

    if not question:
        conv_question, conv_reasoning, conv_answer = conversation_to_question_and_answer(example)
        question = conv_question or question
        reasoning = conv_reasoning or reasoning
        answer = conv_answer or answer

    if reasoning and not answer:
        answer = extract_boxed_answer(reasoning) or answer

    if not answer and "####" in reasoning:
        _, answer = split_gsm8k_answer(reasoning)

    if not question or not reasoning or not answer:
        return None

    payload = DatasetExample(
        source_name=source_name,
        source_id=source_id,
        source_split=split_name,
        question=question,
        reasoning_trace=reasoning,
        answer=answer,
        level=level,
        subject=str(subject).strip() or None,
        difficulty_hint=difficulty_hint,
        metadata={key: value for key, value in example.items() if key not in set(QUESTION_FIELDS + REASONING_FIELDS + ANSWER_FIELDS)},
    )
    return payload


def iter_split_records(dataset: Iterable[dict[str, Any]], source_name: str, source_id: str, split_name: str, cap: int) -> Iterable[dict[str, Any]]:
    emitted = 0
    for row in dataset:
        normalized = normalize_example(source_name, source_id, split_name, dict(row))
        if normalized is None:
            continue
        yield normalized.to_record()
        emitted += 1
        if emitted >= cap:
            break


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raw_dir = ensure_dir(args.output_dir or config["paths"]["raw_dir"])
    summary: dict[str, Any] = {"sources": {}, "totals": defaultdict(int)}

    from datasets import load_dataset, load_dataset_builder

    for source_name, source_spec in config["datasets"]["resolved"].items():
        dataset_id = source_spec["dataset_id"]
        configured_cap = int(config["datasets"]["download_caps"].get(source_name, 0) or 0)
        cap = int(args.cap_override or configured_cap)
        target_file = raw_dir / f"{source_name}.jsonl"
        if target_file.exists() and not args.force:
            print(f"[skip] {source_name}: {target_file} already exists")
            continue
        if target_file.exists():
            target_file.unlink()
        config_names = source_spec.get("config_names") or [source_spec.get("config_name")]
        config_names = [name for name in config_names if name is not None] or [None]
        source_summary = {
            "requested_id": source_spec["requested_id"],
            "resolved_id": dataset_id,
            "config_name": source_spec.get("config_name"),
            "config_names": config_names,
            "note": source_spec.get("note"),
            "splits": {},
        }
        per_config_cap = max(cap // max(len(config_names), 1), 1) if cap else 0
        total_emitted = 0

        for config_name in config_names:
            try:
                builder = load_dataset_builder(dataset_id, name=config_name)
            except Exception as error:
                source_summary["splits"][config_name or "default"] = {"error": str(error)}
                continue
            available_splits = list((builder.info.splits or {}).keys())
            split_names = list(source_spec.get("split_names") or available_splits or ["train"])
            per_split_cap = max((per_config_cap or 0) // max(len(split_names), 1), 1) if per_config_cap else 0

            for split_name in split_names:
                if available_splits and split_name not in available_splits:
                    continue
                config_label = f"{config_name}:{split_name}" if config_name else split_name
                print(f"[download] {source_name}:{config_label} -> {dataset_id}")
                try:
                    dataset = load_dataset(dataset_id, name=config_name, split=split_name, streaming=args.streaming)
                except Exception as error:
                    source_summary["splits"][config_label] = {"error": str(error)}
                    continue
                emitted = 0
                split_cap = per_split_cap or int((builder.info.splits or {}).get(split_name).num_examples if (builder.info.splits or {}).get(split_name) else 10**12)
                for record in iter_split_records(dataset, source_name, dataset_id, split_name, split_cap):
                    append_jsonl(target_file, record)
                    emitted += 1
                    total_emitted += 1
                    if cap and total_emitted >= cap:
                        break
                source_summary["splits"][config_label] = {"records": emitted}
                if cap and total_emitted >= cap:
                    break
            if cap and total_emitted >= cap:
                break

        source_summary["output_file"] = str(target_file)
        source_summary["total_records"] = total_emitted
        summary["sources"][source_name] = source_summary
        summary["totals"]["records"] += total_emitted

    serializable_summary = {"sources": summary["sources"], "totals": dict(summary["totals"])}
    save_json(config["paths"]["artifacts_dir"] + "/dataset_download_summary.json", serializable_summary)
    print(json.dumps(serializable_summary, indent=2))


if __name__ == "__main__":
    main()
