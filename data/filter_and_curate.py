from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    answers_match,
    ensure_dir,
    extract_boxed_answer,
    infer_difficulty,
    is_clean_boxable_answer,
    load_config,
    maybe_seed,
    quality_score,
    read_jsonl,
    render_training_example,
    resolve_model_id,
    save_json,
    token_count,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter and curate a 50k high-quality reasoning dataset.")
    parser.add_argument("--config", default="training/train_config.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    return parser.parse_args()


def load_tokenizer(config: dict[str, Any]):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            resolve_model_id(config),
            trust_remote_code=config["model"].get("trust_remote_code", True),
        )
    except Exception as error:
        print(f"[warn] tokenizer unavailable, falling back to whitespace token counts: {error}")
        return None


def histogram(values: list[float], bins: int = 20) -> list[dict[str, float]]:
    if not values:
        return []
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper):
        return [{"bin_start": lower, "bin_end": upper, "count": float(len(values))}]
    width = (upper - lower) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        index = min(int((value - lower) / width), bins - 1)
        counts[index] += 1
    output = []
    for idx, count in enumerate(counts):
        start = lower + idx * width
        output.append({"bin_start": start, "bin_end": start + width, "count": float(count)})
    return output


def better_record(new_record: dict[str, Any], old_record: dict[str, Any], source_priorities: dict[str, int]) -> bool:
    new_priority = source_priorities.get(new_record["source_name"], 0)
    old_priority = source_priorities.get(old_record["source_name"], 0)
    if new_priority != old_priority:
        return new_priority > old_priority
    if not math.isclose(new_record["quality_score"], old_record["quality_score"]):
        return new_record["quality_score"] > old_record["quality_score"]
    return new_record["reasoning_tokens"] < old_record["reasoning_tokens"]


def proportional_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    ordered = list(ratios.items())
    raw = {name: total * ratio for name, ratio in ordered}
    targets = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(targets.values())
    for name, _ in sorted(ordered, key=lambda item: raw[item[0]] - int(raw[item[0]]), reverse=True):
        if remainder <= 0:
            break
        targets[name] += 1
        remainder -= 1
    return targets


def carve_validation(records: list[dict[str, Any]], validation_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_bucket[record["difficulty_bucket"]].append(record)
    validation_targets = proportional_targets(validation_size, {"easy": 0.2, "medium": 0.4, "hard": 0.4})
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for bucket, rows in by_bucket.items():
        copied = rows[:]
        rng.shuffle(copied)
        val_count = min(len(copied), validation_targets.get(bucket, 0))
        validation_rows.extend(copied[:val_count])
        train_rows.extend(copied[val_count:])
    return train_rows, validation_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    maybe_seed(int(config["project"]["seed"]))
    raw_dir = Path(args.raw_dir or config["paths"]["raw_dir"])
    processed_dir = ensure_dir(args.processed_dir or config["paths"]["processed_dir"])
    artifacts_dir = ensure_dir(config["paths"]["artifacts_dir"])
    tokenizer = load_tokenizer(config)

    source_priorities = {
        name: int(spec.get("priority", 0))
        for name, spec in config["datasets"]["resolved"].items()
    }

    raw_records: list[dict[str, Any]] = []
    for raw_file in sorted(raw_dir.glob("*.jsonl")):
        raw_records.extend(read_jsonl(raw_file))
    if not raw_records:
        raise FileNotFoundError(f"No raw dataset files found in {raw_dir}")

    before_quality_scores: list[float] = []
    after_filter_quality_scores: list[float] = []
    before_reasoning_lengths: list[int] = []
    after_reasoning_lengths: list[int] = []
    deduped: dict[str, dict[str, Any]] = {}
    invalid_reason_counter: Counter[str] = Counter()

    for record in raw_records:
        answer = record["answer"]
        reasoning = record["reasoning_trace"]
        extracted = extract_boxed_answer(reasoning) or answer
        answer_is_correct = answers_match(extracted, answer)
        reasoning_tokens = token_count(reasoning, tokenizer)
        quality = quality_score(answer_is_correct, reasoning, tokenizer)
        before_quality_scores.append(quality)
        before_reasoning_lengths.append(reasoning_tokens)

        if not is_clean_boxable_answer(answer):
            invalid_reason_counter["unclean_answer"] += 1
            continue
        if not (config["curation"]["reasoning_token_min"] <= reasoning_tokens <= config["curation"]["reasoning_token_max"]):
            invalid_reason_counter["length_filter"] += 1
            continue
        if not answer_is_correct:
            invalid_reason_counter["incorrect_answer"] += 1
            continue

        enriched = dict(record)
        enriched["boxed_answer"] = answer
        enriched["reasoning_tokens"] = reasoning_tokens
        enriched["quality_score"] = quality
        enriched["difficulty_bucket"] = infer_difficulty(
            record["source_name"],
            record,
            config["curation"]["olympiad_markers"],
        )
        enriched["answer_is_correct"] = True
        dedup_key = record["question_hash"]
        if dedup_key not in deduped or better_record(enriched, deduped[dedup_key], source_priorities):
            deduped[dedup_key] = enriched

    deduped_records = list(deduped.values())
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in deduped_records:
        bucketed[record["difficulty_bucket"]].append(record)

    keep_fraction = float(config["curation"]["quality_keep_fraction"])
    top_quality_records: list[dict[str, Any]] = []
    for bucket, rows in bucketed.items():
        ordered = sorted(rows, key=lambda row: row["quality_score"], reverse=True)
        keep_count = max(1, int(len(ordered) * keep_fraction))
        top_quality_records.extend(ordered[:keep_count])

    for row in top_quality_records:
        after_filter_quality_scores.append(row["quality_score"])
        after_reasoning_lengths.append(row["reasoning_tokens"])

    final_target = int(config["curation"]["target_size"])
    difficulty_targets = proportional_targets(final_target, config["curation"]["difficulty_mix"])
    selected: list[dict[str, Any]] = []
    carryover: list[dict[str, Any]] = []
    per_bucket_selected: dict[str, int] = {}
    per_bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in top_quality_records:
        per_bucket_rows[row["difficulty_bucket"]].append(row)
    for bucket, rows in per_bucket_rows.items():
        ordered = sorted(rows, key=lambda row: row["quality_score"], reverse=True)
        target = difficulty_targets.get(bucket, 0)
        chosen = ordered[:target]
        selected.extend(chosen)
        carryover.extend(ordered[target:])
        per_bucket_selected[bucket] = len(chosen)
    if len(selected) < final_target:
        carryover = sorted(carryover, key=lambda row: row["quality_score"], reverse=True)
        selected.extend(carryover[: final_target - len(selected)])

    selected = sorted(selected, key=lambda row: row["quality_score"], reverse=True)[:final_target]
    train_rows, validation_rows = carve_validation(selected, int(config["curation"]["validation_size"]), int(config["project"]["seed"]))

    hard_rows = [row for row in train_rows if row["difficulty_bucket"] == "hard"]
    hard_rows = sorted(hard_rows, key=lambda row: row["quality_score"], reverse=True)
    grpo_rows = hard_rows[: int(config["curation"]["grpo_hard_subset_size"])]
    synthetic_seed_rows = hard_rows[: int(config["synthetic"]["hardest_problem_count"])]

    for record in train_rows + validation_rows + grpo_rows:
        record["formatted_text_official"] = render_training_example(
            problem=record["question"],
            reasoning_trace=record["reasoning_trace"],
            boxed_answer=record["boxed_answer"],
            tokenizer=None,
            system_prompt=config["template"]["system_prompt"],
            template_mode="official",
        )
        record["formatted_text_legacy_sft"] = render_training_example(
            problem=record["question"],
            reasoning_trace=record["reasoning_trace"],
            boxed_answer=record["boxed_answer"],
            tokenizer=None,
            system_prompt=config["template"]["system_prompt"],
            template_mode="legacy_sft",
        )

    write_jsonl(processed_dir / "train_sft.jsonl", train_rows)
    write_jsonl(processed_dir / "validation.jsonl", validation_rows)
    write_jsonl(processed_dir / "train_grpo.jsonl", grpo_rows)
    write_jsonl(processed_dir / "synthetic_seed_candidates.jsonl", synthetic_seed_rows)
    write_jsonl(processed_dir / "curated_full.jsonl", selected)

    summary = {
        "raw_records": len(raw_records),
        "deduped_records": len(deduped_records),
        "top_quality_records": len(top_quality_records),
        "final_selected_records": len(selected),
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "grpo_records": len(grpo_rows),
        "synthetic_seed_candidates": len(synthetic_seed_rows),
        "invalid_reasons": dict(invalid_reason_counter),
        "selected_by_bucket": dict(Counter(row["difficulty_bucket"] for row in selected)),
        "train_by_source": dict(Counter(row["source_name"] for row in train_rows)),
        "validation_by_source": dict(Counter(row["source_name"] for row in validation_rows)),
        "quality_histogram_before": histogram(before_quality_scores),
        "quality_histogram_after": histogram(after_filter_quality_scores),
        "reasoning_length_histogram_before": histogram([float(value) for value in before_reasoning_lengths]),
        "reasoning_length_histogram_after": histogram([float(value) for value in after_reasoning_lengths]),
    }
    save_json(artifacts_dir / "filtering_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
