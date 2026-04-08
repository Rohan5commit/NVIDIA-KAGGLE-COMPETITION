from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    PROMPT_VARIANTS,
    answers_match,
    extract_boxed_answer,
    load_config,
    read_jsonl,
    render_generation_prompt,
    resolve_model_id,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local vLLM evaluation for Nemotron reasoning adapters.")
    parser.add_argument("--config", default="training/train_config.yaml")
    parser.add_argument("--validation-file", default="data/processed/validation.jsonl")
    parser.add_argument("--stage1-dir", default=None)
    parser.add_argument("--stage2-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_validation(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    return rows[:max_samples] if max_samples else rows


def make_prompts(config: dict[str, Any], records: list[dict[str, Any]], system_prompt: str, tokenizer=None) -> list[str]:
    template_mode = config["template"]["mode"]
    return [
        render_generation_prompt(
            problem=row["question"],
            tokenizer=tokenizer if template_mode == "official" else None,
            system_prompt=system_prompt,
            template_mode=template_mode,
        )
        for row in records
    ]


def evaluate_predictions(predictions: list[str], records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    results = []
    for prediction, row in zip(predictions, records):
        boxed = extract_boxed_answer(prediction)
        is_correct = answers_match(boxed, row["boxed_answer"])
        correct += int(is_correct)
        results.append(
            {
                "id": row.get("question_hash"),
                "prediction": boxed,
                "ground_truth": row["boxed_answer"],
                "correct": is_correct,
            }
        )
    accuracy = correct / max(len(records), 1)
    return {"accuracy": accuracy, "results": results}


def load_vllm(config: dict[str, Any], adapter_dir: str | None = None):
    from vllm import LLM

    model_id = resolve_model_id(config)
    kwargs = {
        "model": model_id,
        "trust_remote_code": config["model"]["trust_remote_code"],
        "max_model_len": config["evaluation"]["max_model_len"],
        "gpu_memory_utilization": config["evaluation"]["gpu_memory_utilization"],
        "enable_lora": bool(adapter_dir),
    }
    return LLM(**kwargs)


def maybe_lora_request(adapter_dir: str | None):
    if not adapter_dir:
        return None
    from vllm.lora.request import LoRARequest

    return LoRARequest("nemotron-reasoning-adapter", 1, adapter_dir)


def generate_texts(llm, prompts: list[str], temperature: float, max_tokens: int, adapter_dir: str | None = None, n: int = 1) -> list[Any]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
    )
    request = maybe_lora_request(adapter_dir)
    outputs = llm.generate(prompts, sampling_params, lora_request=request)
    if n == 1:
        return [output.outputs[0].text for output in outputs]
    return [[candidate.text for candidate in output.outputs] for output in outputs]


def best_of_n_accuracy(records: list[dict[str, Any]], candidates: list[list[str]]) -> dict[str, Any]:
    successes = 0
    details = []
    for row, options in zip(records, candidates):
        matched = False
        boxed_predictions = []
        for option in options:
            boxed = extract_boxed_answer(option)
            boxed_predictions.append(boxed)
            if answers_match(boxed, row["boxed_answer"]):
                matched = True
        successes += int(matched)
        details.append({"id": row["question_hash"], "matched": matched, "predictions": boxed_predictions})
    return {"accuracy": successes / max(len(records), 1), "results": details}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validation_rows = load_validation(args.validation_file, max_samples=args.max_samples)
    if not validation_rows:
        raise FileNotFoundError("No validation rows found. Run data/filter_and_curate.py first.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_id(config),
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    stage1_dir = args.stage1_dir or config["paths"]["stage1_output_dir"]
    stage2_dir = args.stage2_dir or config["paths"]["stage2_output_dir"]
    eval_dir = Path(config["paths"]["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"model_id": resolve_model_id(config), "records": len(validation_rows)}
    default_prompt = config["template"]["system_prompt"]

    for stage_name, adapter_dir in [("baseline", None), ("stage1_sft", stage1_dir), ("stage2_grpo", stage2_dir)]:
        adapter_path = adapter_dir if adapter_dir and Path(adapter_dir).exists() else None
        llm = load_vllm(config, adapter_dir=adapter_path)
        prompts = make_prompts(config, validation_rows, default_prompt, tokenizer=tokenizer)
        outputs = generate_texts(
            llm,
            prompts,
            temperature=config["evaluation"]["deterministic_temperature"],
            max_tokens=config["evaluation"]["deterministic_max_tokens"],
            adapter_dir=adapter_path,
        )
        results[stage_name] = evaluate_predictions(outputs, validation_rows)

    ablation_adapter = stage2_dir if Path(stage2_dir).exists() else stage1_dir if Path(stage1_dir).exists() else None
    llm = load_vllm(config, adapter_dir=ablation_adapter if Path(ablation_adapter or "").exists() else None)
    ablation_scores = []
    for prompt in PROMPT_VARIANTS:
        prompts = make_prompts(config, validation_rows, prompt, tokenizer=tokenizer)
        outputs = generate_texts(
            llm,
            prompts,
            temperature=config["evaluation"]["deterministic_temperature"],
            max_tokens=config["evaluation"]["deterministic_max_tokens"],
            adapter_dir=ablation_adapter if Path(ablation_adapter or "").exists() else None,
        )
        score = evaluate_predictions(outputs, validation_rows)["accuracy"]
        ablation_scores.append({"prompt": prompt, "accuracy": score})
    best_variant = max(ablation_scores, key=lambda item: item["accuracy"])
    results["prompt_ablation"] = {"variants": ablation_scores, "best": best_variant}

    best_of_n_adapter = ablation_adapter if Path(ablation_adapter or "").exists() else None
    llm = load_vllm(config, adapter_dir=best_of_n_adapter)
    prompts = make_prompts(config, validation_rows, best_variant["prompt"], tokenizer=tokenizer)
    candidates = generate_texts(
        llm,
        prompts,
        temperature=config["evaluation"]["best_of_n_temperature"],
        max_tokens=config["evaluation"]["deterministic_max_tokens"],
        adapter_dir=best_of_n_adapter,
        n=config["evaluation"]["best_of_n"],
    )
    results["best_of_n"] = best_of_n_accuracy(validation_rows, candidates)

    save_json(eval_dir / "local_eval_summary.json", results)
    print(results)


if __name__ == "__main__":
    main()
