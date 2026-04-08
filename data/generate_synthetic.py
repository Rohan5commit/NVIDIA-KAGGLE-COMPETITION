from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    answers_match,
    append_jsonl,
    ensure_dir,
    extract_boxed_answer,
    extract_thinking_section,
    load_config,
    read_jsonl,
    render_generation_prompt,
    render_training_example,
    resolve_attn_implementation,
    resolve_model_id,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning traces for the hardest math problems.")
    parser.add_argument("--config", default="training/train_config.yaml")
    parser.add_argument("--seed-file", default="data/processed/synthetic_seed_candidates.jsonl")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def completion_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        if message is not None:
            if getattr(message, "content", None):
                return str(message.content)
        if getattr(choices[0], "text", None):
            return str(choices[0].text)
    if isinstance(response, dict):
        if response.get("choices"):
            choice = response["choices"][0]
            if isinstance(choice, dict):
                if isinstance(choice.get("message"), dict):
                    return str(choice["message"].get("content", ""))
                return str(choice.get("text", ""))
    return str(response)


def build_messages(problem: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Think carefully and place your final answer in \\boxed{}.",
        },
        {
            "role": "user",
            "content": (
                "Solve this math problem with clear reasoning. Keep the reasoning diverse from previous attempts, "
                "and end with a single final answer in \\boxed{}.\n\n"
                f"{problem}"
            ),
        },
    ]


def build_prompt(problem: str) -> str:
    return (
        "Solve this math problem with clear reasoning. Keep the reasoning diverse from previous attempts, "
        "and end with a single final answer in \\boxed{}.\n\n"
        f"{problem}"
    )


def load_local_generator(config: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = resolve_model_id(config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = None
    if importlib.util.find_spec("bitsandbytes") is not None:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": config["model"]["trust_remote_code"],
        "attn_implementation": resolve_attn_implementation(config["model"]["attn_implementation"]),
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return tokenizer, model


def generate_local_completion(config: dict[str, Any], tokenizer, model, problem: str) -> str:
    import torch

    prompt = render_generation_prompt(
        problem=build_prompt(problem),
        tokenizer=tokenizer if config["template"]["mode"] == "official" else None,
        system_prompt=config["template"]["system_prompt"],
        template_mode=config["template"]["mode"],
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=int(config["synthetic"]["max_new_tokens"]),
            do_sample=True,
            temperature=float(config["synthetic"]["temperature"]),
            top_p=float(config["synthetic"]["top_p"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = encoded["input_ids"].shape[-1]
    return tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_file = Path(args.output_file or config["synthetic"]["output_file"])
    ensure_dir(output_file.parent)
    seeds = read_jsonl(args.seed_file)
    limit = args.limit or int(config["synthetic"]["hardest_problem_count"])
    seeds = seeds[:limit]
    existing = read_jsonl(output_file)
    completed_ids = {row["question_hash"] for row in existing}
    summary = {
        "model_id": resolve_model_id(config),
        "requested": len(seeds),
        "skipped_existing": 0,
        "accepted": len(existing),
        "rejected": 0,
        "backend": "local_transformers" if Path(resolve_model_id(config)).exists() else "hf_inference_api",
    }
    model_id = resolve_model_id(config)
    local_backend = Path(model_id).exists()
    tokenizer = None
    model = None
    client = None
    if local_backend:
        tokenizer, model = load_local_generator(config)
    else:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=None)

    for seed_row in seeds:
        if seed_row["question_hash"] in completed_ids:
            summary["skipped_existing"] += 1
            continue
        accepted_for_problem = 0
        for attempt_index in range(int(config["synthetic"]["traces_per_problem"])):
            time.sleep(0.2)
            if local_backend:
                generated_text = generate_local_completion(config, tokenizer, model, seed_row["question"])
            else:
                response = client.chat_completion(
                    model=model_id,
                    messages=build_messages(seed_row["question"]),
                    temperature=float(config["synthetic"]["temperature"]),
                    top_p=float(config["synthetic"]["top_p"]),
                    max_tokens=int(config["synthetic"]["max_new_tokens"]),
                )
                generated_text = completion_text(response)
            synthetic_answer = extract_boxed_answer(generated_text)
            if not answers_match(synthetic_answer, seed_row["boxed_answer"]):
                summary["rejected"] += 1
                continue
            thinking = extract_thinking_section(generated_text) or generated_text
            synthetic_record = {
                "source_name": "synthetic",
                "source_id": model_id,
                "source_split": "generated",
                "question": seed_row["question"],
                "question_hash": seed_row["question_hash"],
                "reasoning_trace": thinking.strip(),
                "boxed_answer": seed_row["boxed_answer"],
                "quality_score": seed_row["quality_score"],
                "difficulty_bucket": "hard",
                "parent_source_name": seed_row["source_name"],
                "formatted_text_legacy_synthetic": render_training_example(
                    problem=seed_row["question"],
                    reasoning_trace=thinking,
                    boxed_answer=seed_row["boxed_answer"],
                    tokenizer=None,
                    system_prompt=config["template"]["system_prompt"],
                    template_mode="legacy_synthetic",
                ),
                "formatted_text_official": render_training_example(
                    problem=seed_row["question"],
                    reasoning_trace=thinking,
                    boxed_answer=seed_row["boxed_answer"],
                    tokenizer=None,
                    system_prompt=config["template"]["system_prompt"],
                    template_mode="official",
                ),
                "raw_generation": generated_text,
                "attempt_index": attempt_index,
            }
            append_jsonl(output_file, synthetic_record)
            summary["accepted"] += 1
            accepted_for_problem += 1
        if accepted_for_problem:
            completed_ids.add(seed_row["question_hash"])

    save_json(Path(config["paths"]["artifacts_dir"]) / "synthetic_generation_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
