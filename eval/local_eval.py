from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    PROMPT_VARIANTS,
    apply_nemotron_blackwell_compat_fallback,
    answers_match,
    bootstrap_optional_python_paths,
    extract_boxed_answer,
    load_config,
    read_jsonl,
    render_generation_prompt,
    resolve_attn_implementation,
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
    if max_samples is None:
        return rows
    return rows[: max(0, max_samples)]


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


def load_vllm(config: dict[str, Any]):
    from vllm import LLM

    model_id = resolve_model_id(config)
    kwargs = {
        "model": model_id,
        "trust_remote_code": config["model"]["trust_remote_code"],
        "max_model_len": config["evaluation"]["max_model_len"],
        "gpu_memory_utilization": config["evaluation"]["gpu_memory_utilization"],
        "enable_lora": True,
    }
    return LLM(**kwargs)


def load_transformers_backend(config: dict[str, Any], adapter_dir: str | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None
    if config["model"].get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        resolve_model_id(config),
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=config["model"]["trust_remote_code"],
        attn_implementation=resolve_attn_implementation(config["model"]["attn_implementation"]),
    )
    if apply_nemotron_blackwell_compat_fallback(model):
        print("[info] Applied Nemotron Blackwell compatibility fallback kernels (eval path).")
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        apply_nemotron_blackwell_compat_fallback(getattr(model, "base_model", model))
    model.eval()
    return model


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


def generate_with_transformers(model, tokenizer, prompts: list[str], temperature: float, max_tokens: int, n: int = 1) -> list[Any]:
    import torch

    tokenizer.padding_side = "left"
    outputs: list[Any] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = encoded["input_ids"].shape[-1]
        if n == 1:
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            texts = tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
            outputs.append(texts[0])
            continue

        # Sample one completion at a time to avoid a large num_return_sequences KV-cache spike.
        prompt_outputs: list[str] = []
        for _ in range(n):
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_outputs.append(tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        outputs.append(prompt_outputs)
    return outputs


def effective_transformers_max_tokens(config: dict[str, Any], requested_tokens: int) -> int:
    configured_cap = config.get("evaluation", {}).get("transformers_max_new_tokens_cap", 1024)
    try:
        cap = int(configured_cap)
    except (TypeError, ValueError):
        cap = 1024
    return max(1, min(int(requested_tokens), cap))


def release_torch_memory() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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
    bootstrap_optional_python_paths(config)
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
    use_vllm_backend = importlib.util.find_spec("vllm") is not None
    vllm_model = load_vllm(config) if use_vllm_backend else None
    fallback_max_tokens = effective_transformers_max_tokens(config, config["evaluation"]["deterministic_max_tokens"])

    stage_specs: list[tuple[str, str | None]] = [("baseline", None)]
    if Path(stage1_dir).exists():
        stage_specs.append(("stage1_sft", stage1_dir))
    if Path(stage2_dir).exists():
        stage_specs.append(("stage2_grpo", stage2_dir))

    for stage_name, adapter_dir in stage_specs:
        adapter_path = adapter_dir if adapter_dir and Path(adapter_dir).exists() else None
        prompts = make_prompts(config, validation_rows, default_prompt, tokenizer=tokenizer)
        if use_vllm_backend:
            outputs = generate_texts(
                vllm_model,
                prompts,
                temperature=config["evaluation"]["deterministic_temperature"],
                max_tokens=config["evaluation"]["deterministic_max_tokens"],
                adapter_dir=adapter_path,
            )
        else:
            stage_model = load_transformers_backend(config, adapter_dir=adapter_path)
            outputs = generate_with_transformers(
                stage_model,
                tokenizer,
                prompts,
                temperature=config["evaluation"]["deterministic_temperature"],
                max_tokens=fallback_max_tokens,
            )
            stage_model = None
            release_torch_memory()
        results[stage_name] = evaluate_predictions(outputs, validation_rows)

    ablation_adapter = stage2_dir if Path(stage2_dir).exists() else stage1_dir if Path(stage1_dir).exists() else None
    ablation_adapter = ablation_adapter if Path(ablation_adapter or "").exists() else None
    ablation_model = None if use_vllm_backend else load_transformers_backend(config, adapter_dir=ablation_adapter)
    ablation_scores = []
    for prompt in PROMPT_VARIANTS:
        prompts = make_prompts(config, validation_rows, prompt, tokenizer=tokenizer)
        if use_vllm_backend:
            outputs = generate_texts(
                vllm_model,
                prompts,
                temperature=config["evaluation"]["deterministic_temperature"],
                max_tokens=config["evaluation"]["deterministic_max_tokens"],
                adapter_dir=ablation_adapter,
            )
        else:
            outputs = generate_with_transformers(
                ablation_model,
                tokenizer,
                prompts,
                temperature=config["evaluation"]["deterministic_temperature"],
                max_tokens=fallback_max_tokens,
            )
        score = evaluate_predictions(outputs, validation_rows)["accuracy"]
        ablation_scores.append({"prompt": prompt, "accuracy": score})
    best_variant = max(ablation_scores, key=lambda item: item["accuracy"])
    results["prompt_ablation"] = {"variants": ablation_scores, "best": best_variant}

    best_of_n_adapter = ablation_adapter
    prompts = make_prompts(config, validation_rows, best_variant["prompt"], tokenizer=tokenizer)
    if use_vllm_backend:
        candidates = generate_texts(
            vllm_model,
            prompts,
            temperature=config["evaluation"]["best_of_n_temperature"],
            max_tokens=config["evaluation"]["deterministic_max_tokens"],
            adapter_dir=best_of_n_adapter,
            n=config["evaluation"]["best_of_n"],
        )
    else:
        candidates = generate_with_transformers(
            ablation_model,
            tokenizer,
            prompts,
            temperature=config["evaluation"]["best_of_n_temperature"],
            max_tokens=fallback_max_tokens,
            n=config["evaluation"]["best_of_n"],
        )
    results["best_of_n"] = best_of_n_accuracy(validation_rows, candidates)
    if ablation_model is not None:
        ablation_model = None
        release_torch_memory()

    save_json(eval_dir / "local_eval_summary.json", results)
    print(results)


if __name__ == "__main__":
    main()
