from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    TARGET_MODULES,
    bootstrap_optional_python_paths,
    build_generation_config,
    load_config,
    maybe_seed,
    read_jsonl,
    render_training_example,
    resolve_attn_implementation,
    resolve_model_id,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 SFT for Nemotron reasoning LoRA.")
    parser.add_argument("--config", default="training/train_config.yaml")
    parser.add_argument("--train-file", default="data/processed/train_sft.jsonl")
    parser.add_argument("--synthetic-file", default="data/processed/synthetic.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force-fallback", action="store_true", help="Skip Unsloth and use TRL SFTTrainer directly.")
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    return read_jsonl(file_path) if file_path.exists() else []


def build_text_dataset(records: list[dict[str, Any]], system_prompt: str, template_mode: str, tokenizer=None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        rows.append(
            {
                "text": render_training_example(
                    problem=record["question"],
                    reasoning_trace=record["reasoning_trace"],
                    boxed_answer=record["boxed_answer"],
                    tokenizer=tokenizer if template_mode == "official" else None,
                    system_prompt=system_prompt,
                    template_mode=template_mode,
                ),
                "question": record["question"],
                "boxed_answer": record["boxed_answer"],
                "source_name": record["source_name"],
            }
        )
    return rows


def training_args(config: dict[str, Any], output_dir: str):
    from trl import SFTConfig

    stage_config = config["stage1_sft"]
    return build_generation_config(
        SFTConfig,
        output_dir=output_dir,
        learning_rate=stage_config["learning_rate"],
        per_device_train_batch_size=stage_config["per_device_train_batch_size"],
        gradient_accumulation_steps=stage_config["gradient_accumulation_steps"],
        num_train_epochs=stage_config["num_train_epochs"],
        lr_scheduler_type=stage_config["lr_scheduler_type"],
        warmup_ratio=stage_config["warmup_ratio"],
        bf16=stage_config["bf16"],
        optim=stage_config["optim"],
        logging_steps=stage_config["logging_steps"],
        save_strategy=stage_config["save_strategy"],
        eval_strategy=stage_config["eval_strategy"],
        gradient_checkpointing=stage_config["gradient_checkpointing"],
        max_length=stage_config["max_seq_length"],
        max_seq_length=stage_config["max_seq_length"],
        dataset_text_field="text",
        report_to=[],
        packing=False,
        seed=config["project"]["seed"],
    )


def load_tokenizer(model_id: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_sft_trainer(model, tokenizer, config: dict[str, Any], output_dir: str, dataset_rows: list[dict[str, Any]]):
    from datasets import Dataset
    from trl import SFTTrainer

    trainer_kwargs = {
        "model": model,
        "args": training_args(config, output_dir),
        "train_dataset": Dataset.from_list(dataset_rows),
    }
    signature = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in signature:
        trainer_kwargs["dataset_text_field"] = "text"
    return SFTTrainer(**trainer_kwargs)


def run_unsloth(config: dict[str, Any], dataset, tokenizer, output_dir: str):
    import torch
    from peft import replace_lora_weights_loftq
    from unsloth import FastLanguageModel

    stage_config = config["stage1_sft"]
    model_id = resolve_model_id(config)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=stage_config["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=stage_config["load_in_4bit"],
        trust_remote_code=config["model"]["trust_remote_code"],
        attn_implementation=resolve_attn_implementation(stage_config["attn_implementation"]),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=stage_config["lora_r"],
        lora_alpha=stage_config["lora_alpha"],
        lora_dropout=stage_config["lora_dropout"],
        bias="none",
        target_modules=stage_config["target_modules"],
        use_gradient_checkpointing="unsloth",
    )
    if stage_config["loftq_init"]:
        replace_lora_weights_loftq(model)

    trainer = build_sft_trainer(model=model, tokenizer=tokenizer, config=config, output_dir=output_dir, dataset_rows=dataset)
    train_result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, train_result


def run_fallback(config: dict[str, Any], dataset, tokenizer, output_dir: str):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, replace_lora_weights_loftq
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    stage_config = config["stage1_sft"]
    model_id = resolve_model_id(config)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=stage_config["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=config["model"]["trust_remote_code"],
        attn_implementation=resolve_attn_implementation(stage_config["attn_implementation"]),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=stage_config["gradient_checkpointing"])
    peft_config = LoraConfig(
        r=stage_config["lora_r"],
        lora_alpha=stage_config["lora_alpha"],
        lora_dropout=stage_config["lora_dropout"],
        target_modules=stage_config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if stage_config["loftq_init"]:
        replace_lora_weights_loftq(model)

    trainer = build_sft_trainer(model=model, tokenizer=tokenizer, config=config, output_dir=output_dir, dataset_rows=dataset)
    train_result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, train_result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bootstrap_optional_python_paths(config)
    maybe_seed(int(config["project"]["seed"]))
    output_dir = args.output_dir or config["paths"]["stage1_output_dir"]
    system_prompt = config["template"]["system_prompt"]
    template_mode = config["template"]["mode"]
    records = load_records(args.train_file)
    records.extend(load_records(args.synthetic_file))
    if not records:
        raise FileNotFoundError("No training data found. Run data/filter_and_curate.py first.")

    model_id = resolve_model_id(config)
    tokenizer = load_tokenizer(model_id, config["model"]["trust_remote_code"])
    dataset = build_text_dataset(records, system_prompt=system_prompt, template_mode=template_mode, tokenizer=tokenizer)

    artifacts_dir = Path(config["paths"]["artifacts_dir"]) / "stage1_sft"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        artifacts_dir / "dataset_summary.json",
        {
            "record_count": len(dataset),
            "source_counts": {source: sum(1 for row in dataset if row["source_name"] == source) for source in sorted({row["source_name"] for row in dataset})},
            "sample_inputs": [row["text"] for row in dataset[:3]],
            "template_mode": template_mode,
            "model_id": model_id,
        },
    )

    trainer = None
    train_result = None
    if not args.force_fallback:
        try:
            trainer, train_result = run_unsloth(config, dataset, tokenizer, output_dir)
            backend = "unsloth"
        except Exception as error:
            print(f"[warn] Unsloth path failed, falling back to TRL: {error}")
    if trainer is None:
        trainer, train_result = run_fallback(config, dataset, tokenizer, output_dir)
        backend = "trl_fallback"

    save_json(
        artifacts_dir / "training_summary.json",
        {
            "backend": backend,
            "model_id": model_id,
            "requested_model_id": config["model"]["requested_model_id"],
            "resolved_target_modules": TARGET_MODULES,
            "train_result": getattr(train_result, "metrics", {}),
            "log_history": trainer.state.log_history,
            "output_dir": output_dir,
        },
    )
    print({"backend": backend, "output_dir": output_dir})


if __name__ == "__main__":
    main()
