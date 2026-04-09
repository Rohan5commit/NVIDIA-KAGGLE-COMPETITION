from __future__ import annotations

import argparse
import inspect
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    TARGET_MODULES,
    apply_nemotron_blackwell_compat_fallback,
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
from progress import ProgressReporter


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
        save_total_limit=stage_config.get("save_total_limit"),
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


def resolve_resume_checkpoint(output_dir: str, enabled: bool = True) -> str | None:
    if not enabled:
        return None
    from transformers.trainer_utils import get_last_checkpoint

    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    try:
        return get_last_checkpoint(str(output_path))
    except Exception:
        return None


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


def attach_progress_callback(trainer, *, output_dir: str, checkpoint_interval_minutes: float = 15.0) -> None:
    from transformers import TrainerCallback

    reporter = ProgressReporter("stage1_sft")
    checkpoint_interval_seconds = max(float(checkpoint_interval_minutes), 0.0) * 60.0

    class TrainingProgressCallback(TrainerCallback):
        def __init__(self):
            self.last_step = -1
            self.last_logged_step = -1
            self.last_checkpoint_step = -1
            self.next_checkpoint_time = time.time() + checkpoint_interval_seconds if checkpoint_interval_seconds else None

        def _next_checkpoint_due_utc(self) -> str | None:
            if self.next_checkpoint_time is None:
                return None
            return datetime.fromtimestamp(self.next_checkpoint_time, tz=timezone.utc).isoformat()

        def _bump_checkpoint_deadline(self, now: float) -> None:
            if checkpoint_interval_seconds:
                self.next_checkpoint_time = now + checkpoint_interval_seconds

        def on_train_begin(self, args, state, control, **kwargs):
            total_steps = int(getattr(state, "max_steps", 0) or getattr(args, "max_steps", 0) or 0)
            reporter.update(
                status="running",
                message="stage1_training_started",
                phase_percent=0.0,
                current_step=0,
                total_steps=total_steps,
                epoch=float(getattr(state, "epoch", 0.0) or 0.0),
                append_event=True,
                extra={
                    "checkpoint_interval_minutes": checkpoint_interval_minutes,
                    "next_checkpoint_due_utc": self._next_checkpoint_due_utc(),
                    "output_dir": output_dir,
                },
            )
            return control

        def on_step_end(self, args, state, control, **kwargs):
            total_steps = int(getattr(state, "max_steps", 0) or 0)
            current_step = int(getattr(state, "global_step", 0) or 0)
            if current_step == self.last_step:
                return control
            self.last_step = current_step
            phase_percent = (100.0 * current_step / total_steps) if total_steps else 0.0
            reporter.update(
                status="running",
                message="stage1_training_step",
                phase_percent=phase_percent,
                current_step=current_step,
                total_steps=total_steps,
                epoch=float(getattr(state, "epoch", 0.0) or 0.0),
                append_event=current_step in {1, total_steps} or current_step % 25 == 0,
            )

            now = time.time()
            checkpoint_due = self.next_checkpoint_time is not None and now >= self.next_checkpoint_time
            if checkpoint_due and current_step > 0 and current_step != self.last_checkpoint_step:
                self.last_checkpoint_step = current_step
                control.should_save = True
                self._bump_checkpoint_deadline(now)
                reporter.update(
                    status="running",
                    message="stage1_checkpoint_requested",
                    phase_percent=phase_percent,
                    current_step=current_step,
                    total_steps=total_steps,
                    epoch=float(getattr(state, "epoch", 0.0) or 0.0),
                    append_event=True,
                    extra={
                        "checkpoint_interval_minutes": checkpoint_interval_minutes,
                        "next_checkpoint_due_utc": self._next_checkpoint_due_utc(),
                        "output_dir": output_dir,
                    },
                )
            return control

        def on_save(self, args, state, control, **kwargs):
            total_steps = int(getattr(state, "max_steps", 0) or 0)
            current_step = int(getattr(state, "global_step", 0) or 0)
            phase_percent = (100.0 * current_step / total_steps) if total_steps else None
            checkpoint_dir = Path(output_dir) / f"checkpoint-{current_step}"
            reporter.update(
                status="running",
                message="stage1_checkpoint_saved",
                phase_percent=phase_percent,
                current_step=current_step,
                total_steps=total_steps,
                epoch=float(getattr(state, "epoch", 0.0) or 0.0),
                append_event=True,
                extra={
                    "latest_checkpoint": str(checkpoint_dir),
                    "next_checkpoint_due_utc": self._next_checkpoint_due_utc(),
                    "output_dir": output_dir,
                },
            )
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            current_step = int(getattr(state, "global_step", 0) or 0)
            if current_step == self.last_logged_step:
                return control
            self.last_logged_step = current_step
            total_steps = int(getattr(state, "max_steps", 0) or 0)
            phase_percent = (100.0 * current_step / total_steps) if total_steps else None
            reporter.update(
                status="running",
                message="stage1_training_log",
                phase_percent=phase_percent,
                current_step=current_step,
                total_steps=total_steps,
                epoch=float(getattr(state, "epoch", 0.0) or 0.0),
                extra={"last_log": logs or {}},
            )
            return control

        def on_train_end(self, args, state, control, **kwargs):
            total_steps = int(getattr(state, "max_steps", 0) or getattr(state, "global_step", 0) or 0)
            current_step = int(getattr(state, "global_step", 0) or total_steps)
            reporter.update(
                status="running",
                message="stage1_training_finished",
                phase_percent=100.0,
                current_step=current_step,
                total_steps=total_steps,
                epoch=float(getattr(state, "epoch", 0.0) or 0.0),
                append_event=True,
            )
            return control

    trainer.add_callback(TrainingProgressCallback())


def run_unsloth(config: dict[str, Any], dataset, tokenizer, output_dir: str, resume_checkpoint: str | None):
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
    if apply_nemotron_blackwell_compat_fallback(model):
        print("[info] Applied Nemotron Blackwell compatibility fallback kernels (Unsloth path).")
    if stage_config["loftq_init"]:
        replace_lora_weights_loftq(model)

    trainer = build_sft_trainer(model=model, tokenizer=tokenizer, config=config, output_dir=output_dir, dataset_rows=dataset)
    attach_progress_callback(
        trainer,
        output_dir=output_dir,
        checkpoint_interval_minutes=float(stage_config.get("checkpoint_interval_minutes", 15)),
    )
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint) if resume_checkpoint else trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, train_result


def run_fallback(config: dict[str, Any], dataset, tokenizer, output_dir: str, resume_checkpoint: str | None):
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
    if apply_nemotron_blackwell_compat_fallback(model):
        print("[info] Applied Nemotron Blackwell compatibility fallback kernels (TRL path).")
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
    attach_progress_callback(
        trainer,
        output_dir=output_dir,
        checkpoint_interval_minutes=float(stage_config.get("checkpoint_interval_minutes", 15)),
    )
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint) if resume_checkpoint else trainer.train()
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
    ProgressReporter("stage1_sft").update(
        status="running",
        message="stage1_setup_complete",
        phase_percent=0.0,
        current_step=0,
        total_steps=0,
        extra={"dataset_count": len(dataset), "output_dir": output_dir},
        append_event=True,
    )

    stage_config = config["stage1_sft"]
    checkpoint_interval_minutes = float(stage_config.get("checkpoint_interval_minutes", 15))
    resume_checkpoint = resolve_resume_checkpoint(
        output_dir,
        enabled=bool(stage_config.get("resume_from_last_checkpoint", True)),
    )
    if resume_checkpoint:
        print(f"[info] Resuming Stage 1 from checkpoint: {resume_checkpoint}")
        ProgressReporter("stage1_sft").update(
            status="running",
            message="stage1_resume_detected",
            phase_percent=0.0,
            current_step=0,
            total_steps=0,
            extra={
                "resume_checkpoint": resume_checkpoint,
                "checkpoint_interval_minutes": checkpoint_interval_minutes,
                "output_dir": output_dir,
            },
            append_event=True,
        )

    trainer = None
    train_result = None
    if not args.force_fallback:
        try:
            trainer, train_result = run_unsloth(config, dataset, tokenizer, output_dir, resume_checkpoint)
            backend = "unsloth"
        except Exception as error:
            print(f"[warn] Unsloth path failed, falling back to TRL: {error}")
    if trainer is None:
        trainer, train_result = run_fallback(config, dataset, tokenizer, output_dir, resume_checkpoint)
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
            "resume_checkpoint": resume_checkpoint,
            "checkpoint_interval_minutes": checkpoint_interval_minutes,
        },
    )
    ProgressReporter("stage1_sft").update(
        status="running",
        message="stage1_summary_written",
        phase_percent=100.0,
        extra={"backend": backend, "output_dir": output_dir},
        append_event=True,
    )
    print({"backend": backend, "output_dir": output_dir})


if __name__ == "__main__":
    main()
