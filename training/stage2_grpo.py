from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    apply_nemotron_blackwell_compat_fallback,
    answers_match,
    bootstrap_optional_python_paths,
    build_generation_config,
    extract_boxed_answer,
    extract_thinking_section,
    load_config,
    maybe_seed,
    read_jsonl,
    render_generation_prompt,
    resolve_attn_implementation,
    resolve_model_id,
    save_json,
    token_count,
)
from progress import ProgressReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 GRPO for Nemotron reasoning LoRA.")
    parser.add_argument("--config", default="training/train_config.yaml")
    parser.add_argument("--train-file", default="data/processed/train_grpo.jsonl")
    parser.add_argument("--stage1-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def grpo_args(config: dict[str, Any], output_dir: str):
    from trl import GRPOConfig

    stage_config = config["stage2_grpo"]
    use_vllm = stage_config["use_vllm"] and importlib.util.find_spec("vllm") is not None
    return build_generation_config(
        GRPOConfig,
        output_dir=output_dir,
        learning_rate=stage_config["learning_rate"],
        per_device_train_batch_size=stage_config["per_device_train_batch_size"],
        gradient_accumulation_steps=stage_config["gradient_accumulation_steps"],
        num_train_epochs=stage_config["num_train_epochs"],
        beta=stage_config["beta"],
        lr_scheduler_type=stage_config["lr_scheduler_type"],
        num_generations=stage_config["num_generations"],
        max_prompt_length=stage_config["max_prompt_length"],
        max_completion_length=stage_config["max_completion_length"],
        temperature=stage_config["temperature"],
        use_vllm=use_vllm,
        vllm_mode=stage_config["vllm_mode"],
        vllm_gpu_memory_utilization=stage_config["vllm_gpu_memory_utilization"],
        offload_optimizer=stage_config["offload_optimizer"],
        offload_reference_model=stage_config["offload_reference_model"],
        save_total_limit=stage_config.get("save_total_limit"),
        save_strategy=stage_config.get("save_strategy", "steps"),
        save_steps=stage_config.get("save_steps", 500),
        vllm_enable_sleep_mode=False,
        logging_steps=1,
        report_to=[],
        seed=config["project"]["seed"],
    )


def load_tokenizer(model_id: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
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


def build_dataset_rows(config: dict[str, Any], tokenizer, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "prompt": render_generation_prompt(
                    problem=record["question"],
                    tokenizer=tokenizer if config["template"]["mode"] == "official" else None,
                    system_prompt=config["template"]["system_prompt"],
                    template_mode=config["template"]["mode"],
                ),
                "ground_truth": record["boxed_answer"],
                "question": record["question"],
                "difficulty_bucket": record["difficulty_bucket"],
                "source_name": record["source_name"],
            }
        )
    return rows


def load_stage1_model(config: dict[str, Any], stage1_dir: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_id = resolve_model_id(config)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=config["model"]["trust_remote_code"],
        attn_implementation=resolve_attn_implementation(config["model"]["attn_implementation"]),
    )
    if apply_nemotron_blackwell_compat_fallback(base_model):
        print("[info] Applied Nemotron Blackwell compatibility fallback kernels (GRPO path).")
    return PeftModel.from_pretrained(base_model, stage1_dir, is_trainable=True)


def reward_correct_answer(completions, ground_truth, **kwargs) -> list[float]:
    rewards: list[float] = []
    for completion, truth in zip(completions, ground_truth):
        text = completion if isinstance(completion, str) else str(completion)
        boxed = extract_boxed_answer(text)
        if boxed is None:
            rewards.append(-1.0)
        elif answers_match(boxed, truth):
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    return rewards


def reward_boxed_format_present(completions, **kwargs) -> list[float]:
    return [0.3 if "\\boxed{" in (completion if isinstance(completion, str) else str(completion)) else -0.3 for completion in completions]


def make_reward_reasoning_conciseness(tokenizer) -> Callable[..., list[float]]:
    def reward_reasoning_conciseness(completions, **kwargs) -> list[float]:
        rewards: list[float] = []
        for completion in completions:
            text = completion if isinstance(completion, str) else str(completion)
            thinking = extract_thinking_section(text)
            length = token_count(thinking, tokenizer)
            if 400 <= length <= 1200:
                rewards.append(0.2)
            elif 1200 < length <= 2000:
                rewards.append(0.0)
            elif length > 2000:
                rewards.append(-0.1 * (length / 2000.0))
            else:
                rewards.append(-0.05)
        return rewards

    return reward_reasoning_conciseness


def reward_no_answer_leakage(completions, ground_truth, **kwargs) -> list[float]:
    rewards: list[float] = []
    for completion, truth in zip(completions, ground_truth):
        text = completion if isinstance(completion, str) else str(completion)
        thinking = extract_thinking_section(text).lower()
        truth_text = str(truth).strip().lower()
        rewards.append(-0.1 if truth_text and truth_text in thinking else 0.1)
    return rewards


def attach_progress_callback(trainer, *, output_dir: str, checkpoint_interval_minutes: float = 15.0) -> None:
    from transformers import TrainerCallback

    reporter = ProgressReporter("stage2_grpo")
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
                message="stage2_training_started",
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
                message="stage2_training_step",
                phase_percent=phase_percent,
                current_step=current_step,
                total_steps=total_steps,
                epoch=float(getattr(state, "epoch", 0.0) or 0.0),
                append_event=current_step in {1, total_steps} or current_step % 10 == 0,
            )

            now = time.time()
            checkpoint_due = self.next_checkpoint_time is not None and now >= self.next_checkpoint_time
            if checkpoint_due and current_step > 0 and current_step != self.last_checkpoint_step:
                self.last_checkpoint_step = current_step
                control.should_save = True
                self._bump_checkpoint_deadline(now)
                reporter.update(
                    status="running",
                    message="stage2_checkpoint_requested",
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
                message="stage2_checkpoint_saved",
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
                message="stage2_training_log",
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
                message="stage2_training_finished",
                phase_percent=100.0,
                current_step=current_step,
                total_steps=total_steps,
                epoch=float(getattr(state, "epoch", 0.0) or 0.0),
                append_event=True,
            )
            return control

    trainer.add_callback(TrainingProgressCallback())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bootstrap_optional_python_paths(config)
    maybe_seed(int(config["project"]["seed"]))
    stage1_dir = args.stage1_dir or config["paths"]["stage1_output_dir"]
    output_dir = args.output_dir or config["paths"]["stage2_output_dir"]
    records = read_jsonl(args.train_file)
    if not records:
        raise FileNotFoundError("No GRPO training data found. Run data/filter_and_curate.py first.")

    model_id = resolve_model_id(config)
    tokenizer = load_tokenizer(model_id, config["model"]["trust_remote_code"])
    dataset_rows = build_dataset_rows(config, tokenizer, records)
    ProgressReporter("stage2_grpo").update(
        status="running",
        message="stage2_setup_complete",
        phase_percent=0.0,
        current_step=0,
        total_steps=0,
        extra={"dataset_count": len(dataset_rows), "output_dir": output_dir},
        append_event=True,
    )

    stage_config = config["stage2_grpo"]
    checkpoint_interval_minutes = float(stage_config.get("checkpoint_interval_minutes", 15))
    resume_checkpoint = resolve_resume_checkpoint(
        output_dir,
        enabled=bool(stage_config.get("resume_from_last_checkpoint", True)),
    )
    if resume_checkpoint:
        print(f"[info] Resuming Stage 2 from checkpoint: {resume_checkpoint}")
        ProgressReporter("stage2_grpo").update(
            status="running",
            message="stage2_resume_detected",
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

    from datasets import Dataset
    from trl import GRPOTrainer

    model = load_stage1_model(config, stage1_dir)
    trainer_kwargs = {
        "model": model,
        "args": grpo_args(config, output_dir),
        "reward_funcs": [
            reward_correct_answer,
            reward_boxed_format_present,
            make_reward_reasoning_conciseness(tokenizer),
            reward_no_answer_leakage,
        ],
        "train_dataset": Dataset.from_list(dataset_rows),
    }
    signature = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    attach_progress_callback(
        trainer,
        output_dir=output_dir,
        checkpoint_interval_minutes=checkpoint_interval_minutes,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint) if resume_checkpoint else trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    artifacts_dir = Path(config["paths"]["artifacts_dir"]) / "stage2_grpo"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        artifacts_dir / "training_summary.json",
        {
            "model_id": model_id,
            "stage1_dir": stage1_dir,
            "train_result": getattr(train_result, "metrics", {}),
            "log_history": trainer.state.log_history,
            "output_dir": output_dir,
            "dataset_count": len(dataset_rows),
            "resume_checkpoint": resume_checkpoint,
            "checkpoint_interval_minutes": checkpoint_interval_minutes,
        },
    )
    ProgressReporter("stage2_grpo").update(
        status="running",
        message="stage2_summary_written",
        phase_percent=100.0,
        extra={"output_dir": output_dir, "dataset_count": len(dataset_rows)},
        append_event=True,
    )
    print({"output_dir": output_dir, "dataset_count": len(dataset_rows)})


if __name__ == "__main__":
    main()
