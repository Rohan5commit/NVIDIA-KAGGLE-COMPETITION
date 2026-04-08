from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import TARGET_MODULES, extract_boxed_answer, load_config, render_generation_prompt, resolve_model_id


TEST_PROBLEMS = [
    "If x + 7 = 19, what is x?",
    "Compute 3/4 + 1/8.",
    "What is the value of 2^5?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the final GRPO adapter into submission.zip.")
    parser.add_argument("--config", default="training/train_config.yaml")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--output-zip", default=None)
    return parser.parse_args()


def verify_adapter_config(config: dict, adapter_config: dict) -> None:
    expected_base_ids = {
        config["model"]["requested_model_id"],
        config["model"]["canonical_model_id"],
    }
    if adapter_config.get("base_model_name_or_path") not in expected_base_ids:
        raise ValueError(
            "Unexpected base_model_name_or_path in adapter_config.json: "
            f"{adapter_config.get('base_model_name_or_path')}"
        )
    if int(adapter_config.get("r", -1)) != 32:
        raise ValueError("Adapter rank is not 32.")
    if int(adapter_config.get("lora_alpha", -1)) != 64:
        raise ValueError("LoRA alpha is not 64.")
    if sorted(adapter_config.get("target_modules", [])) != sorted(TARGET_MODULES):
        raise ValueError("target_modules do not match the required Nemotron setup.")


def verify_adapter_only(adapter_weights: Path) -> None:
    from safetensors import safe_open

    with safe_open(adapter_weights, framework="pt") as handle:
        keys = list(handle.keys())
    if not keys:
        raise ValueError("adapter_model.safetensors is empty.")
    illegal = [key for key in keys if "lora_" not in key and "modules_to_save" not in key]
    if illegal:
        raise ValueError(f"Found non-adapter tensors in adapter_model.safetensors: {illegal[:5]}")


def run_sanity_check(repo_config: dict, adapter_dir: Path) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_id(repo_config),
        trust_remote_code=repo_config["model"]["trust_remote_code"],
    )
    prompts = [
        render_generation_prompt(
            problem=problem,
            tokenizer=tokenizer if repo_config["template"]["mode"] == "official" else None,
            system_prompt=repo_config["template"]["system_prompt"],
            template_mode=repo_config["template"]["mode"],
        )
        for problem in TEST_PROBLEMS
    ]
    llm = LLM(
        model=resolve_model_id(repo_config),
        trust_remote_code=repo_config["model"]["trust_remote_code"],
        enable_lora=True,
        gpu_memory_utilization=repo_config["evaluation"]["gpu_memory_utilization"],
        max_model_len=repo_config["evaluation"]["max_model_len"],
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=512),
        lora_request=LoRARequest("submission", 1, str(adapter_dir)),
    )
    texts = [output.outputs[0].text for output in outputs]
    if not all("\\boxed{" in text and extract_boxed_answer(text) for text in texts):
        raise ValueError("Sanity check failed: not all vLLM generations contained a boxed answer.")


def main() -> None:
    args = parse_args()
    repo_config = load_config(args.config)
    adapter_dir = Path(args.adapter_dir or repo_config["paths"]["stage2_output_dir"])
    output_zip = Path(args.output_zip or (Path(repo_config["paths"]["submission_dir"]) / repo_config["submission"]["zip_name"]))
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_weights_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_config_path.exists() or not adapter_weights_path.exists():
        raise FileNotFoundError("Expected adapter_config.json and adapter_model.safetensors in the adapter directory.")

    with adapter_config_path.open("r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    verify_adapter_config(repo_config, adapter_config)
    verify_adapter_only(adapter_weights_path)
    run_sanity_check(repo_config, adapter_dir)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(adapter_config_path, arcname="adapter_config.json")
        archive.write(adapter_weights_path, arcname="adapter_model.safetensors")

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    if size_mb >= repo_config["submission"]["max_zip_mb"]:
        raise ValueError(f"{output_zip.name} is too large: {size_mb:.2f} MB")

    print(f"SUBMISSION READY: {output_zip.name} — {size_mb:.2f}MB — Sanity check: PASSED")


if __name__ == "__main__":
    main()
