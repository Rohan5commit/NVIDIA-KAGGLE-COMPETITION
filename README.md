# nemotron-reasoning-lora

LoRA training and evaluation pipeline for the Kaggle NVIDIA Nemotron Model Reasoning Challenge, centered on `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` with rank-32 LoRA, two-stage post-training, adapter-only packaging, and a Kaggle-ready writeup notebook.

## Status

As of April 8, 2026:

- GitHub repository scaffold: implemented and pushed to `https://github.com/Rohan5commit/nemotron-reasoning-lora`.
- Kaggle competition access: confirmed for `nvidia-nemotron-model-reasoning-challenge`.
- Kaggle RTX runtime access: confirmed through Kaggle Jupyter Server on April 8, 2026 with `NVIDIA RTX PRO 6000 Blackwell Server Edition` and 96 GB VRAM.
- Data, SFT, GRPO, evaluation, and packaging code: implemented.
- Smoke-run data download: completed across all six logical sources with 1,474 normalized records.
- Smoke-run curation: completed with 182 selected records, 165 train rows, 17 validation rows, and 165 hard rows.
- Local GPU execution: not available in this environment.
- Kaggle remote bootstrap execution: attempted from the repo on the live RTX session, but blocked by DNS resolution failures to external hosts from inside the Kaggle runtime.
- Final validation accuracy after SFT: not run in this environment.
- Final validation accuracy after GRPO: not run in this environment.
- `submission.zip`: not produced in this environment.
- Public notebook link: `https://www.kaggle.com/code/rohansan1/nemotron-reasoning-lora-solution-writeup` (status observed on April 8, 2026: `COMPLETE`).
- Midpoint submission status for April 9, 2026: not submitted from this environment.

## Kaggle Runtime Notes

The repository now includes Kaggle-specific helper scripts to drive the competition notebook through Kaggle Jupyter Server:

- `training/kaggle_probe.py`
- `training/kaggle_start_bootstrap.py`
- `training/kaggle_start_pipeline.py`

Observed behavior on April 8, 2026:

- the competition draft session exposed the correct RTX hardware
- the session mounted `/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/{train.csv,test.csv}`
- outbound package/model fetches from the runtime failed with `Temporary failure in name resolution`
- because the runtime could not resolve PyPI or Hugging Face hosts, the repo could not complete dependency installation, dataset download, or Nemotron weight retrieval from that Kaggle session

## Live Source Notes

Two parts of the user-supplied spec did not match the live Hugging Face Hub on April 8, 2026, so the repo records both the requested IDs and the working replacements:

- Requested model alias `nvidia/nemotron-3-nano-30b-a3b-bf16` did not resolve. The pipeline uses the live NVIDIA repo `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Requested datasets `lighteval/MATH` and `HuggingFaceH4/numina-math-cots` did not resolve. The downloader maps them to `HuggingFaceH4/MATH` and `HuggingFaceH4/numina_60k_math_verify_correct_2_4gens_with_rm_scores`.

The live Nemotron chat template on Hugging Face uses `<think>...</think>` inside the assistant turn. This repo supports both:

- `official`: current Hugging Face Nemotron template.
- `legacy_sft` / `legacy_synthetic`: the requested `<extra_id_2>` and `<extra_id_1>` reasoning delimiters.

## Repository Layout

- `data/download_datasets.py`: downloads and normalizes the requested Hugging Face datasets into JSONL.
- `data/filter_and_curate.py`: deduplicates, filters, stratifies, and produces the 50k SFT target set plus the 20k GRPO hard subset.
- `data/generate_synthetic.py`: generates and filters synthetic reasoning traces for the hardest 2k problems.
- `training/stage1_sft.py`: rank-32 LoRA SFT with an Unsloth-first, TRL fallback implementation.
- `training/stage2_grpo.py`: GRPO fine-tuning with the four requested reward components.
- `training/kaggle_probe.py`: Kaggle RTX runtime probe for packages, storage, network, and local competition mounts.
- `training/kaggle_start_bootstrap.py`: detached remote dependency bootstrap for a Kaggle Jupyter Server session.
- `training/kaggle_start_pipeline.py`: detached remote launcher for the full data-to-submission pipeline on Kaggle.
- `training/train_config.yaml`: central configuration for model aliases, data quotas, training hyperparameters, and paths.
- `eval/local_eval.py`: deterministic validation, prompt ablation, and best-of-16 ceiling evaluation with vLLM.
- `submission/package_lora.py`: adapter-only verification, vLLM sanity check, and submission zip creation.
- `notebooks/solution_writeup.ipynb`: Kaggle-ready notebook that renders charts and tables from generated artifacts.
- `common.py`: shared answer extraction, scoring, prompt rendering, and config helpers.

## Data Sources

Requested sources:

- `openai/gsm8k`
- `lighteval/MATH`
- `AI-MO/NuminaMath-CoT`
- `nvidia/OpenMathReasoning`
- `HuggingFaceH4/numina-math-cots`
- `EleutherAI/hendrycks_math`

Resolved sources used by the downloader:

| logical source | requested id | resolved id | configured raw cap |
|---|---|---|---:|
| gsm8k | `openai/gsm8k` | `openai/gsm8k` | 9,000 |
| math | `lighteval/MATH` | `HuggingFaceH4/MATH` | 12,500 |
| numina_cot | `AI-MO/NuminaMath-CoT` | `AI-MO/NuminaMath-CoT` | 125,000 |
| open_math_reasoning | `nvidia/OpenMathReasoning` | `nvidia/OpenMathReasoning` | 150,000 |
| numina_h4 | `HuggingFaceH4/numina-math-cots` | `HuggingFaceH4/numina_60k_math_verify_correct_2_4gens_with_rm_scores` | 60,000 |
| hendrycks_math | `EleutherAI/hendrycks_math` | `EleutherAI/hendrycks_math` | 12,500 |

Actual downloaded and curated sizes are written to:

- `artifacts/dataset_download_summary.json`
- `artifacts/filtering_summary.json`

Smoke-run artifact counts currently in the repo from `python data/download_datasets.py --cap-override 250 --streaming`:

| logical source | normalized records |
|---|---:|
| gsm8k | 250 |
| math | 250 |
| numina_cot | 224 |
| open_math_reasoning | 248 |
| numina_h4 | 250 |
| hendrycks_math | 252 |
| total | 1,474 |

Smoke-run curation output:

| artifact | rows |
|---|---:|
| curated_full.jsonl | 182 |
| train_sft.jsonl | 165 |
| validation.jsonl | 17 |
| train_grpo.jsonl | 165 |
| synthetic_seed_candidates.jsonl | 165 |

## Training Plan

Stage 1 SFT:

- model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- LoRA rank: `32`
- alpha: `64`
- dropout: `0.05`
- target modules: `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`
- 4-bit loading: enabled
- LoftQ initialization: enabled
- max sequence length: `8192`
- per-device batch size: `2`
- gradient accumulation: `4`
- learning rate: `2e-4`
- scheduler: `cosine`
- warmup ratio: `0.05`
- epochs: `2`
- bf16: enabled
- optimizer: `paged_adamw_8bit`
- attention backend: `flash_attention_2`

Stage 2 GRPO:

- num generations: `8`
- max prompt length: `1024`
- max completion length: `4096`
- learning rate: `5e-6`
- per-device batch size: `2`
- gradient accumulation: `4`
- epochs: `1`
- beta: `0.01`
- scheduler: `cosine_with_restarts`
- rollout temperature: `0.7`
- vLLM enabled with `gpu_memory_utilization=0.92`
- requested offload settings encoded as `False` when supported by the installed TRL version

## Evaluation And Packaging

- local validation uses vLLM with `gpu_memory_utilization=0.92` and `max_model_len=8192`
- deterministic validation logs baseline, SFT, and GRPO accuracy separately
- prompt ablation compares the three requested system prompt variants
- best-of-N uses `temperature=0.7` and `N=16`
- packaging verifies the adapter config, rejects merged weights, runs a 3-problem vLLM sanity check, and creates `submission/submission.zip`

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python data/download_datasets.py --streaming
python data/filter_and_curate.py
python data/generate_synthetic.py
python training/stage1_sft.py
python eval/local_eval.py --stage1-dir outputs/stage1_sft
python training/stage2_grpo.py
python eval/local_eval.py --stage1-dir outputs/stage1_sft --stage2-dir outputs/stage2_grpo
python submission/package_lora.py --adapter-dir outputs/stage2_grpo
```

## Expected Runtime On Target Hardware

These are planning estimates for the intended RTX PRO 6000 Blackwell 96 GB environment, not measurements from the current machine:

- data download and curation: `1-3h` depending on Hub throughput
- synthetic generation for 2k hard problems x 5 traces: `4-8h` depending on inference endpoint throughput
- stage 1 SFT: `8-14h`
- stage 2 GRPO: `6-12h`
- deterministic evaluation, prompt ablation, packaging: `1-2h`
- total planned GPU time: `19-36h`

## Notebook And Awards

The notebook at `notebooks/solution_writeup.ipynb` is organized to support:

- Best Data: curation score formula and before/after quality distribution charts
- Best RL Method: per-reward GRPO curves and total reward trajectory
- Best Fine-Tuning Method: LoftQ, alpha=`2x rank`, and the SFT→GRPO pipeline rationale

The Open Contribution Award form URL from the user prompt is preserved in the notebook and should only be submitted after a public Kaggle notebook link exists.
