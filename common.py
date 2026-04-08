from __future__ import annotations

import inspect
import json
import math
import os
import random
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
REQUESTED_MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b-bf16"
CANONICAL_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
CANONICAL_BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Think carefully and place your final answer in \\boxed{}."
PROMPT_VARIANTS = [
    "Think step by step. Place your final answer in \\boxed{}.",
    "You are an expert mathematician. Work through this carefully, showing all steps. Your final answer must be in \\boxed{}.",
    "Solve the following problem. Think deeply before answering. Final answer: \\boxed{}.",
]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ALLOWED_ANSWER_CHARS = re.compile(r"^[0-9A-Za-z\s\.\,\-\+\*/\^\=\(\)\[\]\{\}\\]+$")
BOXED_PATTERN = re.compile(r"\\boxed\s*\{", re.MULTILINE)
TRAILING_ANSWER_PREFIX = re.compile(
    r"^(the final answer is|the answer is|answer\s*:|final answer\s*:|so,?|thus,?)\s*",
    re.IGNORECASE,
)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(config_path or ROOT / "training" / "train_config.yaml")


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_question(text: str) -> str:
    normalized = normalize_whitespace(text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("’", "'").replace("“", '"').replace("”", '"')
    return normalized


def stable_question_hash(text: str) -> str:
    return sha256(normalize_question(text).lower().encode("utf-8")).hexdigest()


def try_float(value: str) -> float | None:
    cleaned = normalize_final_answer(value)
    if not cleaned:
        return None
    if cleaned.endswith("%"):
        try:
            return float(cleaned[:-1]) / 100.0
        except ValueError:
            return None
    try:
        return float(cleaned)
    except ValueError:
        pass
    if "/" in cleaned and cleaned.count("/") == 1 and all(part.strip() for part in cleaned.split("/")):
        try:
            numerator, denominator = cleaned.split("/", 1)
            return float(numerator) / float(denominator)
        except ValueError:
            return None
    return None


def sympy_expr(value: str):
    try:
        import sympy as sp
    except ImportError:
        return None

    cleaned = normalize_final_answer(value)
    if not cleaned:
        return None

    replacements = {
        "^": "**",
        "\\cdot": "*",
        "\\times": "*",
        "\\frac": "frac",
        "\\sqrt": "sqrt",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    try:
        return sp.simplify(sp.sympify(cleaned))
    except Exception:
        return None


def answers_match(prediction: str | None, truth: str | None, rel_tol: float = 0.01) -> bool:
    pred = normalize_final_answer(prediction or "")
    gold = normalize_final_answer(truth or "")
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    pred_float = try_float(pred)
    gold_float = try_float(gold)
    if pred_float is not None and gold_float is not None:
        baseline = max(abs(gold_float), 1.0)
        return abs(pred_float - gold_float) <= rel_tol * baseline
    pred_expr = sympy_expr(pred)
    gold_expr = sympy_expr(gold)
    if pred_expr is not None and gold_expr is not None:
        try:
            return bool(pred_expr.equals(gold_expr))
        except Exception:
            return False
    return False


def normalize_final_answer(answer: str) -> str:
    cleaned = (answer or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("$", "")
    boxed = extract_boxed_answer(cleaned)
    if boxed:
        cleaned = boxed
    cleaned = TRAILING_ANSWER_PREFIX.sub("", cleaned).strip()
    cleaned = cleaned.rstrip(". ").strip()
    if cleaned.startswith("{") and cleaned.endswith("}") and len(cleaned) > 2:
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def is_clean_boxable_answer(answer: str) -> bool:
    cleaned = normalize_final_answer(answer)
    if not cleaned:
        return False
    if len(cleaned) > 96:
        return False
    if not ALLOWED_ANSWER_CHARS.match(cleaned):
        return False
    if cleaned.count("=") > 2:
        return False
    return True


def extract_boxed_answer(text: str) -> str | None:
    if not text:
        return None
    matches = list(BOXED_PATTERN.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    chars: list[str] = []
    for character in text[start:]:
        if character == "{":
            depth += 1
            chars.append(character)
            continue
        if character == "}":
            depth -= 1
            if depth == 0:
                break
            chars.append(character)
            continue
        chars.append(character)
    extracted = "".join(chars).strip()
    return normalize_final_answer(extracted) or None


def extract_thinking_section(text: str) -> str:
    if not text:
        return ""
    patterns = [
        (r"<think>\s*(.*?)\s*</think>", re.DOTALL),
        (r"<extra_id_[12]>thinking\s*(.*?)\s*</extra_id_2>", re.DOTALL),
        (r"<extra_id_[12]>\s*(.*?)\s*</extra_id_2>", re.DOTALL),
    ]
    for pattern, flags in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1).strip()
    return ""


def token_count(text: str, tokenizer: Any | None = None) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return len(re.findall(r"\S+", text))


def quality_score(answer_is_correct: bool, reasoning_trace: str, tokenizer: Any | None = None) -> float:
    if not answer_is_correct:
        return 0.0
    return 1.0 / max(token_count(reasoning_trace, tokenizer), 1)


def maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().lower().replace("level", "").replace(" ", "")
    if text.isdigit():
        return int(text)
    return None


def infer_difficulty(source_name: str, record: Mapping[str, Any], olympiad_markers: Sequence[str]) -> str:
    source_name = source_name.lower()
    if "gsm8k" in source_name:
        return "easy"
    level = maybe_int(record.get("level") or record.get("difficulty") or record.get("difficulty_level"))
    if level is not None:
        if level <= 3:
            return "medium"
        return "hard"
    text_blob = " ".join(
        str(record.get(field, "")).lower()
        for field in ("subject", "source_split", "source", "difficulty_hint", "problem", "question")
    )
    if any(marker in text_blob for marker in olympiad_markers):
        return "hard"
    if "math" in source_name or "numina" in source_name or "openmath" in source_name:
        return "hard"
    return "medium"


def maybe_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def pick_first(mapping: Mapping[str, Any], candidates: Sequence[str], default: Any = None) -> Any:
    for candidate in candidates:
        if candidate in mapping and mapping[candidate] not in (None, ""):
            return mapping[candidate]
    return default


def flatten_messages(messages: Sequence[Mapping[str, Any]], role: str) -> str:
    parts: list[str] = []
    for message in messages:
        if message.get("role") != role:
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, Sequence):
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part).strip()


def render_legacy_chat(
    messages: Sequence[Mapping[str, Any]],
    reasoning_open_token: str = "<extra_id_2>thinking",
    add_generation_prompt: bool = False,
) -> str:
    output: list[str] = []
    close_token = "</extra_id_2>"
    for message in messages:
        role = message["role"]
        output.append(f"<|im_start|>{role}\n")
        if role == "assistant" and message.get("reasoning_content"):
            reasoning = str(message["reasoning_content"]).strip()
            content = str(message.get("content", "")).strip()
            rendered = f"{reasoning_open_token}\n{reasoning}\n{close_token}\n{content}".strip()
        else:
            rendered = str(message.get("content", "")).strip()
        output.append(rendered)
        output.append("<|im_end|>\n")
    if add_generation_prompt:
        output.append("<|im_start|>assistant\n")
        output.append(f"{reasoning_open_token}\n")
    return "".join(output)


def render_official_chat(messages: Sequence[Mapping[str, Any]], add_generation_prompt: bool = False) -> str:
    output: list[str] = []
    for message in messages:
        role = message["role"]
        output.append(f"<|im_start|>{role}\n")
        if role == "assistant" and message.get("reasoning_content"):
            reasoning = str(message["reasoning_content"]).strip()
            content = str(message.get("content", "")).strip()
            rendered = f"<think>\n{reasoning}\n</think>\n{content}".strip()
        else:
            rendered = str(message.get("content", "")).strip()
        output.append(rendered)
        output.append("<|im_end|>\n")
    if add_generation_prompt:
        output.append("<|im_start|>assistant\n<think>\n")
    return "".join(output)


def build_messages(
    problem: str,
    reasoning_trace: str | None,
    boxed_answer: str | None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem.strip()},
    ]
    if reasoning_trace is not None and boxed_answer is not None:
        messages.append(
            {
                "role": "assistant",
                "reasoning_content": reasoning_trace.strip(),
                "content": f"The final answer is \\boxed{{{normalize_final_answer(boxed_answer)}}}.",
            }
        )
    return messages


def render_training_example(
    problem: str,
    reasoning_trace: str,
    boxed_answer: str,
    tokenizer: Any | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    template_mode: str = "official",
) -> str:
    messages = build_messages(problem, reasoning_trace, boxed_answer, system_prompt=system_prompt)
    if tokenizer is not None and template_mode == "official":
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    if template_mode == "official":
        return render_official_chat(messages, add_generation_prompt=False)
    if template_mode == "legacy_synthetic":
        return render_legacy_chat(messages, reasoning_open_token="<extra_id_1>thinking", add_generation_prompt=False)
    return render_legacy_chat(messages, reasoning_open_token="<extra_id_2>thinking", add_generation_prompt=False)


def render_generation_prompt(
    problem: str,
    tokenizer: Any | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    template_mode: str = "official",
) -> str:
    messages = build_messages(problem, reasoning_trace=None, boxed_answer=None, system_prompt=system_prompt)
    if tokenizer is not None and template_mode == "official":
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    if template_mode == "official":
        return render_official_chat(messages, add_generation_prompt=True)
    if template_mode == "legacy_synthetic":
        return render_legacy_chat(messages, reasoning_open_token="<extra_id_1>thinking", add_generation_prompt=True)
    return render_legacy_chat(messages, reasoning_open_token="<extra_id_2>thinking", add_generation_prompt=True)


def build_generation_config(config_cls: type[Any], **kwargs: Any) -> Any:
    parameters = inspect.signature(config_cls.__init__).parameters
    filtered = {key: value for key, value in kwargs.items() if key in parameters}
    return config_cls(**filtered)


def resolve_model_id(config: Mapping[str, Any], prefer_base_model: bool = False) -> str:
    model_config = config["model"]
    if prefer_base_model and model_config.get("canonical_base_model_id"):
        return model_config["canonical_base_model_id"]
    if model_config.get("use_canonical_hf_id_by_default", True):
        return model_config["canonical_model_id"]
    return model_config["requested_model_id"]


@dataclass(slots=True)
class DatasetExample:
    source_name: str
    source_id: str
    source_split: str
    question: str
    reasoning_trace: str
    answer: str
    level: str | int | None = None
    subject: str | None = None
    difficulty_hint: str | None = None
    metadata: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        payload = {
            "source_name": self.source_name,
            "source_id": self.source_id,
            "source_split": self.source_split,
            "question": normalize_question(self.question),
            "question_hash": stable_question_hash(self.question),
            "reasoning_trace": self.reasoning_trace.strip(),
            "answer": normalize_final_answer(self.answer),
            "level": self.level,
            "subject": self.subject,
            "difficulty_hint": self.difficulty_hint,
            "metadata": self.metadata or {},
        }
        return payload
