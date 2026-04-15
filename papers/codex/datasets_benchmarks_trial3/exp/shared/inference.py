from __future__ import annotations

import gc
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jinja2.exceptions import TemplateError
from transformers import AutoModelForCausalLM, AutoTokenizer

from .benchmark import normalize_answer
from .utils import write_jsonl


MODEL_SPECS = {
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}


@dataclass
class LoadedModel:
    tokenizer: Any
    model: Any
    device: str


def _model_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_model(model_name: str) -> LoadedModel:
    model_id = MODEL_SPECS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_model_dtype(),
        device_map="auto",
    )
    model.eval()
    return LoadedModel(tokenizer=tokenizer, model=model, device="cuda" if torch.cuda.is_available() else "cpu")


def unload_model(loaded: LoadedModel) -> None:
    del loaded.model
    del loaded.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _answer_style(normalization_rule: str) -> str:
    if normalization_rule == "date":
        return "Return only the final answer as a date."
    if normalization_rule == "numeric":
        return "Return only the final answer as digits."
    return "Return only the final answer."


def build_prompt(question: str, normalization_rule: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a precise question answering assistant. Do not explain your reasoning.",
        },
        {
            "role": "user",
            "content": f"{question}\n{_answer_style(normalization_rule)}",
        },
    ]


def render_prompt(tokenizer: Any, question: str, normalization_rule: str) -> str:
    messages = build_prompt(question, normalization_rule)
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TemplateError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{question}\n{_answer_style(normalization_rule)}"}],
            tokenize=False,
            add_generation_prompt=True,
        )


def extract_final_answer(text: str, normalization_rule: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("```", "").strip()
    boxed = re.findall(r"\\boxed\{([^}]*)\}", cleaned)
    if boxed:
        cleaned = boxed[-1].strip()
    for marker in ["final answer:", "answer:", "result:", "therefore,", "so the answer is"]:
        matches = re.findall(marker + r"\s*(.+)", cleaned, flags=re.IGNORECASE)
        if matches:
            cleaned = matches[-1].strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = lines[-1]
    if normalization_rule == "numeric":
        matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if matches:
            return matches[-1]
    if normalization_rule == "date":
        iso = re.findall(r"\d{4}-\d{2}-\d{2}", cleaned)
        if iso:
            return iso[-1]
        long_date = re.findall(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}", cleaned)
        if long_date:
            return long_date[-1]
    tail = cleaned.split(":")[-1].strip().strip(". ")
    return tail


def run_predictions(
    loaded: LoadedModel,
    benchmark_rows: list[dict[str, Any]],
    output_path: Path,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    max_vram_mb = 0.0
    for cluster in benchmark_rows:
        normalization_rule = cluster["normalization_rule"]
        for q_key in ["q0", "q1", "q2", "q3", "q4"]:
            gold = cluster[f"gold_{q_key}"]
            prompt = render_prompt(loaded.tokenizer, cluster[q_key], normalization_rule)
            inputs = loaded.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(loaded.model.device) for key, value in inputs.items()}
            start = time.time()
            with torch.no_grad():
                outputs = loaded.model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    repetition_penalty=1.0,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=loaded.tokenizer.pad_token_id,
                )
            latency = time.time() - start
            latencies.append(latency)
            generated = outputs[0][inputs["input_ids"].shape[1] :]
            raw_text = loaded.tokenizer.decode(generated, skip_special_tokens=True).strip()
            final_text = extract_final_answer(raw_text, normalization_rule)
            prediction_normalized = normalize_answer(final_text, normalization_rule)
            gold_normalized = normalize_answer(gold, normalization_rule)
            rows.append(
                {
                    "cluster_id": cluster["cluster_id"],
                    "construction_split": cluster["construction_split"],
                    "family": cluster["family"],
                    "flip_template": cluster["flip_template"],
                    "question_id": q_key,
                    "question": cluster[q_key],
                    "prediction_raw": raw_text,
                    "prediction_final": final_text,
                    "prediction_normalized": prediction_normalized,
                    "gold": gold,
                    "gold_normalized": gold_normalized,
                    "normalization_rule": normalization_rule,
                    "correct": prediction_normalized == gold_normalized,
                    "latency_seconds": latency,
                }
            )
            if torch.cuda.is_available():
                max_vram_mb = max(max_vram_mb, torch.cuda.max_memory_allocated() / (1024**2))
    write_jsonl(output_path, rows)
    latencies_sorted = sorted(latencies)
    return {
        "mean_latency_seconds": sum(latencies) / max(1, len(latencies)),
        "p95_latency_seconds": latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))] if latencies else 0.0,
        "peak_vram_mb": max_vram_mb,
    }
