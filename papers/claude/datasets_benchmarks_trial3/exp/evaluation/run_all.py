#!/usr/bin/env python3
"""
SkillStack: Run all model evaluations.

Orchestrates evaluation of all models with direct and CoT prompting.
Time budget: ~7 hours total on 1x A6000.
"""
import sys
import os
import time
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from exp.evaluation.evaluate import evaluate_model

RESULTS_DIR = BASE_DIR / "exp" / "results"
DATA_DIR = BASE_DIR / "data"


# Model configurations
MODELS = [
    {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "short": "qwen0.5b",
        "quantization": None,
        "gpu_mem": 0.30,
    },
    {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "short": "qwen1.5b",
        "quantization": None,
        "gpu_mem": 0.35,
    },
    {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "short": "qwen3b",
        "quantization": None,
        "gpu_mem": 0.40,
    },
    {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "short": "llama8b",
        "quantization": None,
        "gpu_mem": 0.50,
    },
    {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "short": "qwen7b",
        "quantization": None,
        "gpu_mem": 0.50,
    },
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "short": "deepseek7b",
        "quantization": None,
        "gpu_mem": 0.50,
    },
    {
        "name": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "short": "qwen14b",
        "quantization": "awq",
        "gpu_mem": 0.60,
    },
    {
        "name": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "short": "qwen32b",
        "quantization": "awq",
        "gpu_mem": 0.85,
    },
]


def run_evaluation_suite():
    """Run the full evaluation suite."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    benchmark_42 = str(DATA_DIR / "benchmark_seed42.jsonl")
    benchmark_123 = str(DATA_DIR / "benchmark_seed123.jsonl")

    all_summaries = {}
    total_start = time.time()

    # ============================================================
    # Phase 1: Direct prompting on all 6 models (seed 42)
    # ============================================================
    print("=" * 60)
    print("PHASE 1: Direct prompting evaluation (seed 42)")
    print("=" * 60)

    for model_cfg in MODELS:
        short = model_cfg["short"]
        output = str(RESULTS_DIR / f"{short}_direct_seed42.jsonl")
        summary_file = str(RESULTS_DIR / f"{short}_direct_seed42_summary.json")

        if os.path.exists(summary_file):
            print(f"  Skipping {short} direct (already done)")
            with open(summary_file) as f:
                all_summaries[f"{short}_direct"] = json.load(f)
            continue

        print(f"\n--- Evaluating {model_cfg['name']} (direct) ---")
        try:
            summary = evaluate_model(
                model_name=model_cfg["name"],
                benchmark_file=benchmark_42,
                output_file=output,
                prompt_type="direct",
                quantization=model_cfg["quantization"],
                gpu_memory_utilization=model_cfg["gpu_mem"],
            )
            all_summaries[f"{short}_direct"] = summary
        except Exception as e:
            print(f"  ERROR: {e}")
            all_summaries[f"{short}_direct"] = {"error": str(e)}

    # ============================================================
    # Phase 2: CoT prompting on selected models (seed 42)
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Chain-of-Thought evaluation (seed 42)")
    print("=" * 60)

    cot_models = [m for m in MODELS if m["short"] in ["qwen1.5b", "llama8b", "qwen7b", "deepseek7b", "qwen14b", "qwen32b"]]
    for model_cfg in cot_models:
        short = model_cfg["short"]
        output = str(RESULTS_DIR / f"{short}_cot_seed42.jsonl")
        summary_file = str(RESULTS_DIR / f"{short}_cot_seed42_summary.json")

        if os.path.exists(summary_file):
            print(f"  Skipping {short} CoT (already done)")
            with open(summary_file) as f:
                all_summaries[f"{short}_cot"] = json.load(f)
            continue

        print(f"\n--- Evaluating {model_cfg['name']} (CoT) ---")
        try:
            summary = evaluate_model(
                model_name=model_cfg["name"],
                benchmark_file=benchmark_42,
                output_file=output,
                prompt_type="cot",
                quantization=model_cfg["quantization"],
                gpu_memory_utilization=model_cfg["gpu_mem"],
            )
            all_summaries[f"{short}_cot"] = summary
        except Exception as e:
            print(f"  ERROR: {e}")
            all_summaries[f"{short}_cot"] = {"error": str(e)}

    # ============================================================
    # Phase 3: Reliability evaluation (seed 123, 3 models, direct only)
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Reliability evaluation (seed 123)")
    print("=" * 60)

    reliability_models = [m for m in MODELS if m["short"] in ["qwen0.5b", "qwen1.5b", "qwen7b", "qwen14b", "qwen32b"]]
    for model_cfg in reliability_models:
        short = model_cfg["short"]
        output = str(RESULTS_DIR / f"{short}_direct_seed123.jsonl")
        summary_file = str(RESULTS_DIR / f"{short}_direct_seed123_summary.json")

        if os.path.exists(summary_file):
            print(f"  Skipping {short} reliability (already done)")
            with open(summary_file) as f:
                all_summaries[f"{short}_direct_123"] = json.load(f)
            continue

        print(f"\n--- Evaluating {model_cfg['name']} on seed 123 ---")
        try:
            summary = evaluate_model(
                model_name=model_cfg["name"],
                benchmark_file=benchmark_123,
                output_file=output,
                prompt_type="direct",
                quantization=model_cfg["quantization"],
                gpu_memory_utilization=model_cfg["gpu_mem"],
            )
            all_summaries[f"{short}_direct_123"] = summary
        except Exception as e:
            print(f"  ERROR: {e}")
            all_summaries[f"{short}_direct_123"] = {"error": str(e)}

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"All evaluations complete! Total time: {total_time/3600:.1f} hours")

    # Save master summary
    master_summary = str(RESULTS_DIR / "all_summaries.json")
    with open(master_summary, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Master summary saved to {master_summary}")

    return all_summaries


if __name__ == "__main__":
    run_evaluation_suite()
