#!/usr/bin/env python3
"""
SkillStack Model Evaluation Pipeline

Evaluates LLMs on generated benchmark instances using vLLM for efficient batched inference.
Supports both direct and chain-of-thought prompting strategies.
"""
import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_benchmark(filepath: str) -> List[Dict]:
    """Load benchmark instances from JSONL."""
    instances = []
    with open(filepath) as f:
        for line in f:
            instances.append(json.loads(line))
    return instances


def format_prompt_direct(question: str) -> str:
    """Format question for direct (no CoT) prompting."""
    return f"Answer the following question with just the final answer, no explanation.\n\nQ: {question}\nA:"


def format_prompt_cot(question: str) -> str:
    """Format question for chain-of-thought prompting."""
    return f"Q: {question}\nLet me think step by step:"


def extract_answer_direct(output: str, answer_type: str) -> str:
    """Extract answer from direct model output."""
    output = output.strip()
    # Take first line only
    first_line = output.split("\n")[0].strip()

    # Remove common prefixes
    for prefix in ["The answer is ", "Answer: ", "A: ", "Result: "]:
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix):].strip()

    return first_line


def extract_answer_cot(output: str, answer_type: str) -> str:
    """Extract final answer from CoT output."""
    output = output.strip()

    # Look for explicit final answer markers
    patterns = [
        r"(?:the |The )?(?:final )?answer is[:\s]+(.+?)(?:\.|$)",
        r"(?:Therefore|So|Thus|Hence)[,:\s]+(?:the answer is\s+)?(.+?)(?:\.|$)",
        r"\*\*(.+?)\*\*",  # Bold answer
        r"= (.+?)(?:\.|$)",  # After equals sign (last one)
    ]

    for pattern in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # Fallback: last line
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return output


def check_answer(predicted: str, gold: str, answer_type: str, instance: Dict) -> bool:
    """Check if predicted answer matches gold answer."""
    pred = predicted.strip().lower()
    gold_lower = gold.strip().lower()

    if answer_type == "multi_part":
        return check_multi_part(pred, gold_lower, instance)

    if answer_type == "integer":
        # Extract number from prediction
        nums = re.findall(r'-?\d+', pred)
        gold_nums = re.findall(r'-?\d+', gold_lower)
        if nums and gold_nums:
            return nums[0] == gold_nums[0]
        return False

    if answer_type == "yesno":
        pred_yn = "yes" if "yes" in pred else ("no" if "no" in pred else pred)
        return pred_yn == gold_lower

    if answer_type == "set":
        # Extract numbers from both
        pred_nums = set(re.findall(r'\d+', pred))
        gold_nums = set(re.findall(r'\d+', gold_lower))
        return pred_nums == gold_nums

    if answer_type in ("time",):
        # Normalize time format
        pred_clean = re.findall(r'\d+:\d+', pred)
        gold_clean = re.findall(r'\d+:\d+', gold_lower)
        if pred_clean and gold_clean:
            return pred_clean[0] == gold_clean[0]
        return False

    if answer_type == "coordinate":
        pred_coords = re.findall(r'\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?', pred)
        gold_coords = re.findall(r'\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?', gold_lower)
        if pred_coords and gold_coords:
            return pred_coords[0] == gold_coords[0]
        return False

    if answer_type == "fraction":
        # Check fraction equality
        pred_frac = re.findall(r'(\d+)/(\d+)', pred)
        gold_frac = re.findall(r'(\d+)/(\d+)', gold_lower)
        if pred_frac and gold_frac:
            pn, pd = int(pred_frac[0][0]), int(pred_frac[0][1])
            gn, gd = int(gold_frac[0][0]), int(gold_frac[0][1])
            return pn * gd == gn * pd
        return False

    # Default: string containment (relaxed matching)
    return gold_lower in pred or pred in gold_lower


def check_multi_part(pred: str, gold: str, instance: Dict) -> bool:
    """Check multi-part answers (both parts must be correct)."""
    sub_answers = instance.get("sub_answers", [])
    sub_types = instance.get("sub_types", [])
    if not sub_answers:
        return gold in pred

    correct_parts = 0
    for i, (sa, st) in enumerate(zip(sub_answers, sub_types)):
        # Try to find part i answer in prediction
        part_pattern = rf'part\s*{i+1}\s*:\s*(.+?)(?:,\s*part|$)'
        match = re.search(part_pattern, pred, re.IGNORECASE)
        if match:
            part_pred = match.group(1).strip()
            if check_answer(part_pred, sa, st, {}):
                correct_parts += 1
        elif sa.lower() in pred:
            correct_parts += 1

    return correct_parts == len(sub_answers)


def evaluate_model(
    model_name: str,
    benchmark_file: str,
    output_file: str,
    prompt_type: str = "direct",
    batch_size: int = 128,
    max_tokens: int = 256,
    tensor_parallel_size: int = 1,
    quantization: Optional[str] = None,
    gpu_memory_utilization: float = 0.85,
):
    """Evaluate a model on the benchmark."""
    from vllm import LLM, SamplingParams

    instances = load_benchmark(benchmark_file)
    print(f"Loaded {len(instances)} instances from {benchmark_file}")

    # Format prompts
    if prompt_type == "direct":
        prompts = [format_prompt_direct(inst["question"]) for inst in instances]
        max_tokens = 256
    else:
        prompts = [format_prompt_cot(inst["question"]) for inst in instances]
        max_tokens = 512

    print(f"Loading model: {model_name}...")
    t0 = time.time()

    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "max_model_len": 4096,
    }
    if quantization:
        llm_kwargs["quantization"] = quantization

    llm = LLM(**llm_kwargs)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        top_p=1.0,
    )

    print(f"Running inference ({len(prompts)} prompts, batch_size={batch_size})...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    inference_time = time.time() - t0
    print(f"Inference done in {inference_time:.1f}s ({len(prompts)/inference_time:.1f} prompts/s)")

    # Process results
    results = []
    for inst, output in zip(instances, outputs):
        generated_text = output.outputs[0].text
        if prompt_type == "direct":
            predicted = extract_answer_direct(generated_text, inst["answer_type"])
        else:
            predicted = extract_answer_cot(generated_text, inst["answer_type"])

        correct = check_answer(predicted, inst["answer"], inst["answer_type"], inst)
        results.append({
            "id": inst["id"],
            "skill_combo": inst["skill_combo"],
            "level": inst["level"],
            "answer_type": inst["answer_type"],
            "gold_answer": inst["answer"],
            "predicted": predicted,
            "raw_output": generated_text[:500],
            "correct": correct,
        })

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Compute summary
    total_correct = sum(1 for r in results if r["correct"])
    accuracy = total_correct / len(results)
    print(f"Overall accuracy: {accuracy:.3f} ({total_correct}/{len(results)})")

    # Per-category accuracy
    cat_results = {}
    for r in results:
        cat = r["skill_combo"]
        if cat not in cat_results:
            cat_results[cat] = {"correct": 0, "total": 0}
        cat_results[cat]["total"] += 1
        if r["correct"]:
            cat_results[cat]["correct"] += 1

    for cat in sorted(cat_results.keys()):
        cr = cat_results[cat]
        cr["accuracy"] = cr["correct"] / cr["total"]

    summary = {
        "model": model_name,
        "prompt_type": prompt_type,
        "benchmark_file": os.path.basename(benchmark_file),
        "total_instances": len(results),
        "overall_accuracy": accuracy,
        "per_category": cat_results,
        "load_time_s": load_time,
        "inference_time_s": inference_time,
        "throughput_prompts_per_s": len(prompts) / inference_time,
    }

    summary_file = output_file.replace(".jsonl", "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {output_file}")
    print(f"Summary saved to {summary_file}")

    # Clean up GPU memory
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-type", choices=["direct", "cot"], default="direct")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        benchmark_file=args.benchmark,
        output_file=args.output,
        prompt_type=args.prompt_type,
        batch_size=args.batch_size,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
