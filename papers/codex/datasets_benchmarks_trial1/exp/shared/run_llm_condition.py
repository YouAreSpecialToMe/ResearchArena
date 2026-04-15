from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm import LLM, SamplingParams

from exp.shared.benchmark_spec import SEEDS
from exp.shared.eval_lib import (
    aggregate_condition,
    build_prompt,
    load_items,
    normalize_prediction,
    parse_json_object,
    save_seed_outputs,
    score_prediction,
)
from exp.shared.utils import ROOT, ensure_dir, write_json
from exp.shared.utils import timestamp


MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"


def run_condition(condition: str) -> dict:
    log_dir = ensure_dir(ROOT / "exp" / condition / "logs")
    run_log_lines = [
        f"experiment={condition}",
        "action=run_inference",
        f"model={MODEL_NAME}",
        f"seeds={SEEDS}",
        "temperature=0.2",
        "top_p=0.95",
        "max_tokens=512",
        "json_retry=1",
    ]
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        trust_remote_code=True,
    )
    items = load_items()
    per_seed = {}
    for seed in SEEDS:
        sampling = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=512, seed=seed)
        predictions = []
        executions = []
        for item in items:
            print(f"[{condition}] seed={seed} item={item['item_id']}", flush=True)
            prompt = build_prompt(item, condition)
            start = time.time()
            output = llm.generate(prompt, sampling_params=sampling)[0].outputs[0].text
            parsed, ok = parse_json_object(output)
            if not ok:
                retry_prompt = prompt + "\n\nReturn only valid JSON. No markdown fences."
                output = llm.generate(retry_prompt, sampling_params=sampling)[0].outputs[0].text
                parsed, ok = parse_json_object(output)
            pred = normalize_prediction(parsed)
            latency = time.time() - start
            semantic_invalid = bool(pred.get("validation_errors"))
            pred_record = {
                "item_id": item["item_id"],
                "raw_output": output,
                "latency_sec": latency,
                "malformed_output": (not ok) or semantic_invalid,
                "semantic_invalid_output": semantic_invalid,
                **pred,
            }
            score = score_prediction(item, pred)
            score["latency_sec"] = latency
            score["malformed_output"] = (not ok) or semantic_invalid
            score["semantic_invalid_output"] = semantic_invalid
            predictions.append(pred_record)
            executions.append(score)
        save_seed_outputs(condition, seed, predictions, executions)
        per_seed[seed] = executions
        seed_repair_pass = sum(r["needs_update_repair_pass"] for r in executions if r["gold_label"] == "needs_update")
        run_log_lines.append(f"seed={seed} items={len(executions)} needs_update_repair_successes={seed_repair_pass}")

    summary = aggregate_condition(condition)
    write_json(ROOT / "exp" / condition / "results.json", {"experiment": condition, "config": {"model": MODEL_NAME, "seeds": SEEDS}, "metrics": summary})
    run_log_lines.append(f"mean_needs_update_repair_pass_rate={summary['needs_update_repair_pass_rate']['mean']}")
    (log_dir / f"{timestamp()}_run.log").write_text("\n".join(run_log_lines) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["closed_book", "thread_only", "authority_aware", "authority_no_versions"])
    args = parser.parse_args()
    run_condition(args.condition)
