"""Generate parametric and RAG answers using vLLM for all models/datasets/seeds."""
import os
import sys
import json
import time
import gc
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from exp.shared.prompts import PARAMETRIC_PROMPT, RAG_PROMPT, VERBALIZED_CONF_PROMPT
from exp.shared.evaluation import exact_match, token_f1
from exp.shared.confidence import (
    token_probability, token_entropy, verbalized_confidence,
    self_consistency, token_prob_delta
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 43, 44]
DATASETS = ["nq", "triviaqa", "popqa"]
TOP_K_CONTEXT = 5  # use top-5 passages for context
SC_SAMPLES = 5  # self-consistency samples

MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "llama8b"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral7b"),
]


def make_context(passages, top_k=TOP_K_CONTEXT):
    """Concatenate top-k passages into context string."""
    texts = []
    for p in passages[:top_k]:
        title = p.get("title", "")
        text = p.get("text", "")
        if title:
            texts.append(f"{title}: {text}")
        else:
            texts.append(text)
    return "\n\n".join(texts)


def extract_logprobs(output):
    """Extract mean log-prob from vLLM output."""
    logprobs_list = []
    if hasattr(output, 'outputs') and output.outputs:
        o = output.outputs[0]
        if hasattr(o, 'logprobs') and o.logprobs:
            for lp in o.logprobs:
                if lp:
                    # Each element is a dict: {token_id: Logprob}
                    # Get the chosen token's logprob
                    for token_id, logprob_obj in lp.items():
                        logprobs_list.append(logprob_obj.logprob)
                        break
    if not logprobs_list:
        return -5.0, []
    return float(np.mean(logprobs_list)), logprobs_list


def get_answer_text(output):
    """Extract answer text from vLLM output."""
    if hasattr(output, 'outputs') and output.outputs:
        return output.outputs[0].text.strip()
    return ""


def run_model(model_name, model_short):
    """Run all inference for a single model."""
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")

    llm = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        max_model_len=4096,
        trust_remote_code=True,
    )

    greedy_params = SamplingParams(
        temperature=0,
        max_tokens=64,
        logprobs=1,
    )
    sc_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=64,
        logprobs=1,
    )

    timing_data = {}

    for dataset_name in DATASETS:
        for seed in SEEDS:
            data_path = os.path.join(DATA_DIR, f"{dataset_name}_seed{seed}_retrieved.json")
            out_path = os.path.join(RESULTS_DIR, f"generations_{model_short}_{dataset_name}_seed{seed}.json")

            if os.path.exists(out_path):
                print(f"  Skipping {out_path} (already exists)")
                continue

            with open(data_path) as f:
                samples = json.load(f)

            n = len(samples)
            print(f"\n--- {model_short} / {dataset_name} / seed {seed} ({n} questions) ---")

            # 1. Parametric generation (greedy)
            print("  Generating parametric answers...")
            parametric_prompts = [
                PARAMETRIC_PROMPT.format(question=s["question"]) for s in samples
            ]
            t1 = time.time()
            param_outputs = llm.generate(parametric_prompts, greedy_params)
            t_param = time.time() - t1
            print(f"    Done in {t_param:.1f}s ({t_param/n*1000:.0f}ms/query)")

            # 2. RAG generation (greedy)
            print("  Generating RAG answers...")
            rag_prompts = [
                RAG_PROMPT.format(
                    question=s["question"],
                    context=make_context(s["retrieved_passages"])
                ) for s in samples
            ]
            t1 = time.time()
            rag_outputs = llm.generate(rag_prompts, greedy_params)
            t_rag = time.time() - t1
            print(f"    Done in {t_rag:.1f}s ({t_rag/n*1000:.0f}ms/query)")

            # 3. Verbalized confidence
            print("  Generating verbalized confidence...")
            vc_prompts = [
                VERBALIZED_CONF_PROMPT.format(
                    question=s["question"],
                    context=make_context(s["retrieved_passages"])
                ) for s in samples
            ]
            t1 = time.time()
            vc_outputs = llm.generate(vc_prompts, SamplingParams(temperature=0, max_tokens=128))
            t_vc = time.time() - t1
            print(f"    Done in {t_vc:.1f}s")

            # 4. Self-consistency samples
            print(f"  Generating {SC_SAMPLES} self-consistency samples...")
            sc_answers = [[] for _ in range(n)]
            t1 = time.time()
            for sc_i in range(SC_SAMPLES):
                sc_outputs = llm.generate(rag_prompts, sc_params)
                for j, out in enumerate(sc_outputs):
                    sc_answers[j].append(get_answer_text(out))
            t_sc = time.time() - t1
            print(f"    Done in {t_sc:.1f}s")

            # Process results
            print("  Processing results...")
            results = []
            for i in range(n):
                param_answer = get_answer_text(param_outputs[i])
                rag_answer = get_answer_text(rag_outputs[i])
                param_logprob, param_lps = extract_logprobs(param_outputs[i])
                rag_logprob, rag_lps = extract_logprobs(rag_outputs[i])
                vc_text = get_answer_text(vc_outputs[i])
                gold = samples[i]["gold_answers"]

                result = {
                    "question": samples[i]["question"],
                    "gold_answers": gold,
                    "parametric_answer": param_answer,
                    "rag_answer": rag_answer,
                    "parametric_logprob_mean": param_logprob,
                    "rag_logprob_mean": rag_logprob,
                    "parametric_correct_em": int(exact_match(param_answer, gold)),
                    "rag_correct_em": int(exact_match(rag_answer, gold)),
                    "parametric_correct_f1": token_f1(param_answer, gold),
                    "rag_correct_f1": token_f1(rag_answer, gold),
                    "token_prob_rag": rag_logprob,
                    "neg_entropy_rag": -token_entropy(rag_lps),
                    "verbalized_conf": verbalized_confidence(vc_text),
                    "self_consistency": self_consistency(sc_answers[i], rag_answer),
                    "tpd_baseline": token_prob_delta(param_logprob, rag_logprob),
                }

                # Metadata for PopQA
                if "s_pop" in samples[i]:
                    result["s_pop"] = samples[i]["s_pop"]

                results.append(result)

            # Save timing
            timing_key = f"{model_short}_{dataset_name}_seed{seed}"
            timing_data[timing_key] = {
                "parametric_seconds": t_param,
                "rag_seconds": t_rag,
                "verbalized_seconds": t_vc,
                "sc_seconds": t_sc,
                "per_query_param_ms": t_param / n * 1000,
                "per_query_rag_ms": t_rag / n * 1000,
                "per_query_sc_ms": t_sc / n * 1000,
                "n_questions": n,
            }

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved {len(results)} results to {out_path}")

    # Save timing data
    timing_path = os.path.join(RESULTS_DIR, f"timing_{model_short}.json")
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=2)

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n{model_short} complete. GPU memory freed.")


def main():
    t0 = time.time()

    for model_name, model_short in MODELS:
        run_model(model_name, model_short)

    elapsed = time.time() - t0
    print(f"\n=== All inference complete in {elapsed/60:.1f} minutes ===")


if __name__ == "__main__":
    # Allow running a single model via command line arg
    if len(sys.argv) > 1:
        model_idx = int(sys.argv[1])
        model_name, model_short = MODELS[model_idx]
        run_model(model_name, model_short)
    else:
        main()
