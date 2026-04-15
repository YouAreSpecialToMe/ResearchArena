"""Compute PRA-NLI and PRA-TPD scores using DeBERTa NLI model."""
import os
import sys
import json
import time
import glob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from exp.shared.pra_score import pra_em, pra_f1, pra_nli_batch, pra_tpd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def main():
    t0 = time.time()

    # Load NLI model
    nli_model_name = "cross-encoder/nli-deberta-v3-large"
    print(f"Loading NLI model: {nli_model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
    model.eval()
    print(f"NLI model loaded on {device}")

    # Find all generation files
    gen_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "generations_*.json")))
    print(f"Found {len(gen_files)} generation files")

    for gen_file in gen_files:
        basename = os.path.basename(gen_file)
        print(f"\nProcessing {basename}...")

        with open(gen_file) as f:
            results = json.load(f)

        # Check if PRA scores already computed
        if results and "pra_nli" in results[0]:
            print(f"  PRA scores already present, skipping.")
            continue

        # Compute PRA-EM and PRA-F1
        for r in results:
            r["pra_em"] = pra_em(r["parametric_answer"], r["rag_answer"])
            r["pra_f1"] = pra_f1(r["parametric_answer"], r["rag_answer"])

        # Compute PRA-NLI in batch
        premises = [r["parametric_answer"] for r in results]
        hypotheses = [r["rag_answer"] for r in results]

        # Direction 1: parametric entails RAG
        nli_scores_fwd = pra_nli_batch(premises, hypotheses, model, tokenizer, device, batch_size=64)
        # Direction 2: RAG entails parametric (for symmetry, take max)
        nli_scores_rev = pra_nli_batch(hypotheses, premises, model, tokenizer, device, batch_size=64)

        for i, r in enumerate(results):
            r["pra_nli"] = max(nli_scores_fwd[i], nli_scores_rev[i])
            r["pra_tpd"] = pra_tpd(
                r["parametric_logprob_mean"], r["rag_logprob_mean"],
                r["parametric_answer"], r["rag_answer"]
            )

        # Save back
        with open(gen_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Done. PRA-EM agreement rate: {sum(r['pra_em'] for r in results)/len(results):.2%}")
        print(f"  Mean PRA-NLI: {sum(r['pra_nli'] for r in results)/len(results):.3f}")
        print(f"  Mean PRA-F1: {sum(r['pra_f1'] for r in results)/len(results):.3f}")

    elapsed = time.time() - t0
    print(f"\n=== NLI scoring complete in {elapsed/60:.1f} minutes ===")


if __name__ == "__main__":
    main()
