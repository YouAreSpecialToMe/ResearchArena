import argparse

from exp.shared.rerank import score_candidate
from exp.shared.run_core import merge_result_rows
from exp.shared.utils import project_path, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    rows = []
    from exp.shared.run_core import load_split, select_with_global_siglip, SEEDS

    records = load_split(args.split)
    total = len(records) * len(SEEDS)
    done = 0
    print(f"[best_of_4_global_siglip] split={args.split} prompts={len(records)} seeds={len(SEEDS)} k={args.k}", flush=True)
    for record in records:
        for seed in SEEDS:
            best = select_with_global_siglip({"prompt": record["prompt"], "seed": seed}, args.split, args.k)
            structured_eval = score_candidate(
                record=record,
                image_path=project_path("artifacts", "candidate_cache", args.split, record["prompt_id"], f"seed_{seed}", f"cand_{best['candidate_id']}.png"),
                heatmap_path=None,
                method="best_of_4_global_siglip",
                assignment_source="detector",
                use_counterfactual=False,
            )
            rows.append(
                {
                    "method": "best_of_4_global_siglip",
                    "dataset": args.split,
                    "prompt_id": record["prompt_id"],
                    "prompt": record["prompt"],
                    "seed": seed,
                    "selected_candidate_id": best["candidate_id"],
                    "final_score": best["final_score"],
                    "atomic_scores": structured_eval["atomic_scores"],
                    "group_counts": structured_eval["group_counts"],
                    "slot_assignments": structured_eval["slot_assignments"],
                    "all_atoms_pass": structured_eval["all_atoms_pass"],
                    "latency_sec": 0.0,
                    "feature_cache_paths": {},
                }
            )
            done += 1
            print(
                f"[best_of_4_global_siglip] split={args.split} prompt_id={record['prompt_id']} seed={seed} "
                f"selected={best['candidate_id']} score={best['final_score']:.4f} progress={done}/{total}",
                flush=True,
            )
    from exp.shared.utils import write_jsonl

    out_path = project_path("results", "best_of_4_global_siglip.jsonl")
    merged = merge_result_rows(out_path, rows)
    write_json(project_path("exp", "best_of_4_global_siglip", "results.json"), {"num_rows": len(merged)})


if __name__ == "__main__":
    main()
