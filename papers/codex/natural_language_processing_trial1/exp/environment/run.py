import time

from exp.shared.data import build_all_data
from exp.shared.train_eval import run_train
from exp.shared.utils import ROOT, dump_json, run_config, system_info


def main():
    t0 = time.time()
    data = build_all_data()
    pilot_dir = ROOT / "exp" / "pilot_qwen_seed11"
    metrics = run_train(pilot_dir, data, "answer_only", seed=11, example_limit=200)
    projected_training_hours = (metrics["runtime_seconds"] * 9) / 3600.0
    projected_total_hours = projected_training_hours + 1.0 + 1.5 + 0.5
    result = {
        "experiment": "environment",
        "runtime_seconds": time.time() - t0,
        "environment": system_info(),
        "pilot_metrics": metrics,
        "budget_gate": {
            "projected_training_hours_for_9_runs": projected_training_hours,
            "projected_total_hours_with_prep_eval_overhead": projected_total_hours,
            "within_8_hour_budget": projected_total_hours <= 8.0,
            "note": "Projection includes 9 seed runs plus coarse allowances for data prep, evaluation, and plotting.",
        },
        "config": run_config(condition="answer_only", seed=11, extra={"pilot_example_limit": 200}),
    }
    dump_json(result, ROOT / "exp" / "environment" / "results.json")
    dump_json(result["config"], ROOT / "exp" / "environment" / "config.json")


if __name__ == "__main__":
    main()
