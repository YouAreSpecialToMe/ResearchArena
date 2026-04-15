from exp.shared.data import build_all_data
from exp.shared.train_eval import run_train
from exp.shared.utils import ROOT, SEEDS, dump_json, run_config


def main():
    data = build_all_data()
    metrics = []
    for seed in SEEDS:
        metrics.append(run_train(ROOT / "exp" / "answer_only" / f"seed_{seed}", data, "answer_only", seed))
    dump_json({"experiment": "answer_only", "runs": metrics, "config": run_config(condition="answer_only")}, ROOT / "exp" / "answer_only" / "results.json")


if __name__ == "__main__":
    main()
