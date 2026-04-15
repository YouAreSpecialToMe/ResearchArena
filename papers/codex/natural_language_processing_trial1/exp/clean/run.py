from exp.shared.data import build_all_data
from exp.shared.train_eval import run_train
from exp.shared.utils import ROOT, SEEDS, dump_json, run_config


def main():
    data = build_all_data()
    metrics = []
    for seed in SEEDS:
        metrics.append(run_train(ROOT / "exp" / "clean" / f"seed_{seed}", data, "clean", seed))
    dump_json({"experiment": "clean", "runs": metrics, "config": run_config(condition="clean")}, ROOT / "exp" / "clean" / "results.json")


if __name__ == "__main__":
    main()
