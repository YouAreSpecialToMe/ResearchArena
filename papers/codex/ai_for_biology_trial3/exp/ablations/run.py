from pathlib import Path

from exp.core.run import run_condition
from exp.shared.utils import SEEDS


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    for dataset in ["Adamson", "Norman", "Replogle"]:
        run_condition(dataset, 42, "ablation_no_retrieval", root=root, use_retrieval=False)
    for dataset in ["Adamson", "Norman", "Replogle"]:
        for seed in SEEDS:
            run_condition(dataset, seed, "ablation_no_conformal", root=root, disable_conformal=True)
    for dataset in ["Adamson", "Norman", "Replogle"]:
        run_condition(dataset, 42, "ablation_uncertainty_only", root=root, uncertainty_only=True)
    for dataset in ["Adamson", "Norman", "Replogle"]:
        run_condition(dataset, 42, "ablation_no_novelty", root=root, omit_novelty=True)
    for dataset in ["Adamson", "Norman", "Replogle"]:
        run_condition(dataset, 42, "ablation_direct_prediction", root=root, direct_prediction=True)


if __name__ == "__main__":
    main()
