from pathlib import Path

from exp.core.run import run_condition
from exp.shared.utils import SEEDS


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    for dataset in ["Adamson", "Norman", "Replogle"]:
        for seed in SEEDS:
            for alpha in [0.1, 0.2, 0.3]:
                run_condition(dataset, seed, f"sensitivity_alpha_{str(alpha).replace('.', '_')}", root=root, alpha=alpha)
    for dataset in ["Adamson", "Norman", "Replogle"]:
        for route_ratio, cal_ratio in [(0.2, 0.1), (0.15, 0.15), (0.1, 0.2)]:
            tag = f"sensitivity_split_r{int(route_ratio*100)}_c{int(cal_ratio*100)}"
            run_condition(dataset, 42, tag, root=root, route_ratio=route_ratio, cal_ratio=cal_ratio)
    for dataset in ["Adamson", "Norman", "Replogle"]:
        run_condition(dataset, 42, "sensitivity_crossfit", root=root, cross_fit_conformal=True)
    for dataset in ["Adamson", "Norman", "Replogle"]:
        for gain_key in ["topde", "pearson", "pathway"]:
            run_condition(dataset, 42, f"sensitivity_gain_{gain_key}", root=root, gain_key=gain_key)
    for dataset in ["Adamson", "Norman", "Replogle"]:
        for frac in [0.5, 0.75, 1.0]:
            tag = f"sensitivity_calibration_size_{str(frac).replace('.', '_')}"
            run_condition(dataset, 42, tag, root=root, calibration_subsample_fraction=frac)


if __name__ == "__main__":
    main()
