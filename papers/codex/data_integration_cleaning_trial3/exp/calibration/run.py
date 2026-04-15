import pandas as pd

from exp.shared.stressaudit import (
    METHODS,
    PRIMARY_SEED,
    freeze_inventory,
    run_calibration_experiment,
    run_method,
    split_inventory,
    tune_threshold,
)


if __name__ == "__main__":
    inventory = split_inventory(freeze_inventory())
    frames = []
    thresholds = {}
    for method in METHODS:
        threshold = tune_threshold(method, inventory)
        thresholds[method] = threshold
        pred_df, _ = run_method(method, inventory, threshold, run_seed=PRIMARY_SEED, repeat_tag="headline", persist_outputs=False)
        frames.append(pred_df)
    run_calibration_experiment(pd.concat(frames, ignore_index=True), inventory, thresholds)
