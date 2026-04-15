import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.pipeline import (
    ARTIFACTS_DIR,
    BUDGET_SECONDS,
    CPU_WORKER_LIMIT,
    MAIN_SETTINGS,
    RUNS_DIR,
    SEEDS,
    SOURCE_LOG_BUDGET_SECONDS,
    ExperimentRunner,
    persist_run_outputs,
    prepare_one_setting,
    read_json,
    fit_estimators_from_rows,
    source_rows_from_result,
    set_env_threads,
)


def load_bundles():
    bundles = []
    prepared = read_json(ARTIFACTS_DIR / "prepared_manifest.json")
    thr_by_key = {(row["setting"], row["seed"]): row["threshold"] for row in prepared}
    for setting in MAIN_SETTINGS:
        base = setting.replace("_corrupted", "")
        corrupted = setting.endswith("_corrupted")
        for seed in SEEDS:
            bundle = prepare_one_setting(base, seed, corrupted)
            bundle.threshold = thr_by_key[(bundle.setting, bundle.seed)]
            bundles.append(bundle)
    return bundles


def run_one(bundle, method, budget, threshold, estimator=None, ablation=None, reference_actions=None):
    runner = ExperimentRunner(
        bundle,
        method,
        budget,
        threshold,
        estimator=estimator,
        ablation=ablation,
        reference_actions=reference_actions,
    )
    return runner.run()


def run_all() -> None:
    set_env_threads()
    bundles = load_bundles()
    source_rows = []
    with ProcessPoolExecutor(max_workers=CPU_WORKER_LIMIT) as pool:
        source_futures = {
            pool.submit(run_one, bundle, "MutableGreedy", SOURCE_LOG_BUDGET_SECONDS, bundle.threshold): bundle for bundle in bundles
        }
        for future in as_completed(source_futures):
            bundle = source_futures[future]
            result = future.result()
            run_dir = RUNS_DIR / bundle.setting / "MutableGreedy_source_logs" / str(bundle.seed)
            persist_run_outputs(
                run_dir,
                result,
                {
                    "setting": bundle.setting,
                    "seed": bundle.seed,
                    "method": "MutableGreedy_source_logs",
                    "budget_seconds": SOURCE_LOG_BUDGET_SECONDS,
                    "threshold": bundle.threshold,
                    "cpu_workers": CPU_WORKER_LIMIT,
                    "gpus": 0,
                },
            )
            source_rows.extend(source_rows_from_result(bundle, result))
    estimators = fit_estimators_from_rows(source_rows)
    methods = ["RawPEM", "HybridStatic", "FullClean+PEM", "LocalHeuristic", "MutableGreedy", "MutableRandom", "CanopyER"]
    canopy_runs = {}
    main_jobs = []
    for bundle in bundles:
        for method in methods:
            estimator = estimators.get(bundle.setting) if method == "CanopyER" else None
            main_jobs.append((bundle, method, estimator))
    with ProcessPoolExecutor(max_workers=CPU_WORKER_LIMIT) as pool:
        future_map = {
            pool.submit(run_one, bundle, method, BUDGET_SECONDS, bundle.threshold, estimator=estimator): (bundle, method)
            for bundle, method, estimator in main_jobs
        }
        for future in as_completed(future_map):
            bundle, method = future_map[future]
            result = future.result()
            run_dir = RUNS_DIR / bundle.setting / method / str(bundle.seed)
            persist_run_outputs(
                run_dir,
                result,
                {
                    "setting": bundle.setting,
                    "seed": bundle.seed,
                    "method": method,
                    "budget_seconds": BUDGET_SECONDS,
                    "threshold": bundle.threshold,
                    "gpus": 0,
                    "cpu_workers": CPU_WORKER_LIMIT,
                },
            )
            if method == "CanopyER":
                canopy_runs[(bundle.setting, bundle.seed)] = result["clean_actions"]
    ablation_settings = {"amazon_google_corrupted", "dblp_acm"}
    ablations = ["NoMicroSim", "NoRisk", "FormatOnly", "FullReblock"]
    ablation_jobs = []
    for bundle in bundles:
        if bundle.setting not in ablation_settings:
            continue
        estimator = estimators.get(bundle.setting)
        for ablation in ablations:
            reference_actions = canopy_runs.get((bundle.setting, bundle.seed), []) if ablation == "FullReblock" else None
            ablation_jobs.append((bundle, ablation, estimator, reference_actions))
    with ProcessPoolExecutor(max_workers=CPU_WORKER_LIMIT) as pool:
        future_map = {
            pool.submit(
                run_one,
                bundle,
                "CanopyER",
                BUDGET_SECONDS,
                bundle.threshold,
                estimator=estimator,
                ablation=ablation,
                reference_actions=reference_actions,
            ): (bundle, ablation)
            for bundle, ablation, estimator, reference_actions in ablation_jobs
        }
        for future in as_completed(future_map):
            bundle, ablation = future_map[future]
            result = future.result()
            run_dir = RUNS_DIR / bundle.setting / ablation / str(bundle.seed)
            persist_run_outputs(
                run_dir,
                result,
                {
                    "setting": bundle.setting,
                    "seed": bundle.seed,
                    "method": "CanopyER",
                    "ablation": ablation,
                    "budget_seconds": BUDGET_SECONDS,
                    "threshold": bundle.threshold,
                    "gpus": 0,
                    "cpu_workers": CPU_WORKER_LIMIT,
                },
            )


if __name__ == "__main__":
    run_all()
