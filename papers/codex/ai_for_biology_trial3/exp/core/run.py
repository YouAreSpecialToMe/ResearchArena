from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import traceback
from pathlib import Path
import time

from exp.shared.pipeline import (
    build_seed_bundle,
    choose_frozen_baseline,
    conformal_interval_stats,
    evaluate_methods,
    fit_router_models,
    save_run_artifacts,
    train_residual_corrector,
)
from exp.shared.utils import RUN_VERSION, ensure_dir, get_process_peak_rss_bytes, get_system_info


def run_condition(dataset: str, seed: int, tag: str, root: Path, direct_prediction: bool = False, use_retrieval: bool = True, omit_novelty: bool = False, uncertainty_only: bool = False, gain_key: str = "rmse", alpha: float = 0.2, route_ratio: float = 0.2, cal_ratio: float = 0.1, disable_conformal: bool = False, cross_fit_conformal: bool = False, calibration_subsample_fraction: float = 1.0) -> None:
    out_dir = root / "exp" / dataset / tag / f"seed_{seed}"
    log_path = ensure_dir(out_dir / "logs") / "run.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle, contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
        def log(message: str) -> None:
            timestamp = dt.datetime.now().isoformat(timespec="seconds")
            print(f"[{timestamp}] {message}", flush=True)

        log(
            f"start dataset={dataset} seed={seed} tag={tag} gain_key={gain_key} alpha={alpha} "
            f"route_ratio={route_ratio} cal_ratio={cal_ratio} direct_prediction={int(direct_prediction)} "
            f"use_retrieval={int(use_retrieval)} omit_novelty={int(omit_novelty)} "
            f"uncertainty_only={int(uncertainty_only)} disable_conformal={int(disable_conformal)} "
            f"cross_fit_conformal={int(cross_fit_conformal)} calibration_subsample_fraction={calibration_subsample_fraction}"
        )
        try:
            _run_condition_impl(
                dataset=dataset,
                seed=seed,
                tag=tag,
                root=root,
                direct_prediction=direct_prediction,
                use_retrieval=use_retrieval,
                omit_novelty=omit_novelty,
                uncertainty_only=uncertainty_only,
                gain_key=gain_key,
                alpha=alpha,
                route_ratio=route_ratio,
                cal_ratio=cal_ratio,
                disable_conformal=disable_conformal,
                cross_fit_conformal=cross_fit_conformal,
                calibration_subsample_fraction=calibration_subsample_fraction,
                out_dir=out_dir,
                log=log,
            )
        except Exception:
            log("run_failed")
            print(traceback.format_exc(), flush=True)
            raise


def _run_condition_impl(dataset: str, seed: int, tag: str, root: Path, direct_prediction: bool = False, use_retrieval: bool = True, omit_novelty: bool = False, uncertainty_only: bool = False, gain_key: str = "rmse", alpha: float = 0.2, route_ratio: float = 0.2, cal_ratio: float = 0.1, disable_conformal: bool = False, cross_fit_conformal: bool = False, calibration_subsample_fraction: float = 1.0, out_dir: Path | None = None, log: callable | None = None) -> None:
    total_start = time.time()
    total_cpu_start = time.process_time()
    if log is not None:
        log("building_seed_bundle")
    bundle = build_seed_bundle(dataset, seed, root=root, route_ratio=route_ratio, cal_ratio=cal_ratio)
    if log is not None:
        split_counts = {split: int((bundle.split_labels == split).sum()) for split in ["train", "route_dev", "calibration", "test"]}
        log(f"bundle_ready retained_genes={len(bundle.retained_genes)} responsive_panel={len(bundle.responsive_panel)} split_counts={split_counts}")
        log("choosing_frozen_baseline")
    y0, baseline_info = choose_frozen_baseline(bundle, log_fn=log)
    if log is not None:
        log("training_residual_corrector")
    residual = train_residual_corrector(bundle, y0, direct_prediction=direct_prediction, use_retrieval=use_retrieval, root=root, log_fn=log)
    if log is not None:
        log("fitting_router_models")
    routers = fit_router_models(
        bundle,
        y0,
        residual,
        gain_key=gain_key,
        alpha=alpha,
        omit_novelty=omit_novelty,
        uncertainty_only=uncertainty_only,
        disable_conformal=disable_conformal,
        cross_fit_conformal=cross_fit_conformal,
        calibration_subsample_fraction=calibration_subsample_fraction,
        log_fn=log,
    )
    if log is not None:
        log("evaluating_methods")
    per_perturbation, metrics = evaluate_methods(bundle, y0, residual, routers)
    coverage = conformal_interval_stats(bundle, residual["y1"])
    metrics["baseline_info"] = baseline_info
    metrics["conformal_interval_stats"] = coverage
    metrics["router_info"] = {
        "adjustment": routers["adjustment"],
        "swap_adjustment": routers["swap_adjustment"],
        "feature_names": routers["feature_names"],
    }
    config = {
        "run_version": RUN_VERSION,
        "dataset": dataset,
        "seed": seed,
        "tag": tag,
        "direct_prediction": direct_prediction,
        "use_retrieval": use_retrieval,
        "omit_novelty": omit_novelty,
        "uncertainty_only": uncertainty_only,
        "gain_key": gain_key,
        "alpha": alpha,
        "route_ratio": route_ratio,
        "cal_ratio": cal_ratio,
        "disable_conformal": disable_conformal,
        "cross_fit_conformal": cross_fit_conformal,
        "calibration_subsample_fraction": calibration_subsample_fraction,
        "split_config": bundle.split_config,
    }
    runtime = {
        "run_version": RUN_VERSION,
        "system": get_system_info(),
        "baseline_runtime_seconds": baseline_info["runtime_seconds"],
        "baseline_cpu_seconds": baseline_info["cpu_seconds"],
        "baseline_peak_rss_bytes": baseline_info["peak_rss_bytes"],
        "baseline_parameter_count": baseline_info["selected_parameter_count"],
        "residual_runtime_seconds": residual["runtime"]["seconds"],
        "residual_cpu_seconds": residual["runtime"]["cpu_seconds"],
        "peak_gpu_bytes": residual["runtime"]["peak_gpu_bytes"],
        "residual_peak_rss_bytes": residual["runtime"]["peak_rss_bytes"],
        "residual_parameter_count": residual["runtime"]["parameters"],
        "router_runtime_seconds": routers["runtime"]["seconds"],
        "router_cpu_seconds": routers["runtime"]["cpu_seconds"],
        "router_peak_rss_bytes": routers["runtime"]["peak_rss_bytes"],
        "classifier_parameter_count": routers["runtime"]["classifier_parameter_count"],
        "uncertainty_parameter_count": routers["runtime"]["uncertainty_parameter_count"],
        "gain_regressor_parameter_count": routers["runtime"]["gain_regressor_parameter_count"],
        "conformal_parameter_count": routers["runtime"]["conformal_parameter_count"],
        "total_runtime_seconds": time.time() - total_start,
        "total_cpu_seconds": time.process_time() - total_cpu_start,
        "total_peak_rss_bytes": get_process_peak_rss_bytes(),
    }
    out_dir = out_dir or (root / "exp" / dataset / tag / f"seed_{seed}")
    save_run_artifacts(
        out_dir,
        config=config,
        metrics=metrics,
        per_perturbation=per_perturbation,
        predictions={
            "y0": y0,
            "y1": residual["y1"],
            "mc": residual["mc"],
            "router_classifier_score": routers["scores"]["classifier_gate"],
            "router_uncertainty_score": routers["scores"]["uncertainty_gate"],
            "router_gain_regressor_score": routers["scores"]["gain_regressor"],
            "router_conformal_score": routers["scores"]["conformal_gate"],
            "router_quantile_raw_score": routers["scores"]["quantile_raw_gate"],
            "router_test_gain": routers["test_gains"],
        },
        runtime=runtime,
    )
    if log is not None:
        for method_name in ["uncertainty_gate@40", "gain_regressor@40", "conformal_gate@40"]:
            if method_name in metrics:
                method_metrics = metrics[method_name]
                log(
                    f"final_metrics method={method_name} "
                    f"accepted_fraction={method_metrics['accepted_fraction']:.6f} "
                    f"mean_gain={method_metrics['mean_gain']:.6f} "
                    f"all_gene_rmse={method_metrics['all_gene_rmse']:.6f} "
                    f"pathway_corr={method_metrics['pathway_corr']:.6f}"
                )
        log(
            f"completed selected_baseline={baseline_info['selected']} "
            f"conformal_adjustment={routers['adjustment']:.6f} total_runtime_seconds={runtime['total_runtime_seconds']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--direct-prediction", action="store_true")
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--omit-novelty", action="store_true")
    parser.add_argument("--uncertainty-only", action="store_true")
    parser.add_argument("--gain-key", default="rmse")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--route-ratio", type=float, default=0.2)
    parser.add_argument("--cal-ratio", type=float, default=0.1)
    parser.add_argument("--disable-conformal", action="store_true")
    parser.add_argument("--cross-fit-conformal", action="store_true")
    parser.add_argument("--calibration-subsample-fraction", type=float, default=1.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run_condition(
        args.dataset,
        args.seed,
        args.tag,
        root=root,
        direct_prediction=args.direct_prediction,
        use_retrieval=not args.no_retrieval,
        omit_novelty=args.omit_novelty,
        uncertainty_only=args.uncertainty_only,
        gain_key=args.gain_key,
        alpha=args.alpha,
        route_ratio=args.route_ratio,
        cal_ratio=args.cal_ratio,
        disable_conformal=args.disable_conformal,
        cross_fit_conformal=args.cross_fit_conformal,
        calibration_subsample_fraction=args.calibration_subsample_fraction,
    )


if __name__ == "__main__":
    main()
