from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from pymc.blocking import DictToArrayBijection, RaveledVars

from exp.shared.core import ensure_dir, save_csv, save_json


RADON_URL = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/radon.csv"


def download_and_prepare_radon(data_dir: str | Path) -> dict:
    data_dir = ensure_dir(data_dir)
    raw_path = data_dir / "radon.csv"
    if not raw_path.exists():
        df = pd.read_csv(RADON_URL)
        df.to_csv(raw_path, index=False)
    else:
        df = pd.read_csv(raw_path)
    top_counties = (
        df.groupby("county")
        .size()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    subset = df[df["county"].isin(top_counties)].copy()
    subset["county_idx"] = subset["county"].astype("category").cat.codes
    subset["floor"] = subset["floor"].fillna(0).astype(int)
    subset["log_radon"] = subset["log_radon"].astype(float)
    subset["log_uppm"] = np.log1p(subset["Uppm"].clip(lower=0.0).astype(float))
    uppm_mean = float(subset["log_uppm"].mean())
    uppm_std = float(subset["log_uppm"].std(ddof=0))
    subset["log_uppm_std"] = (
        subset["log_uppm"] - uppm_mean
    ) / max(uppm_std, 1e-8)
    prep_path = data_dir / "radon_top12.csv"
    subset.to_csv(prep_path, index=False)
    meta = {
        "raw_path": str(raw_path),
        "prepared_path": str(prep_path),
        "rows": int(len(subset)),
        "n_counties": int(subset["county_idx"].nunique()),
        "counties": sorted(subset["county"].unique().tolist()),
        "standardized_columns": ["log_uppm_std"],
        "available_predictors": ["floor", "log_uppm_std"],
        "model_predictors": ["floor"],
        "log_uppm_mean": uppm_mean,
        "log_uppm_std": uppm_std,
    }
    save_json(data_dir / "radon_metadata.json", meta)
    return meta


def build_radon_model(df: pd.DataFrame) -> pm.Model:
    county_idx = df["county_idx"].to_numpy()
    floor = df["floor"].to_numpy()
    y = df["log_radon"].to_numpy()
    n_counties = int(df["county_idx"].nunique())
    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=0.0, sigma=5.0)
        sigma_a = pm.HalfNormal("sigma_a", sigma=2.0)
        beta = pm.Normal("beta", mu=0.0, sigma=5.0)
        sigma_y = pm.HalfNormal("sigma_y", sigma=2.0)
        a_offset = pm.Normal("a_offset", mu=0.0, sigma=1.0, shape=n_counties)
        a = pm.Deterministic("a", mu_a + sigma_a * a_offset)
        mu = a[county_idx] + beta * floor
        pm.Normal("obs", mu=mu, sigma=sigma_y, observed=y)
    return model


def flatten_idata(idata) -> np.ndarray:
    posterior = idata.posterior
    mu_a = posterior["mu_a"].values.reshape(-1, 1)
    beta = posterior["beta"].values.reshape(-1, 1)
    sigma_a = posterior["sigma_a"].values.reshape(-1, 1)
    sigma_y = posterior["sigma_y"].values.reshape(-1, 1)
    a = posterior["a"].values.reshape(-1, posterior["a"].shape[-1])
    return np.concatenate([mu_a, beta, sigma_a, sigma_y, a], axis=1)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _write_fit_log(log_path: Path, payload: dict) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def _summary_records(idata) -> list[dict]:
    summary = az.summary(idata, kind="stats", round_to=None)
    summary = summary.reset_index().rename(columns={"index": "parameter"})
    return summary.to_dict(orient="records")


def fit_reference_posterior(
    df: pd.DataFrame, seed: int, output_dir: str | Path
) -> dict:
    output_dir = ensure_dir(output_dir)
    start = time.perf_counter()
    log_path = output_dir / "nuts_log.jsonl"
    with build_radon_model(df) as model:
        idata = pm.sample(
            draws=750,
            tune=750,
            chains=4,
            target_accept=0.9,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=True,
            return_inferencedata=True,
        )
    flat = flatten_idata(idata)
    np.save(output_dir / "reference_samples.npy", flat)
    idata.to_netcdf(output_dir / "reference_idata.nc")
    sample_stats = idata.sample_stats
    summary_records = _summary_records(idata)
    save_json(output_dir / "reference_summary.json", {"rows": summary_records})
    save_csv(output_dir / "reference_summary.csv", pd.DataFrame(summary_records))
    runtime_seconds = time.perf_counter() - start
    _write_fit_log(
        log_path,
        {
            "event": "nuts_complete",
            "seed": seed,
            "runtime_seconds": runtime_seconds,
            "draws": 750,
            "tune": 750,
            "chains": 4,
            "target_accept": 0.9,
            "n_divergences": int(np.asarray(sample_stats["diverging"]).sum()),
            "mean_step_size": float(np.asarray(sample_stats["step_size_bar"]).mean()),
            "mean_tree_depth": float(np.asarray(sample_stats["tree_depth"]).mean()),
            "mean_acceptance_rate": float(np.asarray(sample_stats["acceptance_rate"]).mean()),
        },
    )
    meta = {
        "seed": seed,
        "n_samples": int(flat.shape[0]),
        "dimension": int(flat.shape[1]),
        "draws": 750,
        "tune": 750,
        "chains": 4,
        "target_accept": 0.9,
        "runtime_seconds": runtime_seconds,
        "n_divergences": int(np.asarray(sample_stats["diverging"]).sum()),
        "max_rhat": float(max(row.get("r_hat", 1.0) for row in summary_records)),
        "min_ess_bulk": float(min(row.get("ess_bulk", float("inf")) for row in summary_records)),
        "saved_artifacts": {
            "idata_netcdf": str(output_dir / "reference_idata.nc"),
            "summary_csv": str(output_dir / "reference_summary.csv"),
            "sampler_log": str(log_path),
        },
        "model_spec": {
            "response": "log_radon",
            "predictors": ["floor"],
            "hierarchy": "county varying intercepts",
        },
    }
    save_json(output_dir / "reference_meta.json", meta)
    return {"samples": flat, "meta": meta}


def fit_advi_samples(
    df: pd.DataFrame,
    seed: int,
    method: str,
    draws: int,
    output_dir: str | Path,
) -> dict:
    output_dir = ensure_dir(output_dir)
    cache_path = output_dir / f"{method}_samples.npy"
    meta_path = output_dir / f"{method}_meta.json"
    if cache_path.exists():
        flat = np.load(cache_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
            "method": method,
            "draws": int(flat.shape[0]),
            "cache_hit": True,
        }
        return {"samples": flat, "meta": meta}
    start = time.perf_counter()
    log_path = output_dir / f"{method}_optimizer_log.jsonl"
    with build_radon_model(df) as model:
        approx = pm.fit(
            n=6000,
            method=method,
            random_seed=seed,
            progressbar=False,
        )
        idata = approx.sample(draws=draws)
    flat = flatten_idata(idata)
    np.save(cache_path, flat)
    hist = np.asarray(approx.hist, dtype=float).reshape(-1)
    hist_df = pd.DataFrame(
        {
            "iteration": np.arange(1, hist.size + 1, dtype=int),
            "objective": hist,
        }
    )
    save_csv(output_dir / f"{method}_history.csv", hist_df)
    runtime_seconds = time.perf_counter() - start
    _write_fit_log(
        log_path,
        {
            "event": "advi_complete",
            "method": method,
            "seed": seed,
            "iterations": int(hist.size),
            "runtime_seconds": runtime_seconds,
            "objective_start": float(hist[0]),
            "objective_end": float(hist[-1]),
            "objective_best": float(hist.min()),
        },
    )
    meta = {
        "method": method,
        "draws": int(flat.shape[0]),
        "optimizer": method,
        "iterations": int(hist.size),
        "runtime_seconds": runtime_seconds,
        "objective_start": float(hist[0]),
        "objective_end": float(hist[-1]),
        "objective_best": float(hist.min()),
        "cache_hit": False,
        "saved_artifacts": {
            "history_csv": str(output_dir / f"{method}_history.csv"),
            "optimizer_log": str(log_path),
        },
        "model_spec": {
            "response": "log_radon",
            "predictors": ["floor"],
            "hierarchy": "county varying intercepts",
        },
    }
    save_json(meta_path, meta)
    return {"samples": flat, "meta": meta}


def fit_laplace_samples(
    df: pd.DataFrame, seed: int, draws: int, output_dir: str | Path
) -> dict:
    output_dir = ensure_dir(output_dir)
    cache_path = output_dir / "laplace_samples.npy"
    meta_path = output_dir / "laplace_meta.json"
    if cache_path.exists():
        flat = np.load(cache_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
            "method": "laplace",
            "draws": int(flat.shape[0]),
            "cache_hit": True,
        }
        return {"samples": flat, "meta": meta}
    start = time.perf_counter()
    log_path = output_dir / "laplace_log.jsonl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with build_radon_model(df) as model:
            map_est = pm.find_MAP(progressbar=False, seed=seed, include_transformed=True)
            try:
                point = {
                    var.name: np.asarray(map_est[var.name], dtype=float)
                    for var in model.free_RVs
                }
                mapped = DictToArrayBijection.map(point)
                hess = np.asarray(
                    pm.find_hessian(
                        point=map_est, vars=model.free_RVs, model=model
                    ),
                    dtype=float,
                )
                hess = 0.5 * (hess + hess.T)
                cov = np.linalg.pinv(hess)
                cov = 0.5 * (cov + cov.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                clipped = np.clip(eigvals, 1e-8, None)
                cov = eigvecs @ np.diag(clipped) @ eigvecs.T
                raw = np.random.default_rng(seed).multivariate_normal(
                    mapped.data, cov, size=draws
                )
                samples = []
                for row in raw:
                    sample_point = DictToArrayBijection.rmap(
                        RaveledVars(np.asarray(row, dtype=float), mapped.point_map_info)
                    )
                    mu_a = float(np.asarray(sample_point["mu_a"]))
                    beta = float(np.asarray(sample_point["beta"]))
                    sigma_a = float(max(1e-6, np.asarray(sample_point["sigma_a"])))
                    sigma_y = float(max(1e-6, np.asarray(sample_point["sigma_y"])))
                    a_offset = np.asarray(sample_point["a_offset"], dtype=float)
                    a = mu_a + sigma_a * a_offset
                    samples.append(np.concatenate([[mu_a, beta, sigma_a, sigma_y], a]))
                flat = np.asarray(samples, dtype=float)
                runtime_seconds = time.perf_counter() - start
                meta = {
                    "method": "laplace",
                    "draws": int(flat.shape[0]),
                    "seed": seed,
                    "runtime_seconds": runtime_seconds,
                    "value_dimension": int(mapped.data.shape[0]),
                    "covariance_min_eigenvalue_before_clip": float(eigvals.min()),
                    "covariance_max_eigenvalue_after_clip": float(np.linalg.eigvalsh(cov).max()),
                    "map_keys": sorted(map_est.keys()),
                    "cache_hit": False,
                    "saved_artifacts": {"laplace_log": str(log_path)},
                    "model_spec": {
                        "response": "log_radon",
                        "predictors": ["floor"],
                        "hierarchy": "county varying intercepts",
                    },
                }
                save_json(meta_path, meta)
                _write_fit_log(
                    log_path,
                    {
                        "event": "laplace_complete",
                        "seed": seed,
                        "runtime_seconds": runtime_seconds,
                        "draws": int(flat.shape[0]),
                        "value_dimension": int(mapped.data.shape[0]),
                        "covariance_min_eigenvalue_before_clip": float(eigvals.min()),
                    },
                )
            except Exception as exc:
                save_json(output_dir / "laplace_skipped.json", {"reason": repr(exc)})
                _write_fit_log(
                    log_path,
                    {"event": "laplace_failed", "seed": seed, "reason": repr(exc)},
                )
                flat = None
    if flat is not None:
        np.save(cache_path, flat)
    meta = (
        json.loads(meta_path.read_text())
        if flat is not None and meta_path.exists()
        else {"method": "laplace", "draws": 0 if flat is None else int(flat.shape[0])}
    )
    return {"samples": flat, "meta": meta}
