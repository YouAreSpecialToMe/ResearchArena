from __future__ import annotations

import json
import math
import os
import resource
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit
from scipy.stats import chi2, kendalltau, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


SEEDS = [11, 23, 37]
ALPHAS = [0.05, 0.10]
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def set_thread_env() -> None:
    for key, value in THREAD_ENV.items():
        os.environ[key] = value


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))


def save_csv(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_array(path: str | Path, values: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, values)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def append_jsonl(path: str | Path, rows: list[dict] | dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = rows if isinstance(rows, list) else [rows]
    with path.open("a", encoding="utf-8") as handle:
        for row in payload:
            handle.write(json.dumps(row) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def log_message(path: str | Path, message: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now_iso()}] {message}\n")


def init_experiment(experiment: str) -> dict[str, Path]:
    exp_dir = ensure_dir(Path("exp") / experiment)
    results_dir = ensure_dir(Path("results") / experiment)
    logs_dir = ensure_dir(exp_dir / "logs")
    runtime_dir = ensure_dir("results/runtime")
    return {
        "exp_dir": exp_dir,
        "results_dir": results_dir,
        "logs_dir": logs_dir,
        "runtime_dir": runtime_dir,
    }


def summarize_mean_std_ci(
    df: pd.DataFrame, value: str, group_cols: list[str]
) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[value]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    se = out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    out[f"{value}_mean"] = out["mean"]
    out[f"{value}_std"] = out["std"].fillna(0.0)
    out[f"{value}_ci95"] = 1.96 * se
    return out.drop(columns=["mean", "std", "count"])


def pair_indices(d: int) -> list[tuple[int, int]]:
    return [(j, k) for j in range(d) for k in range(j + 1, d)]


def randomized_ranks(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    jitter = rng.uniform(0.0, 1e-9, size=values.shape)
    order = np.argsort(values + jitter, kind="mergesort")
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = np.arange(values.size)
    return ranks


def pooled_transform(
    x: np.ndarray, rng: np.random.Generator, quantize: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    values = np.array(x, dtype=float, copy=True)
    if quantize is not None:
        values = np.round(values / quantize) * quantize
    m_plus_1, d = values.shape
    u = np.empty_like(values, dtype=float)
    for j in range(d):
        ranks = randomized_ranks(values[:, j], rng)
        u[:, j] = (ranks + 0.5) / m_plus_1
    u = np.clip(u, 1e-6, 1 - 1e-6)
    z = norm.ppf(u)
    return u, z


def build_family_features(
    u: np.ndarray,
    z: np.ndarray,
    family: str,
    target_pair: tuple[int, int] | None = None,
    alpha: float = 0.9,
) -> np.ndarray:
    pairs = [target_pair] if target_pair is not None else pair_indices(u.shape[1])
    feats: list[np.ndarray] = []
    if family == "cov":
        for j, k in pairs:
            feats.append((z[:, j] * z[:, k])[:, None])
    elif family == "tail":
        upper = alpha
        lower = 1.0 - alpha
        for j, k in pairs:
            feats.append(((u[:, j] > upper) & (u[:, k] > upper)).astype(float)[:, None])
            feats.append(((u[:, j] < lower) & (u[:, k] < lower)).astype(float)[:, None])
    elif family == "coord":
        feats = [z]
    elif family == "radius":
        feats = [np.sum(z**2, axis=1, keepdims=True)]
    else:
        raise ValueError(f"Unknown family: {family}")
    return np.concatenate(feats, axis=1)


def candidate_scores_from_features(
    feats: np.ndarray, method: str, rng: np.random.Generator, score_kind: str
) -> np.ndarray:
    if method in {"cosbc", "energy_distance"}:
        dist = cdist(feats, feats, metric="euclidean")
        if score_kind == "energy":
            scores = dist.sum(axis=1) / max(1, feats.shape[0] - 1)
        elif score_kind == "kernel":
            med = np.median(dist[np.triu_indices_from(dist, k=1)])
            bw = max(med, 1e-6)
            sim = np.exp(-(dist**2) / (2.0 * bw**2))
            scores = -sim.sum(axis=1) / max(1, feats.shape[0] - 1)
        else:
            raise ValueError(f"Unknown score_kind: {score_kind}")
        return scores
    if method in {"enriched", "scalar"}:
        feature_ranks = np.empty_like(feats, dtype=float)
        for k in range(feats.shape[1]):
            ranks = randomized_ranks(feats[:, k], rng)
            feature_ranks[:, k] = (ranks + 0.5) / feats.shape[0]
        return np.mean((feature_ranks - 0.5) ** 2, axis=1)
    raise ValueError(f"Unknown method: {method}")


def family_rank_matrix(
    x: np.ndarray,
    family: str,
    method: str,
    seed: int | None = None,
    score_kind: str = "energy",
    quantize: float | None = None,
    target_pair: tuple[int, int] | None = None,
    transform_seeds: np.ndarray | list[int] | None = None,
    score_seeds: np.ndarray | list[int] | None = None,
) -> dict:
    r, m_plus_1, _ = x.shape
    rank_matrix = np.empty((r, m_plus_1), dtype=int)
    candidate_scores = np.empty((r, m_plus_1), dtype=float)
    if transform_seeds is None:
        assert seed is not None
        base_rng = np.random.default_rng(seed)
        transform_seeds = base_rng.integers(0, 2**32 - 1, size=r, dtype=np.uint32)
    if score_seeds is None:
        assert seed is not None
        base_rng = np.random.default_rng(seed + 1)
        score_seeds = base_rng.integers(0, 2**32 - 1, size=r, dtype=np.uint32)
    for i in range(r):
        transform_rng = np.random.default_rng(int(transform_seeds[i]))
        score_rng = np.random.default_rng(int(score_seeds[i]))
        rank_rng = np.random.default_rng(int(score_seeds[i]) + 1)
        u, z = pooled_transform(x[i], transform_rng, quantize=quantize)
        feats = build_family_features(u, z, family=family, target_pair=target_pair)
        scores = candidate_scores_from_features(
            feats=feats, method=method, rng=score_rng, score_kind=score_kind
        )
        candidate_scores[i] = scores
        rank_matrix[i] = randomized_ranks(scores, rank_rng)
    pits = (rank_matrix + 0.5) / m_plus_1
    return {"ranks": rank_matrix, "pits": pits, "scores": candidate_scores}


def statistic_from_pits(pits: np.ndarray) -> float:
    return float(np.sum((pits - 0.5) ** 2))


def monte_carlo_family_pvalue(
    pits: np.ndarray,
    rng: np.random.Generator | None = None,
    draws: int = 199,
    relabel_indices: np.ndarray | None = None,
) -> tuple[float, float]:
    observed = statistic_from_pits(pits[:, 0])
    exceed = 0
    if relabel_indices is None:
        assert rng is not None
        relabel_indices = rng.integers(0, pits.shape[1], size=(draws, pits.shape[0]))
    for idx in relabel_indices:
        stat = statistic_from_pits(pits[np.arange(pits.shape[0]), idx])
        exceed += int(stat >= observed)
    return observed, (1 + exceed) / (len(relabel_indices) + 1)


def monte_carlo_global_pvalue(
    family_pits: dict[str, np.ndarray],
    rng: np.random.Generator | None = None,
    draws: int = 199,
    relabel_indices: np.ndarray | None = None,
) -> tuple[float, float]:
    observed = {
        family: statistic_from_pits(pits[:, 0]) for family, pits in family_pits.items()
    }
    observed_tmax = max(observed.values())
    exceed = 0
    keys = list(family_pits)
    if relabel_indices is None:
        assert rng is not None
        relabel_indices = rng.integers(
            0,
            next(iter(family_pits.values())).shape[1],
            size=(draws, next(iter(family_pits.values())).shape[0]),
        )
    for idx in relabel_indices:
        tmax = max(
            statistic_from_pits(family_pits[key][np.arange(family_pits[key].shape[0]), idx])
            for key in keys
        )
        exceed += int(tmax >= observed_tmax)
    return observed_tmax, (1 + exceed) / (len(relabel_indices) + 1)


def bonferroni_global_pvalue(family_pvalues: dict[str, float]) -> tuple[float, float]:
    observed = float(min(family_pvalues.values()))
    return observed, float(min(1.0, len(family_pvalues) * observed))


def fisher_global_pvalue(family_pvalues: dict[str, float]) -> tuple[float, float]:
    clipped = np.clip(np.asarray(list(family_pvalues.values()), dtype=float), 1e-12, 1.0)
    observed = float(-2.0 * np.log(clipped).sum())
    pvalue = float(chi2.sf(observed, df=2 * len(clipped)))
    return observed, pvalue


def exhaustive_global_pvalue(family_pits: dict[str, np.ndarray]) -> tuple[float, float]:
    keys = list(family_pits)
    r, m_plus_1 = next(iter(family_pits.values())).shape
    observed = max(statistic_from_pits(family_pits[key][:, 0]) for key in keys)
    exceed = 0
    total = m_plus_1**r
    for idx in product(range(m_plus_1), repeat=r):
        idx_arr = np.array(idx)
        tmax = max(
            statistic_from_pits(family_pits[key][np.arange(r), idx_arr]) for key in keys
        )
        exceed += int(tmax >= observed)
    return observed, exceed / total


def summarize_rejection(pvals: np.ndarray, alphas: list[float] | None = None) -> dict:
    alphas = alphas or ALPHAS
    out = {}
    for alpha in alphas:
        out[f"reject@{alpha:.2f}"] = float(np.mean(pvals <= alpha))
    return out


def toeplitz_cov(d: int, scale: float = 2.0, rho: float = 0.7) -> np.ndarray:
    idx = np.arange(d)
    return scale * rho ** np.abs(idx[:, None] - idx[None, :])


def block_pair_cov(d: int, var: float = 2.0, rho: float = 0.8) -> np.ndarray:
    cov = np.eye(d) * var
    for j in range(0, d, 2):
        if j + 1 < d:
            cov[j, j + 1] = cov[j + 1, j] = rho * var
    return cov


def exact_posterior(prior_cov: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv_prior = np.linalg.inv(prior_cov)
    post_cov = np.linalg.inv(inv_prior + np.eye(prior_cov.shape[0]))
    post_mean = post_cov @ y
    return post_mean, post_cov


def shrink_correlation(cov: np.ndarray, lam: float) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    shrunk = np.eye(cov.shape[0]) + lam * (corr - np.eye(cov.shape[0]))
    return shrunk * np.outer(sd, sd)


def tailmix_weights(scale_small: float = 0.6, scale_large: float = 1.6) -> tuple[float, float, float]:
    large2 = scale_large**2
    small2 = scale_small**2
    weight_small = (large2 - 1.0) / (large2 - small2)
    return weight_small, scale_small, scale_large


def sample_tailmix(
    mean: np.ndarray, cov: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    weight_small, scale_small, scale_large = tailmix_weights()
    base = rng.multivariate_normal(np.zeros(mean.size), cov, size=size)
    choose_small = rng.uniform(size=size) < weight_small
    scales = np.where(choose_small, scale_small, scale_large)
    return mean + base * scales[:, None]


def corrupted_pair_cov(
    cov: np.ndarray, pair: tuple[int, int], mode: str
) -> np.ndarray:
    out = cov.copy()
    j, k = pair
    if mode == "ZeroPair":
        out[j, k] = out[k, j] = 0.0
    elif mode == "FlipPair":
        out[j, k] = out[k, j] = -out[j, k]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return out


def sample_from_approx(
    mean: np.ndarray,
    cov: np.ndarray,
    approx: str,
    rng: np.random.Generator,
    m: int,
    pair: tuple[int, int] | None = None,
) -> np.ndarray:
    if approx == "exact":
        return rng.multivariate_normal(mean, cov, size=m)
    if approx == "Diag":
        return rng.multivariate_normal(mean, np.diag(np.diag(cov)), size=m)
    if approx.startswith("Shrink("):
        lam = float(approx.split("(")[1].rstrip(")"))
        return rng.multivariate_normal(mean, shrink_correlation(cov, lam), size=m)
    if approx == "TailMix":
        return sample_tailmix(mean, cov, size=m, rng=rng)
    if approx == "ZeroPair":
        assert pair is not None
        return rng.multivariate_normal(mean, corrupted_pair_cov(cov, pair, approx), size=m)
    if approx == "FlipPair":
        assert pair is not None
        return rng.multivariate_normal(mean, corrupted_pair_cov(cov, pair, approx), size=m)
    raise ValueError(f"Unknown approx: {approx}")


def generate_gaussian_replicates(
    seed: int,
    d: int,
    r: int,
    m: int,
    prior_cov: np.ndarray,
    approx: str,
    quantize: float | None = None,
    pair: tuple[int, int] | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    x = np.empty((r, m + 1, d), dtype=float)
    theta = np.empty((r, d), dtype=float)
    y_obs = np.empty((r, d), dtype=float)
    meta = []
    for i in range(r):
        theta_star = rng.multivariate_normal(np.zeros(d), prior_cov)
        y = rng.multivariate_normal(theta_star, np.eye(d))
        post_mean, post_cov = exact_posterior(prior_cov, y)
        draws = sample_from_approx(post_mean, post_cov, approx=approx, rng=rng, m=m, pair=pair)
        theta[i] = theta_star
        y_obs[i] = y
        x[i, 0] = theta_star
        x[i, 1:] = draws
        meta.append(
            {
                "replicate": i,
                "post_mean_norm": float(np.linalg.norm(post_mean)),
                "post_trace": float(np.trace(post_cov)),
                "quantized": quantize is not None,
            }
        )
    return {"x": x, "theta": theta, "y": y_obs, "metadata": meta}


def paired_ci(values: pd.Series | np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    std = float(arr.std(ddof=0))
    se = std / math.sqrt(max(1, arr.size))
    return {"mean": float(arr.mean()), "std": std, "ci95": float(1.96 * se)}


def summarize_paired_difference(
    df: pd.DataFrame,
    index_cols: list[str],
    method_col: str,
    value_col: str,
    method_a: str,
    method_b: str,
    diff_name: str,
) -> pd.DataFrame:
    pivot = (
        df[df[method_col].isin([method_a, method_b])]
        .pivot_table(index=index_cols, columns=method_col, values=value_col)
        .reset_index()
    )
    if method_a not in pivot.columns or method_b not in pivot.columns:
        return pd.DataFrame(columns=index_cols + [diff_name, f"{diff_name}_mean", f"{diff_name}_std", f"{diff_name}_ci95"])
    pivot[diff_name] = pivot[method_a] - pivot[method_b]
    summary = (
        pivot.groupby([col for col in index_cols if col != "seed"], dropna=False)[diff_name]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    se = summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
    summary[f"{diff_name}_mean"] = summary["mean"]
    summary[f"{diff_name}_std"] = summary["std"].fillna(0.0)
    summary[f"{diff_name}_ci95"] = 1.96 * se
    return summary.drop(columns=["mean", "std", "count"])


def select_relabel_indices(randomization_spec: dict | None, b: int) -> np.ndarray | None:
    if randomization_spec is None:
        return None
    key = f"relabel_indices_{b}"
    relabel = randomization_spec.get(key)
    if relabel is None:
        return None
    return np.asarray(relabel, dtype=int)


def evaluate_family_bundle(
    x: np.ndarray,
    method: str,
    families: list[str],
    b: int,
    seed: int | None = None,
    score_kind: str = "energy",
    quantize: float | None = None,
    target_pair: tuple[int, int] | None = None,
    randomization_spec: dict | None = None,
) -> dict:
    start = time.perf_counter()
    family_results = {}
    family_pits = {}
    rng = np.random.default_rng(0 if seed is None else seed + 1000)
    relabel_indices = select_relabel_indices(randomization_spec, b=b)
    transform_seeds = None
    score_seeds = {}
    if randomization_spec is not None:
        transform_seeds = np.asarray(randomization_spec["transform_seeds"], dtype=np.uint32)
        score_seeds = {
            name: np.asarray(values, dtype=np.uint32)
            for name, values in randomization_spec["score_seeds"].items()
        }
    for offset, family in enumerate(families):
        bundle = family_rank_matrix(
            x=x,
            family=family,
            method=method,
            seed=None if seed is None else seed + offset,
            score_kind=score_kind,
            quantize=quantize,
            target_pair=target_pair,
            transform_seeds=transform_seeds,
            score_seeds=score_seeds.get(family),
        )
        stat, pval = monte_carlo_family_pvalue(
            bundle["pits"], rng=rng, draws=b, relabel_indices=relabel_indices
        )
        family_results[family] = {
            "statistic": stat,
            "pvalue": pval,
            "mean_pit": float(np.mean(bundle["pits"][:, 0])),
            "ranks": bundle["ranks"][:, 0].tolist(),
            "pits": bundle["pits"][:, 0].tolist(),
        }
        family_pits[family] = bundle["pits"]
    global_stat, global_p = monte_carlo_global_pvalue(
        family_pits, rng=rng, draws=b, relabel_indices=relabel_indices
    )
    bonf_stat, bonf_p = bonferroni_global_pvalue(
        {family: result["pvalue"] for family, result in family_results.items()}
    )
    fisher_stat, fisher_p = fisher_global_pvalue(
        {family: result["pvalue"] for family, result in family_results.items()}
    )
    runtime = (time.perf_counter() - start) / 60.0
    return {
        "method": method,
        "families": family_results,
        "global": {
            "statistic": global_stat,
            "pvalue": global_p,
            **summarize_rejection(np.array([global_p])),
        },
        "aggregations": {
            "tmax": {"statistic": global_stat, "pvalue": global_p},
            "bonferroni": {"statistic": bonf_stat, "pvalue": bonf_p},
            "fisher": {"statistic": fisher_stat, "pvalue": fisher_p},
        },
        "family_pits": {family: pits.tolist() for family, pits in family_pits.items()},
        "runtime_minutes": runtime,
        "peak_memory_mb": peak_memory_mb(),
    }


def evaluate_contextual_baseline(
    x: np.ndarray,
    seed: int,
    method: str,
    b: int,
    randomization_spec: dict | None = None,
    selection_log_path: str | Path | None = None,
    selection_context: dict | None = None,
) -> dict:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    family_results = {}
    selection_records = []
    transform_seeds = None
    if randomization_spec is not None:
        transform_seeds = np.asarray(randomization_spec["transform_seeds"], dtype=np.uint32)
        negative_indices = np.asarray(randomization_spec["context_negative_indices"], dtype=int)
        label_permutations = np.asarray(
            randomization_spec[f"context_label_permutations_{b}"], dtype=int
        )
    else:
        negative_indices = None
        label_permutations = None
    observed_stats = {}
    perm_stats_by_family: dict[str, list[float]] = {}
    for family in ["cov", "tail"]:
        feats_pos = []
        feats_neg = []
        for i in range(x.shape[0]):
            transform_rng = (
                rng if transform_seeds is None else np.random.default_rng(int(transform_seeds[i]))
            )
            u, z = pooled_transform(x[i], transform_rng)
            feats = build_family_features(u, z, family=family)
            feats_pos.append(feats[0])
            if negative_indices is None:
                neg_idx = int(rng.integers(1, feats.shape[0]))
            else:
                neg_idx = int(negative_indices[i])
            feats_neg.append(feats[neg_idx])
        xp = np.asarray(feats_pos)
        xn = np.asarray(feats_neg)
        n = min(len(xp), len(xn))
        xp = xp[:n]
        xn = xn[:n]
        joined = np.concatenate([xp, xn], axis=0)
        labels = np.concatenate([np.ones(n), np.zeros(n)])
        if method == "discriminative":
            x_train, x_val, y_train, y_val = train_test_split(
                joined, labels, test_size=0.2, random_state=seed, stratify=labels
            )
            best_auc = -np.inf
            best_model = None
            for c_val in [0.1, 1.0, 10.0]:
                model = LogisticRegression(
                    solver="lbfgs", max_iter=2000, C=c_val
                )
                model.fit(x_train, y_train)
                auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
                selection_records.append(
                    {
                        **(selection_context or {}),
                        "family": family,
                        "candidate_c": float(c_val),
                        "validation_auc": float(auc),
                    }
                )
                if auc > best_auc:
                    best_auc = auc
                    best_model = model
            observed = best_auc
            perm_stats = []
            perms = (
                label_permutations
                if label_permutations is not None
                else np.asarray([rng.permutation(joined.shape[0]) for _ in range(b)], dtype=int)
            )
            for perm in perms:
                perm_labels = labels[perm]
                y_train_perm = perm_labels[: y_train.shape[0]]
                y_val_perm = perm_labels[y_train.shape[0] :]
                model = LogisticRegression(
                    solver="lbfgs", max_iter=2000, C=best_model.C
                )
                model.fit(x_train, y_train_perm)
                auc = roc_auc_score(y_val_perm, model.predict_proba(x_val)[:, 1])
                perm_stats.append(float(auc))
        elif method == "energy_distance":
            def energy_stat(a: np.ndarray, c: np.ndarray) -> float:
                aa = cdist(a, a).mean()
                cc = cdist(c, c).mean()
                ac = cdist(a, c).mean()
                return float(2 * ac - aa - cc)

            observed = energy_stat(xp, xn)
            perm_stats = []
            perms = (
                label_permutations
                if label_permutations is not None
                else np.asarray([rng.permutation(joined.shape[0]) for _ in range(b)], dtype=int)
            )
            for perm in perms:
                ap = joined[perm[:n]]
                cn = joined[perm[n : 2 * n]]
                stat = energy_stat(ap, cn)
                perm_stats.append(float(stat))
        else:
            raise ValueError(method)
        exceed = sum(int(stat >= observed) for stat in perm_stats)
        pval = (1 + exceed) / (len(perm_stats) + 1)
        family_results[family] = {"statistic": observed, "pvalue": pval}
        observed_stats[family] = float(observed)
        perm_stats_by_family[family] = perm_stats
    if selection_log_path is not None and selection_records:
        append_jsonl(selection_log_path, selection_records)
    observed_tmax = max(observed_stats.values())
    exceed = 0
    for perm_idx in range(len(next(iter(perm_stats_by_family.values())))):
        tmax = max(perm_stats_by_family[family][perm_idx] for family in ["cov", "tail"])
        exceed += int(tmax >= observed_tmax)
    global_p = (1 + exceed) / (len(next(iter(perm_stats_by_family.values()))) + 1)
    return {
        "method": method,
        "families": family_results,
        "global": {
            "statistic": observed_tmax,
            "pvalue": global_p,
            **summarize_rejection(np.array([global_p])),
        },
        "runtime_minutes": (time.perf_counter() - start) / 60.0,
        "peak_memory_mb": peak_memory_mb(),
    }


def summarize_seed_table(rows: list[dict], group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped[metric_cols].agg(["mean", "std"]).reset_index()
    out.columns = [
        "__".join(col).strip("_") if isinstance(col, tuple) else col for col in out.columns
    ]
    return out


def ks_uniform_distance(ranks: np.ndarray, m: int) -> float:
    hist = np.bincount(ranks, minlength=m + 1) / len(ranks)
    target = np.ones(m + 1) / (m + 1)
    return float(np.abs(np.cumsum(hist) - np.cumsum(target)).max())


def boolean_mean(values: list[bool]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def kendall_safe(x: list[float], y: list[float]) -> float:
    tau = kendalltau(x, y, nan_policy="omit").correlation
    return float(0.0 if np.isnan(tau) else tau)


@dataclass
class ExperimentConfig:
    seed: int
    r: int
    m: int
    b: int
