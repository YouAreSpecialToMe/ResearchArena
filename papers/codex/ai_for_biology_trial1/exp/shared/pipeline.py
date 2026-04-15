from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import config
from .data import DatasetSplit, prepare_dataset_split
from .metrics import mean_perturbed_reference_pearson, nearest_centroid_metrics, rmse
from .models import fit_mlp, fit_pls, fit_ridge
from .utils import Timer, append_jsonl, max_rss_mb, set_global_seed


@dataclass
class RunResult:
    model_name: str
    dataset: str
    seed: int
    metrics: dict[str, float]
    predictions: np.ndarray
    true: np.ndarray
    labels: list[str]
    hyperparams: dict[str, float | int | str]
    runtime_minutes: float
    peak_memory_mb: float
    peak_gpu_memory_mb: float | None


def _compute_metrics(split: DatasetSplit, pred: np.ndarray, true: np.ndarray, labels: list[str]) -> dict[str, float]:
    pearson, _ = mean_perturbed_reference_pearson(pred, true, split.mu_pert_train)
    rmse_value, _ = rmse(pred, true)
    top1, median_rank = nearest_centroid_metrics(pred, true, labels)
    top100 = split.top100_gene_idx
    pearson_top100, _ = mean_perturbed_reference_pearson(
        pred[:, top100], true[:, top100], split.mu_pert_train[top100]
    )
    return {
        "perturbed_reference_pearson": pearson,
        "rmse": rmse_value,
        "top1_accuracy": top1,
        "median_rank": median_rank,
        "pearson_top100_hvg": pearson_top100,
    }


def _select_best(candidates):
    return max(
        candidates,
        key=lambda item: (
            item["metrics"]["perturbed_reference_pearson"],
            -item["metrics"]["rmse"],
        ),
    )


def _weighted_neighbor_summary(
    similarities: np.ndarray,
    indices: np.ndarray,
    residual_bank: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    weighted = []
    max_sim = []
    for i in range(indices.shape[0]):
        idx = indices[i, :k]
        sims = np.clip(similarities[i, :k], 0, None)
        if sims.sum() == 0:
            sims = np.ones_like(sims)
        sims = sims / sims.sum()
        weighted.append((residual_bank[idx] * sims[:, None]).sum(axis=0))
        max_sim.append(float(similarities[i, 0]))
    return np.stack(weighted).astype(np.float32), np.array(max_sim, dtype=np.float32)[:, None]


def _leave_one_out_retrieval(
    descriptor_train: np.ndarray,
    residual_train: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    sims = descriptor_train @ descriptor_train.T
    norms = np.linalg.norm(descriptor_train, axis=1, keepdims=True)
    denom = norms @ norms.T
    denom[denom == 0] = 1.0
    sims = sims / denom
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1)[:, :k]
    sorted_sims = np.take_along_axis(sims, order, axis=1)
    return _weighted_neighbor_summary(sorted_sims, order, residual_train, k)


def _log(path: Path | None, payload: dict[str, object]) -> None:
    if path is not None:
        append_jsonl(path, payload)


def _fit_mlp_with_protocol(
    split: DatasetSplit,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    latent_to_gene_fn,
    seed: int,
    log_path: Path | None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_val_t = torch.from_numpy(x_val.astype(np.float32)).to(device)

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            pred_val_latent = model(x_val_t).cpu().numpy()
        pred_val = latent_to_gene_fn(pred_val_latent)
        metrics = _compute_metrics(split, pred_val, split.val_matrix, split.val_perts)
        return metrics["perturbed_reference_pearson"], metrics["rmse"]

    def log_fn(row):
        _log(log_path, {"event": "epoch", **row})

    model, train_info = fit_mlp(
        x_train=x_train,
        y_train=y_train,
        seed=seed,
        device=device,
        eval_fn=eval_fn,
        log_fn=log_fn,
    )
    return model, train_info, device


def run_model(
    model_name: str,
    dataset: str,
    seed: int,
    log_path: Path | None = None,
    preprocess_runtime_minutes: float = 0.0,
) -> RunResult:
    set_global_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    split = prepare_dataset_split(dataset, seed)
    _log(
        log_path,
        {
            "event": "start",
            "model": model_name,
            "dataset": dataset,
            "seed": seed,
            "preprocess_runtime_minutes": preprocess_runtime_minutes,
        },
    )

    with Timer() as timer:
        if model_name == "Train Perturbed Mean":
            pred = np.repeat(split.mu_pert_train[None, :], len(split.test_perts), axis=0)
            hyper = {}
        elif model_name == "Train Perturbed Median":
            median = np.median(split.train_matrix, axis=0).astype(np.float32)
            pred = np.repeat(median[None, :], len(split.test_perts), axis=0)
            hyper = {}
        elif model_name == "Non-residualized Ridge":
            candidates = []
            for alpha in config.RIDGE_ALPHA_GRID:
                model = fit_ridge(split.descriptor_train, split.full_train_pca, alpha)
                pred_val = split.full_pca.inverse_transform(model.predict(split.descriptor_val))
                metrics = _compute_metrics(split, pred_val, split.val_matrix, split.val_perts)
                candidates.append({"metrics": metrics, "model": model, "hyperparams": {"alpha": alpha}})
                _log(log_path, {"event": "candidate", "alpha": alpha, "metrics": metrics})
            best = _select_best(candidates)
            pred = split.full_pca.inverse_transform(best["model"].predict(split.descriptor_test))
            hyper = best["hyperparams"]
        elif model_name == "Residualized Ridge":
            candidates = []
            for alpha in config.RIDGE_ALPHA_GRID:
                model = fit_ridge(split.descriptor_train, split.residual_train_pca, alpha)
                pred_val = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(
                    model.predict(split.descriptor_val)
                )
                metrics = _compute_metrics(split, pred_val, split.val_matrix, split.val_perts)
                candidates.append({"metrics": metrics, "model": model, "hyperparams": {"alpha": alpha}})
                _log(log_path, {"event": "candidate", "alpha": alpha, "metrics": metrics})
            best = _select_best(candidates)
            pred = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(
                best["model"].predict(split.descriptor_test)
            )
            hyper = best["hyperparams"]
        elif model_name == "Residualized PLS":
            candidates = []
            max_comp = max(
                1,
                min(
                    len(split.train_perts) - 1,
                    split.residual_train_pca.shape[1],
                    split.descriptor_train.shape[1],
                ),
            )
            for n_comp in [c for c in config.PLS_COMPONENTS_GRID if c <= max_comp]:
                model = fit_pls(split.descriptor_train, split.residual_train_pca, n_comp)
                pred_val = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(
                    model.predict(split.descriptor_val)
                )
                metrics = _compute_metrics(split, pred_val, split.val_matrix, split.val_perts)
                candidates.append(
                    {"metrics": metrics, "model": model, "hyperparams": {"n_components": n_comp}}
                )
                _log(log_path, {"event": "candidate", "n_components": n_comp, "metrics": metrics})
            best = _select_best(candidates)
            pred = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(
                best["model"].predict(split.descriptor_test)
            )
            hyper = best["hyperparams"]
        elif model_name == "Residualized Linear Embedding":
            descriptor_proj = split.descriptor_train
            x_val = split.descriptor_val
            x_test = split.descriptor_test
            from .models import fit_pca

            proj = fit_pca(descriptor_proj, config.LINEAR_EMBED_DIM)
            x_train = proj.transform(split.descriptor_train)
            x_val = proj.transform(split.descriptor_val)
            x_test = proj.transform(split.descriptor_test)
            candidates = []
            for alpha in config.RIDGE_ALPHA_GRID:
                model = fit_ridge(x_train, split.residual_train_pca, alpha)
                pred_val = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(
                    model.predict(x_val)
                )
                metrics = _compute_metrics(split, pred_val, split.val_matrix, split.val_perts)
                candidates.append(
                    {
                        "metrics": metrics,
                        "model": model,
                        "hyperparams": {"alpha": alpha, "descriptor_pca_dim": x_train.shape[1]},
                    }
                )
                _log(log_path, {"event": "candidate", "alpha": alpha, "metrics": metrics})
            best = _select_best(candidates)
            pred = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(
                best["model"].predict(x_test)
            )
            hyper = best["hyperparams"]
        elif model_name == "Retrieval-only Residual kNN":
            candidates = []
            for k in config.RETRIEVAL_K_GRID:
                pred_val_latent, _ = _weighted_neighbor_summary(
                    split.retrieval_cache_val["similarities"],
                    split.retrieval_cache_val["indices"],
                    split.residual_train_pca,
                    k,
                )
                pred_val = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(pred_val_latent)
                metrics = _compute_metrics(split, pred_val, split.val_matrix, split.val_perts)
                candidates.append({"metrics": metrics, "k": k, "hyperparams": {"k": k}})
                _log(log_path, {"event": "candidate", "k": k, "metrics": metrics})
            best = _select_best(candidates)
            pred_latent, _ = _weighted_neighbor_summary(
                split.retrieval_cache_test["similarities"],
                split.retrieval_cache_test["indices"],
                split.residual_train_pca,
                best["k"],
            )
            pred = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(pred_latent)
            hyper = best["hyperparams"]
        elif model_name in {"ReSRP-Linear", "ReSRP-MLP"}:
            candidates = []
            for k in config.RETRIEVAL_K_GRID:
                train_weighted, train_max = _leave_one_out_retrieval(
                    split.descriptor_train, split.residual_train_pca, k
                )
                val_weighted, val_max = _weighted_neighbor_summary(
                    split.retrieval_cache_val["similarities"],
                    split.retrieval_cache_val["indices"],
                    split.residual_train_pca,
                    k,
                )
                x_train = np.concatenate([split.descriptor_train, train_weighted, train_max], axis=1)
                x_val = np.concatenate([split.descriptor_val, val_weighted, val_max], axis=1)
                if model_name == "ReSRP-Linear":
                    for alpha in config.RETRIEVAL_RIDGE_ALPHA_GRID:
                        model = fit_ridge(x_train, split.residual_train_pca, alpha)
                        pred_val = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(
                            model.predict(x_val)
                        )
                        metrics = _compute_metrics(split, pred_val, split.val_matrix, split.val_perts)
                        candidates.append(
                            {
                                "metrics": metrics,
                                "model": model,
                                "hyperparams": {"alpha": alpha, "k": k, "input_dim": x_train.shape[1]},
                            }
                        )
                        _log(log_path, {"event": "candidate", "alpha": alpha, "k": k, "metrics": metrics})
                else:
                    model, train_info, device = _fit_mlp_with_protocol(
                        split=split,
                        x_train=x_train,
                        y_train=split.residual_train_pca,
                        x_val=x_val,
                        latent_to_gene_fn=lambda pred_latent: split.mu_pert_train[None, :]
                        + split.residual_pca.inverse_transform(pred_latent),
                        seed=seed,
                        log_path=log_path,
                    )
                    with torch.no_grad():
                        pred_val_latent = model(torch.from_numpy(x_val.astype(np.float32)).to(device)).cpu().numpy()
                    pred_val = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(pred_val_latent)
                    metrics = _compute_metrics(split, pred_val, split.val_matrix, split.val_perts)
                    candidates.append(
                        {
                            "metrics": metrics,
                            "model": model,
                            "hyperparams": {"k": k, "input_dim": x_train.shape[1], **train_info},
                        }
                    )
                    _log(log_path, {"event": "candidate", "k": k, "metrics": metrics, **train_info})
            best = _select_best(candidates)
            test_weighted, test_max = _weighted_neighbor_summary(
                split.retrieval_cache_test["similarities"],
                split.retrieval_cache_test["indices"],
                split.residual_train_pca,
                best["hyperparams"]["k"],
            )
            x_test = np.concatenate([split.descriptor_test, test_weighted, test_max], axis=1)
            if model_name == "ReSRP-Linear":
                pred_latent = best["model"].predict(x_test)
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                best["model"].eval()
                with torch.no_grad():
                    pred_latent = best["model"](torch.from_numpy(x_test.astype(np.float32)).to(device)).cpu().numpy()
            pred = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(pred_latent)
            hyper = best["hyperparams"]
        else:
            raise ValueError(model_name)

        metrics = _compute_metrics(split, pred, split.test_matrix, split.test_perts)
        runtime = preprocess_runtime_minutes + timer.minutes

    _log(
        log_path,
        {
            "event": "finish",
            "metrics": metrics,
            "hyperparams": hyper,
            "runtime_minutes": runtime,
            "fit_runtime_minutes": timer.minutes,
            "peak_memory_mb": max_rss_mb(),
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else None
            ),
        },
    )
    return RunResult(
        model_name=model_name,
        dataset=dataset,
        seed=seed,
        metrics=metrics,
        predictions=pred.astype(np.float32),
        true=split.test_matrix.astype(np.float32),
        labels=split.test_perts,
        hyperparams=hyper,
        runtime_minutes=runtime,
        peak_memory_mb=max_rss_mb(),
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else None
        ),
    )


def bootstrap_difference(
    seed_results_a: list[RunResult],
    seed_results_b: list[RunResult],
    metric_name: str,
    draws: int = config.BOOTSTRAP_DRAWS,
) -> dict[str, float]:
    refs = [prepare_dataset_split(r.dataset, r.seed).mu_pert_train for r in seed_results_a]
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(draws):
        seed_deltas = []
        for split_ref, ra, rb in zip(refs, seed_results_a, seed_results_b):
            n = len(ra.labels)
            idx = rng.integers(0, n, size=n)
            if metric_name == "perturbed_reference_pearson":
                vals_a = [
                    float(np.corrcoef(ra.predictions[i] - split_ref, ra.true[i] - split_ref)[0, 1])
                    for i in idx
                ]
                vals_b = [
                    float(np.corrcoef(rb.predictions[i] - split_ref, rb.true[i] - split_ref)[0, 1])
                    for i in idx
                ]
                delta = float(np.nanmean(vals_a) - np.nanmean(vals_b))
            elif metric_name == "rmse":
                delta = float(
                    np.sqrt(((ra.predictions[idx] - ra.true[idx]) ** 2).mean())
                    - np.sqrt(((rb.predictions[idx] - rb.true[idx]) ** 2).mean())
                )
            else:
                raise ValueError(metric_name)
            seed_deltas.append(delta)
        deltas.append(float(np.mean(seed_deltas)))
    arr = np.array(deltas, dtype=np.float64)
    return {
        "mean_difference": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }
