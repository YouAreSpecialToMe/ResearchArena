from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .models import PCAWrapper, fit_pca, fit_truncated_svd
from .utils import ensure_dir, save_json


@dataclass
class DatasetSplit:
    dataset: str
    seed: int
    genes: list[str]
    target_genes: list[str]
    train_perts: list[str]
    val_perts: list[str]
    test_perts: list[str]
    train_matrix: np.ndarray
    val_matrix: np.ndarray
    test_matrix: np.ndarray
    descriptor_train: np.ndarray
    descriptor_val: np.ndarray
    descriptor_test: np.ndarray
    descriptor_degree_train: np.ndarray
    descriptor_degree_val: np.ndarray
    descriptor_degree_test: np.ndarray
    residual_train: np.ndarray
    residual_val: np.ndarray
    residual_test: np.ndarray
    residual_train_pca: np.ndarray
    residual_val_pca: np.ndarray
    residual_test_pca: np.ndarray
    full_train_pca: np.ndarray
    full_val_pca: np.ndarray
    full_test_pca: np.ndarray
    mu_pert_train: np.ndarray
    x_ctrl_train: np.ndarray
    residual_pca: PCAWrapper
    full_pca: PCAWrapper
    retrieval_cache_val: dict[str, np.ndarray]
    retrieval_cache_test: dict[str, np.ndarray]
    top100_gene_idx: np.ndarray
    audit: dict[str, object]


def _load_expression_matrix(adata: ad.AnnData) -> sparse.csr_matrix | np.ndarray:
    if "counts" in adata.layers:
        x = adata.layers["counts"]
        if not sparse.issparse(x):
            x = sparse.csr_matrix(np.asarray(x, dtype=np.float32))
        else:
            x = x.tocsr().astype(np.float32)
        libsize = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
        libsize[libsize == 0] = 1.0
        scaled = x.multiply((1e4 / libsize)[:, None]).tocsr()
        scaled.data = np.log1p(scaled.data)
        return scaled
    if "logNor" in adata.layers:
        x = adata.layers["logNor"]
    else:
        x = adata.X
    if sparse.issparse(x):
        return x.tocsr().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _mean_of_cells(x_all: sparse.csr_matrix | np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = x_all[mask]
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(x.mean(axis=0), dtype=np.float32).ravel()


def _variance_of_cells(x_all: sparse.csr_matrix | np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = x_all[mask]
    if sparse.issparse(x):
        sq = x.copy()
        sq.data **= 2
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(sq.mean(axis=0)).ravel()
        return (mean_sq - mean**2).astype(np.float32)
    arr = np.asarray(x, dtype=np.float32)
    return arr.var(axis=0).astype(np.float32)


def _released_single_simulation_split(perturbations: list[str], frac: float, seed: int) -> tuple[list[str], list[str]]:
    unique_genes = np.unique(np.asarray(perturbations, dtype=object))
    np.random.seed(seed=seed)
    n_train = max(1, int(len(unique_genes) * frac))
    train_genes = np.random.choice(unique_genes, n_train, replace=False)
    train = sorted([pert for pert in perturbations if pert in set(train_genes.tolist())])
    test = sorted([pert for pert in perturbations if pert not in set(train_genes.tolist())])
    return train, test


def _split_perturbations_released(perturbations: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    train_total, test = _released_single_simulation_split(perturbations, config.TRAIN_TEST_FRAC, seed)
    train, val = _released_single_simulation_split(train_total, 0.9, seed)
    if not val:
        val = train[-1:]
        train = train[:-1] or val
    if not test:
        test = val[-1:]
        val = val[:-1] or test
    return sorted(train), sorted(val), sorted(test)


def _select_genes(
    x_all: sparse.csr_matrix | np.ndarray,
    var_names: np.ndarray,
    train_mask: np.ndarray,
    target_genes: list[str],
) -> tuple[list[str], np.ndarray]:
    gene_to_idx = {gene: idx for idx, gene in enumerate(var_names.tolist())}
    variances = _variance_of_cells(x_all, train_mask)
    ranked = np.argsort(-variances)
    forced = [gene for gene in target_genes if gene in gene_to_idx]

    selected: list[int] = []
    forced_set = set()
    for gene in forced:
        idx = gene_to_idx[gene]
        if idx not in forced_set:
            forced_set.add(idx)
            selected.append(idx)

    hvg_selected = 0
    for idx in ranked:
        idx_i = int(idx)
        if idx_i in forced_set:
            continue
        if hvg_selected >= config.MAX_HVGS:
            break
        selected.append(idx_i)
        hvg_selected += 1

    if len(selected) > config.FORCED_GENE_CAP:
        kept_forced = [idx for idx in selected if idx in forced_set]
        kept_nonforced = [idx for idx in selected if idx not in forced_set][: config.FORCED_GENE_CAP - len(kept_forced)]
        selected = kept_forced + kept_nonforced

    selected = sorted(selected)
    genes = var_names[selected].tolist()
    return genes, np.array(selected, dtype=np.int64)


def _compute_centroids(
    x_all: sparse.csr_matrix | np.ndarray,
    pert_series: np.ndarray,
    genes_idx: np.ndarray,
    perturbations: list[str],
) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for pert in perturbations:
        mask = pert_series == pert
        x = x_all[mask][:, genes_idx]
        if sparse.issparse(x):
            centroids[pert] = np.asarray(x.mean(axis=0)).ravel().astype(np.float32)
        else:
            centroids[pert] = np.asarray(x.mean(axis=0), dtype=np.float32).ravel()
    return centroids


def _load_string_gene_embeddings(
    gene_universe: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, set[str]]]:
    info_path = config.DATA_EXTERNAL / "9606.protein.info.v12.0.txt.gz"
    links_path = config.DATA_EXTERNAL / "9606.protein.links.v12.0.txt.gz"
    info = pd.read_csv(info_path, sep="\t", compression="gzip")
    info = info[["#string_protein_id", "preferred_name"]].drop_duplicates()
    info.columns = ["protein", "gene"]
    info = info[info["gene"].isin(gene_universe)]
    protein_to_gene = dict(zip(info["protein"], info["gene"]))
    genes = sorted(info["gene"].unique().tolist())
    gene_index = {g: i for i, g in enumerate(genes)}
    adjacency = np.zeros((len(genes), len(genes)), dtype=np.float32)
    connected: dict[str, set[str]] = {gene: set() for gene in genes}

    with gzip.open(links_path, "rt", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            p1, p2, score = line.rstrip().split()
            score_i = int(score)
            if score_i < config.STRING_SCORE_THRESHOLD:
                continue
            g1 = protein_to_gene.get(p1)
            g2 = protein_to_gene.get(p2)
            if g1 is None or g2 is None or g1 == g2:
                continue
            i = gene_index[g1]
            j = gene_index[g2]
            adjacency[i, j] = max(adjacency[i, j], score_i)
            adjacency[j, i] = max(adjacency[j, i], score_i)
            connected[g1].add(g2)
            connected[g2].add(g1)

    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adjacency = adjacency / row_sums
    embeddings: dict[str, np.ndarray] = {
        gene: np.zeros(config.STRING_EMBED_DIM, dtype=np.float32) for gene in gene_universe
    }
    degrees: dict[str, float] = {gene: 0.0 for gene in gene_universe}
    if adjacency.shape[0] >= 3:
        n_comp = min(config.STRING_EMBED_DIM, max(2, adjacency.shape[0] - 1), adjacency.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        emb = svd.fit_transform(adjacency)
        for gene, idx in gene_index.items():
            vec = np.zeros(config.STRING_EMBED_DIM, dtype=np.float32)
            vec[:n_comp] = emb[idx].astype(np.float32)
            embeddings[gene] = vec
            degrees[gene] = float(len(connected[gene]))
    return embeddings, degrees, connected


def _make_descriptor_matrix(
    targets: list[str],
    embeddings: dict[str, np.ndarray],
    degrees: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(
        [embeddings.get(t, np.zeros(config.STRING_EMBED_DIM, dtype=np.float32)) for t in targets]
    )
    deg = np.array([degrees.get(t, 0.0) for t in targets], dtype=np.float32)[:, None]
    deg_rep = np.repeat(deg, config.STRING_EMBED_DIM, axis=1)
    return x.astype(np.float32), deg_rep.astype(np.float32)


def _neighbor_cache(
    train_desc: np.ndarray,
    query_desc: np.ndarray,
    train_resid_pca: np.ndarray,
) -> dict[str, np.ndarray]:
    sims = cosine_similarity(query_desc, train_desc)
    order = np.argsort(-sims, axis=1)[:, : max(config.RETRIEVAL_K_GRID)]
    sorted_sims = np.take_along_axis(sims, order, axis=1)
    weighted = np.zeros((query_desc.shape[0], config.RESIDUAL_PCA_DIM), dtype=np.float32)
    for i in range(query_desc.shape[0]):
        weights = np.clip(sorted_sims[i], 0, None)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        weighted[i] = (train_resid_pca[order[i]] * weights[:, None]).sum(axis=0)
    max_sim = sorted_sims[:, :1].astype(np.float32)
    return {
        "indices": order.astype(np.int64),
        "similarities": sorted_sims.astype(np.float32),
        "weighted_residual": weighted,
        "max_similarity": max_sim,
    }


def _audit_dir(dataset: str, seed: int):
    cache_dir = ensure_dir(config.DATA_PROCESSED / dataset / f"seed_{seed}")
    audit_dir = ensure_dir(cache_dir / "audit")
    return cache_dir, audit_dir


def prepare_dataset_split(dataset: str, seed: int) -> DatasetSplit:
    cache_dir, audit_dir = _audit_dir(dataset, seed)
    cache_path = cache_dir / f"prepared_{config.PREP_CACHE_VERSION}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    adata = ad.read_h5ad(config.DATASETS[dataset])
    x_all = _load_expression_matrix(adata)
    obs = adata.obs.copy()
    obs["perturbation"] = obs["perturbation"].astype(str)
    obs["gene"] = obs["gene"].astype(str)

    control_mask = obs["perturbation"].str.lower().isin(config.CONTROL_LABELS).to_numpy()
    all_perts = sorted(
        [pert for pert in obs.loc[~control_mask, "perturbation"].astype(str).unique().tolist() if "+" not in pert]
    )
    pert_counts = obs.loc[~control_mask, "perturbation"].value_counts()

    dropped: list[dict[str, object]] = []
    kept_perts: list[str] = []
    for pert in all_perts:
        count = int(pert_counts.get(pert, 0))
        if count < config.MIN_PERT_CELLS:
            dropped.append({"perturbation": pert, "reason": "too_few_cells", "n_cells": count})
            continue
        kept_perts.append(pert)

    if int(control_mask.sum()) < config.MIN_TRAIN_CONTROL_CELLS:
        raise RuntimeError(f"{dataset} has too few control cells")

    train_perts, val_perts, test_perts = _split_perturbations_released(kept_perts, seed)
    train_mask = obs["perturbation"].isin(train_perts).to_numpy()
    train_with_ctrl_mask = train_mask | control_mask

    target_genes = sorted({pert for pert in kept_perts if pert.lower() not in config.CONTROL_LABELS})
    var_names = np.asarray(adata.var_names.astype(str))
    genes, gene_idx = _select_genes(x_all, var_names, train_with_ctrl_mask, target_genes)
    pert_series = obs["perturbation"].to_numpy()
    centroids = _compute_centroids(x_all, pert_series, gene_idx, train_perts + val_perts + test_perts)
    x_ctrl_train = _mean_of_cells(x_all[:, gene_idx], control_mask)
    mu_pert_train = np.stack([centroids[p] for p in train_perts]).mean(axis=0).astype(np.float32)

    train_matrix = np.stack([centroids[p] for p in train_perts]).astype(np.float32)
    val_matrix = np.stack([centroids[p] for p in val_perts]).astype(np.float32)
    test_matrix = np.stack([centroids[p] for p in test_perts]).astype(np.float32)

    residual_train = train_matrix - mu_pert_train[None, :]
    residual_val = val_matrix - mu_pert_train[None, :]
    residual_test = test_matrix - mu_pert_train[None, :]

    residual_pca = fit_pca(residual_train, config.RESIDUAL_PCA_DIM)
    full_pca = fit_truncated_svd(train_matrix, config.FULL_PCA_DIM)
    residual_train_pca = residual_pca.transform(residual_train)
    residual_val_pca = residual_pca.transform(residual_val)
    residual_test_pca = residual_pca.transform(residual_test)
    full_train_pca = full_pca.transform(train_matrix)
    full_val_pca = full_pca.transform(val_matrix)
    full_test_pca = full_pca.transform(test_matrix)

    string_embeddings, degrees, connected = _load_string_gene_embeddings(genes)
    descriptor_train, descriptor_degree_train = _make_descriptor_matrix(train_perts, string_embeddings, degrees)
    descriptor_val, descriptor_degree_val = _make_descriptor_matrix(val_perts, string_embeddings, degrees)
    descriptor_test, descriptor_degree_test = _make_descriptor_matrix(test_perts, string_embeddings, degrees)

    retrieval_cache_val = _neighbor_cache(descriptor_train, descriptor_val, residual_train_pca)
    retrieval_cache_test = _neighbor_cache(descriptor_train, descriptor_test, residual_train_pca)

    top100_gene_idx = np.argsort(-residual_train.var(axis=0))[: min(100, residual_train.shape[1])]
    train_set = set(train_perts)
    connected_test = sum(1 for pert in test_perts if len(connected.get(pert, set()) & train_set) > 0)
    nonzero_test = sum(1 for pert in test_perts if np.linalg.norm(string_embeddings.get(pert, 0)) > 0)

    audit = {
        "dataset": dataset,
        "seed": seed,
        "cache_version": config.PREP_CACHE_VERSION,
        "n_cells": int(adata.n_obs),
        "n_genes_available": int(adata.n_vars),
        "n_retained_genes": int(len(genes)),
        "n_kept_perturbations": int(len(kept_perts)),
        "n_train_perturbations": int(len(train_perts)),
        "n_val_perturbations": int(len(val_perts)),
        "n_test_perturbations": int(len(test_perts)),
        "n_control_cells": int(control_mask.sum()),
        "mean_cells_per_perturbation": float(pert_counts.loc[kept_perts].mean()) if kept_perts else 0.0,
        "fraction_test_nonzero_string": float(nonzero_test / max(1, len(test_perts))),
        "fraction_test_with_connected_neighbor": float(connected_test / max(1, len(test_perts))),
        "residual_pca_dim_active": int(residual_pca.active_components),
        "residual_pca_dim_fixed": int(config.RESIDUAL_PCA_DIM),
        "full_target_basis": "truncated_svd_no_centering",
        "full_pca_dim_active": int(full_pca.active_components),
        "full_pca_dim_fixed": int(config.FULL_PCA_DIM),
        "train_perturbations": train_perts,
        "val_perturbations": val_perts,
        "test_perturbations": test_perts,
        "dropped_perturbations": dropped,
        "isolated_test_perturbations": [
            pert for pert in test_perts if np.linalg.norm(string_embeddings.get(pert, 0)) == 0
        ],
    }
    save_json(audit_dir / "dataset_audit.json", audit)

    payload = DatasetSplit(
        dataset=dataset,
        seed=seed,
        genes=genes,
        target_genes=target_genes,
        train_perts=train_perts,
        val_perts=val_perts,
        test_perts=test_perts,
        train_matrix=train_matrix,
        val_matrix=val_matrix,
        test_matrix=test_matrix,
        descriptor_train=descriptor_train,
        descriptor_val=descriptor_val,
        descriptor_test=descriptor_test,
        descriptor_degree_train=descriptor_degree_train,
        descriptor_degree_val=descriptor_degree_val,
        descriptor_degree_test=descriptor_degree_test,
        residual_train=residual_train,
        residual_val=residual_val,
        residual_test=residual_test,
        residual_train_pca=residual_train_pca,
        residual_val_pca=residual_val_pca,
        residual_test_pca=residual_test_pca,
        full_train_pca=full_train_pca,
        full_val_pca=full_val_pca,
        full_test_pca=full_test_pca,
        mu_pert_train=mu_pert_train,
        x_ctrl_train=x_ctrl_train,
        residual_pca=residual_pca,
        full_pca=full_pca,
        retrieval_cache_val=retrieval_cache_val,
        retrieval_cache_test=retrieval_cache_test,
        top100_gene_idx=top100_gene_idx,
        audit=audit,
    )
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload
