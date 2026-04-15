from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances

from .data import build_datasets, get_dataset_bundle
from .utils import device


@torch.no_grad()
def extract_dataset_features(model, dataset, batch_size: int = 256) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    model.eval()
    feats, coarse, fine, sample_ids = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device(), non_blocking=True)
        features = model.forward_backbone(images).cpu().numpy()
        feats.append(features)
        coarse.append(batch["coarse_label"].numpy())
        fine.append(batch["fine_label"].numpy())
        sample_ids.append(batch["sample_id"].numpy())
    return {
        "features": np.concatenate(feats),
        "coarse_labels": np.concatenate(coarse),
        "fine_labels": np.concatenate(fine),
        "sample_ids": np.concatenate(sample_ids),
    }


def linear_probe(train_X, train_y, val_X, val_y, test_X, test_y) -> Dict:
    best = None
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1]:
        clf = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=2000,
            tol=1e-3,
            random_state=0,
            n_jobs=4,
        )
        clf.fit(train_X, train_y)
        val_acc = clf.score(val_X, val_y)
        if best is None or val_acc > best["val_accuracy"]:
            best = {"alpha": alpha, "val_accuracy": float(val_acc), "clf": clf}
    test_acc = best["clf"].score(test_X, test_y)
    return {"val_accuracy": best["val_accuracy"], "test_accuracy": float(test_acc), "alpha": best["alpha"]}


def cosine_knn(train_X, train_y, test_X, test_y, k: int = 20) -> Dict:
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    return {"test_accuracy": float(accuracy_score(test_y, pred)), "k": k}


def overlap_at_k(train_features: np.ndarray, sample_ids: np.ndarray, coarse_labels: np.ndarray, graph_ids: np.ndarray, k: int = 10) -> float:
    overlaps = []
    for coarse in np.unique(coarse_labels):
        idx = np.where(coarse_labels == coarse)[0]
        feats = train_features[idx]
        ids = sample_ids[idx]
        sims = 1.0 - pairwise_distances(feats, metric="cosine")
        np.fill_diagonal(sims, -np.inf)
        topk = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        topk_sims = np.take_along_axis(sims, topk, axis=1)
        order = np.argsort(-topk_sims, axis=1)
        topk = np.take_along_axis(topk, order, axis=1)
        student_ids_block = ids[topk]
        teacher_ids_block = graph_ids[idx][:, :k]
        for student_row, teacher_row in zip(student_ids_block, teacher_ids_block):
            overlaps.append(len(set(map(int, student_row)) & set(map(int, teacher_row))) / float(k))
    return float(np.mean(overlaps))


def clustering_scores(features: np.ndarray, fine_labels: np.ndarray, coarse_labels: np.ndarray) -> Dict:
    aris, nmis = [], []
    for coarse in np.unique(coarse_labels):
        idx = coarse_labels == coarse
        y = fine_labels[idx]
        n_clusters = len(np.unique(y))
        if len(y) < n_clusters:
            continue
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        pred = km.fit_predict(features[idx])
        aris.append(adjusted_rand_score(y, pred))
        nmis.append(normalized_mutual_info_score(y, pred))
    return {"ari_mean": float(np.mean(aris)), "nmi_mean": float(np.mean(nmis))}


def evaluate_model(model, dataset_name: str, graph_npz: str | None = None) -> Dict:
    bundle = get_dataset_bundle(dataset_name)
    datasets = build_datasets(bundle)
    train_eval = extract_dataset_features(model, datasets["train_eval"])
    val_eval = extract_dataset_features(model, datasets["val_eval"])
    test_eval = extract_dataset_features(model, datasets["test_eval"])
    probe = linear_probe(
        train_eval["features"], train_eval["fine_labels"],
        val_eval["features"], val_eval["fine_labels"],
        test_eval["features"], test_eval["fine_labels"],
    )
    coarse_probe = linear_probe(
        train_eval["features"], train_eval["coarse_labels"],
        val_eval["features"], val_eval["coarse_labels"],
        test_eval["features"], test_eval["coarse_labels"],
    )
    knn20 = cosine_knn(train_eval["features"], train_eval["fine_labels"], test_eval["features"], test_eval["fine_labels"], k=20)
    knn5 = cosine_knn(train_eval["features"], train_eval["fine_labels"], test_eval["features"], test_eval["fine_labels"], k=5)
    coarse_pred = coarse_probe["test_accuracy"]
    coarse_acc = float(coarse_probe["test_accuracy"])
    coarse_clf = SGDClassifier(loss="log_loss", alpha=coarse_probe["alpha"], max_iter=2000, tol=1e-3, random_state=0)
    coarse_clf.fit(train_eval["features"], train_eval["coarse_labels"])
    coarse_test_pred = coarse_clf.predict(test_eval["features"])
    coarse_macro_f1 = float(f1_score(test_eval["coarse_labels"], coarse_test_pred, average="macro"))
    cluster = clustering_scores(test_eval["features"], test_eval["fine_labels"], test_eval["coarse_labels"])
    overlap = None
    if graph_npz is not None:
        graph = np.load(graph_npz)
        overlap = overlap_at_k(train_eval["features"], train_eval["sample_ids"], train_eval["coarse_labels"], graph["neighbor_ids"], k=10)
    per_coarse = defaultdict(dict)
    fine_knn20_model = KNeighborsClassifier(n_neighbors=20, metric="cosine")
    fine_knn20_model.fit(train_eval["features"], train_eval["fine_labels"])
    for coarse in np.unique(test_eval["coarse_labels"]):
        idx = test_eval["coarse_labels"] == coarse
        per_coarse[str(int(coarse))]["fine_knn20_acc"] = float(
            accuracy_score(
                test_eval["fine_labels"][idx],
                fine_knn20_model.predict(test_eval["features"][idx]),
            )
        )
    return {
        "fine_linear_probe": probe,
        "coarse_linear_probe": coarse_probe,
        "fine_knn20": knn20,
        "fine_knn5": knn5,
        "coarse_acc": coarse_acc,
        "coarse_macro_f1": coarse_macro_f1,
        "overlap_at_10": overlap,
        "ari_nmi": cluster,
        "per_coarse": per_coarse,
    }
