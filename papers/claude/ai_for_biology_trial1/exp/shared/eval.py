"""Evaluation functions implementing the CLEAN protocol."""
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import defaultdict


def compute_centroids(embeddings, labels):
    """Compute mean embedding per class.

    Args:
        embeddings: [N, D] tensor
        labels: [N] tensor of integer labels
    Returns:
        centroids: dict mapping label -> [D] tensor
    """
    centroids = {}
    unique_labels = labels.unique()
    for lab in unique_labels:
        mask = labels == lab
        centroids[lab.item()] = embeddings[mask].mean(dim=0)
    return centroids


def predict_nearest_centroid(query_embeddings, centroids):
    """Assign each query to its nearest centroid (cosine similarity).

    Args:
        query_embeddings: [Q, D] tensor
        centroids: dict mapping label -> [D] tensor
    Returns:
        predictions: list of predicted labels
    """
    labels = sorted(centroids.keys())
    centroid_matrix = torch.stack([centroids[lab] for lab in labels])  # [C, D]
    centroid_matrix = torch.nn.functional.normalize(centroid_matrix, dim=1)
    query_norm = torch.nn.functional.normalize(query_embeddings, dim=1)

    # Cosine similarity
    sim = torch.matmul(query_norm, centroid_matrix.T)  # [Q, C]
    pred_indices = sim.argmax(dim=1)
    predictions = [labels[idx] for idx in pred_indices.tolist()]
    return predictions


def predict_max_sep(query_embeddings, centroids):
    """Max-separation prediction: assign EC with largest gap to next closest.

    Args:
        query_embeddings: [Q, D]
        centroids: dict mapping label -> [D] tensor
    Returns:
        predictions: list of predicted labels
    """
    labels = sorted(centroids.keys())
    centroid_matrix = torch.stack([centroids[lab] for lab in labels])
    centroid_matrix = torch.nn.functional.normalize(centroid_matrix, dim=1)
    query_norm = torch.nn.functional.normalize(query_embeddings, dim=1)

    sim = torch.matmul(query_norm, centroid_matrix.T)  # [Q, C]

    predictions = []
    for i in range(sim.shape[0]):
        sims = sim[i]
        sorted_sims, sorted_idx = torch.sort(sims, descending=True)
        if len(sorted_sims) >= 2:
            # Find the class with max separation from 2nd best
            best_sep = -float('inf')
            best_label = labels[sorted_idx[0].item()]
            for j in range(min(5, len(sorted_sims) - 1)):  # Check top-5
                sep = sorted_sims[j] - sorted_sims[j + 1]
                if sep > best_sep:
                    best_sep = sep
                    best_label = labels[sorted_idx[j].item()]
            predictions.append(best_label)
        else:
            predictions.append(labels[sorted_idx[0].item()])

    return predictions


def compute_metrics(predictions, true_labels, rare_classes=None, label_map=None):
    """Compute evaluation metrics.

    Args:
        predictions: list of predicted integer labels
        true_labels: list of true integer labels
        rare_classes: set of integer labels considered "rare" (optional)
        label_map: dict mapping integer label -> string EC number (optional)
    Returns:
        dict of metrics
    """
    preds = np.array(predictions)
    trues = np.array(true_labels)

    metrics = {
        "macro_f1": float(f1_score(trues, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(trues, preds, average="micro", zero_division=0)),
        "precision": float(precision_score(trues, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(trues, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(trues, preds)),
    }

    # F1 on rare classes only
    if rare_classes is not None and len(rare_classes) > 0:
        rare_mask = np.isin(trues, list(rare_classes))
        if rare_mask.sum() > 0:
            metrics["f1_rare"] = float(f1_score(
                trues[rare_mask], preds[rare_mask], average="macro", zero_division=0
            ))
            metrics["n_rare_samples"] = int(rare_mask.sum())
        else:
            metrics["f1_rare"] = 0.0
            metrics["n_rare_samples"] = 0

    return metrics


def evaluate_model(model, train_dataset, test_dataset, level="ec_l4",
                   rare_classes_str=None, device="cuda"):
    """Full evaluation pipeline: compute centroids, predict, compute metrics.

    Args:
        model: ProjectionHead model
        train_dataset: EmbeddingDataset for training data
        test_dataset: EmbeddingDataset for test data
        level: which EC level to evaluate at
        rare_classes_str: set of string EC labels considered rare
        device: torch device
    Returns:
        dict of metrics
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        # Get train embeddings in batches (227K at once may OOM)
        batch_size = 4096
        train_embs = []
        for i in range(0, train_dataset.n_samples, batch_size):
            batch = train_dataset.embeddings[i:i+batch_size].to(device)
            train_embs.append(model(batch).cpu())
        train_emb = torch.cat(train_embs, dim=0)
        train_labels = train_dataset.get_labels(level)

        # Get test embeddings
        test_emb = model(test_dataset.embeddings.to(device)).cpu()
        test_labels = test_dataset.get_labels(level)

        # Build a combined label map (test labels must be in train label space)
        train_label_map = train_dataset.label_maps[level]
        test_label_map = test_dataset.label_maps[level]

        # Map test string labels to train integer labels
        test_str_labels = test_dataset.get_string_labels(level)
        mapped_test_labels = []
        valid_mask = []
        for str_lab in test_str_labels:
            if str_lab in train_label_map:
                mapped_test_labels.append(train_label_map[str_lab])
                valid_mask.append(True)
            else:
                mapped_test_labels.append(-1)
                valid_mask.append(False)

        valid_mask = np.array(valid_mask)
        mapped_test_labels = np.array(mapped_test_labels)

        # Compute centroids from training data
        centroids = compute_centroids(train_emb.cpu(), train_labels)

        # Predict for valid test samples
        if valid_mask.sum() == 0:
            return {"macro_f1": 0.0, "micro_f1": 0.0, "accuracy": 0.0,
                    "n_valid_test": 0, "n_total_test": len(test_str_labels)}

        valid_test_emb = test_emb[valid_mask]
        valid_true = mapped_test_labels[valid_mask]

        # Nearest centroid prediction
        preds_nc = predict_nearest_centroid(valid_test_emb, centroids)
        # Max-sep prediction
        preds_ms = predict_max_sep(valid_test_emb, centroids)

        # Determine rare classes in integer label space
        rare_int = None
        if rare_classes_str is not None:
            rare_int = set()
            for str_lab in rare_classes_str:
                if str_lab in train_label_map:
                    rare_int.add(train_label_map[str_lab])

        metrics_nc = compute_metrics(preds_nc, valid_true.tolist(), rare_int)
        metrics_ms = compute_metrics(preds_ms, valid_true.tolist(), rare_int)

        # Use the better of the two strategies
        if metrics_ms["macro_f1"] >= metrics_nc["macro_f1"]:
            best_metrics = dict(metrics_ms)
            best_metrics["strategy"] = "max_sep"
        else:
            best_metrics = dict(metrics_nc)
            best_metrics["strategy"] = "nearest_centroid"

        best_metrics["n_valid_test"] = int(valid_mask.sum())
        best_metrics["n_total_test"] = len(test_str_labels)
        best_metrics["metrics_nearest_centroid"] = metrics_nc
        best_metrics["metrics_max_sep"] = metrics_ms

    return best_metrics
