"""Data loading: cached embeddings + label management for contrastive learning."""
import os
import json
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EMB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings")

class EmbeddingDataset(Dataset):
    """Dataset that loads cached ESM-2 embeddings with multi-level EC labels."""
    def __init__(self, split="train"):
        self.embeddings = torch.load(os.path.join(EMB_DIR, f"{split}_embeddings.pt"),
                                      map_location="cpu", weights_only=True)
        with open(os.path.join(EMB_DIR, f"{split}_labels.json")) as f:
            self.labels_dict = json.load(f)

        # Build integer label mappings for each EC level
        self.label_maps = {}
        for level in ["ec_l1", "ec_l2", "ec_l3", "ec_l4"]:
            unique_labels = sorted(set(self.labels_dict[level]))
            self.label_maps[level] = {lab: idx for idx, lab in enumerate(unique_labels)}

        self.n_samples = self.embeddings.shape[0]

    def __len__(self):
        return self.n_samples

    def get_labels(self, level):
        """Return integer labels for a given EC level."""
        mapping = self.label_maps[level]
        return torch.tensor([mapping[lab] for lab in self.labels_dict[level]], dtype=torch.long)

    def get_string_labels(self, level):
        """Return string labels for a given EC level."""
        return self.labels_dict[level]

    def __getitem__(self, idx):
        return self.embeddings[idx]

    def n_classes(self, level):
        return len(self.label_maps[level])


class BalancedBatchSampler(Sampler):
    """Sampler ensuring each batch has multiple examples per class for contrastive learning."""
    def __init__(self, labels, batch_size, classes_per_batch=32, samples_per_class=16):
        self.labels = labels.numpy()
        self.batch_size = batch_size
        self.classes_per_batch = min(classes_per_batch, len(set(self.labels)))
        self.samples_per_class = samples_per_class

        # Build index for each class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)
        self.classes = list(self.class_indices.keys())

        # Effective batch size
        self.effective_batch = self.classes_per_batch * self.samples_per_class
        self.n_batches = max(1, len(self.labels) // self.effective_batch)

    def __iter__(self):
        for _ in range(self.n_batches):
            # Sample classes
            selected_classes = np.random.choice(
                self.classes, size=self.classes_per_batch, replace=False
            ) if len(self.classes) >= self.classes_per_batch else self.classes

            batch = []
            for cls in selected_classes:
                indices = self.class_indices[cls]
                if len(indices) >= self.samples_per_class:
                    sampled = np.random.choice(indices, size=self.samples_per_class, replace=False)
                else:
                    sampled = np.random.choice(indices, size=self.samples_per_class, replace=True)
                batch.extend(sampled.tolist())

            # Trim or pad to batch_size
            if len(batch) > self.batch_size:
                batch = batch[:self.batch_size]
            yield batch

    def __len__(self):
        return self.n_batches


def get_dataloader(split, level, batch_size=512, balanced=True):
    """Get a DataLoader for a specific split and EC level."""
    dataset = EmbeddingDataset(split)
    labels = dataset.get_labels(level)

    if balanced and split == "train":
        # For contrastive learning, use balanced sampling
        n_classes = dataset.n_classes(level)
        if n_classes <= 32:
            classes_per_batch = n_classes
            samples_per_class = min(batch_size // n_classes, 64)
        else:
            classes_per_batch = min(32, n_classes)
            samples_per_class = batch_size // classes_per_batch

        sampler = BalancedBatchSampler(
            labels, batch_size,
            classes_per_batch=classes_per_batch,
            samples_per_class=samples_per_class
        )
        return dataset, DataLoader(
            dataset, batch_sampler=sampler, num_workers=0, pin_memory=True
        ), labels
    else:
        return dataset, DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=0, pin_memory=True
        ), labels
