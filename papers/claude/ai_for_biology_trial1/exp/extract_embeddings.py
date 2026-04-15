"""Extract and cache ESM-2 (650M) embeddings for all sequences."""
import os
import sys
import json
import csv
import time
import torch
import numpy as np
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "exp", "data")
EMB_DIR = os.path.join(BASE_DIR, "exp", "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

def load_csv(filepath):
    """Load processed CSV file."""
    records = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records

def extract_embeddings(records, model, alphabet, batch_converter, device, batch_size=8, max_len=1022):
    """Extract mean-pooled ESM-2 embeddings."""
    embeddings = []
    n_total = len(records)

    # Sort by sequence length for efficient batching
    indexed_records = [(i, r) for i, r in enumerate(records)]
    indexed_records.sort(key=lambda x: len(x[1]["sequence"]))

    # Process in batches
    batch_data = []
    batch_original_indices = []

    result_embeddings = [None] * n_total

    for orig_idx, record in tqdm(indexed_records, desc="Extracting"):
        seq = record["sequence"][:max_len]  # Truncate very long sequences
        batch_data.append((record["id"], seq))
        batch_original_indices.append(orig_idx)

        if len(batch_data) >= batch_size:
            _process_batch(batch_data, batch_original_indices, result_embeddings,
                          model, alphabet, batch_converter, device)
            batch_data = []
            batch_original_indices = []

    # Process remaining
    if batch_data:
        _process_batch(batch_data, batch_original_indices, result_embeddings,
                      model, alphabet, batch_converter, device)

    return torch.stack(result_embeddings)

def _process_batch(batch_data, batch_indices, result_embeddings,
                   model, alphabet, batch_converter, device):
    """Process a single batch."""
    _, _, batch_tokens = batch_converter(batch_data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        representations = results["representations"][33]  # [B, L, 1280]

    # Mean pool (excluding BOS and EOS tokens)
    for i, orig_idx in enumerate(batch_indices):
        seq_len = len(batch_data[i][1])
        # ESM-2 adds BOS token at position 0
        emb = representations[i, 1:seq_len+1, :].mean(dim=0)  # [1280]
        result_embeddings[orig_idx] = emb.float().cpu()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load ESM-2 model
    print("Loading ESM-2 (650M)...")
    start = time.time()
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    print(f"Model loaded in {time.time()-start:.1f}s")

    # Process each split
    for split_name, csv_file in [("train", "train.csv"), ("new392", "new392.csv"), ("price149", "price149.csv")]:
        emb_path = os.path.join(EMB_DIR, f"{split_name}_embeddings.pt")
        labels_path = os.path.join(EMB_DIR, f"{split_name}_labels.json")

        if os.path.exists(emb_path) and os.path.exists(labels_path):
            print(f"\n{split_name}: Already cached, skipping.")
            continue

        print(f"\nProcessing {split_name}...")
        records = load_csv(os.path.join(DATA_DIR, csv_file))

        # Adjust batch size based on split
        if split_name == "train":
            batch_size = 8  # More conservative for larger dataset
        else:
            batch_size = 16

        start = time.time()
        embeddings = extract_embeddings(records, model, alphabet, batch_converter,
                                        device, batch_size=batch_size)
        elapsed = time.time() - start

        print(f"  Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]} "
              f"in {elapsed:.1f}s ({elapsed/60:.1f}min)")

        # Verify no NaN/Inf
        assert not torch.isnan(embeddings).any(), "NaN in embeddings!"
        assert not torch.isinf(embeddings).any(), "Inf in embeddings!"

        # Save embeddings
        torch.save(embeddings, emb_path)

        # Save labels
        labels = {
            "ec_l1": [r["ec_l1"] for r in records],
            "ec_l2": [r["ec_l2"] for r in records],
            "ec_l3": [r["ec_l3"] for r in records],
            "ec_l4": [r["ec_l4"] for r in records],
            "ids": [r["id"] for r in records],
        }
        with open(labels_path, "w") as f:
            json.dump(labels, f)

        print(f"  Saved to {emb_path} and {labels_path}")

    print("\nDone! All embeddings extracted.")

if __name__ == "__main__":
    main()
