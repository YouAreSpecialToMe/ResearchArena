"""Preprocess CLEAN data: parse EC levels, compute stats, save processed CSVs."""
import os
import json
import csv
from collections import Counter, defaultdict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(DATA_DIR, "CLEAN_repo", "app", "data")

def parse_ec_levels(ec_full):
    parts = ec_full.strip().split(".")
    if len(parts) != 4:
        return None
    # Skip incomplete EC numbers (contain '-')
    if "-" in ec_full:
        return None
    return {
        "ec_full": ec_full,
        "ec_l1": parts[0],
        "ec_l2": f"{parts[0]}.{parts[1]}",
        "ec_l3": f"{parts[0]}.{parts[1]}.{parts[2]}",
        "ec_l4": ec_full
    }

def load_and_process(filepath, name):
    """Load a CLEAN CSV file, parse EC levels, return list of dicts."""
    records = []
    skipped_multi = 0
    skipped_incomplete = 0
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entry = row["Entry"]
            ec_raw = row["EC number"].strip()
            seq = row["Sequence"].strip()

            # Handle multiple EC numbers: take the first one
            if ";" in ec_raw:
                ec_raw = ec_raw.split(";")[0].strip()
                skipped_multi += 1

            levels = parse_ec_levels(ec_raw)
            if levels is None:
                skipped_incomplete += 1
                continue

            records.append({
                "id": entry,
                "sequence": seq,
                **levels
            })

    print(f"  {name}: {len(records)} valid records "
          f"(multi-EC: {skipped_multi}, incomplete: {skipped_incomplete} skipped)")
    return records

def compute_stats(records, name):
    """Compute dataset statistics."""
    stats = {"name": name, "n_sequences": len(records)}
    for level in ["ec_l1", "ec_l2", "ec_l3", "ec_l4"]:
        classes = Counter(r[level] for r in records)
        stats[f"n_classes_{level}"] = len(classes)
        stats[f"min_count_{level}"] = min(classes.values())
        stats[f"max_count_{level}"] = max(classes.values())
        stats[f"median_count_{level}"] = sorted(classes.values())[len(classes) // 2]

    # Rare classes at L4 (fewer than 10 examples)
    l4_counts = Counter(r["ec_l4"] for r in records)
    rare_classes = {ec for ec, count in l4_counts.items() if count < 10}
    stats["n_rare_l4_classes"] = len(rare_classes)
    stats["pct_rare_l4_classes"] = len(rare_classes) / len(l4_counts) * 100

    return stats, rare_classes

def save_csv(records, filepath):
    if not records:
        return
    fieldnames = ["id", "sequence", "ec_full", "ec_l1", "ec_l2", "ec_l3", "ec_l4"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def main():
    print("Processing CLEAN benchmark data...")

    # Load splits
    train_records = load_and_process(os.path.join(REPO_DIR, "split100.csv"), "train")
    new392_records = load_and_process(os.path.join(REPO_DIR, "new.csv"), "new392")
    price149_records = load_and_process(os.path.join(REPO_DIR, "datasets", "price.csv"), "price149")

    # Compute statistics
    train_stats, rare_classes = compute_stats(train_records, "train")
    new392_stats, _ = compute_stats(new392_records, "new392")
    price149_stats, _ = compute_stats(price149_records, "price149")

    all_stats = {
        "train": train_stats,
        "new392": new392_stats,
        "price149": price149_stats,
        "rare_l4_classes": sorted(list(rare_classes))
    }

    print(f"\n=== Dataset Statistics ===")
    for split_name, stats in [("train", train_stats), ("new392", new392_stats), ("price149", price149_stats)]:
        print(f"\n{split_name}:")
        print(f"  Sequences: {stats['n_sequences']}")
        for level in ["ec_l1", "ec_l2", "ec_l3", "ec_l4"]:
            print(f"  {level}: {stats[f'n_classes_{level}']} classes "
                  f"(min={stats[f'min_count_{level}']}, max={stats[f'max_count_{level}']})")
    print(f"\nRare L4 classes (<10 examples): {len(rare_classes)} / {train_stats['n_classes_ec_l4']} "
          f"({train_stats['pct_rare_l4_classes']:.1f}%)")

    # Save processed data
    save_csv(train_records, os.path.join(DATA_DIR, "train.csv"))
    save_csv(new392_records, os.path.join(DATA_DIR, "new392.csv"))
    save_csv(price149_records, os.path.join(DATA_DIR, "price149.csv"))

    with open(os.path.join(DATA_DIR, "dataset_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nSaved: train.csv, new392.csv, price149.csv, dataset_stats.json")

if __name__ == "__main__":
    main()
