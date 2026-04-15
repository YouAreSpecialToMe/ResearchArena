"""BLASTp sequence homology baseline."""
import os
import sys
import json
import csv
import subprocess
import tempfile
import time
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "exp", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "exp", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_csv(filepath):
    records = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records

def write_fasta(records, filepath):
    with open(filepath, "w") as f:
        for r in records:
            f.write(f">{r['id']}\n{r['sequence']}\n")

def run_blastp_baseline():
    """Run BLASTp as baseline."""
    print("Loading data...")
    train = load_csv(os.path.join(DATA_DIR, "train.csv"))
    new392 = load_csv(os.path.join(DATA_DIR, "new392.csv"))
    price149 = load_csv(os.path.join(DATA_DIR, "price149.csv"))

    # Build EC mapping from training data
    id_to_ec = {r["id"]: r["ec_l4"] for r in train}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write train FASTA
        train_fasta = os.path.join(tmpdir, "train.fasta")
        write_fasta(train, train_fasta)

        # Create BLAST database
        db_path = os.path.join(tmpdir, "train_db")
        print("Creating BLAST database...")
        subprocess.run([
            "makeblastdb", "-in", train_fasta, "-dbtype", "prot",
            "-out", db_path
        ], check=True, capture_output=True)

        results = {}
        for test_name, test_data in [("new392", new392), ("price149", price149)]:
            print(f"\nRunning BLASTp for {test_name}...")
            query_fasta = os.path.join(tmpdir, f"{test_name}.fasta")
            write_fasta(test_data, query_fasta)

            out_file = os.path.join(tmpdir, f"{test_name}_blast.tsv")
            subprocess.run([
                "blastp", "-query", query_fasta, "-db", db_path,
                "-out", out_file, "-outfmt", "6 qseqid sseqid evalue bitscore",
                "-evalue", "1e-10", "-max_target_seqs", "1",
                "-num_threads", "4"
            ], check=True, capture_output=True)

            # Parse results
            hits = {}
            with open(out_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    query_id = parts[0]
                    subject_id = parts[1]
                    if query_id not in hits:
                        hits[query_id] = subject_id

            # Compute metrics
            true_labels = []
            pred_labels = []
            for r in test_data:
                true_ec = r["ec_l4"]
                pred_ec = id_to_ec.get(hits.get(r["id"], ""), "unknown")
                true_labels.append(true_ec)
                pred_labels.append(pred_ec)

            # Compute F1
            from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

            # For unknown predictions, they'll just be wrong
            correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
            n_hits = sum(1 for r in test_data if r["id"] in hits)

            # Macro/micro F1
            macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
            micro_f1 = f1_score(true_labels, pred_labels, average="micro", zero_division=0)
            precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
            recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)

            results[test_name] = {
                "macro_f1": float(macro_f1),
                "micro_f1": float(micro_f1),
                "precision": float(precision),
                "recall": float(recall),
                "accuracy": float(correct / len(test_data)),
                "n_hits": n_hits,
                "n_total": len(test_data),
                "hit_rate": float(n_hits / len(test_data)),
            }
            print(f"  {test_name}: F1={macro_f1:.4f}, Acc={correct/len(test_data):.4f}, "
                  f"Hits={n_hits}/{len(test_data)}")

    return results

if __name__ == "__main__":
    # Check if BLAST is installed
    try:
        subprocess.run(["blastp", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("BLAST+ not found. Installing...")
        os.system("sudo apt-get install -y ncbi-blast+ 2>/dev/null || conda install -y -c bioconda blast 2>/dev/null")

    start = time.time()
    results = run_blastp_baseline()
    results["method"] = "blastp"
    results["runtime_seconds"] = time.time() - start

    result_path = os.path.join(RESULTS_DIR, "blastp_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {result_path}")
