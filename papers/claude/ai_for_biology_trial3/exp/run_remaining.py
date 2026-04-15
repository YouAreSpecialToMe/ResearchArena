#!/usr/bin/env python3
"""Run remaining steps: GB1 attention, statistical analysis, figures, results aggregation.
Baselines, REN, and ablations are already completed and saved."""
import sys
import time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from run_all_v2 import (
    load_cached_data, run_gb1_attention_analysis,
    run_statistical_analysis, generate_figures, aggregate_results,
    DEVICE
)

start = time.time()
print(f"Device: {DEVICE}")

# Load data
selected_assays, assay_stats, struct_data, embeddings, llr_cache = load_cached_data()

# Step 4: GB1 attention analysis (fixed OOM)
print("\n\nSTEP 4: GB1 attention analysis")
import torch
torch.cuda.empty_cache()
gb1_attention = run_gb1_attention_analysis(struct_data, embeddings, llr_cache)

# Step 5: Statistical analysis
print("\n\nSTEP 5: Statistical analysis")
analysis_results = run_statistical_analysis(selected_assays, assay_stats)

# Step 6: Figures
print("\n\nSTEP 6: Figures")
generate_figures(selected_assays, assay_stats, analysis_results)

# Step 7: Aggregate
print("\n\nSTEP 7: Aggregate results")
final = aggregate_results(selected_assays, assay_stats, analysis_results, gb1_attention)

elapsed = time.time() - start
print(f"\nRemaining steps completed in {elapsed/60:.1f} minutes")
