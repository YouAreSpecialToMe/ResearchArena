#!/usr/bin/env python3
"""Ablation: Sampling-based confidence estimation with N=5, 10, 20 samples.
Run independently for all 3 models on FActScore (200 claims subsample).
Also run N=10 on TruthfulQA and LongFact for cross-dataset comparison.

Key fix from self-review: Each (claim, level, N) combination uses a unique
seed derived from hash(claim_id, level, N) to ensure independent sampling.
Previous version used identical seeds for N=5 and N=10, producing identical results.

See ../fix_optimized.py for implementation (run_sampling_ablation function).
"""
print("This experiment was already run. See results/sampling_N*_*.json for outputs.")
