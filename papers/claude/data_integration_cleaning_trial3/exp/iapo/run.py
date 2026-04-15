#!/usr/bin/env python3
"""
Experiment: Interaction-Aware Pipeline Optimizer (IAPO)
Leave-one-dataset-out evaluation of the IAPO optimizer.

See run_experiment.py (iapo_optimize function) for implementation.

Architecture:
  - Tier 1: Rule-based heuristics (6 rules derived from interaction study)
  - Tier 2: Dataset-similarity-weighted lookup (cosine similarity on 8 features)
  - Fallback: Increase K or beam search when confidence is low
  - Pipeline generation: Greedy max-synergy path + local perturbations + random fill

Config:
  - Leave-one-dataset-out: train on 17 datasets, test on held-out
  - K=10 candidates evaluated per optimization
  - 18 datasets * 3 seeds = 54 runs

Results: results/iapo/
  - IAPO mean F1 = 0.697 (std 0.132), 10 evaluations
  - Quality ratio vs exhaustive: 98.6%
  - Search cost: 10/120 = 8.3%
  - Wilcoxon test vs random: p=0.005 (IAPO slightly lower than random)
  - IAPO beats greedy (0.697 vs 0.626) and canonical (0.697 vs 0.660)
  - IAPO is comparable to but slightly below random search (0.697 vs 0.705)

Interpretation:
  With only 5 operators (120 total permutations), the search space is small enough
  that random search with 50 samples is highly effective. IAPO's interaction-guided
  search would show more advantage with larger operator sets where the search space
  grows factorially.

Runtime: ~132s (2.2 min)
"""
