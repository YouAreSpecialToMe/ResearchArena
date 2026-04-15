#!/usr/bin/env python3
"""
Experiment: Baseline Pipeline Construction Methods
4 baselines compared against IAPO.

See run_experiment.py for implementation.

Methods:
  1. Exhaustive Search: All 5!=120 permutations (true exhaustive, not sampling)
  2. Random Search: 50 random permutations, return best
  3. Greedy Forward: Greedily append best operator (15 evaluations = 5+4+3+2+1)
  4. Canonical Order: Fixed textbook ordering (1 evaluation)

Config:
  - 18 datasets * 3 seeds = 54 runs per baseline method
  - Seed-dependent splits and error injection
  - Evaluation: LogisticRegression (fast=True)

Results: results/baselines/
  - Exhaustive:  mean F1 = 0.706 (std 0.128), 120 evals
  - Random(50):  mean F1 = 0.705 (std 0.128), 50 evals
  - Greedy:      mean F1 = 0.626 (std 0.170), 15 evals
  - Canonical:   mean F1 = 0.660 (std 0.162), 1 eval

Key findings:
  - Random search with 50 samples nearly matches exhaustive (99.9%)
  - With 5 operators, 5!=120 is small enough that random 50 covers search space well
  - Greedy is worst due to local optima (pipeline ordering is non-monotonic)
  - Canonical order is better than greedy but worse than search-based methods

Runtime: ~2691s (44.9 min)
"""
