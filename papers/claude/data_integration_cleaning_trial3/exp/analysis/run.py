#!/usr/bin/env python3
"""
Experiment: Statistical Analysis & Hypothesis Testing.

See run_experiment.py (Step 5) for implementation.

Hypothesis Results:
  H1 (Systematicity): FAIL
    - 15% of operator pairs significant (needed >= 50%)
    - But interactions DO exist: 22% order-sensitive, meaningful synergistic/antagonistic pairs
    - The majority of interactions are weak/dataset-specific, not systematic across all datasets

  H2 (Predictability): PARTIAL
    - Rule accuracy: 93.6% (PASS, needed >= 70%)
    - Spearman rho: 0.380 (FAIL, needed > 0.5)
    - Rules work well for sign prediction but magnitude prediction is weak

  H3 (Efficiency): PASS
    - IAPO achieves 98.6% of exhaustive quality (needed >= 95%)
    - Search cost: 8.3% (needed <= 10%)
    - But IAPO does NOT beat random search (0.697 vs 0.705, Wilcoxon p=0.005)
    - The pass is because IAPO achieves near-optimal quality, not because it
      outperforms other methods

Scientific conclusions:
  1. Operator interactions exist but are weaker than hypothesized
  2. Most pairs (63%) are effectively independent
  3. The strongest interactions involve ValueNormalizer and CategoricalEncoder
  4. Interaction effects are partially predictable from dataset features
  5. With 5 operators (small search space), random search is hard to beat
  6. IAPO's value lies in principled search rather than brute-force, though
     the benefit is marginal for small operator sets

Results: results/hypothesis_tests/
"""
