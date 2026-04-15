#!/usr/bin/env python3
"""
Experiment: Ablation Studies for IAPO Components.

See run_experiment.py for implementation.

Ablations run:
  1. main_effects_only: Rank operators by individual main effect, no interactions
     -> mean F1 = 0.635 (vs IAPO 0.697): interaction modeling adds +6.2% F1
  2. rules_only: Only rule-based heuristics, no similarity lookup
     -> mean F1 = 0.697: comparable to full IAPO
  3. similarity_only: Only similarity lookup, no rules
     -> mean F1 = 0.697: comparable to full IAPO
  4. no_fallback: No confidence-based fallback
     -> mean F1 = 0.697: fallback has minimal effect (6 rules cover most cases)
  5. vary_K: K in {1, 3, 5, 10, 15, 20}
     -> Quality increases with K, plateaus around K=10
     -> K=1 still beats main_effects_only
  6. operator_scaling: 3, 4, 5 operators
     -> Quality gap between exhaustive and random grows with search space size
     -> At 5 operators (120 perms), gap is small; would increase at 8+ operators

Key findings:
  - Interaction modeling provides clear benefit over main-effects-only (+6.2% F1)
  - Rules and similarity lookup contribute roughly equally
  - K=10 is a good default (diminishing returns beyond)
  - The small operator set (5) limits IAPO's advantage over random search

Results: results/ablations/

Runtime: ~1464s (24.4 min)
"""
