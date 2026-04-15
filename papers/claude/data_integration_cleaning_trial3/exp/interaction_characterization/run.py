#!/usr/bin/env python3
"""
Experiment: Interaction Characterization
Measures pairwise interaction effects between 5 cleaning operators across 18 datasets.

See run_experiment.py (compute_pairwise_interactions function) for implementation.

Config:
  - 5 operators: Imputer, OutlierRemover, DuplicateRemover, Normalizer, CategoricalEncoder
  - 18 OpenML datasets with controlled error injection
  - 3 seeds: [42, 123, 456] - each creates different train/test split + error pattern
  - 20 ordered operator pairs per dataset-seed (5*4=20)
  - Total: 18 * 3 * 20 = 1080 interaction measurements

Results: results/interaction_profiles/
  - 15% of operator pairs show statistically significant interactions (3/20)
  - 22% of observations are order-sensitive (OS > 0.01)
  - Category: 63% Independent, 13% Synergistic, 11% Order-Critical, 9% Antagonistic
  - Top synergistic: Normalizer->CatEnc (IE=0.053), Normalizer->Outlier (IE=0.050)
  - Top antagonistic: Dedup->CatEnc (IE=-0.049)
  - Most order-sensitive: Outlier<->Normalizer (OS=0.055)
  - 6 interaction rules derived

Runtime: ~380s
"""
