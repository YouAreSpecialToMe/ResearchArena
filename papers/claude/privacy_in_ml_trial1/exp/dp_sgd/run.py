#!/usr/bin/env python3
"""DP-SGD combination experiments.

Evaluates pruning + differential privacy (DP-SGD via Opacus).
Configurations:
  - Magnitude pruning (s=0.9) + DP-SGD at epsilon={1, 4, 8}
  - MemPrune (s=0.9) + DP-SGD at epsilon={1, 4, 8}
  - Dense (no pruning) + DP-SGD at epsilon={1, 4, 8}

This experiment is run as Phase 6 of run_experiments_v2.py.
See that script for the full implementation.
"""
print("DP-SGD experiments are run via run_experiments_v2.py Phase 6.")
print("Run: python run_experiments_v2.py")
