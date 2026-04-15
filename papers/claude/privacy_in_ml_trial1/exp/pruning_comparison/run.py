#!/usr/bin/env python3
"""Primary pruning comparison: 4 methods x 4 sparsities x 3 seeds on ResNet-18/CIFAR-10."""
# This experiment was run as Step 3 of run_experiments.py
# Methods: random, magnitude, grad_sensitivity, memprune (GDS-based)
# Sparsities: 50%, 70%, 90%, 95%
# Seeds: 42, 43, 44
# 20 epochs fine-tuning after pruning
# See run_experiments.py::step3_pruning_comparison()
