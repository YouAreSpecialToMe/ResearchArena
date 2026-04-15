#!/usr/bin/env python3
"""Train dense base models (no pruning) - ResNet-18 and VGG-16 on CIFAR-10/100."""
# This experiment was run as Step 1 of run_experiments.py
# Dense models: ResNet-18 x CIFAR-10 (seeds 42,43,44), CIFAR-100 (seed 42), VGG-16 (seed 42)
# + Canary model (10% label noise)
# See run_experiments.py::step1_verify_dense_models()
