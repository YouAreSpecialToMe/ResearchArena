#!/usr/bin/env python3
"""Experiment 6: Safety Level Ablation.
Run from workspace root: python exp/exp6_ablation/run.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from run_experiments import run_exp6
run_exp6()
