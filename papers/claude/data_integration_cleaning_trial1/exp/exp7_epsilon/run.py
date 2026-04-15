#!/usr/bin/env python3
"""Experiment 7: Epsilon Sensitivity Analysis.
Run from workspace root: python exp/exp7_epsilon/run.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from run_experiments import run_exp7
run_exp7()
