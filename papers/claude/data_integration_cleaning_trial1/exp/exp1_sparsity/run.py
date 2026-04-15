#!/usr/bin/env python3
"""Experiment 1: Constraint Interaction Sparsity Analysis.
Run from workspace root: python exp/exp1_sparsity/run.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from run_experiments import run_exp1
run_exp1()
