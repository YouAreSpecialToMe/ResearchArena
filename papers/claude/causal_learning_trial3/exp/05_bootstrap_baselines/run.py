#!/usr/bin/env python3
"""Run bootstrap-PC and bootstrap-GES on ALL 80 settings x 3 seeds."""
import sys
sys.path.insert(0, '../..')
from run_all import run_bootstrap_baselines
run_bootstrap_baselines()
