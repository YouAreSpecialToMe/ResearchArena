"""Beta ablation sweep for CC-SupCon."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
# Training handled by run_continue.py Phase 4 (beta ablation)
# beta values: 0.0, 0.5, 1.0, 2.0, 5.0
# See results/ablations/ for per-beta outputs
