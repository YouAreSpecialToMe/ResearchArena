"""
Finalize results: re-run combination analysis, evaluation, and figures
with the fixed SelfCheck baseline data.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Import functions from the main rerun script
from rerun_full import (
    run_combination_analysis, compute_all_metrics, generate_figures,
    save_canonical_results, RESULTS_DIR, DATASETS
)

print("Step 1: Re-running combination analysis with fixed SelfCheck...")
for mshort in ["llama", "mistral", "qwen"]:
    for ds in DATASETS:
        print(f"  {mshort}/{ds}")
        run_combination_analysis(mshort, ds)

print("\nStep 2: Computing comprehensive evaluation metrics...")
eval_results = compute_all_metrics()

print("\nStep 3: Generating figures...")
generate_figures(eval_results)

print("\nStep 4: Saving canonical results.json...")
save_canonical_results(eval_results)

print("\nDone!")
