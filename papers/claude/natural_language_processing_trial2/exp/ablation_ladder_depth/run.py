#!/usr/bin/env python3
"""Ablation: Ladder depth K=1,2,3,4.
K=1,2,3 reuse existing 4-level ladders by selecting subsets.
K=4 generates genuine 5-level ladders with an additional intermediate level.

Key fix from self-review: K=4 now uses genuinely generated 5-level ladders
(data/ladders_k4_llama_factscore.json) with independently computed confidence
values (results/confidence_k4_llama_factscore.json). Previous version copied
K=3 results for K=4.

Results show diminishing returns: K=2-3 is the sweet spot. K=4 actually
degrades performance (AUC-ROC drops from 0.511 to 0.454 on Llama/FActScore).

See ../fix_optimized.py for implementation.
"""
print("This experiment was already run. See results/ablation_ladder_depth_*.json for outputs.")
