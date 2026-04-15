"""Run baseline experiments (LRU, TPP, ARMS, ALTO, Oracle)."""
import sys
sys.path.insert(0, '../..')
from run_experiments import run_experiment_batch, save_results, SEEDS, TRACE_TYPES, ADVERSARIAL_TYPES

all_traces = TRACE_TYPES + ADVERSARIAL_TYPES
baseline_policies = ['lru_reactive', 'tpp_like', 'arms_like', 'alto_like', 'oracle']
experiments = []
for policy in baseline_policies:
    for trace in all_traces:
        for seed in SEEDS:
            experiments.append((policy, trace, seed, '../../traces', 0.5, None))

results = run_experiment_batch(experiments, max_workers=2, desc="Baselines")
save_results(results, 'results.json')
