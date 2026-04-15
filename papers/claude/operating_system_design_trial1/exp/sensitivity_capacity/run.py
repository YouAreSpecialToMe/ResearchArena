"""Run capacity sensitivity experiments."""
import sys
sys.path.insert(0, '../..')
from run_experiments import run_experiment_batch, save_results, SEEDS

capacity_ratios = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
cap_traces = ['regular', 'gcc_like']
cap_policies = ['lru_reactive', 'alto_like', 'markovtier', 'oracle']
experiments = []
for ratio in capacity_ratios:
    for policy in cap_policies:
        for trace in cap_traces:
            for seed in SEEDS:
                experiments.append((policy, trace, seed, '../../traces', ratio, None))

results = run_experiment_batch(experiments, max_workers=2, desc="Capacity sensitivity")
save_results(results, 'results.json')
