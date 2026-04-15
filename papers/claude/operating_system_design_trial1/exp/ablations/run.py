"""Run ablation study experiments."""
import sys
sys.path.insert(0, '../..')
from run_experiments import run_experiment_batch, save_results, SEEDS, TRACE_TYPES

ablation_policies = ['markovtier_phase_only', 'markovtier_no_confidence',
                    'markovtier_no_rollback', 'markovtier_random']
experiments = []
for policy in ablation_policies:
    for trace in TRACE_TYPES:
        for seed in SEEDS:
            experiments.append((policy, trace, seed, '../../traces', 0.5, None))

results = run_experiment_batch(experiments, max_workers=2, desc="Ablations")
save_results(results, 'results.json')
