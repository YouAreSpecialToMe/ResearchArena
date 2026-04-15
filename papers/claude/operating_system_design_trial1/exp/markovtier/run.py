"""Run MarkovTier experiments."""
import sys
sys.path.insert(0, '../..')
from run_experiments import run_experiment_batch, save_results, SEEDS, TRACE_TYPES, ADVERSARIAL_TYPES

all_traces = TRACE_TYPES + ADVERSARIAL_TYPES
experiments = []
for trace in all_traces:
    for seed in SEEDS:
        experiments.append(('markovtier', trace, seed, '../../traces', 0.5, None))

results = run_experiment_batch(experiments, max_workers=2, desc="MarkovTier")
save_results(results, 'results.json')
