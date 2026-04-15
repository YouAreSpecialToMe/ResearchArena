"""Run confidence threshold and migration budget sensitivity experiments."""
import sys
sys.path.insert(0, '../..')
from run_experiments import run_experiment_batch, save_results, SEEDS

conf_thresholds = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
budgets = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
traces = ['regular', 'semi_regular', 'irregular']

# Confidence sweep
conf_experiments = []
for thresh in conf_thresholds:
    for trace in traces:
        for seed in SEEDS:
            conf_experiments.append(('markovtier', trace, seed, '../../traces', 0.5,
                                    {'confidence_threshold': thresh}))

# Budget sweep
budget_experiments = []
for budget in budgets:
    for trace in traces:
        for seed in SEEDS:
            budget_experiments.append(('markovtier', trace, seed, '../../traces', 0.5,
                                      {'anticipatory_budget': budget}))

results = run_experiment_batch(conf_experiments + budget_experiments, max_workers=2,
                              desc="Sensitivity sweep")
save_results(results, 'results.json')
