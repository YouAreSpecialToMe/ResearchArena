# Experiment Guidelines

Distilled from Karpathy's Recipe for Training Neural Networks, Google's Rules of ML,
NeurIPS reproducibility checklist, and REFORMS consensus framework.

## Phase 1: Data & Setup

- Spend time exploring your data before writing model code. Visualize distributions,
  check for class imbalances, corrupted labels, duplicates.
- Use standard benchmarks when possible — they make comparison easier.
- Document: data source, collection method, train/val/test split sizes,
  class distributions, any preprocessing applied.
- Keep raw data intact. Create derived versions separately.

## Phase 2: Baseline First

- Start with the simplest possible baseline (linear model, random, majority class).
- Get it running end-to-end before building anything complex.
- Copy proven architectures from related papers before inventing new ones.
- Verify your training loop:
  - Loss at initialization should match theory (e.g., -log(1/n_classes) for softmax)
  - Overfit a single batch of 2-3 examples to zero loss
  - If you can't overfit a tiny batch, your code has a bug
- Turn off all regularization initially (no dropout, no augmentation, no weight decay).

## Phase 3: Your Method

- Add complexity one component at a time. Evaluate each change independently.
- If your method has multiple novel components, verify each one matters
  before combining them.
- Use Adam optimizer with lr=3e-4 as a starting point.
- Disable learning rate decay until final tuning.
- Use random search over grid search for hyperparameters.

## Phase 4: Rigorous Evaluation

### Multiple Seeds (non-negotiable)
- Run every experiment with at least 3 different random seeds.
- Report mean +/- standard deviation across seeds.
- Use the SAME seeds for your method and all baselines (paired comparison).
- Never report best-of-N runs — always report the average.

### Baselines
- Include at least 2 meaningful baselines.
- One should be simple (linear, random, majority class).
- One should be a recent published method.
- Train baselines with equivalent effort (same compute budget, same tuning).

### Ablation Studies (required)
- Remove each novel component one at a time.
- Show quantitative impact: "without component X, performance drops from Y to Z."
- This proves every part of your method contributes.

### Statistical Significance
- Report 95% confidence intervals when claiming one method beats another.
- If the confidence intervals overlap, you cannot claim superiority.

## Phase 5: Common Pitfalls to Avoid

- DO NOT tune hyperparameters on the test set. Use a validation set.
- DO NOT compare against baselines with different preprocessing.
- DO NOT report only the metric where your method wins — report all standard
  metrics for your task.
- DO NOT forget to check for data leakage: ensure no test data is seen
  during training or preprocessing.
- DO NOT use batch normalization statistics from the training set at test
  time without proper handling.

## Phase 6: What to Save

Save everything needed to reproduce and write the paper:

```
results.json          # structured results (see format below)
figures/              # training curves, comparison plots, ablation charts
```

### results.json format

```json
{
  "method": {
    "metric1": {"mean": 0.8734, "std": 0.0021},
    "metric2": {"mean": 0.8521, "std": 0.0034}
  },
  "baselines": {
    "baseline_name_1": {
      "metric1": {"mean": 0.8102, "std": 0.0018},
      "metric2": {"mean": 0.7856, "std": 0.0042}
    },
    "baseline_name_2": { ... }
  },
  "ablations": {
    "without_component_A": {
      "metric1": {"mean": 0.8401, "std": 0.0025}
    },
    "without_component_B": { ... }
  },
  "config": {
    "dataset": "dataset_name",
    "seeds": [42, 123, 456],
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.0003,
    "optimizer": "Adam",
    "hardware": "1x NVIDIA GPU",
    "training_time_minutes": 45
  }
}
```

## Reproducibility Checklist

Before finishing, verify:
- [ ] Fixed random seeds used throughout
- [ ] Loss at initialization verified
- [ ] At least 2 baselines compared
- [ ] Results from 3+ seeds with mean +/- std
- [ ] Ablation study for each novel component
- [ ] No data leakage (preprocessing uses only training data)
- [ ] All hyperparameters documented in config
- [ ] Training curves saved as figures
