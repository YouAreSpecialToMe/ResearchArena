# Experiment Execution Guidelines

How to execute your experiment plan efficiently and rigorously.
Experiment design principles are in idea_guidelines.md — you should
have already applied them when creating plan.json.

## Phase 1: Resource Usage

### Use ALL available GPUs
Check how many GPUs you have (`nvidia-smi`) and distribute work across them.
Use `CUDA_VISIBLE_DEVICES=N` to pin different experiments to different GPUs:
```bash
CUDA_VISIBLE_DEVICES=0 python exp/baseline1/run.py &
CUDA_VISIBLE_DEVICES=1 python exp/baseline2/run.py &
CUDA_VISIBLE_DEVICES=2 python exp/method/run.py &
# ... etc
wait  # wait for all to finish
```
If one experiment only needs 1 GPU, you can run 8 experiments in parallel.

### Parallelize independent experiments
If experiments don't depend on each other (different seeds, baselines,
ablations), run them in parallel across GPUs or using `subprocess` / shell `&`.

### Estimate runtime first
Time a single short run on one GPU and extrapolate. Divide by the number
of GPUs for parallel runtime. If your estimate STILL exceeds the budget
after parallelization, then reduce scope.

### Do NOT scope down prematurely
If your pilot shows a single experiment takes N minutes on 1 GPU, and you
have K GPUs, total parallel runtime is roughly N/K — check this before
dropping experiments from the plan.

### Use all available GPU memory
If your model only uses 10GB of a 48GB GPU, consider running multiple
experiments simultaneously on the same GPU, or increasing batch size.

### Prioritize
Run the most important experiments first (method vs. strongest baseline).
If time runs out, you'll at least have the core comparison.

## Phase 2: Implementation

### General principles
- Start simple. Get a minimal version working end-to-end first.
- Add complexity one piece at a time. Evaluate each change independently.
- Copy proven implementations from related papers before writing from scratch.
- Use well-tested libraries (PyTorch, HuggingFace, scikit-learn, etc.)
- Fix random seeds for reproducibility.

### If training models
- Verify loss at initialization matches theory (e.g., -log(1/n_classes) for softmax)
- Overfit a single batch first — if you can't, your code has a bug
- Turn off regularization initially (no dropout, augmentation, weight decay)
- Use Adam optimizer with lr=3e-4 as a starting point
- Use random search over grid search for hyperparameters
- Disable learning rate decay until final tuning

### If doing analysis/probing
- Clearly document what you're measuring and why
- Use controlled setups where possible (synthetic data with known properties)
- Verify your measurement tool doesn't interfere with what you're measuring

### If doing systems experiments
- Run multiple times to account for variance
- Report median and percentiles, not just mean
- Warm up the system before measuring (avoid cold-start effects)
- Control for background processes, other workloads, thermal throttling

## Phase 3: Workspace Structure

Organize experiments so that each step has its own folder with code, results,
and logs. This makes it easy to verify which code produced which results.

```
exp/
├── <experiment_name>/              # one folder per experiment/condition
│   ├── run.py                      # experiment script
│   ├── config.yaml (or .json)      # hyperparameters, settings
│   ├── results.json                # per-experiment results
│   └── logs/                       # training/eval logs, stdout
│
├── <baseline_name>/
│   ├── run.py
│   ├── results.json
│   └── logs/
│
├── <ablation_name>/
│   ├── run.py
│   ├── results.json
│   └── logs/
│
└── shared/                         # shared utilities across experiments
    ├── data_loader.py              # data loading, preprocessing
    ├── metrics.py                  # evaluation metrics
    ├── models.py                   # model definitions
    └── utils.py                    # common helpers

data/                               # downloaded/processed datasets
figures/                            # generated figures for the paper
```

### Per-experiment results

Each `exp/<name>/results.json` should capture that experiment's output:
```json
{
  "experiment": "<name>",
  "metrics": {"metric1": {"mean": 0.87, "std": 0.002}, ...},
  "config": {"lr": 0.001, "epochs": 50, "seed": [42, 123, 456], ...},
  "runtime_minutes": 45
}
```

### Figures

Save publication-ready figures to `figures/`:
- Comparison plots (your method vs baselines)
- Ablation charts (impact of each component)
- Training curves (loss/metric over epochs)
- Analysis visualizations (distributions, embeddings, etc.)

Each figure should be self-contained with axis labels, legends, and titles.

## Phase 4: Plan Compliance

Follow plan.json step by step:
- Execute every step in order
- Create a subfolder under `exp/` for each plan step
- If a step is infeasible, document why in that step's folder (create a
  `SKIPPED.md` with the reason) and move on
- After all steps, verify that the plan's success criteria are met
- If results contradict the hypothesis, report this honestly — negative
  results with good analysis are valuable

## Reproducibility Checklist

Before finishing, verify:
- [ ] Fixed random seeds used throughout
- [ ] At least 2 meaningful baselines compared fairly
- [ ] Results from 3+ runs with mean +/- std
- [ ] Ablation study for each novel component
- [ ] No data leakage (verified)
- [ ] All configuration documented in per-experiment results
- [ ] Each experiment has its own folder under exp/ with code and results
- [ ] Figures saved for key results
- [ ] Negative results reported honestly (if any)
- [ ] All available GPUs utilized
