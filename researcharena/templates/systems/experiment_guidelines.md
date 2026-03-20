# Experiment Guidelines

Distilled from Michael Lones' "How to Avoid ML Pitfalls", NeurIPS reproducibility
checklist, REFORMS consensus framework, and Google's Rules of ML.

## Phase 1: Experiment Design (do this BEFORE writing code)

### 1.1 Formulate your claim

Before any implementation, write down:
- **What is your hypothesis?** State it as a testable claim.
  Example: "Method X improves accuracy over baseline Y on task Z because of property W."
- **What evidence would convince a skeptical reader?**
- **What would DISPROVE your claim?** Design experiments that could fail — if your
  experiment cannot possibly produce a negative result, it's not informative.

### 1.2 Choose the right experiment type

Not all research requires training a model. Choose what fits your claim:

| Claim type | Experiment type | What to measure |
|---|---|---|
| "Our method outperforms X" | Empirical comparison | Metrics on shared benchmarks |
| "Component A is critical" | Ablation study | Performance with/without A |
| "This scales better" | Scaling experiment | Performance vs. data/compute/params |
| "Our theory predicts X" | Theoretical validation | Synthetic setup with known ground truth |
| "This property holds" | Analysis/probing | Measurements on existing models/data |
| "This is faster/cheaper" | Systems experiment | Latency, throughput, memory, FLOPs |
| "This benchmark is better" | Benchmark evaluation | Existing methods on new benchmark |
| "This failure mode exists" | Failure analysis | Controlled examples that trigger failure |

### 1.3 Select what to measure

- Use standard metrics for your task (accuracy, F1, BLEU, FID, perplexity, etc.)
- Report ALL standard metrics, not just the one where you win
- If you propose a new metric, also report standard ones for comparison
- Consider both performance AND cost: FLOPs, latency, memory, training time
- For systems claims, report percentiles (p50, p95, p99), not just mean

### 1.4 Choose datasets that test your claim

- Use standard benchmarks when possible — they enable comparison with published work
- Choose datasets that are relevant to your claim, not just convenient
- If your claim is about robustness, test on distribution-shifted data
- If your claim is about efficiency, test at multiple scales
- If your claim is about generalization, test on multiple datasets
- Document: data source, size, splits, preprocessing, any filtering applied

### 1.5 Select baselines fairly

- Include at least 2 meaningful baselines:
  - One simple baseline (random, majority class, linear model, naive approach)
  - One strong baseline (recent published method or established approach)
- Run all baselines with equivalent effort (same compute, same tuning)
- If a baseline is too expensive to run yourself, cite published numbers
  and clearly state you didn't rerun it
- Never compare against intentionally weak baselines to inflate your results

### 1.6 Plan ablation studies

- For each novel component in your method, plan to remove it and measure impact
- If your contribution is a single technique, vary its key parameters instead
- Plan which components to ablate BEFORE running experiments, not after seeing results

### 1.7 Think about confounders

- What else could explain your results besides your method?
- Are you comparing with the same preprocessing, data splits, and compute budget?
- Could the improvement come from more parameters, more data, or more compute
  rather than from your actual contribution?
- If using published baselines, are the setups truly comparable?

## Phase 2: Implementation

### Efficient use of resources
You have a fixed time budget and compute resources — use them wisely:
- **Use ALL available GPUs.** Check how many GPUs you have (e.g., `nvidia-smi`)
  and distribute work across them. Use `CUDA_VISIBLE_DEVICES=N` to pin
  different experiments to different GPUs. For example, with 8 GPUs:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python exp/baseline1/run.py &
  CUDA_VISIBLE_DEVICES=1 python exp/baseline2/run.py &
  CUDA_VISIBLE_DEVICES=2 python exp/method/run.py &
  # ... etc
  wait  # wait for all to finish
  ```
  If one experiment only needs 1 GPU, you can run 8 experiments in parallel.
- **Parallelize independent experiments.** If experiments don't depend on each
  other (e.g., different seeds, different baselines, different ablations),
  run them in parallel across GPUs or using `subprocess` / shell `&`.
- **Estimate runtime first.** Before launching the full experiment suite, time
  a single short run on one GPU and extrapolate. Divide by the number of GPUs
  for parallel runtime. If your estimate STILL exceeds the budget after
  parallelization, reduce scope.
- **Use all available GPU memory.** If your model only uses 10GB of a 48GB GPU,
  consider running multiple experiments simultaneously on the same GPU, or
  increasing batch size for faster convergence.
- **Prioritize.** Run the most important experiments first (method vs. strongest
  baseline). If time runs out, you'll at least have the core comparison.
- **Do NOT scope down prematurely.** If your pilot shows a single experiment
  takes N minutes on 1 GPU, and you have K GPUs, total parallel runtime is
  roughly N/K — check this before dropping experiments from the plan.

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

## Phase 3: Rigorous Evaluation

### Multiple runs (non-negotiable)
- Run every experiment with at least 3 different random seeds
- Report mean +/- standard deviation across runs
- Use the SAME seeds for your method and all baselines (paired comparison)
- Never report best-of-N runs — always report the average

### Ablation studies (required)
- Remove each novel component one at a time
- Show quantitative impact: "without component X, metric drops from Y to Z"
- This proves every part of your method contributes

### Statistical significance
- Report 95% confidence intervals when claiming superiority
- If confidence intervals overlap, you cannot claim your method is better
- For multiple comparisons, apply correction (Bonferroni or similar)
- Distinguish statistical significance from practical significance

### Avoid data leakage (REFORMS checklist)
- Preprocessing statistics (mean, std, scaling) computed from training data ONLY
- Feature selection done on training data ONLY, not full dataset
- Data augmentation applied AFTER train/test split, not before
- For time series, use temporal splits (no future data in training)
- Test set used ONCE for final evaluation, not for iterative model selection

## Phase 4: Common Pitfalls

From Michael Lones' "How to Avoid ML Pitfalls":

- DO NOT tune hyperparameters on the test set — use a validation set
- DO NOT compare against baselines with different preprocessing or splits
- DO NOT report only the metric where your method wins
- DO NOT claim SOTA without comparing against actual SOTA methods
- DO NOT treat benchmark results as ground truth — small improvements may be noise
- DO NOT ignore negative results — report them honestly with analysis
- DO NOT draw conclusions beyond your tested conditions
- DO NOT assume deep learning is always better — test simpler alternatives too
- DO NOT use a single train/test split — use cross-validation or multiple seeds
- DO NOT forget to inspect your model — verify it learns meaningful patterns,
  not spurious correlations

## Phase 5: Workspace Structure

Organize experiments so that each step has its own folder with code, results,
and logs. This makes it easy to verify which code produced which results and
ensures reproducibility.

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
results.json                        # aggregated final results (see below)
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

### Aggregated results.json (workspace root)

After all experiments complete, compile a summary `results.json` at the
workspace root that aggregates across all experiments:

```json
{
  "method": {
    "metric1": {"mean": 0.8734, "std": 0.0021},
    "metric2": {"mean": 0.8521, "std": 0.0034}
  },
  "baselines": {
    "baseline_name_1": {
      "metric1": {"mean": 0.8102, "std": 0.0018}
    }
  },
  "ablations": {
    "without_component_A": {
      "metric1": {"mean": 0.8401, "std": 0.0025}
    }
  },
  "config": {
    "experiment_type": "empirical_evaluation",
    "dataset": "dataset_name",
    "seeds": [42, 123, 456],
    "hardware": "1x GPU",
    "total_runtime_minutes": 120
  }
}
```

Adapt the structure to your experiment type. The key requirement:
structured, machine-readable, complete, and honest.

### Figures

Save publication-ready figures to `figures/`:
- Comparison plots (your method vs baselines)
- Ablation charts (impact of each component)
- Training curves (loss/metric over epochs)
- Analysis visualizations (distributions, embeddings, etc.)

Each figure should be self-contained with axis labels, legends, and titles.

## Phase 6: Plan Compliance

If you have a `plan.json` from the ideation stage:
- Execute every step in order
- Create a subfolder under `exp/` for each plan step
- If a step is infeasible, document why in that step's folder (create a
  `SKIPPED.md` with the reason) and move on
- After all steps, verify that the plan's success criteria are met
- If results contradict the hypothesis, report this honestly — negative
  results with good analysis are valuable

## Reproducibility Checklist

Before finishing, verify:
- [ ] Claim is clearly stated and testable
- [ ] Experiment type matches the claim
- [ ] Fixed random seeds used throughout
- [ ] At least 2 meaningful baselines compared fairly
- [ ] Results from 3+ runs with mean +/- std
- [ ] Ablation study for each novel component
- [ ] No data leakage (verified)
- [ ] All configuration documented in results.json
- [ ] Each experiment has its own folder under exp/ with code and results
- [ ] Figures saved for key results
- [ ] Negative results reported honestly (if any)
- [ ] Aggregated results.json at workspace root matches per-experiment results
