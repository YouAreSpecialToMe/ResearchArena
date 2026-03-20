# Experiment Execution Guidelines

How to execute your experiment plan efficiently and rigorously.
Experiment design principles are in plan_guidelines.md — you should
have already applied them when creating plan.json.

## Phase 1: Maximize Resource Usage

Your goal is to use ALL available resources efficiently. Check what you
have before starting:
```bash
nvidia-smi          # GPUs: count, memory, current usage
nproc               # CPU cores
free -h             # RAM
```

### Parallel execution strategy

Identify independent experiments in your plan (different seeds, baselines,
ablations, datasets) and run them simultaneously:

```bash
# Example: 8 GPUs, 6 independent experiments
CUDA_VISIBLE_DEVICES=0 python exp/baseline1/run.py &
CUDA_VISIBLE_DEVICES=1 python exp/baseline2/run.py &
CUDA_VISIBLE_DEVICES=2 python exp/method/run.py &
CUDA_VISIBLE_DEVICES=3 python exp/ablation1/run.py &
CUDA_VISIBLE_DEVICES=4 python exp/ablation2/run.py &
CUDA_VISIBLE_DEVICES=5 python exp/dataset2/run.py &
wait  # wait for all to finish
```

For CPU-bound work (data preprocessing, evaluation, API calls):
```bash
# Run multiple CPU tasks in parallel
python exp/preprocess_dataset1.py &
python exp/preprocess_dataset2.py &
python exp/preprocess_dataset3.py &
wait
```

### GPU utilization
- **Pin experiments to GPUs** with `CUDA_VISIBLE_DEVICES=N`
- If a model uses only part of GPU memory, run multiple experiments
  per GPU (e.g., 2 small models on one 48GB GPU)
- Increase batch size to fill GPU memory — larger batches = faster training
- For inference-only experiments (embedding, evaluation), consider
  running several on the same GPU

### CPU utilization
- Use `multiprocessing` or `concurrent.futures.ProcessPoolExecutor` for
  CPU-bound data processing
- Parallelize data loading with `num_workers` in PyTorch DataLoaders
- For API-based experiments (LLM scoring), use `asyncio` or thread pools
  to make concurrent API calls

### Follow the plan — do not scope down
Your plan.json was designed with the available resources in mind.
Execute ALL steps. If a step truly cannot run (dependency failure,
out-of-memory), document it in that step's `SKIPPED.md` and move on.
Do NOT drop experiments just because a single-GPU pilot looks slow —
use parallel execution across all GPUs.

### Prioritize execution order
Run experiments in dependency order:
1. Data preparation (must finish first)
2. Baselines + method (can run in parallel)
3. Ablations (can run in parallel after method works)
4. Analysis + visualization (after results are in)

## Phase 2: Workspace Structure

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
  "config": {"lr": 0.001, "epochs": 50, "seed": 42, ...},
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
- [ ] Fixed random seed used for reproducibility
- [ ] Ablation study for each novel component
- [ ] No data leakage (verified)
- [ ] All configuration documented in per-experiment results
- [ ] Each experiment has its own folder under exp/ with code and results
- [ ] Figures saved for key results
- [ ] Negative results reported honestly (if any)
- [ ] All available GPUs utilized
