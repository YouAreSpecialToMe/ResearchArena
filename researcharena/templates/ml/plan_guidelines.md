# Experiment Plan Guidelines

How to design a rigorous experiment plan based on your research proposal.

Read proposal.md and idea.json first — your plan must test the claims
and hypothesis described there.

## Plan Format

Save your plan as plan.json — a JSON array of experiment steps:
```json
[
  {
    "category": "<category>",
    "title": "short descriptive title",
    "description": "what this step does and why",
    "steps": {
      "step1": "detailed instruction with specifics",
      "step2": "...",
      ...
    }
  },
  ...
]
```

Suggested categories (add your own as needed):
- **Environment Configuration** — dependencies, setup
- **Data Preparation** — download, preprocess, splits, statistics
- **Baseline Experiment** — existing methods to compare against
- **Main Experiment** — your proposed method
- **Analysis Experiment** — ablations, robustness, sensitivity
- **Effectiveness Evaluation** — success criteria, statistical tests
- **Visualization** — figures, tables, plots for the paper

## Experiment Design Principles

### Formulate testable claims
- State your hypothesis as a testable claim
- Design experiments that could fail — if they can't produce a negative
  result, they're not informative
- Define what would DISPROVE your claim

### Choose the right experiment type

| Claim type | Experiment type | What to measure |
|---|---|---|
| "Our method outperforms X" | Empirical comparison | Metrics on shared benchmarks |
| "Component A is critical" | Ablation study | Performance with/without A |
| "This scales better" | Scaling experiment | Performance vs. data/compute/params |
| "Our theory predicts X" | Theoretical validation | Synthetic setup with known ground truth |
| "This property holds" | Analysis/probing | Measurements on existing models/data |
| "This is faster/cheaper" | Systems experiment | Latency, throughput, memory, FLOPs |
| "This benchmark is better" | Benchmark evaluation | Existing methods on new benchmark |

### Select metrics carefully
- Use standard metrics for your task
- Report ALL standard metrics, not just the one where you win
- Consider both performance AND cost (FLOPs, latency, memory)

### Choose datasets that test your claim
- Use standard benchmarks when possible
- If claiming robustness, test on distribution-shifted data
- If claiming generalization, test on multiple datasets
- Document: data source, size, splits, preprocessing

### Select baselines fairly
- At least 2 meaningful baselines (one simple, one strong/recent)
- Run all baselines with equivalent effort (same compute, same tuning)
- Never compare against intentionally weak baselines

### Plan ablation studies
- For each novel component, plan to remove it and measure impact
- Plan ablations BEFORE running experiments, not after seeing results

### Think about confounders
- Could the improvement come from more parameters/data/compute?
- Are comparisons fair (same preprocessing, splits, compute budget)?
- If using published baselines, are the setups truly comparable?

### Rigorous evaluation
- At least 3 different random seeds with mean ± std
- Use the SAME seeds for method and all baselines (paired comparison)
- Report 95% confidence intervals when claiming superiority
- Avoid data leakage (preprocessing stats from train only, etc.)
- Test set used ONCE for final evaluation, not for model selection

### Common pitfalls to avoid
- Don't tune hyperparameters on the test set
- Don't compare against baselines with different preprocessing or splits
- Don't report only the metric where you win
- Don't claim SOTA without comparing against actual SOTA methods
- Don't ignore negative results — report them honestly
- Don't use a single train/test split
- Don't assume deep learning is always better — test simpler alternatives

## Plan Quality Checklist

Before finalizing plan.json, verify:
- [ ] Each step has a clear category, title, description, and sub-steps
- [ ] Sub-steps are detailed enough to follow without ambiguity
- [ ] Specific datasets, metrics, and hyperparameters are named
- [ ] At least 2 meaningful baselines are included
- [ ] Ablation studies planned for each novel component
- [ ] At least 3 random seeds specified
- [ ] Success criteria clearly defined
- [ ] Plan is feasible within the resource constraints
- [ ] Total runtime estimate accounts for parallel GPU usage
- [ ] Visualization step included for paper figures
