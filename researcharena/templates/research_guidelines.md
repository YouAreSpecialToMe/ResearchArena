# Research Guidelines

These guidelines are distilled from top ML conference reviewer criteria (NeurIPS,
ICML, ICLR), and research methodology guides by Simon Peyton Jones, Andrej Karpathy,
John Schulman, and the REFORMS checklist.

## Idea & Problem Selection

- Clearly state the problem, population, and why ML is the right approach
- Prefer goal-driven research over idea-driven: ask "what problem needs solving?"
  then find the method, not the other way around
- Your idea needs a clear hypothesis that can be tested experimentally
- Check if the idea already exists: search Semantic Scholar, arXiv, Google Scholar
  before committing to an approach

## Experimental Design

### Baselines & Comparisons
- Compare against at least 2 established baselines
- Include a simple/trivial baseline (e.g., random, majority class, linear model)
- Use published conference/journal versions of baselines, not just arXiv preprints
- Ensure fair comparison: same data splits, same compute budget where possible

### Ablation Studies (required)
- Remove each component of your method one at a time
- Show quantitative impact of each component
- This proves each part of your method contributes meaningfully

### Reproducibility
- Fix random seeds
- Document: hardware specs, software versions, hyperparameters
- Provide an automated script that reproduces all results
- Report total runtime and compute requirements

### Hyperparameter Tuning
- Use random search over grid search for high-dimensional spaces
- For neural networks: start with Adam optimizer, lr=3e-4
- Disable learning rate decay until final tuning phase
- Don't plug multiple changes simultaneously — isolate each variable

### Results Integrity (CRITICAL)
- ALL numbers in the paper MUST come directly from actual experiment runs
- DO NOT fabricate, inflate, or manually adjust experimental results
- DO NOT hardcode or invent numbers in results.json — every value must be
  produced by actually running the experiment code
- The review system cross-checks results.json against experiment logs.
  If results cannot be traced back to actual code execution, the paper
  is automatically rejected
- If your method doesn't beat the baseline, report that honestly.
  Negative results with honest analysis are acceptable; fabricated
  positive results are not

### Statistical Rigor
- ALWAYS include error bars (standard error of the mean, not just std dev)
- Run experiments with at least 3-5 different random seeds
- Report mean ± std across runs
- Use statistical significance tests when claiming one method beats another

### Data
- Use standard benchmarks when possible (easier to compare)
- Report dataset sizes, class distributions, train/val/test splits
- Document any preprocessing or data cleaning steps
- Inspect your data manually before writing model code

### Model Development Process
- Start simple: get a trivial baseline working first
- Copy proven architectures from related papers before inventing new ones
- Verify loss at initialization matches theory (e.g., -log(1/n) for softmax)
- Overfit a single batch first to verify the training loop works
- Add complexity incrementally, not all at once

## Writing the Paper

### Structure
1. Abstract (1 paragraph, 6-7 sentences max)
2. Introduction (motivation, contributions list)
3. Related Work (funnel: broad → narrow, end with your positioning)
4. Method (detailed, reproducible description)
5. Experiments (setup, results, ablations)
6. Discussion (limitations, broader impact)
7. Conclusion
8. References

### Key Writing Rules
- State contributions explicitly as a numbered list in the introduction
- Put implementation details before results
- Every claim must be backed by evidence (numbers, figures, citations)
- Discuss limitations honestly — reviewers reward honesty, punish omission
- Use active voice and first person where appropriate

### Figures & Tables
- Write self-contained captions (readable without the main text)
- Bold the best values in comparison tables
- Include ↑/↓ arrows indicating whether higher or lower is better
- Reference and discuss every figure/table in the text

### References (CRITICAL)
- EVERY reference must be a real, verifiable publication
- Search Semantic Scholar (semanticscholar.org) or Google Scholar to find real papers
- Cite with correct titles, authors, venues, and years
- DO NOT fabricate or hallucinate citations — this causes automatic rejection
- Prefer citing published conference/journal papers over arXiv preprints
- Include 15-30 references for a typical ML paper

## Common Rejection Reasons

- Fabricated or manipulated results (automatic reject)
- Fabricated references (automatic reject)
- No ablation study
- Missing error bars / no statistical significance testing
- Unfair baseline comparisons
- Insufficient experimental rigor
- Vague or unsupported claims
- Poor writing quality
- No discussion of limitations
- Lack of novelty (incremental over existing work without sufficient justification)
