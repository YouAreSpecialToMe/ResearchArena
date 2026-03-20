# Idea Generation Guidelines

How to go from a seed field to a novel, feasible research idea.

Distilled from John Schulman's "Opinionated Guide to ML Research", the
ResearchAgent methodology, "Can LLMs Generate Novel Research Ideas?" (Si et al.),
and standard academic research practices.

## Step 1: Explore the field

Start by understanding what already exists. DO NOT skip this step.

### Search for existing work (newest to oldest)
- Search arXiv (arxiv.org), Semantic Scholar (semanticscholar.org), and
  Google Scholar (scholar.google.com) for papers in your seed field
- **Start with the newest papers first** — sort by date, read the most
  recent work before going to older foundational papers. This ensures
  you know the current frontier before proposing something new.
- Recommended search order:
  1. Last 6 months — what's happening right now?
  2. Last 1-2 years — what are the current state-of-the-art methods?
  3. Foundational papers — what are the classic approaches?
- Look for:
  - Survey papers — they summarize the landscape and list open problems
  - Highly-cited recent papers — they define the current state of the art
  - Workshop papers — they often contain early-stage ideas and emerging trends

### Build a mental map
- What are the main approaches in this area?
- What are the established benchmarks and metrics?
- What are the known limitations of current methods?
- What problems are considered "open" or "unsolved"?
- What recent techniques from OTHER fields could apply here?

### Find the gaps
- Read the "Limitations" and "Future Work" sections of recent papers
- Look for recurring complaints in reviews (on OpenReview, if available)
- Identify assumptions that current methods make — can you relax them?
- Look for problems where simple baselines still perform surprisingly well
  (this signals the community hasn't cracked it yet)

## Step 2: Generate candidate ideas

### Two approaches (choose one or combine)

**Goal-driven** (recommended): Start with a problem you want to solve.
- "Current methods for X fail when Y happens. How can we fix that?"
- "Task Z requires too much labeled data. Can we do it with less?"
- The goal constrains your search and makes the contribution clear.

**Idea-driven**: Start with a technique and find where it applies.
- "Technique A works well for B. Could it also work for C?"
- Riskier — you may find the idea already exists or doesn't work.

### What makes a good research idea
- **Novel**: Not already done. You MUST verify this (Step 3).
- **Feasible**: Can be implemented and tested within your resource constraints.
- **Clear**: The contribution is easy to explain in one sentence.
- **Testable**: There's a concrete way to evaluate whether it works.
- **Significant**: If it works, the community would care.

### What makes a BAD research idea
- Too broad ("improve NLP") — needs a specific problem and approach
- Too incremental ("change hyperparameter X from 0.1 to 0.01")
- Not verifiable (no way to test if it worked)
- Requires resources you don't have (100 GPUs, proprietary data)
- Already exists (you didn't check the literature)

## Step 3: Verify novelty (CRITICAL — do not skip)

Before committing to an idea, verify it hasn't been done:

### Search specifically for your idea
- Search Semantic Scholar and arXiv with keywords from your proposed method
- Search for the PROBLEM you're solving, not just your approach
- Check if your idea is a special case of something more general that exists
- Look at the "Related Work" sections of papers closest to your idea

### Common novelty traps
- Your idea exists but under a different name (jargon varies across subfields)
- Your idea was tried and didn't work (check for negative results too)
- Your idea is a minor variation of an existing approach
- A concurrent paper (posted in the last few months) does the same thing

### If your idea already exists
- DON'T give up immediately. Ask: can you improve on it? Apply it to a
  new domain? Combine it with something else? Scale it up?
- If it truly exists with no room for improvement, go back to Step 2

## Step 4: Produce structured outputs

You must produce FOUR outputs. Each serves a different purpose:

### 4.1 proposal.md — Research Proposal

A thorough document with these sections:
- **Introduction**: Context, problem statement, key insight, hypothesis
- **Proposed Approach**: Overview, method details, key innovations
- **Related Work**: Key papers, how your idea differs, positioning
- **Experiments**: Planned setup, benchmarks, metrics, expected results
- **Success Criteria**: What would confirm or refute your hypothesis
- **References**: Full citation list (all must be real, verifiable papers)

### 4.2 plan.json — Experiment Plan

A JSON array of experiment steps that will be followed in the experiment stage:
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

The plan should be comprehensive enough to produce a publishable paper.
Each step must be detailed enough to follow without ambiguity — include
specific datasets, model architectures, hyperparameters, evaluation metrics,
and expected output formats.

**When designing your experiment plan, apply these principles:**

**Formulate testable claims:**
- State your hypothesis as a testable claim
- Design experiments that could fail — if they can't produce a negative
  result, they're not informative
- Define what would DISPROVE your claim

**Choose the right experiment type:**

| Claim type | Experiment type | What to measure |
|---|---|---|
| "Our method outperforms X" | Empirical comparison | Metrics on shared benchmarks |
| "Component A is critical" | Ablation study | Performance with/without A |
| "This scales better" | Scaling experiment | Performance vs. data/compute/params |
| "Our theory predicts X" | Theoretical validation | Synthetic setup with known ground truth |
| "This property holds" | Analysis/probing | Measurements on existing models/data |

**Select metrics carefully:**
- Use standard metrics for your task
- Report ALL standard metrics, not just the one where you win
- Consider both performance AND cost (FLOPs, latency, memory)

**Choose datasets that test your claim:**
- Use standard benchmarks when possible
- If claiming robustness, test on distribution-shifted data
- If claiming generalization, test on multiple datasets

**Select baselines fairly:**
- At least 2 meaningful baselines (one simple, one strong/recent)
- Run all baselines with equivalent effort
- Never compare against intentionally weak baselines

**Plan ablation studies:**
- For each novel component, plan to remove it and measure impact
- Plan ablations BEFORE running experiments

**Think about confounders:**
- Could the improvement come from more parameters/data/compute?
- Are comparisons fair (same preprocessing, splits, compute budget)?

**Rigorous evaluation:**
- At least 3 different random seeds with mean ± std
- Use the SAME seeds for method and all baselines
- Report 95% confidence intervals when claiming superiority
- Avoid data leakage (preprocessing stats from train only, etc.)

**Common pitfalls to avoid in your plan:**
- Don't tune hyperparameters on the test set
- Don't report only the metric where you win
- Don't ignore negative results
- Don't use a single train/test split

### 4.3 idea.json — Structured Summary

A JSON object with at least these fields:
- **description**: 1-3 sentences explaining what you're proposing
- **title**: paper title
- **motivation**: why this problem matters, what gap you're filling
- **proposed_approach**: your high-level method and why it should work
- **related_work**: key existing papers and how your idea differs
  (use REAL papers you found in Steps 1 and 3 — include titles and authors)
- **hypothesis**: testable hypothesis
- **success_criteria**: what would confirm/refute the hypothesis

### 4.4 references/ — Parsed Reference Papers

Create a directory with key reference papers. For each paper, create a
subdirectory containing the paper's content parsed into sections:
```
references/
├── Paper-Title-One/
│   ├── meta/
│   │   ├── meta_info.txt       # title, authors, venue, year, URL
│   │   └── bibtex.txt          # BibTeX entry
│   └── sections/
│       ├── abstract.md
│       ├── 1 Introduction.md
│       ├── 2 Related Work.md
│       └── ...
├── Paper-Title-Two/
│   └── ...
```

This grounds your proposal in real literature and ensures references are
verifiable by reviewers.

### Sanity checks before moving on
- Can you explain the idea in one sentence to a non-expert?
- Is there a clear experiment that would test the idea?
- Do you have the resources (data, compute, time) to do it?
- Is the expected contribution large enough for a paper?
- Is the experiment plan detailed enough to follow step by step?
- Are all references real, verifiable publications?

## General principles

### From John Schulman
- Your ability to choose the right problem is more important than raw skill
- Watch which ideas prosper and which are forgotten — this develops taste
- Goal-driven research has lower scooping risk than idea-driven research
- There's no shame in working on ideas suggested by others or by the literature

### From "Can LLMs Generate Novel Research Ideas?" (Si et al.)
- AI-generated ideas tend to be novel but lack feasibility — ground yours
  in practical constraints
- Vague implementation details are the #1 weakness — be specific about how
  your method actually works
- Missing baselines and unrealistic assumptions are common failures
- Verify your idea against existing work — 80% of reviewer rejections
  cite existing papers that the authors missed

### From ResearchAgent (Baek et al.)
- Connect ideas across papers, not just within one paper
- Look for shared concepts across different subfields
- Iterative refinement improves idea quality — but diminishing returns
  after 2-3 rounds
- Both citation relationships AND underlying concepts matter for novelty
