# Paper Writing Guidelines

Distilled from Simon Peyton Jones, NeurIPS/ICML/ICLR formatting requirements,
and technical writing best practices.

## Core Principle

Your paper tells a story: problem → why it matters → your approach → evidence
it works → what it means. Every section serves this narrative.

## Structure

Write in this order (not the order they appear in the paper):

1. Methods → Experiments → Contributions list → Conclusion
2. Then Introduction (now you know what to introduce)
3. Then Related Work
4. Abstract LAST (summarize the completed paper)

Final paper order:

```
1. Title
2. Abstract (150-250 words, one paragraph)
3. Introduction (problem, gap, contributions list, paper roadmap)
4. Related Work (funnel: broad → narrow, end with your positioning)
5. Method (complete, reproducible description)
6. Experiments (setup, results tables, ablations, analysis)
7. Discussion / Limitations
8. Conclusion
9. References
```

## Abstract

- ONE paragraph, 150-250 words
- Structure: context → problem → method → key result → implication
- Must be self-contained — readable without the rest of the paper
- No citations in the abstract
- Include one concrete quantitative result if possible

## Introduction

- Start with what is known (context)
- Identify the gap (what's missing or broken)
- State your approach (one sentence)
- List contributions explicitly:
  ```latex
  Our contributions are:
  \begin{itemize}
    \item We propose X, which addresses Y.
    \item We show that Z through experiments on A and B.
    \item We release our code and data at [URL].
  \end{itemize}
  ```
- End with a roadmap: "Section 2 reviews..., Section 3 describes..., Section 4 presents..."

## Related Work

- Organize by approach/concept, NOT chronologically
- Funnel structure: broad field → specific subproblem → directly competing methods
- For each group of related papers, explain:
  1. What they do
  2. How your work differs
- End with: "Unlike [prior work], our approach..."
- Every cited paper must be REAL and verifiable. Search Semantic Scholar
  (semanticscholar.org) to confirm papers exist before citing them.

## Method

- Complete enough that an expert can reimplement from the paper alone
- State all assumptions explicitly
- Include: model architecture, loss function, training algorithm
- Use clear notation, define every symbol on first use
- Include a method overview figure if the approach has multiple components

## Experiments

- Structure: Setup → Main results → Ablations → Analysis

### Setup subsection
- Datasets: name, size, splits, preprocessing
- Baselines: what they are, why chosen, how trained (fair comparison)
- Metrics: which ones, why appropriate
- Implementation: optimizer, lr, epochs, batch size, hardware, training time
- Seeds: how many, which values

### Results subsection
- Main comparison table with your method vs all baselines
- Bold the best value in each column
- Include ↑ or ↓ to indicate if higher/lower is better
- Report mean ± std from multiple runs
- Every number in the paper must match results.json exactly

### Ablation subsection
- One table showing: full method, then remove each component
- Proves every component contributes

### Analysis subsection (optional but strengthens paper)
- Failure cases: where does your method fail and why?
- Qualitative examples: show what the model actually produces
- Training curves: show convergence behavior

## Tables

```latex
\begin{table}[t]
\caption{Comparison on [Dataset]. Best results in \textbf{bold}. ↑ means higher is better.}
\label{tab:main}
\centering
\begin{tabular}{lccc}
\toprule
Method & Accuracy ↑ & F1 ↑ & Latency (ms) ↓ \\
\midrule
Baseline A & 82.1 ± 0.3 & 79.4 ± 0.5 & 12.3 \\
Baseline B & 84.7 ± 0.2 & 81.2 ± 0.4 & 15.7 \\
\textbf{Ours} & \textbf{87.3 ± 0.2} & \textbf{84.1 ± 0.3} & 14.1 \\
\bottomrule
\end{tabular}
\end{table}
```

Rules:
- Use booktabs (\toprule, \midrule, \bottomrule) — no vertical lines
- Caption goes ABOVE the table
- Self-contained caption: readable without main text
- Reference every table in the text: "As shown in Table~\ref{tab:main}..."
- Numbers must match results.json exactly

## Figures

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.8\linewidth]{figures/training_curve.pdf}
\caption{Training loss over epochs. Our method (blue) converges faster
than Baseline A (orange) and Baseline B (green).}
\label{fig:training}
\end{figure}
```

Rules:
- Caption goes BELOW the figure
- Self-contained caption
- Use PDF or vector format when possible (not low-res PNG)
- Readable at print size (font ≥ 8pt in the figure)
- Reference every figure: "Figure~\ref{fig:training} shows..."
- Use consistent colors across all figures

## Discussion / Limitations

- Discuss what the results mean, not just what they are
- Honestly acknowledge limitations:
  - "Our method assumes X, which may not hold in Y scenarios"
  - "We evaluated on Z datasets; generalization to other domains is untested"
- Reviewers reward honesty — hiding limitations gets papers rejected

## Conclusion

- Restate the problem and your approach (one sentence each)
- Summarize key findings with concrete numbers
- State broader implications
- Suggest future work
- DO NOT introduce new results or claims here
- 0.5-1 page

## References (CRITICAL)

- EVERY reference must be a REAL, VERIFIABLE publication
- Search Semantic Scholar (semanticscholar.org) to find and verify papers
- Fake or hallucinated citations undermine scientific integrity
- Use correct format: authors, title, venue, year
- Prefer published conference/journal papers over arXiv preprints
- Include 15-30 references for a typical ML paper
- Use \citep{} for parenthetical: "(Smith et al., 2023)"
- Use \citet{} for textual: "Smith et al. (2023) showed..."

## LaTeX Best Practices

- Use the venue's official style file (neurips_2025.sty, etc.)
- Use booktabs for tables (no vertical lines)
- Use \usepackage{hyperref} for clickable references
- Define notation with \newcommand for consistency
- Use ~ for non-breaking spaces before references: Table~\ref{tab:main}
- Compile at least twice to resolve references
- 8-10 pages for main content (excluding references and appendix)

## Common Mistakes That Get Papers Rejected

- No explicit contributions list in the introduction
- Claims not supported by evidence in the experiments
- Numbers in text don't match tables
- Missing error bars / single-run results
- No ablation study
- Unfair baseline comparisons
- Fabricated references
- No limitations discussion
- Poor writing quality / unclear main contribution
