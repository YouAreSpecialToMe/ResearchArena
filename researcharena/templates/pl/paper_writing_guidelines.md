# Paper Writing Guidelines (Programming Languages)

Distilled from Simon Peyton Jones' "How to Write a Great Research Paper",
SIGPLAN formatting requirements (PACMPL and ACM SIGPLAN styles), and
conventions in PLDI, POPL, OOPSLA, and ICFP papers.

## Core Principle

A PL paper tells a story: here is a problem with real programs (or a gap in
our formal understanding) -> here is a precise, elegant solution -> here is
evidence it works (proof and/or evaluation). Every section serves this
narrative. The key insight should be understandable from examples BEFORE the
reader encounters formalism.

## Venue-Specific Formatting

- **POPL, OOPSLA, ICFP**: PACMPL format (Proceedings of the ACM on
  Programming Languages). Use the `acmart` document class with
  `\documentclass[acmsmall]{acmart}`.
- **PLDI**: ACM SIGPLAN conference format. Use the `acmart` document class
  with `\documentclass[sigplan,screen]{acmart}`.
- All venues: follow ACM formatting guidelines strictly. Do not alter margins,
  font sizes, or spacing.

## Structure

PL papers follow different structures depending on the type of contribution.
Write sections in this order (not the order they appear in the paper):

### For formal papers (type systems, semantics, verification):
Write: Formalization -> Metatheory -> Implementation -> Evaluation ->
Motivation/Examples -> Contributions list -> Conclusion -> Introduction ->
Related Work -> Abstract LAST.

Final paper order:
```
1. Title
2. Abstract (150-250 words, one paragraph)
3. Introduction (problem, motivating example, contributions list)
4. Overview / Motivation (key examples that build intuition)
5. Formalization (syntax, semantics, type rules)
6. Metatheory (soundness, completeness, decidability theorems)
7. Implementation
8. Evaluation
9. Related Work
10. Conclusion
11. References
```

### For systems-oriented papers (compilers, analyses, tools):
Write: Design -> Implementation -> Evaluation -> Contributions list ->
Conclusion -> Introduction -> Related Work -> Abstract LAST.

Final paper order:
```
1. Title
2. Abstract (150-250 words, one paragraph)
3. Introduction (problem, gap, contributions list)
4. Overview (high-level design, architecture diagram)
5. Design / Approach (technical details)
6. Implementation (engineering decisions, LOC, effort)
7. Evaluation (benchmarks, comparison, analysis)
8. Related Work
9. Conclusion
10. References
```

## Abstract

- ONE paragraph, 150-250 words
- Structure: context -> problem -> approach (with key insight) -> main result
  -> implication
- Must be self-contained
- No citations in the abstract
- Include one concrete result: "Our analysis finds 23 previously unknown bugs
  in 1.2M lines of Java code with a 12% false positive rate" or "We prove
  soundness and implement a type checker that handles all of Rust's borrow
  checking rules"

## Introduction

- Start with a concrete, relatable problem. If possible, show a small code
  example that illustrates the problem in the first page.
- Identify the gap: what can't existing tools/theories handle?
- State your key insight in one sentence
- List contributions explicitly:
  ```latex
  Our contributions are:
  \begin{itemize}
    \item A type system for X that enforces Y (Section~\ref{sec:formal}).
    \item A proof of soundness, mechanized in Coq (Section~\ref{sec:meta}).
    \item An implementation and evaluation on Z real-world programs,
          showing A\% improvement over baseline (Section~\ref{sec:eval}).
  \end{itemize}
  ```
- End with a paper roadmap

## Overview / Motivation Section (critical for PL papers)

This section is arguably the most important in a PL paper. It builds intuition
BEFORE the formalism.

- Present 2-3 carefully chosen examples that illustrate:
  1. The problem: a program where existing approaches fail
  2. The key idea: how your approach handles it
  3. The subtlety: a tricky case that motivates your design choices
- Use real or realistic code, not abstract mathematical examples
- Walk the reader through the examples step by step
- End with: "The remainder of this paper formalizes and proves correct the
  intuition developed in this section."
- Reviewers will judge your paper heavily on this section. If they can't
  understand the key insight from the examples, the paper will likely be
  rejected regardless of how correct the formalism is.

## Formalization Section

### Syntax
- Present the abstract syntax using standard BNF notation:
  ```latex
  \[
  \begin{array}{rcl}
  e & ::= & x \mid \lambda x{:}\tau.\, e \mid e_1\; e_2
            \mid \texttt{let}\; x = e_1\; \texttt{in}\; e_2 \\
  \tau & ::= & \alpha \mid \tau_1 \to \tau_2 \mid \forall \alpha.\, \tau \\
  v & ::= & \lambda x{:}\tau.\, e
  \end{array}
  \]
  ```
- Use a core calculus that captures the essential features. Explicitly state
  what is omitted and why (e.g., "We omit mutable references for clarity;
  the extension is straightforward and included in the appendix").

### Typing rules
- Use standard inference rule notation:
  ```latex
  \begin{mathpar}
  \inferrule
    {\Gamma \vdash e_1 : \tau_1 \to \tau_2 \\
     \Gamma \vdash e_2 : \tau_1}
    {\Gamma \vdash e_1\; e_2 : \tau_2}
    \textsc{T-App}
  \end{mathpar}
  ```
- Name every rule (T-App, T-Abs, T-Let, etc.)
- Use the `mathpartir` package for inference rules
- Highlight the NEW rules that are your contribution (annotate or separate
  them from standard rules)
- Define the judgment forms before presenting rules

### Operational semantics
- Choose small-step or big-step and state which you use
- Distinguish values from expressions
- Define evaluation contexts if using contextual semantics

## Metatheory Section

- State main theorems precisely (progress, preservation, soundness,
  completeness, decidability)
- For each theorem:
  - State the theorem formally
  - Give a proof sketch in the main text (key ideas and interesting cases)
  - Put the full proof in the appendix (pen-and-paper) or reference the
    mechanized proof artifact
- If proofs are mechanized:
  ```latex
  \begin{theorem}[Soundness]
  If\/ $\Gamma \vdash e : \tau$ and $e \longrightarrow^* e'$ then either
  $e'$ is a value or there exists $e''$ such that $e' \longrightarrow e''$.
  \end{theorem}
  \begin{proof}
  By induction on the derivation. Mechanized in Coq; see the accompanying
  artifact (\texttt{Soundness.v}, 847 lines).
  \end{proof}
  ```
- Clearly state any restrictions or limitations of the formal results

## Implementation Section

- Describe what you built: language, LOC, architecture
- Discuss key engineering decisions and why you made them
- State what subset of the formalism is implemented (if not all of it)
- Discuss any gaps between the formalized core calculus and the implementation
  (e.g., "The implementation handles the full Java type system; the formalization
  covers a core calculus without generics")
- Implementation details that reviewers care about: is this a standalone tool?
  An LLVM pass? A GraalVM extension? A Coq plugin?

## Evaluation Section

- Structure: Setup -> Main results -> Detailed analysis

### Setup subsection
- Benchmarks: which suite, why chosen, how many programs, total LOC
- Baselines: which tools/techniques, versions, configurations
- Metrics: what you measure and why it's appropriate
- Hardware and software: CPU, RAM, OS, compiler/runtime versions
- Timeouts: time limit per benchmark (if applicable)

### Results subsection
- Main comparison table with your tool vs all baselines
- Bold the best value in each column
- Report per-benchmark results (in appendix if too many)
- For performance experiments: report geometric mean speedup
- For analysis experiments: report precision, recall, false positive rate,
  and analysis time
- For synthesis experiments: report cactus plots (solved vs. time)

### Analysis subsection
- Where does your approach do well and why?
- Where does it fail and why?
- Scalability: how does performance change with input size?
- Case studies: walk through interesting examples in detail

## Tables

```latex
\begin{table}[t]
\caption{Comparison on DaCapo benchmarks. Best results in \textbf{bold}.
FP = false positives.}
\label{tab:main}
\centering
\begin{tabular}{lrrrr}
\toprule
Benchmark & LOC & Bugs Found & FP & Time (s) \\
\midrule
antlr     & 35K  & 12 & 3  & 24.1 \\
bloat     & 81K  & 8  & 5  & 67.3 \\
\midrule
\textbf{Total} & \textbf{580K} & \textbf{47} & \textbf{12} & \textbf{312} \\
\bottomrule
\end{tabular}
\end{table}
```

Rules:
- Use booktabs (\toprule, \midrule, \bottomrule) -- no vertical lines
- Caption goes ABOVE the table
- Self-contained caption: readable without main text
- Reference every table: "As shown in Table~\ref{tab:main}..."
- Numbers must match results.json exactly

## Figures

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.8\linewidth]{figures/scalability.pdf}
\caption{Analysis time vs.\ program size (LOC). Our analysis (blue) scales
linearly while Tool~A (orange) scales quadratically.}
\label{fig:scalability}
\end{figure}
```

Rules:
- Caption goes BELOW the figure
- Self-contained caption
- Use PDF or vector format (not low-res PNG)
- Readable at print size (font >= 8pt in the figure)
- Reference every figure: "Figure~\ref{fig:scalability} shows..."
- Use consistent colors across all figures
- Cactus plots for synthesis/verification: x-axis = time, y-axis = number
  of benchmarks solved

## Related Work

- Organize by technique or approach, NOT chronologically
- Funnel structure: broad area -> specific subproblem -> directly competing
  methods
- For each group, explain:
  1. What they do
  2. How your work differs (be precise about the technical distinction)
- PL papers are expected to cite foundational work even if it's decades old
- End with: "Unlike [prior work], our approach..."
- Every cited paper must be REAL and verifiable. Search Semantic Scholar
  (semanticscholar.org) or ACM DL (dl.acm.org) to confirm papers exist
  before citing them.

## Discussion / Limitations

- Discuss what the results mean, not just what they are
- Honestly acknowledge limitations:
  - "Our type system currently handles a core calculus without exceptions;
    extending to full Java is future work"
  - "Our analysis is intraprocedural; interprocedural extension would improve
    precision at the cost of scalability"
  - "Our soundness proof assumes a sequential execution model"
- Reviewers reward honesty -- hiding limitations gets papers rejected

## Conclusion

- Restate the problem and your approach (one sentence each)
- Summarize key findings with concrete numbers or formal results
- State broader implications
- Suggest future work
- DO NOT introduce new results or claims here
- 0.5-1 page

## References (CRITICAL)

- EVERY reference must be a REAL, VERIFIABLE publication
- Search Semantic Scholar (semanticscholar.org) or ACM DL (dl.acm.org) to
  find and verify papers
- Fake or hallucinated citations undermine scientific integrity
- Use correct ACM reference format: authors, title, venue, year
- Prefer published conference/journal papers over arXiv preprints
- PL papers typically have 30-50 references (more than ML papers, because
  PL has a longer history and formal lineage matters)
- Use \citep{} for parenthetical: "(Pierce, 2002)"
- Use \citet{} for textual: "Pierce (2002) defines..."
- Cite foundational work (Milner, Hoare, Plotkin, Reynolds, etc.) where
  appropriate

## LaTeX Best Practices

- Use `acmart` document class with the correct format option
- Use `mathpartir` for inference rules
- Use `booktabs` for tables (no vertical lines)
- Use `listings` or `minted` for code listings
- Define notation with \newcommand for consistency:
  ```latex
  \newcommand{\tj}[3]{#1 \vdash #2 : #3}          % typing judgment
  \newcommand{\step}[2]{#1 \longrightarrow #2}     % reduction step
  \newcommand{\mstep}[2]{#1 \longrightarrow^* #2}  % multi-step reduction
  ```
- Use ~ for non-breaking spaces before references: Table~\ref{tab:main}
- Compile at least twice to resolve references

## Common Mistakes That Get PL Papers Rejected

- No motivating examples before formalism (readers get lost)
- Unsound type system or incorrect proofs (reviewers WILL find bugs)
- Formalism without implementation or evaluation ("is this practical?")
- Evaluation on toy benchmarks only ("does this scale to real code?")
- No comparison with existing tools or techniques
- Missing inference rules or semantic cases (incomplete formalization)
- Numbers in text don't match tables
- Fabricated references
- No limitations discussion
- Overly complex formalism when a simpler one would suffice
- Poor code examples that don't illustrate the key insight
- Claiming soundness without a proof (or with a hand-wavy argument)
