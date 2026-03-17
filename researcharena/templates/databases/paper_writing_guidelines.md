# Paper Writing Guidelines (Databases)

Distilled from SIGMOD/VLDB/ICDE formatting requirements, advice from
senior database researchers, and technical writing best practices for
systems and data management papers.

## Core Principle

Your paper tells a story: workload problem or system limitation -> why it
matters -> your technique -> evidence it works -> what it means for practice.
Every section serves this narrative. Database papers must bridge theory
and practice — showing both intellectual depth and real-world relevance.

## Structure

Write in this order (not the order they appear in the paper):

1. Algorithm/Design -> Experiments -> Analysis -> Contributions list -> Conclusion
2. Then Introduction (now you know what to introduce)
3. Then Related Work
4. Preliminaries / Problem Statement (refine once the method section is stable)
5. Abstract LAST (summarize the completed paper)

Final paper order:

```
1. Title
2. Abstract (150-250 words, one paragraph)
3. Introduction (problem, motivation, contributions list, paper roadmap)
4. Preliminaries / Problem Statement (formal definitions, notation, problem formulation)
5. Algorithm / System Design (your technique, with pseudocode)
6. Analysis (complexity bounds, correctness arguments)
7. Experiments (setup, main comparison, scalability, sensitivity, ablation)
8. Related Work (positioned after experiments in most database papers)
9. Conclusion
10. References
```

Note: database papers typically place Related Work near the end (before
Conclusion), not after the Introduction. This is the convention at SIGMOD
and VLDB, though both orderings are acceptable.

## Page Budget

- SIGMOD: 12 pages + unlimited references (ACM format, double-column)
- VLDB: 12 pages + up to 4 additional pages for references/appendix
  (VLDB format, double-column)
- ICDE: 12 pages + references (IEEE format, double-column)
- Plan your content to use the full 12 pages — shorter papers signal
  incomplete evaluation

## Abstract

- ONE paragraph, 150-250 words
- Structure: context (what system/workload problem) -> gap (what's missing) ->
  approach (your technique, one sentence) -> key result (concrete speedup
  or improvement) -> implication (what this means for system design)
- Must be self-contained — readable without the rest of the paper
- No citations in the abstract
- Include one concrete quantitative result:
  "Our approach achieves 3.2x speedup over PostgreSQL on TPC-H at SF100
  while using 40% less memory."

## Introduction

- Start with practical motivation: what workload or system problem exists?
  - "As data volumes grow, query X becomes a bottleneck because..."
  - "Modern hardware (NVMe, large DRAM) invalidates assumption Y..."
- Identify the gap: what's wrong with existing approaches?
- State your approach: one sentence describing your technique
- List contributions explicitly:
  ```latex
  Our contributions are:
  \begin{itemize}
    \item We identify the problem of X in Y workloads and formalize
          it as [formal problem statement].
    \item We propose \textsc{SystemName}, a new algorithm/structure/protocol
          that addresses X by exploiting property Z.
    \item We prove that \textsc{SystemName} achieves O(f(n)) time and
          O(g(n)) space complexity.
    \item We evaluate \textsc{SystemName} on TPC-H, JOB, and real-world
          datasets, showing N$\times$ improvement over [baseline].
  \end{itemize}
  ```
- End with a roadmap: "Section 2 defines the problem. Section 3 describes
  our algorithm. Section 4 analyzes its complexity. Section 5 presents
  experiments. Section 6 discusses related work. Section 7 concludes."

## Preliminaries / Problem Statement

This section is expected in database papers. It should include:

- **Notation table**: define all symbols used in the paper
- **Formal problem definition**: state the input, output, objective, and
  constraints precisely
  ```latex
  \begin{definition}[Problem Name]
  Given a relation $R$ with attributes $A_1, \ldots, A_n$ and a query
  workload $W = \{q_1, \ldots, q_m\}$, find a configuration $C$ that
  minimizes $\sum_{q \in W} \text{cost}(q, C)$ subject to space
  constraint $|C| \leq B$.
  \end{definition}
  ```
- **Complexity of the problem** (if known): is it NP-hard? Is there a
  known approximation ratio? Cite the relevant results.
- **Running example**: a small concrete example that illustrates the
  problem and will be referenced throughout the paper

## Algorithm / System Design

- Complete enough that an expert can reimplement from the paper alone
- **Pseudocode is essential** for any new algorithm:
  ```latex
  \begin{algorithm}[t]
  \caption{Our Algorithm}
  \label{alg:ours}
  \begin{algorithmic}[1]
  \Require Input relation $R$, parameter $k$
  \Ensure Output result $S$
  \State Initialize data structure $D$
  \For{each tuple $t \in R$}
    \State Process $t$ using $D$
  \EndFor
  \State \Return $S$
  \end{algorithmic}
  \end{algorithm}
  ```
- Walk through the pseudocode with the running example
- State all assumptions explicitly
- For systems contributions: include an architecture diagram showing
  how your component fits into the larger system
- Discuss edge cases and how they are handled
- Explain WHY design decisions were made, not just WHAT they are

## Analysis

Database papers benefit strongly from formal analysis. Include when possible:

- **Time complexity**: worst-case and expected, in terms of input size,
  selectivity, or other relevant parameters
- **Space complexity**: memory and storage requirements
- **I/O complexity**: number of disk/page accesses (especially for
  external-memory algorithms)
- **Correctness**: for transaction protocols, prove serializability or
  the claimed isolation level; for query processing, prove result
  correctness
- **Optimality** (if applicable): prove approximation ratio or show
  optimality under stated assumptions

A strong theoretical analysis can compensate for a smaller experimental
section, and vice versa.

## Experiments

- Structure: Setup -> Main Comparison -> Scalability -> Sensitivity ->
  Ablation -> (optional) Case Study

### Setup subsection
- **Hardware**: CPU model, cores, RAM size, storage type (SSD/HDD,
  model), OS version
- **Software**: system versions, compiler and flags, relevant libraries
- **Benchmarks**: which benchmarks, which queries/transactions, scale factors
- **Baselines**: what systems, how configured (buffer pool size, thread
  count, etc.), why chosen
- **Methodology**: number of runs, warmup procedure, warm/cold cache,
  how timing is measured

### Main comparison subsection
- Primary comparison table or figure: your method vs. all baselines on
  the main benchmark
- Report both aggregate (total time, average throughput) and per-query/
  per-transaction breakdown when space permits
- Bold the best value in each column
- Include both performance metrics AND resource usage (memory, storage)

### Scalability subsection
- Show how performance changes as data size grows (at least 3 scale points)
- Show how performance changes with concurrency (vary thread count)
- Use line plots with data size or thread count on x-axis
- Discuss scaling behavior: does it scale linearly? Where does it break?

### Sensitivity subsection
- Vary key parameters one at a time:
  - Selectivity (fraction of data accessed)
  - Skew (Zipfian parameter)
  - Read/write ratio (for mixed workloads)
  - Key size, value size, number of attributes (for data structures)
- Show which parameters significantly affect performance and why

### Ablation subsection
- One table showing: full method, then disable each novel component
- Proves every component contributes to the overall result

## Tables

```latex
\begin{table}[t]
\caption{TPC-H execution time (seconds) at SF10. Best results in \textbf{bold}.
$\downarrow$ means lower is better.}
\label{tab:main}
\centering
\begin{tabular}{lcccc}
\toprule
Method & Total Time $\downarrow$ & Memory (MB) $\downarrow$ & Build Time (s) $\downarrow$ \\
\midrule
PostgreSQL & 28.9 $\pm$ 0.8 & 512 & --- \\
DuckDB & 15.7 $\pm$ 0.3 & 1024 & --- \\
\textbf{Ours} & \textbf{12.3 $\pm$ 0.5} & \textbf{384} & 8.3 \\
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
- For per-query breakdowns with many queries, consider a bar chart instead

## Figures

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{figures/scalability.pdf}
\caption{Execution time vs. data size (TPC-H, SF1 to SF100).
Our method (blue) scales sublinearly while PostgreSQL (orange)
scales superlinearly due to increasing sort-merge cost.}
\label{fig:scalability}
\end{figure}
```

Common figure types in database papers:
- **Bar charts**: comparing methods on a fixed benchmark
- **Line plots**: scalability (x = data size, y = time or throughput)
- **CDF/ECDF plots**: latency distributions (essential for OLTP papers)
- **Heatmaps**: parameter sensitivity across two dimensions
- **Architecture diagrams**: system design overview
- **Throughput-over-time plots**: showing warmup, steady state, and any stalls

Rules:
- Caption goes BELOW the figure
- Self-contained caption: describe what is shown and the key takeaway
- Use PDF or vector format (not low-res PNG)
- Readable at print size (font >= 8pt in the figure)
- Reference every figure: "Figure~\ref{fig:scalability} shows..."
- Use consistent colors across all figures
- Use log scale for axes spanning multiple orders of magnitude

## Related Work

- Organize by approach or problem area, NOT chronologically
- Common categories for database papers:
  - Query optimization (cost-based, adaptive, learned)
  - Indexing (tree-based, hash-based, learned, multi-dimensional)
  - Storage layouts (row, column, hybrid, compressed)
  - Concurrency control (pessimistic, optimistic, MVCC, deterministic)
  - The specific subarea your contribution addresses
- For each group: explain what they do, then how your work differs
- End with: "Unlike [prior work], our approach..."
- Cite both academic papers AND relevant systems papers from industry
- Every cited paper must be REAL and verifiable. Search DBLP (dblp.org)
  or Semantic Scholar (semanticscholar.org) to confirm papers exist.

## Conclusion

- Restate the problem and your approach (one sentence each)
- Summarize key findings with concrete numbers
- State broader implications for database system design
- Suggest future work
- DO NOT introduce new results or claims here
- 0.5-1 page

## References (CRITICAL)

- EVERY reference must be a REAL, VERIFIABLE publication
- Search DBLP (dblp.org) or Semantic Scholar (semanticscholar.org)
  to find and verify papers
- Fake or hallucinated citations undermine scientific integrity
- Use correct format: authors, title, venue, year
- Prefer published conference/journal papers (SIGMOD, VLDB, ICDE,
  TODS, VLDB Journal) over arXiv preprints
- Include 25-40 references for a typical database paper (database papers
  tend to cite more extensively than ML papers)
- Use \citep{} for parenthetical and \citet{} for textual citations

## LaTeX Best Practices

- Use the venue's official style file (sig-alternate.cls for SIGMOD,
  vldb.cls for VLDB, IEEEtran.cls for ICDE)
- Use booktabs for tables (no vertical lines)
- Use the algorithm/algorithmic packages for pseudocode
- Use \usepackage{hyperref} for clickable references
- Define notation with \newcommand for consistency
- Use ~ for non-breaking spaces before references: Table~\ref{tab:main}
- Use \textsc{} for system names
- Compile at least twice to resolve references
- 12 pages for main content (excluding references)

## Common Mistakes That Get Database Papers Rejected

- No formal problem definition — jumping straight to the solution
- Missing pseudocode for the core algorithm
- No complexity analysis (time, space, I/O)
- Unfair baseline comparison (untuned systems, different hardware)
- Evaluation on only toy-sized data (a few MB)
- No scalability experiment
- Missing system configuration details (buffer sizes, thread counts)
- Claims not supported by the experimental evidence
- Numbers in text don't match tables
- No ablation study — unclear which component provides the benefit
- Fabricated references
- No limitations discussion
- Paper too short (significantly under 12 pages) — signals incomplete work
