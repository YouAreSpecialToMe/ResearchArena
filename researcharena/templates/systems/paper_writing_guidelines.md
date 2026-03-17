# Paper Writing Guidelines (Systems)

Distilled from "How to Write a Great Research Paper" (Peyton Jones), OSDI/SOSP/EuroSys
author guidelines, USENIX and ACM formatting requirements, and established systems
paper conventions.

## Core Principle

A systems paper tells the story of a real artifact you built: the problem it
solves, WHY the design is the way it is, evidence that it works, and what
others can learn from it. The key insight is in the DESIGN, not the code.

## Structure

Write in this order (not the order they appear in the paper):

1. Design -> Implementation -> Evaluation -> Contributions list -> Conclusion
2. Then Introduction (now you know what to introduce)
3. Then Related Work (typically at the END in systems papers)
4. Abstract LAST (summarize the completed paper)

Final paper order:

```
1. Title
2. Abstract (200-300 words, one paragraph)
3. Introduction (problem, motivation, key insight, contributions, roadmap)
4. Background / Motivation (why the problem is hard, what existing systems get wrong)
5. Design (the core of the paper -- system architecture and design rationale)
6. Implementation (key implementation details, lines of code, effort)
7. Evaluation (benchmarks, comparisons, microbenchmarks, scalability)
8. Related Work (positioned AFTER evaluation in most systems papers)
9. Conclusion
10. References
```

Note: systems papers typically place Related Work near the end, after Evaluation.
This is the opposite of ML papers. The reasoning is that readers need to
understand your design before they can appreciate how it differs from prior work.

## Format and Length

- **USENIX venues** (OSDI, SOSP, ATC, NSDI, FAST): USENIX format, typically
  12-14 pages excluding references. Check the specific CFP for exact limits.
- **ACM venues** (ASPLOS, ISCA, MICRO, EuroSys): ACM sigplan or sigarch format,
  typically 12-14 pages excluding references.
- Systems papers are longer than ML papers. Use the space for thorough
  design explanation and evaluation. Do NOT pad -- use it productively.

## Abstract

- ONE paragraph, 200-300 words (systems abstracts tend to be slightly longer
  than ML paper abstracts)
- Structure: problem -> why existing systems fail -> your key insight ->
  your system name -> key result -> implication
- Must be self-contained -- readable without the rest of the paper
- No citations in the abstract
- Include one or two concrete quantitative results:
  "SystemName achieves 3.2x higher throughput than BaselineSystem on
  workload W while reducing tail latency by 58%."
- Name your system in the abstract

## Introduction

- **Paragraph 1**: What is the problem and why does it matter?
  Ground it in a real scenario.
- **Paragraph 2**: Why is this hard? What do existing systems do wrong?
  Be specific about the limitation.
- **Paragraph 3**: What is your key insight? One sentence that captures
  the core design idea. This is the most important sentence in the paper.
- **Paragraph 4**: What did you build? Brief system overview (2-3 sentences).
- **Paragraph 5**: What are the results? Headline numbers.
- **Paragraph 6**: Contributions list:
  ```latex
  Our contributions are:
  \begin{itemize}
    \item We identify [problem/bottleneck] in existing systems through
          [measurement/analysis] (\S\ref{sec:motivation}).
    \item We design and implement \textsc{SystemName}, which addresses
          this through [key technique] (\S\ref{sec:design}).
    \item We evaluate \textsc{SystemName} on [workloads] and show it
          achieves [key result] compared to [baselines] (\S\ref{sec:eval}).
  \end{itemize}
  ```
- **Final sentence**: Paper roadmap using section references.

## Background / Motivation

This section is critical in systems papers. It must convince the reader
that the problem is real and that existing solutions are inadequate.

- Explain the domain and relevant background (assume a systems-literate
  but not domain-expert reader)
- Show evidence that the problem exists:
  - Measurements from real systems or production traces
  - Performance breakdown showing where time/resources are wasted
  - Example scenarios that expose the limitation
- Explain WHY existing systems cannot solve this problem -- what
  fundamental design choice prevents them from doing better?
- End with a clear statement of your design goal

## Design (the core of a systems paper)

This is where systems papers succeed or fail. The design section must
explain not just WHAT you built, but WHY you made each design decision.

### System architecture
- Start with an architecture overview figure (this is expected in every
  systems paper). Show the major components and how they interact.
- Walk through the architecture top-down: overall structure first, then
  dive into each component
- Use a running example or a request/operation walkthrough to make the
  design concrete

### Design rationale (CRITICAL)
- For each major design decision, explain:
  1. What alternatives you considered
  2. Why you chose this approach
  3. What tradeoff it embodies
- Example: "We use a log-structured design rather than in-place updates
  because workload W is write-heavy and sequential I/O on SSDs is 10x
  faster than random I/O."
- Reviewers reject papers where design decisions seem arbitrary

### Key techniques
- Describe each novel technique or mechanism in detail
- Use pseudocode for algorithms (keep it concise -- not full source code)
- Include diagrams for data structures and protocols
- Explain correctness arguments for protocols (especially for distributed
  systems, concurrency control, or crash recovery)

### Handling corner cases
- Address failure modes: what happens when a node crashes, a disk fails,
  the network partitions?
- Discuss consistency and correctness guarantees explicitly
- Do not sweep hard cases under the rug -- reviewers will find them

## Implementation

- Brief but informative (typically 0.5-1 page)
- Report: language, lines of code, key libraries used, development effort
- Describe implementation challenges and how you solved them
- Mention any limitations of the prototype vs. a production system
- State what hardware features you exploit (if any)

## Evaluation

Systems evaluations must be thorough. A typical evaluation has 5-8 figures
spanning 3-5 pages. Structure as:

### Setup subsection
- **Hardware**: CPU model, core count, memory (size and type), storage
  (model, capacity), network (NIC, bandwidth), any accelerators
- **Software**: OS, kernel version, compiler and flags, key library versions
- **Baselines**: what systems, what versions, how configured
- **Workloads**: which benchmarks, what parameters (key count, value size,
  read/write ratio, skew distribution, number of clients)
- **Methodology**: warmup duration, measurement duration, number of iterations,
  how percentiles are computed

### End-to-end performance (the headline result)
- Full system comparison on representative workloads
- Show throughput AND latency (not just one)
- Include a throughput-latency curve (throughput on X, latency on Y, varying
  offered load) -- this is the gold standard for systems evaluation
- Bold the best result; explain significant differences

### Microbenchmarks (understanding the design)
- Isolate each key component and measure its contribution
- Example: "To understand the benefit of our new index structure, we
  replaced it with a B-tree and measured throughput."
- This is the systems equivalent of an ablation study

### Scalability
- Show performance as you vary: number of threads/cores, number of nodes,
  data size, number of clients
- Include a scalability plot (linear X axis, performance on Y axis)
- Show ideal linear scaling as a reference line
- Explain where and why scalability plateaus

### Sensitivity analysis
- Vary key workload parameters one at a time:
  - Read/write ratio (100/0, 95/5, 50/50, 0/100)
  - Key distribution (uniform vs. Zipfian with varying skew)
  - Value size (small, medium, large)
  - Number of concurrent clients
- Show that your system is robust, not tuned for one specific workload

### Comparison fairness
- Use the same hardware for all systems
- Tune baselines fairly (use recommended configurations)
- If baselines have different feature sets, acknowledge and explain

## Tables

```latex
\begin{table}[t]
\caption{End-to-end throughput (Kops/sec) on YCSB workloads. Best in \textbf{bold}.}
\label{tab:throughput}
\centering
\begin{tabular}{lcccc}
\toprule
System & YCSB-A & YCSB-B & YCSB-C & YCSB-F \\
\midrule
Redis & 312 & 485 & 520 & 289 \\
RocksDB & 198 & 423 & 467 & 176 \\
\textsc{OurSystem} & \textbf{1,024} & \textbf{1,380} & \textbf{1,455} & \textbf{892} \\
\bottomrule
\end{tabular}
\end{table}
```

Rules:
- Use booktabs (\toprule, \midrule, \bottomrule) -- no vertical lines
- Caption goes ABOVE the table
- Self-contained caption: readable without main text
- Reference every table in the text: "As shown in Table~\ref{tab:throughput}..."
- Numbers must match results.json exactly

## Figures

Systems papers are figure-heavy. Common figure types:

- **Architecture diagram**: required. Show components and data flow.
- **Throughput-latency curve**: the standard systems performance figure.
- **Scalability plot**: performance vs. cores/nodes/data size.
- **CDF of latency**: shows full latency distribution.
- **Breakdown chart**: stacked bar showing where time is spent.
- **Time-series plot**: behavior over time (e.g., during failure recovery).

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/throughput_latency.pdf}
\caption{Throughput vs. p99 latency under increasing load (YCSB-A,
Zipfian, 16B keys, 1KB values). \textsc{OurSystem} sustains
1M ops/sec at sub-50\textmu{}s p99 latency.}
\label{fig:throughput_latency}
\end{figure}
```

Rules:
- Caption goes BELOW the figure
- Self-contained caption with workload parameters
- Use PDF or vector format (not low-res PNG)
- Readable at print size (font >= 8pt, distinguishable line styles)
- Reference every figure: "Figure~\ref{fig:throughput_latency} shows..."
- Use consistent colors and line styles across all figures

## Related Work (near the end)

- Organize by category of system, not chronologically:
  - "Key-value stores: ..."
  - "Log-structured systems: ..."
  - "Hardware-accelerated systems: ..."
- For each category, explain what those systems do and how yours differs
- Be respectful but clear about limitations of prior work
- End with: "Unlike [closest prior system], \textsc{OurSystem}..."
- Every cited paper must be REAL and verifiable. Search ACM DL, USENIX
  proceedings, IEEE Xplore, or Google Scholar to confirm papers exist.

## Conclusion

- Restate the problem and your system (one sentence each)
- Summarize key results with concrete numbers
- State the broader lesson or design principle that others can learn from
- Suggest future work
- DO NOT introduce new results or claims here
- 0.5-1 page

## References (CRITICAL)

- EVERY reference must be a REAL, VERIFIABLE publication
- Search ACM Digital Library, USENIX proceedings, IEEE Xplore, or Google
  Scholar to find and verify papers
- Fake or hallucinated citations undermine scientific integrity and are
  grounds for automatic rejection
- Use correct format: authors, title, venue, year
- Prefer published conference proceedings over technical reports or arXiv
- Include 30-50 references for a typical systems paper (more than ML papers)
- Use \cite{} for parenthetical citations and \citet{} for textual

## LaTeX Best Practices

- Use the venue's official style file (check the CFP)
  - USENIX: usenix-2020-09.sty or current equivalent
  - ACM: acmart.cls with sigplan or sigarch option
- Use booktabs for tables (no vertical lines)
- Use \usepackage{hyperref} for clickable references
- Use \textsc{SystemName} consistently throughout the paper
- Use ~ for non-breaking spaces: Table~\ref{tab:main}, Figure~\ref{fig:arch}
- Use \S for section symbol: \S\ref{sec:design}
- Use siunitx for units: \SI{1.5}{\micro\second}, \SI{100}{Gbps}
- Compile at least twice to resolve references

## Common Mistakes That Get Systems Papers Rejected

- No clear problem statement or motivation (why should anyone care?)
- Design decisions without rationale ("we use X" but not "we use X because...")
- No system architecture diagram
- Evaluation only at low load (not showing saturation or overload behavior)
- Only mean latency reported (no percentiles)
- Unfair baseline comparison (misconfigured or outdated baselines)
- No microbenchmarks (reviewer cannot tell which design decisions matter)
- No scalability evaluation
- Evaluation on only one workload
- Fabricated or unverifiable references
- Implementation described but no design insight explained
- Claims about production readiness without production-scale evaluation
- Ignoring failure handling and correctness
