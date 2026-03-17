# Paper Writing Guidelines (Theory)

How to write a theoretical computer science paper for venues like
STOC, FOCS, SODA, and COLT.

Distilled from Oded Goldreich's writing advice, Ian Parberry's guide to
TCS paper writing, STOC/FOCS formatting conventions, and standard practices
in the algorithms and complexity community.

## Core Principle

A theory paper presents a precise mathematical result and a rigorous proof.
The paper must convince the reader that (1) the result is significant,
(2) the result is correct, and (3) the techniques are interesting. Every
section serves this goal.

## Structure

Write in this order (not the order they appear in the paper):

1. Theorem statements and proofs (the core work)
2. Preliminaries (what the reader needs)
3. Introduction (now you know what to motivate and summarize)
4. Abstract LAST (distill the completed paper)

Final paper order:

```
1. Title
2. Abstract (one paragraph, 100-200 words)
3. Introduction
   3a. Problem statement and motivation
   3b. Our results (precise theorem statements)
   3c. Our techniques / proof overview
   3d. Related work
   3e. Paper organization
4. Preliminaries (definitions, notation, known tools)
5. Main Result (theorem + proof, possibly split into sections)
6. Extensions / Additional Results (if any)
7. Conclusion and Open Problems
8. References
9. Appendix (deferred proofs, full calculations)
```

## Format and Length

- STOC/FOCS: extended abstract format, typically 10-12 pages in ACM
  two-column format. The paper should be self-contained but proofs of
  secondary lemmas can be deferred to an appendix.
- SODA: similar to STOC/FOCS, up to 12 pages in SIAM format
- COLT: JMLR/PMLR format, main body typically 12-15 pages plus appendix
- Use the official style file for the target venue
- The main body must contain the key ideas. The appendix is for
  completeness, not for essential content that the reviewer must read.

## Abstract

- ONE paragraph, 100-200 words
- Structure: problem statement -> main result (with precise bound) ->
  brief mention of technique -> significance
- Must state the main theorem, including the bound, explicitly
- Example: "We give an O(m sqrt(n))-time algorithm for computing maximum
  bipartite matching, improving the O(mn)-time algorithm of Hopcroft and
  Karp (1973). Our approach is based on a new decomposition of augmenting
  paths using persistent data structures. This resolves a question posed
  by [Author] and implies improved bounds for several related problems."
- No citations in the abstract (refer to results by author names and year
  if needed, but no \cite commands)

## Introduction

The introduction is the most important part of a theory paper. Most
reviewers form their opinion primarily from the introduction.

### Problem statement and motivation (0.5-1 page)

- Define the problem precisely but accessibly
- Explain why it matters: is it a fundamental problem? Does it have
  applications? Is it a long-standing open question?
- State the previous best bounds and who achieved them
- Frame the gap: "The best known algorithm runs in O(n^2) time. No
  super-linear lower bound is known. Can we do better?"

### Our results (0.5-1 page)

- State your main theorem formally in the introduction. Use a Theorem
  environment. This is the most important part of the paper.
- If you have multiple results, state them all here, ordered by importance
- Explicitly compare with prior work: "This improves the O(n^2) bound of
  [Author, Year] to O(n^{3/2}). Previously, the O(n^{3/2}) bound was
  known only for the special case of [restricted class]."
- If the result is tight, state the matching lower bound
- If the result has corollaries or applications, list them

### Our techniques / proof overview (1-2 pages)

- This subsection is crucial and often makes or breaks the paper
- Explain the high-level proof strategy in intuitive terms
- What is the key new idea? Why do previous approaches fail, and what
  new ingredient do you introduce?
- Walk through the proof at a conceptual level: "We proceed in three
  steps. First, we show that... This allows us to... Finally, we
  combine these to obtain..."
- Use small examples or toy cases to illustrate the key idea
- A reader who reads only the introduction should come away understanding
  WHAT you proved, WHY it matters, and HOW the proof works at a high level

### Related work (0.5-1 page)

- Organize by problem/technique, not chronologically
- Give a history of results on your problem: who proved what bound, when,
  and using what technique
- Discuss related problems and techniques
- Be generous with citations -- theory values proper attribution of ideas
- Cite the original source of a result, not just the most recent paper
  that uses it

### Paper organization

- One paragraph: "In Section 2 we introduce notation and state
  preliminary results. In Section 3 we prove our main theorem. In
  Section 4 we extend the result to weighted graphs. Section 5
  concludes with open problems."

## Preliminaries

- Define the computational model (word RAM, comparison model, real RAM,
  Turing machine, etc.)
- State all notation: G = (V, E) for graphs, [n] = {1, ..., n}, etc.
- State definitions for all concepts used in the proof (formally, with
  Definition environments)
- State known results you will use as black boxes (with Theorem or Lemma
  environments and citations)
- Keep this section short -- only include what the reader needs for
  the proof. Do not write a textbook chapter.

## Main Result

### Theorem-Lemma-Proof structure

Theory papers follow a strict formal structure:

```latex
\begin{theorem}\label{thm:main}
Let $G = (V, E)$ be an undirected graph with $n$ vertices and $m$ edges.
Algorithm~\ref{alg:main} computes a maximum matching in $G$ in
$O(m \sqrt{n})$ time.
\end{theorem}

The proof proceeds by establishing the following two lemmas.

\begin{lemma}\label{lem:phases}
The algorithm terminates after at most $O(\sqrt{n})$ phases.
\end{lemma}

\begin{proof}
[Proof of Lemma~\ref{lem:phases}]
...
\end{proof}

\begin{lemma}\label{lem:phase-time}
Each phase can be implemented in $O(m)$ time.
\end{lemma}

\begin{proof}
[Proof of Lemma~\ref{lem:phase-time}]
...
\end{proof}

\begin{proof}[Proof of Theorem~\ref{thm:main}]
By Lemma~\ref{lem:phases}, there are $O(\sqrt{n})$ phases. By
Lemma~\ref{lem:phase-time}, each takes $O(m)$ time. The total time
is $O(m\sqrt{n})$.
\end{proof}
```

### Proof writing guidelines

- **Start with the proof outline**: before diving into details, tell the
  reader the plan ("We prove this in three steps...")
- **Highlight the key idea**: when you reach the novel part of the proof,
  signal it ("The key observation is that...")
- **Be rigorous but readable**: every step must be justified, but routine
  calculations should be deferred to the appendix if they are long
- **Use proof sketches for minor results**: for lemmas whose proofs use
  standard techniques, a sketch in the main body with a full proof in
  the appendix is acceptable
- **Label and number everything**: every theorem, lemma, claim,
  definition, and equation should have a label for cross-referencing
- **Case analysis**: when doing case analysis, clearly state what the
  cases are and why they are exhaustive before proving each case

### Algorithms

If presenting an algorithm:

```latex
\begin{algorithm}[t]
\caption{Maximum Matching via Augmenting Paths}\label{alg:main}
\begin{algorithmic}[1]
\Require Graph $G = (V, E)$
\Ensure Maximum matching $M$
\State $M \gets \emptyset$
\While{there exists an augmenting path $P$ w.r.t.\ $M$}
    \State $M \gets M \oplus P$
\EndWhile
\State \Return $M$
\end{algorithmic}
\end{algorithm}
```

- Use pseudocode, not implementation code
- The pseudocode should match the proof exactly -- if the proof analyzes
  a while loop, the algorithm should have a while loop
- Include a correctness proof (often a loop invariant argument) and a
  runtime analysis

## Extensions and Corollaries

- State each extension as a formal corollary or theorem
- If the extension requires significant new ideas, give the proof
- If it follows routinely, sketch the proof and note what changes

## Conclusion and Open Problems

- Summarize the main result in one sentence
- State concrete open problems arising from your work:
  - "Can the running time be improved to O(m log n)?"
  - "Does our technique extend to directed graphs?"
  - "Is the O(n^{1/3}) gap between our upper and lower bound tight?"
- Open problems should be specific and well-posed, not vague wishes
- 0.5-1 page

## References (CRITICAL)

- EVERY reference must be a REAL, VERIFIABLE publication
- Search DBLP (dblp.org) and Semantic Scholar (semanticscholar.org) to
  verify papers exist with the stated titles, authors, and venues
- Fake or hallucinated citations are unacceptable in any context
- Prefer published conference or journal versions over arXiv preprints
- Cite the original source of a result, not just the most recent use
- For classic results, cite the original paper (even if decades old)
- Include 15-40 references for a typical theory paper
- Use consistent formatting: authors, title, venue, year

## LaTeX Best Practices for Theory Papers

- Use the venue's official style file
- Use AMS math environments: theorem, lemma, proposition, corollary,
  definition, remark, proof
- Use \newtheorem to define numbered theorem environments
- Use booktabs for any tables (no vertical lines)
- Define macros for repeated notation:
  ```latex
  \newcommand{\OPT}{\mathrm{OPT}}
  \newcommand{\poly}{\mathrm{poly}}
  \newcommand{\polylog}{\mathrm{polylog}}
  \newcommand{\eps}{\varepsilon}
  ```
- Use ~ for non-breaking spaces: Theorem~\ref{thm:main},
  Lemma~\ref{lem:key}, Algorithm~\ref{alg:main}
- Use \qedhere at the end of proofs that end in a displayed equation
  or item list
- Do NOT include training curves, model architectures, or ML-specific
  visualizations -- this is a mathematics/algorithms paper

## Common Mistakes That Get Theory Papers Rejected

- Main theorem not stated precisely in the introduction
- No proof overview / techniques section -- the reviewer cannot tell if
  the ideas are novel
- Proof has gaps ("it is easy to see" for something that is not easy)
- Result is correct but incremental -- small improvement with no new ideas
- Result is correct but technique is a straightforward application of
  a known method
- Computational model not specified or inappropriate for the claimed bound
- Poor comparison with prior work -- failing to cite relevant results or
  misstating previous bounds
- Overly long or unfocused -- the paper tries to do too much instead of
  presenting one clean result
- Appendix contains essential material that should be in the main body
- Fabricated or hallucinated references
