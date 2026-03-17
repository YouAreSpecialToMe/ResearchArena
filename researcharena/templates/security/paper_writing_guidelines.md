# Paper Writing Guidelines (Computer Security)

Distilled from CCS, IEEE S&P, USENIX Security, and NDSS formatting requirements,
combined with best practices for security research writing.

## Core Principle

Your paper tells a story: threat → why it matters → your attack/defense/analysis
→ evidence it works → what it means for the community. Every section serves
this narrative.

## Venue-Specific Formatting

| Venue | Format | Page limit | Style file |
|---|---|---|---|
| ACM CCS | ACM sigconf | 12 pages + unlimited references | acmart.cls (sigconf) |
| IEEE S&P | IEEE conference | 13 pages + 5 pages references/appendices | IEEEtran.cls |
| USENIX Security | USENIX | 13 pages + unlimited references/appendices | usenix-2019_v3.sty |
| NDSS | NDSS | 15 pages + unlimited references | ndss.cls |

Page limits exclude references and (where noted) appendices. Check the
current call for papers -- limits change occasionally.

## Structure

Write in this order (not the order they appear in the paper):

1. Threat Model → Design/Attack → Implementation → Evaluation → Contributions list
2. Then Introduction (now you know what to introduce)
3. Then Background and Related Work
4. Then Discussion and Conclusion
5. Abstract LAST (summarize the completed paper)

Final paper order:

```
1. Title
2. Abstract (200-300 words, one paragraph)
3. Introduction (threat, gap, contributions list, paper roadmap)
4. Background (technical context needed to understand the paper)
5. Threat Model (attacker capabilities, trust assumptions, scope)
6. Design / Attack Description (your approach, step by step)
7. Implementation (systems details, engineering decisions)
8. Evaluation (setup, results, ablations, evasion analysis)
9. Discussion (limitations, ethical considerations, future work)
10. Related Work (positioning against prior art)
11. Conclusion
12. References
```

Note: in security papers, Related Work often comes near the end (after
Evaluation), unlike ML papers where it typically follows the Introduction.
Both placements are acceptable.

## Abstract

- ONE paragraph, 200-300 words
- Structure: threat/problem → why it matters → your approach → key results
  → implications
- Must be self-contained -- readable without the rest of the paper
- No citations in the abstract
- Include concrete quantitative results: "detects 97% of attacks with
  0.2% false positive rate and 3% throughput overhead"
- For attacks: state what is compromised and on what system
- For defenses: state what is detected/prevented and at what cost

## Introduction

- Start with the threat landscape (what is at risk?)
- Describe the current state of defenses/understanding (what exists?)
- Identify the gap (what's missing, broken, or insufficient?)
- State your approach (one sentence)
- List contributions explicitly:
  ```latex
  Our contributions are:
  \begin{itemize}
    \item We identify a new vulnerability in X that allows Y.
    \item We design and implement Z, a defense that detects/prevents
      the attack with A\% accuracy and B\% overhead.
    \item We evaluate Z on realistic workloads and show it outperforms
      existing defenses C and D.
    \item We responsibly disclosed the vulnerability to [vendor].
  \end{itemize}
  ```
- End with a roadmap: "Section 2 provides background..., Section 3 defines
  our threat model..., Section 4 describes..."

## Background

- Provide the technical context a security researcher needs to understand
  your paper
- Cover: relevant protocols, system architectures, cryptographic primitives,
  or prior techniques your work builds on
- Define key terms precisely
- Include diagrams of system architecture or protocol flows as needed
- Keep it focused -- only include background that is directly used later

## Threat Model (MANDATORY)

This is the most security-specific section and reviewers scrutinize it closely.

- **Attacker goals**: What does the attacker want to achieve? (steal credentials,
  exfiltrate data, escalate privileges, deny service, violate privacy)
- **Attacker capabilities**: What can the attacker do? (send network packets,
  execute unprivileged code, control a malicious server, tamper with hardware,
  poison training data)
- **Attacker knowledge**: What does the attacker know? (network topology,
  defense mechanisms, system configuration, partial secrets)
- **Trust assumptions**: What is trusted? (OS kernel, hardware, specific
  network segments, certificate authorities, specific parties)
- **Scope**: What is explicitly out of scope? (physical access, social
  engineering, denial of service, side channels -- whatever you don't address)

Be precise. "The attacker can observe all network traffic between client and
server but cannot modify it" is good. "The attacker is powerful" is not.

A weak or vague threat model is the single most common reason security papers
are rejected.

## Design / Attack Description

**For attacks:**
- Describe the attack step by step, in enough detail to reproduce
- Include concrete examples with real or realistic parameters
- Explain why the attack works (root cause of the vulnerability)
- Show the attack flow with a diagram or pseudocode
- Discuss prerequisites: what conditions must hold for the attack to succeed?
- Describe any required setup or preparation by the attacker

**For defenses:**
- Explain the high-level architecture with a system diagram
- Describe each component and its role
- Explain the security guarantees: what attacks are prevented and why
- State what attacks are NOT prevented (scope limitations)
- Describe the detection/prevention mechanism formally if possible
- Include pseudocode or algorithm descriptions for key components

**For both:**
- Use clear notation, define every symbol on first use
- Include overview figures showing the system/attack architecture
- Make the description complete enough that an expert can reimplement it

## Implementation

- Describe the implementation: language, libraries, lines of code, platform
- Discuss engineering decisions and tradeoffs
- Mention any modifications required to existing systems
- Describe deployment requirements
- State what will be open-sourced (code, tools, datasets)

## Evaluation

Structure: Setup → Main Results → Ablations → Evasion Analysis →
Performance Overhead → Case Studies

### Setup subsection
- Target systems: name, version, configuration, why chosen
- Datasets: source, size, collection methodology, time period
- Baselines: what they are, why chosen, how configured (fair comparison)
- Metrics: which ones, why appropriate for this evaluation
- Environment: hardware, OS, network setup, virtualization
- Seeds: how many, which values (for non-deterministic experiments)

### Main results subsection
- Main comparison table with your method vs all baselines
- Bold the best value in each column
- Include arrows to indicate if higher/lower is better
- Report mean +/- std from multiple runs where applicable
- Every number in the paper must match results.json exactly
- For detection systems: include ROC curves or precision-recall curves

### Ablation subsection
- One table showing: full system, then disable each component
- Proves every component contributes to the overall result

### Evasion analysis subsection (REQUIRED for defenses)
- Describe specific evasion strategies you tested
- Show detection rates under adaptive attacks
- Discuss the arms race: what would the attacker try next?
- Be honest about limitations

### Performance overhead subsection (REQUIRED for defenses)
- Table showing: unprotected system, your defense, existing defenses
- Metrics: latency, throughput, memory, CPU utilization
- Test under multiple load conditions
- Show that overhead is acceptable for real-world deployment

### Case studies subsection (recommended)
- Walk through specific attack instances or detection examples
- Show real traces, logs, or packet captures
- Illustrate how the system works in practice

## Tables

```latex
\begin{table}[t]
\caption{Detection performance on [Dataset]. Best results in \textbf{bold}.}
\label{tab:detection}
\centering
\begin{tabular}{lcccc}
\toprule
Method & Det. Rate $\uparrow$ & FPR $\downarrow$ & Latency (ms) $\downarrow$ & Overhead (\%) $\downarrow$ \\
\midrule
Baseline A & 91.2\% & 1.3\% & 0.8 & 2.1 \\
Baseline B & 94.7\% & 0.8\% & 2.1 & 5.7 \\
\textbf{Ours} & \textbf{97.3\%} & \textbf{0.2\%} & 1.4 & 3.2 \\
\bottomrule
\end{tabular}
\end{table}
```

Rules:
- Use booktabs (\toprule, \midrule, \bottomrule) -- no vertical lines
- Caption goes ABOVE the table
- Self-contained caption: readable without main text
- Reference every table in the text: "As shown in Table~\ref{tab:detection}..."
- Numbers must match results.json exactly
- Always include both detection metrics AND overhead metrics

## Figures

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.8\linewidth]{figures/roc_curve.pdf}
\caption{ROC curves for our defense (blue) vs. Baseline A (orange) and
Baseline B (green). Our method achieves higher true positive rates at
all false positive rates.}
\label{fig:roc}
\end{figure}
```

Rules:
- Caption goes BELOW the figure
- Self-contained caption
- Use PDF or vector format when possible
- Readable at print size (font >= 8pt in the figure)
- Reference every figure: "Figure~\ref{fig:roc} shows..."
- Use consistent colors across all figures
- Common security figures: system architecture diagrams, attack flow diagrams,
  ROC curves, overhead vs. load plots, CDF plots, timeline charts

## Discussion

This section is especially important in security papers:

- **Limitations**: Be honest about what your system does not handle.
  Reviewers respect honesty; hiding limitations gets papers rejected.
- **Potential evasions**: Discuss how an adaptive adversary might try to
  bypass your defense, and why those evasions are difficult or out of scope.
- **Deployment considerations**: What would it take to deploy this in practice?
  What are the operational costs?
- **Ethical considerations**: If you discovered vulnerabilities, describe
  your responsible disclosure process. If you collected data involving
  users, describe your IRB approval and privacy protections.
- **Broader impact**: How might this research affect security practices,
  policy, or the arms race between attackers and defenders?

## Related Work

- Organize by approach or problem area, NOT chronologically
- Cover: directly competing approaches, related but different threat models,
  techniques you build upon
- For each group, explain:
  1. What they do
  2. How your work differs (different threat model, better performance,
     stronger guarantees, more realistic evaluation)
- End with: "Unlike [prior work], our approach..."
- Every cited paper must be REAL and verifiable. Search IEEE Xplore, ACM DL,
  USENIX proceedings, and Semantic Scholar to confirm papers exist before
  citing them.

## Conclusion

- Restate the problem and your approach (one sentence each)
- Summarize key findings with concrete numbers
- State broader implications for security practices
- Suggest future work
- DO NOT introduce new results or claims here
- 0.5-1 page

## References (CRITICAL)

- EVERY reference must be a REAL, VERIFIABLE publication
- Search IEEE Xplore, ACM DL, USENIX proceedings, IACR ePrint, and
  Semantic Scholar to find and verify papers
- Fake or hallucinated citations undermine scientific integrity
- Use correct format: authors, title, venue, year
- Prefer published conference/journal papers over preprints
- Include 30-50 references for a typical security paper (security papers
  tend to cite more broadly than ML papers)
- Use appropriate citation commands for your format:
  - ACM: \cite{}, or \citet{}/\citep{} with natbib
  - IEEE: \cite{}
  - USENIX: \cite{}

## LaTeX Best Practices

- Use the venue's official style file
- Use booktabs for tables (no vertical lines)
- Use \usepackage{hyperref} for clickable references
- Define notation with \newcommand for consistency
- Use ~ for non-breaking spaces before references: Table~\ref{tab:detection}
- Compile at least twice to resolve references
- 12-15 pages for main content (check venue-specific limits)

## Common Mistakes That Get Security Papers Rejected

- No explicit threat model or a vague threat model
- Unrealistic threat model (too strong or too weak attacker)
- Defense not evaluated against adaptive adversaries
- No performance overhead measurement for defenses
- Attack demonstrated only on toy/outdated systems
- False positive rate not reported for detection systems
- No explicit contributions list in the introduction
- Claims not supported by evidence in the experiments
- Numbers in text don't match tables
- Missing error bars / single-run results
- No ablation study
- Unfair baseline comparisons
- Fabricated references
- No limitations or ethics discussion
- Poor writing quality / unclear main contribution
- No responsible disclosure for discovered vulnerabilities
