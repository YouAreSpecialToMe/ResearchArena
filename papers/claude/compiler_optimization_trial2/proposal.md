# ShapleyPass: Game-Theoretic Attribution and Interaction Analysis of Compiler Optimization Passes

## Introduction

### Context

Modern optimizing compilers like LLVM and GCC apply dozens of transformation passes---dead code elimination, loop unrolling, constant propagation, function inlining, and many more---in a carefully orchestrated pipeline. The standard optimization levels (e.g., LLVM's `-O2` with ~70 passes and `-O3` with ~100 passes) represent decades of hand-tuned heuristics for pass ordering. Yet it is well known that these fixed pipelines are suboptimal: different programs benefit from different pass combinations, and the interactions between passes are complex and poorly understood.

The **phase-ordering problem**---finding the optimal sequence of compiler optimization passes for a given program---has been studied for over four decades. Recent approaches have applied reinforcement learning (CompilerDream, Deng et al. 2025), evolutionary strategies (Shackleton, Brownlee et al. 2022), and machine learning on pass sub-sequences (MiCOMP, Jantz & Kulkarni 2017; DCO, Mammadli et al. 2024). While these methods achieve significant speedups over default pipelines, they treat the pass pipeline as a **black box**: they search for good sequences without providing insight into *why* certain pass combinations work and others do not.

### Problem Statement

Despite extensive work on phase ordering, a fundamental question remains largely unanswered: **What is the individual contribution of each optimization pass, and how do passes interact with each other?** Current approaches to understanding pass contributions are limited to:
- Ad-hoc ablation (remove one pass, measure the change) which ignores interaction effects
- Source-code dependency analysis (DCO) which captures structural but not performance relationships
- Pairwise co-occurrence statistics which miss higher-order interactions

None of these approaches provide a principled, axiomatic framework for attributing performance gains to individual passes and their combinations.

### Key Insight

We observe that a set of compiler optimization passes applied to a program constitutes a **cooperative game** in the sense of game theory: each pass is a "player," the "coalition" is the subset of passes applied, and the "value" of a coalition is the resulting performance improvement (or code size reduction). This framing immediately suggests the use of **Shapley values**---the unique attribution method satisfying efficiency (total gain is fully distributed), symmetry (equally contributing passes get equal credit), additivity, and the dummy player property (useless passes get zero credit).

Beyond individual attribution, **Shapley interaction indices** (Grabisch & Roubens, 1999; Sundararajan et al., 2020) extend this framework to quantify pairwise and higher-order synergies and antagonisms between passes. A positive interaction index indicates synergy (passes are more valuable together), while a negative index indicates antagonism (passes interfere with each other).

### Hypothesis

We hypothesize that:
1. Compiler optimization passes exhibit significant non-additive interactions that Shapley interaction indices can quantify
2. The interaction landscape reveals structured patterns---synergistic pass clusters and antagonistic pass pairs---that are consistent across programs with similar structure
3. A Shapley-guided greedy pass selector, informed by pass values and interactions, can match or outperform standard optimization levels while using fewer passes, and can outperform random search baselines given equal computational budget

## Proposed Approach

### Overview

We propose **ShapleyPass**, a framework for analyzing compiler optimization passes through the lens of cooperative game theory. The framework has three components:

1. **ShapleyPass-Attribute**: Compute approximate Shapley values for LLVM optimization passes to quantify each pass's marginal contribution to optimization outcomes
2. **ShapleyPass-Interact**: Compute Shapley interaction indices to identify synergistic and antagonistic pass pairs and higher-order interactions
3. **ShapleyPass-Select**: Use the interaction landscape to guide efficient, program-specific pass selection

### Method Details

#### Defining the Cooperative Game

For a program $P$ and a set of $n$ optimization passes $N = \{p_1, p_2, \ldots, p_n\}$, we define a characteristic function $v: 2^N \to \mathbb{R}$ where $v(S)$ is the performance metric achieved by applying the passes in $S$ to $P$. The metric can be:
- **Code size reduction**: measured as reduction in LLVM IR instruction count relative to unoptimized code
- **Execution speedup**: measured as wall-clock time improvement on representative inputs

For pass ordering within a coalition $S$, we use a fixed canonical ordering (the order in which passes appear in LLVM's `-O3` pipeline). This is a standard simplification that isolates pass *selection* from pass *ordering*, reducing the combinatorial complexity while still capturing the core interaction effects.

#### Approximating Shapley Values

The exact Shapley value requires evaluating $2^n$ coalitions, which is infeasible for $n > 20$ passes. We employ **Monte Carlo permutation sampling** (Castro et al., 2009):

1. Sample a random permutation $\pi$ of the passes
2. For each pass $p_i$ in position $k$ of $\pi$, compute $v(\{p_{\pi(1)}, \ldots, p_{\pi(k)}\}) - v(\{p_{\pi(1)}, \ldots, p_{\pi(k-1)}\})$
3. Average the marginal contributions over $M$ sampled permutations

With $M = 1000$ permutations and $n = 50$ passes, this requires $M \times n = 50{,}000$ compilations per program---feasible since individual LLVM `opt` invocations on small benchmarks take milliseconds.

We also investigate **stratified sampling** and **antithetic variates** to reduce variance, and report confidence intervals for all Shapley value estimates.

#### Computing Shapley Interaction Indices

For pairwise interactions, we compute the **Shapley Interaction Index (SII)** (Grabisch & Roubens, 1999):

$$I_{ij} = \sum_{S \subseteq N \setminus \{i,j\}} \frac{|S|!(n-|S|-2)!}{(n-1)!} \left[ v(S \cup \{i,j\}) - v(S \cup \{i\}) - v(S \cup \{j\}) + v(S) \right]$$

This measures the average excess (or deficit) of having both passes $i$ and $j$ together beyond their individual contributions. We approximate this using the same Monte Carlo framework, leveraging the efficient KernelSHAP-IQ method (Fumagalli et al., 2024) when applicable.

For computational efficiency, we focus on:
- All pairwise interactions among the top-$k$ passes (by Shapley value)
- Selective higher-order (3-way) interactions for the most synergistic pairs

#### ShapleyPass-Select: Interaction-Guided Pass Selection

We propose a greedy forward-selection algorithm informed by Shapley values and interactions:

1. **Initialize**: Start with the empty set $S = \emptyset$
2. **Score**: For each candidate pass $p_i \notin S$, compute a score combining its Shapley value and its interaction with the current set: $\text{score}(p_i) = \phi_i + \lambda \sum_{p_j \in S} I_{ij}$
3. **Select**: Add the pass with the highest score to $S$
4. **Terminate**: Stop when adding any pass decreases performance or a pass budget is reached

The hyperparameter $\lambda$ balances individual contribution and interaction synergy. We also compare against a portfolio approach where we pre-compute Shapley-optimal pass subsets for representative program clusters.

### Key Innovations

1. **First application of Shapley values to compiler pass attribution**: We provide the first axiomatic, game-theoretically grounded attribution of optimization contributions to individual compiler passes
2. **Quantification of pass interactions**: Unlike prior work that captures pairwise co-occurrence (MiCOMP) or source-code dependency (DCO), we measure the actual performance-level synergy and antagonism between passes
3. **Interaction-guided selection**: Our greedy selector exploits the interaction landscape, providing an interpretable alternative to black-box ML approaches
4. **Empirical pass interaction atlas**: We produce a reusable dataset of pass interaction measurements across standard benchmarks

## Related Work

### Phase Ordering and Pass Selection

The phase ordering problem---finding the best order of compiler optimization passes---dates to the 1980s (Whitfield & Soffa, 1997). **Kulkarni & Cavazos (2012)** applied neuro-evolution (NEAT) to learn pass orderings in the Jikes RVM, demonstrating that ML could outperform fixed orderings. **MiCOMP (Jantz & Kulkarni, 2017)** introduced optimization sub-sequences, clustering related passes and using ML to select sub-sequence combinations, achieving 90% of available speedup while exploring <0.001% of the search space. **DCO (Mammadli et al., 2024)** modeled pass dependencies via source-code and performance dependence graphs to construct sub-sequences, achieving 22% speedup over `-O3`.

Our work differs fundamentally: rather than searching for good pass sequences, we *explain* why certain combinations work via axiomatic attribution. Our Shapley interaction indices capture performance-level synergies that source-code dependency (DCO) cannot detect.

### ML and RL for Compiler Optimization

**MLGO (Trofin et al., 2021)** integrated ML into LLVM for inlining decisions, achieving 7% code size reduction. **CompilerGym (Cummins et al., 2022)** provided RL environments for compiler optimization. **CompilerDream (Deng et al., 2025)** learned a world model of compiler passes for RL-based optimization, leading the CompilerGym leaderboard.

These methods optimize pass sequences but do not explain pass contributions. ShapleyPass is complementary: our interaction analysis could inform the reward shaping and state representation of these RL approaches.

### Theoretical Foundations of Phase Ordering

**Wang et al. (2024)** proved that solving the phase ordering problem is not equivalent to generating globally optimal code, introducing IIBO (infinitive iterative bi-directional optimization) which applies both forward and reverse optimizations. Their work motivates our approach: if phase ordering alone is insufficient, understanding *why* passes interact is essential for progress.

### Shapley Values and Interaction Indices

The Shapley value (Shapley, 1953) is the cornerstone of cooperative game theory attribution. **SHAP (Lundberg & Lee, 2017)** popularized Shapley values for ML feature attribution. **The Shapley Interaction Index** (Grabisch & Roubens, 1999) and **Shapley-Taylor Interaction Index** (Sundararajan et al., 2020) extended attribution to interactions. **KernelSHAP-IQ (Fumagalli et al., 2024)** and the **shapiq library (Muschalik et al., 2024)** made higher-order interaction computation practical.

Our work is the first to apply this rich game-theoretic framework to compiler optimization, bridging two previously disconnected fields.

## Experiments

### Setup

**Compiler**: LLVM 17+ via the `opt` tool for IR-level optimization
**Pass Set**: The ~50 most common LLVM transform passes in the `-O3` pipeline (excluding analysis-only passes)
**Benchmarks**:
- **cBench** (23 programs): Standard benchmark for compiler optimization research, used in CompilerGym
- **PolyBench/C 4.2** (30 kernels): Polyhedral benchmarks for loop-heavy numerical code
- **MiBench** (subset): Embedded systems benchmarks for diversity

**Metrics**:
- Primary: LLVM IR instruction count reduction (deterministic, fast to measure)
- Secondary: Execution time (where feasible, with multiple runs for stability)

### Experiment 1: Shapley Value Attribution

For each benchmark program, compute approximate Shapley values for all ~50 passes using $M = 1000$ Monte Carlo permutations.

**Analysis**:
- Rank passes by Shapley value; identify the most and least valuable passes per program and across programs
- Compute the variance of Shapley values across programs (which passes are universally useful vs. program-specific?)
- Compare with naive ablation (leave-one-out) to quantify the gap between additive attribution and Shapley attribution

**Expected results**: A small set of passes (dead code elimination, constant propagation, GVN) will have consistently high Shapley values, while most passes will have near-zero values for any given program. The Shapley ranking will differ significantly from leave-one-out ranking, demonstrating the importance of interaction effects.

### Experiment 2: Interaction Landscape

Compute pairwise Shapley interaction indices for the top-20 passes (by average Shapley value).

**Analysis**:
- Visualize the interaction matrix as a heatmap
- Identify the strongest synergistic and antagonistic pass pairs
- Cluster passes by interaction pattern using hierarchical clustering
- Compare interaction-based clusters with DCO's source-code dependency clusters

**Expected results**: Strong synergies between passes that create and exploit optimization opportunities (e.g., constant propagation + dead code elimination, inlining + GVN). Strong antagonisms between passes that compete for the same opportunities (e.g., different loop transformations).

### Experiment 3: Shapley-Guided Pass Selection

Evaluate ShapleyPass-Select against baselines:
- **LLVM -O3**: Default pipeline (~100 passes)
- **LLVM -O2**: Reduced pipeline (~70 passes)
- **LLVM -Oz**: Size-optimized pipeline
- **Random search**: Random subsets of passes, same compilation budget
- **Greedy ablation**: Start from -O3, greedily remove passes that hurt least
- **Top-k Shapley**: Simply take the top-$k$ passes by Shapley value (no interaction term)

**Metrics**: Code size reduction and speedup, measured against unoptimized baseline. Report mean and standard deviation across benchmarks.

**Expected results**: ShapleyPass-Select with interaction term ($\lambda > 0$) outperforms Top-k Shapley ($\lambda = 0$), demonstrating the value of interaction information. Both should match or approach `-O3` performance with significantly fewer passes.

### Experiment 4: Transferability Analysis

Analyze whether the interaction landscape transfers across programs:
- Compute interaction matrices for each program
- Measure correlation between per-program interaction matrices
- Cluster programs by their interaction fingerprints
- Evaluate: can a cluster-average interaction matrix guide pass selection for new programs in the cluster?

**Expected results**: Programs with similar structure (e.g., loop-heavy PolyBench kernels) share similar interaction patterns, enabling transfer of pass selection strategies.

### Experiment 5: Computational Cost Analysis

Report:
- Wall-clock time for Shapley value computation per program
- Convergence of Shapley estimates as a function of sample count $M$
- Comparison with iterative compilation baselines in terms of compilations needed to reach a given performance level

### Resource Estimates

- Each `opt` invocation: ~10-100ms on small benchmarks
- Per program, Experiment 1: 50,000 compilations x 50ms = ~42 minutes
- Total for 53 programs: ~37 hours at full sequential
- With 2 CPU cores: ~18.5 hours
- **Optimization**: Many programs are small (cBench averages <1000 LOC); use program-level parallelism and early stopping for converged estimates
- **Realistic estimate with optimizations**: 6-8 hours total

## Success Criteria

The hypothesis is **confirmed** if:
1. Shapley interaction indices reveal statistically significant non-zero interactions (p < 0.05 after multiple-testing correction) for at least 30% of pass pairs
2. The interaction-based clustering of passes differs meaningfully from source-code dependency clustering (adjusted Rand index < 0.5), demonstrating that performance interactions are not captured by structural analysis alone
3. ShapleyPass-Select matches `-O3` code size reduction to within 2% while using at most 50% of the passes
4. The interaction landscape shows measurable cross-program consistency (average Spearman correlation > 0.3 between per-program interaction matrices within clusters)

The hypothesis is **refuted** if:
- Pass interactions are negligible (all interaction indices near zero), suggesting passes are essentially additive
- Shapley-guided selection performs no better than Top-k selection by leave-one-out ranking

## References

1. Wang, Y., Chen, H., & Wang, K. (2024). Beyond the Phase Ordering Problem: Finding the Globally Optimal Code w.r.t. Optimization Phases. arXiv:2410.03120.
2. Mammadli, R., et al. (2024). Efficient compiler optimization by modeling passes dependence. CCF Transactions on High Performance Computing.
3. Deng, C., Wu, J., Feng, N., Wang, J., & Long, M. (2025). CompilerDream: Learning a Compiler World Model for General Code Optimization. KDD 2025. arXiv:2404.16077.
4. Jantz, M. R. & Kulkarni, P. A. (2017). MiCOMP: Mitigating the Compiler Phase-Ordering Problem Using Optimization Sub-Sequences and Machine Learning. ACM TACO, 14(3).
5. Kulkarni, S. & Cavazos, J. (2012). Mitigating the compiler optimization phase-ordering problem using machine learning. OOPSLA 2012.
6. Trofin, M., Qian, Y., Brevdo, E., et al. (2021). MLGO: A Machine Learning Guided Compiler Optimizations Framework. arXiv:2101.04808.
7. Cummins, C., et al. (2022). CompilerGym: Robust, Performant Compiler Optimization Environments for AI Research. CGO 2022.
8. Muschalik, M., Baniecki, H., Fumagalli, F., et al. (2024). shapiq: Shapley Interactions for Machine Learning. NeurIPS 2024.
9. Fumagalli, F., Muschalik, M., Kolpaczki, P., Hüllermeier, E., & Hammer, B. (2024). KernelSHAP-IQ: Weighted Least-Square Optimization for Shapley Interactions. ICML 2024.
10. Sundararajan, M., Dhamdhere, K., & Agarwal, A. (2020). The Shapley Taylor Interaction Index. ICML 2020.
11. Grabisch, M. & Roubens, M. (1999). An axiomatic approach to the concept of interaction among players in cooperative games. International Journal of Game Theory, 28(4), 547-565.
12. Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games, 2(28), 307-317.
13. Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
14. Castro, J., Gómez, D., & Tejada, J. (2009). Polynomial calculation of the Shapley value based on sampling. Computers & Operations Research, 36(5), 1726-1730.
15. Brownlee, A. E. I., Callan, J., Even-Mendoza, K., et al. (2022). Optimizing LLVM Pass Sequences with Shackleton. arXiv:2201.13305.
