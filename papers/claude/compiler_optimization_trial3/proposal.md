# ShapleyPass: Quantifying Higher-Order Interactions Among Compiler Optimization Passes via Shapley Interaction Indices

## Introduction

### Context

Modern optimizing compilers such as LLVM offer over 100 optimization passes that transform intermediate representations (IR) to improve metrics like code size, execution speed, and energy consumption. The **phase-ordering problem** — selecting which passes to apply and in what order — remains one of the most challenging open problems in compiler optimization. The search space is combinatorially explosive: even restricting to a binary on/off selection among 60 passes yields 2^60 possible configurations, and considering ordering makes the space even larger.

### Problem Statement

Existing approaches to pass selection fall into two broad categories, each with significant limitations:

1. **Search-based methods** (random search, genetic algorithms, Bayesian optimization) explore the configuration space without understanding *why* certain combinations work. They are expensive and their solutions do not transfer across programs.

2. **Learning-based methods** (MiCOMP, CompilerDream, Compiler-R1) use machine learning to predict good configurations, but treat the pass selection problem as a black box — they learn correlations without providing interpretable insights into pass interactions.

3. **Dependence modeling methods** (ODG-based approaches, GroupTuner) analyze relationships between passes, but are limited to **pairwise** interactions. They model whether pass A enables or disables pass B, but cannot capture **higher-order synergies** — situations where passes A, B, and C together produce effects that no pair of them achieves.

This limitation to pairwise analysis is critical because compiler passes often exhibit complex multi-way interactions. For example, loop unrolling may create opportunities that are only exploitable when *both* constant propagation *and* dead code elimination are subsequently applied — a three-way synergy invisible to pairwise analysis.

### Key Insight

We propose to bridge the gap between **interpretability** and **effectiveness** in compiler pass selection by applying **Shapley Interaction Indices** — a principled framework from cooperative game theory and machine learning explainability — to decompose the performance contribution of compiler optimization passes into individual effects, pairwise interactions, and higher-order synergies (up to order k).

The Shapley value is the unique attribution method satisfying four desirable axioms (efficiency, symmetry, linearity, null player). The Shapley-Taylor Interaction Index extends this to interactions among groups of features (here, passes), providing a mathematically grounded decomposition of *why* certain pass combinations outperform others.

### Hypothesis

**Higher-order interactions (order ≥ 3) among compiler optimization passes account for a significant fraction of the performance variance unexplained by individual pass effects and pairwise interactions alone. An interaction-aware pass selection algorithm that explicitly exploits these higher-order synergies will outperform methods that consider only individual or pairwise pass effects.**

## Proposed Approach

### Overview

We frame compiler pass selection as a **cooperative game** where:
- **Players** are the individual optimization passes (e.g., `mem2reg`, `instcombine`, `loop-unroll`, `gvn`, `licm`, etc.)
- **Value function** v(S) maps a subset S of passes to a performance metric (e.g., IR instruction count reduction relative to unoptimized code)
- **Shapley Interaction Indices** decompose v into contributions from individual passes, pairs, triples, and higher-order groups

### Method Details

#### Phase 1: Game Construction

For each benchmark program P in our evaluation suite:
1. Select the top-K most impactful LLVM optimization passes (K ≈ 20-25) based on individual marginal contribution from -O3
2. Define the value function v_P(S) = reduction in IR instruction count when applying pass subset S (via LLVM's `opt` tool or CompilerGym)
3. Evaluate v_P on a strategically sampled collection of pass subsets (using the sampling strategies from the shapiq framework)

#### Phase 2: Interaction Computation

Using the shapiq library's efficient algorithms (KernelSHAP-IQ, permutation sampling):
1. Compute **first-order Shapley values** φ_i for each pass i — the average marginal contribution
2. Compute **second-order Shapley interaction indices** φ_{i,j} for pass pairs — capturing pairwise synergy (+) or redundancy (-)
3. Compute **third-order interaction indices** φ_{i,j,k} for pass triples — capturing irreducible three-way effects
4. Optionally compute fourth-order interactions for selected high-synergy groups

#### Phase 3: Interaction Analysis

1. **Variance decomposition**: What fraction of performance variance is explained by order-1 (individual), order-2 (pairwise), and order-3+ (higher-order) interactions?
2. **Synergy detection**: Which pass groups exhibit the strongest positive higher-order interactions (synergies where the group effect exceeds the sum of sub-group effects)?
3. **Redundancy detection**: Which pass groups exhibit negative interactions (diminishing returns from combining them)?
4. **Program-dependence analysis**: How stable are interaction patterns across different programs? Are there universal synergies vs. program-specific ones?

#### Phase 4: Interaction-Guided Pass Selection

Design an algorithm that exploits the discovered interaction structure:

1. **Greedy interaction-aware selection**: Start with the pass having the highest individual Shapley value. At each step, add the pass that maximizes not just its individual contribution but also its positive interactions with already-selected passes (considering both pairwise and higher-order terms).

2. **Synergy-group seeding**: Identify the top-k highest-synergy groups (e.g., triples with the largest positive third-order interaction). Seed the selection with complete synergy groups rather than individual passes.

3. **Redundancy pruning**: Remove passes from the candidate set if they have strong negative interactions with already-selected passes.

### Key Innovations

1. **First application of Shapley interaction indices to compiler optimization** — bringing a principled game-theoretic framework to a combinatorial optimization problem traditionally addressed by heuristics or black-box learning.

2. **Higher-order interaction analysis** — going beyond the pairwise dependence models of prior work (ODG, MiCOMP) to capture irreducible multi-way synergies among passes.

3. **Interpretable pass selection** — unlike RL-based approaches (CompilerDream, Compiler-R1), our method produces human-readable interaction maps that explain *why* certain pass combinations work well.

4. **Interaction-guided algorithm** — a practical pass selection algorithm that exploits the discovered synergy structure for improved optimization without expensive search.

## Related Work

### Phase Ordering and Pass Selection

The phase-ordering problem has been studied extensively. **Ashouri et al. (2017)** introduced MiCOMP, which clusters LLVM -O3 passes into sub-sequences and uses ML to predict speedups, achieving 90% of available speedup while exploring <0.001% of the space. **Gao et al. (2024)** proposed modeling pass dependence through an Optimization Dependence Graph (ODG) capturing source-code and pairwise performance dependencies, achieving 22% runtime improvement over -O3. **Gao et al. (2025)** introduced GroupTuner, which applies localized mutations to coherent option groups, achieving 12.39% improvement over -O3. All these methods are limited to pairwise or heuristic groupings — our work provides a principled higher-order interaction analysis.

### ML-Based Compiler Optimization

**Deng et al. (2025)** proposed CompilerDream, a model-based RL approach that learns a "world model" of compiler transformations, leading the CompilerGym leaderboard. **Pan et al. (2025)** introduced Compiler-R1, the first RL-driven LLM framework for compiler auto-tuning, achieving 8.46% IR reduction over -Oz. **VenkataKeerthy et al. (2024)** presented ML-Compiler-Bridge for efficient ML-compiler integration. These approaches treat pass selection as a black box; our work provides interpretable decomposition of pass contributions.

### Shapley Values and Interaction Indices

**Dhamdhere et al. (2020)** introduced the Shapley-Taylor Interaction Index, extending Shapley values to attribute predictions to feature interactions up to order k. **Muschalik et al. (2024)** developed shapiq, an open-source Python package unifying algorithms for computing Shapley values and any-order interactions efficiently. **Wever et al. (2026)** applied Shapley interactions to hyperparameter optimization in HyperSHAP. Our work is the first to apply these tools to compiler pass selection.

### Compiler Infrastructure

**Lattner & Adve (2004)** designed LLVM, which provides the pass infrastructure we build on. **Cummins et al. (2022)** created CompilerGym, providing RL environments for compiler optimization with millions of programs. **Lopes & Regehr (2018)** surveyed future directions for optimizing compilers, identifying the need for better understanding of pass interactions.

## Experiments

### Experimental Setup

**Compiler Infrastructure**: LLVM via CompilerGym's LLVM environment (or direct `opt` invocation if CompilerGym is unavailable)

**Benchmark Programs**:
- cBench suite (23 programs covering diverse domains: signal processing, cryptography, text processing, numerical computation)
- PolyBench/C suite (30 numerical kernels) if time permits

**Pass Selection**: Top 20-25 most impactful passes from LLVM's -O3 pipeline, selected by individual marginal contribution in a preliminary screening

**Performance Metric**:
- Primary: IR instruction count reduction (fast to measure, deterministic)
- Secondary: Binary size reduction

**Interaction Computation**:
- shapiq library with KernelSHAP-IQ and permutation sampling
- Interaction orders: k = 1, 2, 3 (and k = 4 for selected high-synergy groups)
- Sufficient samples per program for convergence (guided by shapiq's built-in convergence diagnostics)

### Planned Experiments

**Experiment 1: Variance Decomposition**
- For each benchmark program, decompose the performance variance into contributions from order-1, order-2, and order-3+ interactions
- Report the fraction of variance explained at each order
- **Expected result**: Order-3+ interactions explain 10-25% of variance beyond what pairwise analysis captures

**Experiment 2: Interaction Structure Analysis**
- Visualize the interaction network (passes as nodes, edges weighted by pairwise Shapley interaction, hyperedges for higher-order)
- Identify the top-10 most synergistic and most redundant pass groups at each order
- Analyze stability across programs (which interactions are universal vs. program-specific)
- **Expected result**: Certain pass triples (e.g., loop transformations + scalar optimizations + dead code elimination) consistently show strong three-way synergy

**Experiment 3: Interaction-Guided Pass Selection**
- Compare our interaction-guided selection algorithm against:
  - **Baselines**: LLVM -O1, -O2, -O3, -Oz, -Os
  - **Individual Shapley greedy**: Select passes by decreasing individual Shapley value (no interactions)
  - **Pairwise-only greedy**: Use only first and second-order interactions
  - **Random search** (with equivalent computational budget)
  - **Genetic algorithm** (with equivalent computational budget)
- **Expected result**: Interaction-guided selection (using order-3 interactions) outperforms pairwise-only greedy by 3-8% on average

**Experiment 4: Computational Cost Analysis**
- Measure the time to compute Shapley interactions at each order
- Compare the cost of interaction computation + guided selection vs. direct search methods with equivalent wall-clock budget
- **Expected result**: Interaction computation amortizes well — computed once per program family, reused for all programs in that family

**Experiment 5: Transferability Analysis**
- Cluster programs by their interaction structure (using cosine similarity of interaction vectors)
- Test whether interaction-guided selections from one program transfer to similar programs
- **Expected result**: Programs with similar interaction fingerprints benefit from transferred selections

### Ablation Studies

1. **Order ablation**: Compare selection using only order-1, order-1+2, order-1+2+3 interactions
2. **Number of passes**: Vary K from 10 to 30 and measure impact on interaction quality and selection effectiveness
3. **Sample budget**: Vary the number of value function evaluations and measure convergence of interaction estimates

## Success Criteria

The hypothesis is **confirmed** if:
1. Third-order (or higher) Shapley interaction indices are statistically significant for ≥30% of pass triples tested
2. The variance decomposition shows ≥10% additional variance explained by order-3+ interactions beyond pairwise
3. The interaction-guided pass selection algorithm outperforms the pairwise-only greedy baseline on ≥60% of benchmark programs

The hypothesis is **refuted** if:
1. Higher-order interactions are negligible (explain <5% additional variance)
2. Pairwise analysis is sufficient to capture essentially all pass interaction structure
3. The interaction-guided algorithm does not outperform simple greedy selection

Even if refuted, the negative result is publishable — it would demonstrate that pairwise modeling is sufficient for compiler pass selection, validating the design choices of ODG and similar methods.

## References

1. Ashouri, A. H., Bignoli, A., Palermo, G., Silvano, C., Kulkarni, S., & Cavazos, J. (2017). MiCOMP: Mitigating the Compiler Phase-Ordering Problem Using Optimization Sub-Sequences and Machine Learning. *ACM Transactions on Architecture and Code Optimization*, 14(3), Article 29.

2. Gao, B., Yao, M., Wang, Z., Liu, D., Li, D., Chen, X., & Guo, Y. (2024). Efficient compiler optimization by modeling passes dependence. *CCF Transactions on High Performance Computing*, 6, 197-211.

3. Gao, B., Yao, M., Wang, Z., Liu, D., Li, D., Chen, X., & Guo, Y. (2025). GroupTuner: Efficient Group-Aware Compiler Auto-Tuning. In *Proceedings of the 26th ACM SIGPLAN/SIGBED International Conference on Languages, Compilers, and Tools for Embedded Systems (LCTES '25)*.

4. Deng, C., Wu, J., Feng, N., Wang, J., & Long, M. (2025). CompilerDream: Learning a Compiler World Model for General Code Optimization. In *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '25)*.

5. Pan, H., Lin, H., Luo, H., Liu, Y., Yao, K., Zhang, L., Xing, M., & Wu, Y. (2025). Compiler-R1: Towards Agentic Compiler Auto-tuning with Reinforcement Learning. In *Advances in Neural Information Processing Systems (NeurIPS '25)*.

6. VenkataKeerthy, S., Jain, S., Kalvakuntla, U., Gorantla, P. S., Chitale, R. S., Brevdo, E., Cohen, A., Trofin, M., & Upadrasta, R. (2024). The Next 700 ML-Enabled Compiler Optimizations. In *Proceedings of the 33rd ACM SIGPLAN International Conference on Compiler Construction (CC '24)*.

7. Dhamdhere, K., Agarwal, A., & Sundararajan, M. (2020). The Shapley Taylor Interaction Index. In *Proceedings of the 37th International Conference on Machine Learning (ICML '20)*.

8. Muschalik, M., Baniecki, H., Fumagalli, F., Kolpaczki, P., Hammer, B., & Hüllermeier, E. (2024). shapiq: Shapley Interactions for Machine Learning. In *Advances in Neural Information Processing Systems (NeurIPS '24)*, Datasets and Benchmarks Track.

9. Wever, M., Muschalik, M., Fumagalli, F., & Lindauer, M. (2026). HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization. In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI '26)*.

10. Cummins, C., Wasti, B., Guo, J., Cui, B., Ansel, J., Gomez, S., Jain, S., Liu, J., Teytaud, O., Steiner, B., Tian, Y., & Leather, H. (2022). CompilerGym: Robust, Performant Compiler Optimization Environments for AI Research. In *Proceedings of the IEEE/ACM International Symposium on Code Generation and Optimization (CGO '22)*.

11. Lopes, N. P., & Regehr, J. (2018). Future Directions for Optimizing Compilers. *arXiv preprint arXiv:1809.02161*.

12. Lattner, C., & Adve, V. (2004). LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation. In *Proceedings of the International Symposium on Code Generation and Optimization (CGO '04)*.

13. Tsai, C.-P., Yeh, C.-K., & Ravikumar, P. (2023). Faith-Shap: The Faithful Shapley Interaction Index. *Journal of Machine Learning Research*, 24(94), 1-42.
