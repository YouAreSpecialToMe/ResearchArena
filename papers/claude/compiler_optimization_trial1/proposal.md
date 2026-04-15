# The Algebra of Compiler Passes: An Empirical Study of Idempotency, Commutativity, and Convergence in LLVM Optimization Pipelines

## Introduction

### Context

Modern compilers such as LLVM apply dozens of optimization passes in carefully orchestrated sequences to transform intermediate representations (IR) into efficient machine code. The LLVM infrastructure offers over 100 general-purpose optimization passes, each implementing a specific transformation such as dead code elimination, loop unrolling, constant propagation, or instruction combining. The effectiveness of the overall optimization pipeline depends critically on which passes are selected and in what order they are applied -- the well-known *phase-ordering problem* (Kulkarni and Cavazos, 2012).

Despite decades of research on phase ordering, the fundamental mathematical properties governing how passes compose remain poorly understood. Compiler engineers intuitively know that some passes "enable" others, some passes "undo" each other's work, and some passes have no effect when applied twice. These observations correspond to well-defined algebraic properties -- constructive/destructive interference, commutativity, and idempotency -- yet no systematic empirical characterization of these properties exists for real-world compiler passes.

### Problem Statement

The phase-ordering problem has been attacked primarily through search-based methods (genetic algorithms, reinforcement learning, random search) that treat the compiler as a black box. Recent approaches include model-based RL (CompilerDream; Deng et al., 2024), structure-aware genetic algorithms (Pan et al., 2025), and sub-sequence decomposition (MiCOMP; Ashouri et al., 2017). While these methods achieve impressive results, they share a common limitation: they do not exploit the *intrinsic algebraic structure* of the pass space. Understanding which passes commute, which are idempotent, and which interfere constructively or destructively could dramatically reduce the effective search space and provide principled guidance for pass ordering.

Furthermore, practitioners have observed that iteratively applying optimization pipelines can lead to oscillating behavior -- passes that "flip-flop" the IR without converging (Regehr, 2018). This oscillation phenomenon has been noted anecdotally but never systematically characterized.

### Key Insight

We propose to treat LLVM optimization passes as operators on a structured space of IR programs and systematically characterize their algebraic properties. By measuring idempotency, pairwise commutativity, and interference across diverse benchmarks, we can build a *pass interaction algebra* that reveals the compositional structure of the optimization space. This algebra can then be used to (1) explain observed phenomena like pass oscillation, (2) predict good orderings without exhaustive search, and (3) identify redundant or conflicting passes in existing pipelines.

### Hypothesis

We hypothesize that LLVM optimization passes exhibit rich algebraic structure -- specifically, that a significant fraction of passes are empirically idempotent, that commutativity is highly non-uniform across pass pairs, and that this algebraic structure can be exploited to construct competitive optimization sequences with dramatically reduced search effort compared to black-box methods.

## Proposed Approach

### Overview

We will conduct a large-scale empirical study of the algebraic properties of LLVM optimization passes, comprising five interconnected analyses:

1. **Idempotency Characterization**: For each pass, determine whether applying it twice produces the same IR as applying it once, across a diverse benchmark suite.

2. **Pairwise Commutativity Analysis**: For all pairs of passes, determine whether swapping their order produces equivalent IR, building a commutativity graph.

3. **Interference Quantification**: For all pass pairs, measure whether their combined effect on code quality (instruction count) is greater than (constructive), less than (destructive), or equal to (neutral) the sum of their individual effects.

4. **Convergence and Cycle Analysis**: Apply standard optimization pipelines iteratively and characterize the convergence behavior -- how quickly does the IR reach a fixed point, and when does oscillation occur?

5. **Algebra-Guided Ordering**: Use the discovered algebraic properties to construct a lightweight pass ordering heuristic and compare it against standard optimization levels and state-of-the-art methods.

### Method Details

#### Pass Selection and Benchmark Suite

We will study the ~50 most commonly used LLVM transform passes available through the `opt` tool with the LLVM New Pass Manager. Benchmarks will include:
- **cBench**: A standard compiler optimization benchmark suite used in CompilerGym
- **PolyBench**: Polyhedral benchmark suite for loop-intensive kernels
- **LLVM test-suite programs**: Diverse real-world programs

We target 50-100 benchmark programs to ensure statistical robustness.

#### Idempotency Measurement

For each pass P and each benchmark program B:
1. Compile B to LLVM IR at -O0 (unoptimized baseline)
2. Apply P once: IR_1 = P(B)
3. Apply P twice: IR_2 = P(P(B))
4. Compare IR_1 and IR_2 using both structural hash comparison and instruction count

A pass is **strongly idempotent** if IR_1 = IR_2 structurally for all benchmarks. It is **weakly idempotent** if the instruction counts match but IR differs in naming or ordering. We report the idempotency rate across benchmarks to characterize how program-dependent this property is.

#### Pairwise Commutativity Analysis

For each pair of passes (P_i, P_j) and each benchmark B:
1. Apply P_i then P_j: IR_ij = P_j(P_i(B))
2. Apply P_j then P_i: IR_ji = P_i(P_j(B))
3. Compare IR_ij and IR_ji structurally and by instruction count

We classify pairs as: **strongly commutative** (structurally identical IR for all benchmarks), **weakly commutative** (same instruction count), or **non-commutative** (different instruction counts). We build a commutativity graph where edges indicate commuting pairs and analyze its structure (clustering coefficient, connected components).

#### Interference Quantification

For each pass pair (P_i, P_j) and benchmark B:
1. Measure the instruction count reduction from P_i alone: delta_i
2. Measure the instruction count reduction from P_j alone: delta_j
3. Measure the instruction count reduction from P_i followed by P_j: delta_ij
4. Compute the interference score: I(P_i, P_j) = delta_ij - (delta_i + delta_j)

Positive interference indicates synergy (passes enable each other), negative interference indicates destructive interaction (passes undo each other's work), and zero indicates independence. We construct a weighted pass interaction graph and analyze clusters of synergistic and antagonistic passes.

#### Convergence and Cycle Analysis

For each benchmark B:
1. Apply the standard -O2 pipeline as a single "mega-pass" iteratively up to 20 times
2. Record the instruction count after each iteration
3. Fit a convergence model: exponential decay, damped oscillation, or limit cycle
4. Identify programs that exhibit oscillation (non-monotonic instruction count sequences)

Additionally, for small subsets of passes known to interact (e.g., jump threading + loop canonicalization + SimplifyCFG), we apply them cyclically and detect minimal cycling sequences.

#### Algebra-Guided Pass Ordering

Using the measured algebraic properties, we construct a pass ordering heuristic:
1. **Pruning**: Remove redundant passes (strongly idempotent passes that appear consecutively; commutative passes that can be deduplicated)
2. **Synergy chaining**: Order passes to maximize constructive interference by placing synergistic pairs adjacent in the pipeline
3. **Anti-interference**: Separate destructively interfering passes with buffer passes that "protect" their respective optimizations

We evaluate this heuristic against:
- LLVM -O1, -O2, -O3, -Oz baselines
- Random pass ordering (averaged over multiple seeds)
- Greedy ordering (always apply the pass with the greatest marginal benefit)

### Key Innovations

1. **First systematic algebraic characterization**: No prior work has measured idempotency and commutativity of LLVM passes at scale across diverse benchmarks.

2. **Interference as a first-class metric**: While pass "dependencies" have been studied, quantifying constructive vs. destructive interference as a continuous metric is novel.

3. **Convergence analysis of iterative compilation**: Formalizing when optimization pipelines reach fixed points vs. oscillate fills a gap between anecdotal observations and rigorous understanding.

4. **Algebra-guided ordering**: Using intrinsic pass properties rather than black-box search to guide ordering is a fundamentally different approach to the phase-ordering problem.

## Related Work

### Phase-Ordering Problem

The phase-ordering problem -- selecting and ordering compiler optimization passes -- has been studied for decades. Kulkarni and Cavazos (2012) formulated it as a Markov process and used neural networks to predict the next pass based on code features. MiCOMP (Ashouri et al., 2017) decomposed the problem into optimization sub-sequences, achieving 1.31x average speedup over baselines. POSET-RL (Jain et al., 2022) used reinforcement learning with partially ordered sets to guide subsequence generation. CompilerDream (Deng et al., 2024) learned a world model of the compiler using model-based RL, achieving state-of-the-art results on CompilerGym. Pan et al. (2025) introduced a synergy-guided genetic algorithm for nested LLVM pipelines, achieving 13.62% instruction count reduction over -Oz. DeCOS (2025) combined LLM-guided initialization with RL for data-efficient pass sequence search.

**How our work differs**: All these methods treat passes as opaque actions in a search space. We characterize the *intrinsic properties* of individual passes and their pairwise interactions, providing a complementary analytical foundation that could improve any of these search methods.

### Pass Dependence Modeling

Liu et al. (2024) constructed optimization dependence graphs (ODGs) capturing source code dependence and pairwise performance dependence between passes. This is the closest work to ours, but focuses on sequential dependence (which pass must come before which) rather than algebraic properties (commutativity, idempotency, interference magnitude).

### Program Representation for Compiler Optimization

ProGraML (Cummins et al., 2021) introduced graph-based program representations capturing control, data, and call relations for ML-driven optimization. CompilerGym (Cummins et al., 2021) provided standardized RL environments for compiler optimization. These provide infrastructure we build upon, but focus on program representation rather than pass characterization.

### Compiler Pass Interactions

The phenomenon of passes undoing each other's work has been documented informally. Regehr (2018) observed oscillating behavior in LLVM's optimization pipeline, where jump threading and loop canonicalization repeatedly flip IR structure. The Faultlore blog documented how passes can destroy information needed by subsequent passes. Our work provides the first systematic quantification of these phenomena.

## Experiments

### Setup

- **Compiler**: LLVM 18.1.x (latest stable release), using `opt` with the New Pass Manager
- **Passes**: ~50 transform passes commonly used in -O1 through -O3 pipelines
- **Benchmarks**: cBench (23 programs), PolyBench (30 kernels), selected LLVM test-suite programs (20-30 programs)
- **Metrics**: Instruction count (primary), structural IR hash, basic block count, branch count
- **Hardware**: 2-core CPU, 128GB RAM (all experiments are I/O and compilation-based, not compute-intensive)

### Experiment 1: Idempotency Survey

- Apply each of ~50 passes to each of ~75 benchmarks, once and twice
- Report: fraction of passes that are strongly/weakly idempotent, per-benchmark variation
- Expected: Most simplification passes (DCE, CSE) are idempotent; canonicalization passes may not be

### Experiment 2: Commutativity Matrix

- Test all ~1,225 unique pass pairs on ~75 benchmarks
- Report: commutativity rate, graph structure, clustering of commutative/non-commutative passes
- Expected: Orthogonal passes (e.g., loop optimizations vs. algebraic simplifications) commute more often

### Experiment 3: Interference Measurement

- Measure constructive/destructive interference for all ~1,225 pairs on representative subset (~20 benchmarks)
- Report: interference distribution, most constructive/destructive pairs, interference heatmap
- Expected: Known enabling relationships (e.g., mem2reg enables SROA) will show strong positive interference

### Experiment 4: Convergence Analysis

- Apply -O2 iteratively (up to 20 iterations) on all benchmarks
- Report: convergence curves, time-to-fixpoint distribution, oscillation detection
- Expected: Most programs converge within 2-3 iterations; some exhibit persistent oscillation

### Experiment 5: Cycle Detection

- Systematically apply small pass subsets (2-4 passes) cyclically, detect IR state cycles
- Report: minimal cycling sequences, cycle lengths, affected program characteristics
- Expected: Cycles involve canonicalization vs. transformation passes (e.g., SimplifyCFG + JumpThreading)

### Experiment 6: Algebra-Guided Ordering

- Construct pass sequences using algebraic properties (synergy chaining + interference avoidance)
- Compare against -O1, -O2, -O3, -Oz, random ordering, greedy ordering
- Report: instruction count reduction, number of passes needed, compilation time
- Expected: Competitive with -O2/-O3 using fewer passes, or modest improvement (5-10%)

### Computational Budget

- Experiments 1-3: ~50 passes x 75 benchmarks x ~3 configurations each = ~11,250 compilations (estimated ~4 hours)
- Experiments 4-5: ~75 benchmarks x 20 iterations + cycle detection = ~2,000 compilations (estimated ~1 hour)
- Experiment 6: Ordering comparison on ~75 benchmarks x 6 methods x 5 seeds = ~2,250 compilations (estimated ~1 hour)
- Total: ~6 hours, well within the 8-hour budget

## Success Criteria

### Primary (confirms hypothesis)
1. At least 60% of passes are strongly or weakly idempotent on the majority of benchmarks
2. Commutativity is non-uniform: at least 30% of pairs are non-commutative (order matters)
3. The interference distribution is non-trivial: at least 10% of pairs show significant constructive or destructive interference (|I| > 5% instruction count change)
4. At least some benchmarks exhibit oscillation under iterative pipeline application

### Secondary (demonstrates practical value)
5. The algebra-guided ordering achieves instruction count reduction within 5% of -O2 using fewer passes, OR achieves measurable improvement over -O2
6. The commutativity and interference structures reveal interpretable clusters aligned with pass categories (loop opts, scalar opts, etc.)

### What would refute the hypothesis
- If all passes commute and are idempotent (the space has trivial algebra), the research question is answered but the practical implications are limited
- If algebraic properties are entirely program-dependent with no generalizable patterns, the framework has limited predictive value

## References

1. Kulkarni, S. and Cavazos, J. "Mitigating the Compiler Optimization Phase-Ordering Problem using Machine Learning." In Proceedings of the ACM International Conference on Object Oriented Programming Systems Languages and Applications (OOPSLA), 2012.

2. Ashouri, A. H., Bignoli, A., Palermo, G., Silvano, C., Kulkarni, S., and Cavazos, J. "MiCOMP: Mitigating the Compiler Phase-Ordering Problem Using Optimization Sub-Sequences and Machine Learning." ACM Transactions on Architecture and Code Optimization, 14(3), 2017.

3. Jain, S., Andaluri, Y., VenkataKeerthy, S., and Upadrasta, R. "POSET-RL: Phase Ordering for Optimizing Size and Execution Time using Reinforcement Learning." In IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), 2022.

4. Deng, C., Wu, J., Feng, N., Wang, J., and Long, M. "CompilerDream: Learning a Compiler World Model for General Code Optimization." In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2025.

5. Pan, H., Dong, J., Xing, M., and Wu, Y. "Synergy-Guided Compiler Auto-Tuning of Nested LLVM Pass Pipelines." arXiv:2510.13184, 2025.

6. Cummins, C., Fisches, Z. V., Ben-Nun, T., Hoefler, T., O'Boyle, M. F. P., and Leather, H. "ProGraML: A Graph-based Program Representation for Data Flow Analysis and Compiler Optimizations." In Proceedings of the 38th International Conference on Machine Learning (ICML), 2021.

7. Cummins, C., Wasti, B., Guo, J., Cui, B., Ansel, J., Gomez, S., Jain, S., Liu, J., Teytaud, O., Steiner, B., Tian, Y., and Leather, H. "CompilerGym: Robust, Performant Compiler Optimization Environments for AI Research." arXiv:2109.08267, 2021.

8. Liu, Y., Fang, J., Wang, W., Xue, H., Huang, H., and Wang, J. "Efficient Compiler Optimization by Modeling Passes Dependence." CCF Transactions on High Performance Computing, 2024.

9. Liang, J., Ma, L., Qian, C., and Yu, Y. "Learning Compiler Pass Orders using Coreset and Normalized Value Prediction." In Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.

10. Regehr, J. "How LLVM Optimizes a Function." Embedded in Academia blog, 2018. https://blog.regehr.org/archives/1603
