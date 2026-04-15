# Research Proposal: CleanBP: Making Belief Propagation Practical for Holistic Data Repair via FD-Specific Sparsification

## 1. Introduction

### 1.1 Context and Problem Statement

Data cleaning is a critical bottleneck in modern data pipelines, with studies showing that data scientists spend up to 80% of their time on data preparation tasks. The problem of repairing inconsistent data—correcting errors while satisfying integrity constraints—remains challenging due to the inherent uncertainty in identifying which values are erroneous and what the correct values should be.

Existing approaches to data cleaning fall into three categories:
1. **Deterministic methods** (e.g., based on minimum repairs [Kolahi & Lakshmanan, 2009]) that produce a single repair but cannot express uncertainty
2. **Probabilistic sampling-based methods** (e.g., HoloClean [Rekatsinas et al., 2017]) that use Gibbs sampling, which is computationally expensive and slow to converge
3. **Early probabilistic inference methods** (e.g., Chu et al. [2005], ERACER [Mayfield et al., 2010]) that applied belief propagation to data cleaning but did not exploit constraint-specific factor graph structure for holistic FD-based repair

HoloClean, the current state-of-the-art probabilistic data cleaning system, formulates repairs as inference in a factor graph but relies on Gibbs sampling—a Markov Chain Monte Carlo (MCMC) method. While principled, Gibbs sampling requires thousands of iterations to converge, making it impractical for large datasets. Prior belief propagation approaches (Chu et al., ERACER) focused on imputation and outlier detection in sensor networks and genealogical data using relational dependency networks, but did not address the specific challenges of functional dependency (FD) violation repair in relational databases.

### 1.2 Key Insight and Hypothesis

Our key insight is that **prior belief propagation approaches for data cleaning failed to exploit the specific structure of functional dependency constraints**. By designing factor graph sparsification techniques that are tailored to FD-based holistic repair, we can make belief propagation practical for this important class of data cleaning problems.

Specifically, we observe that:
1. **Violation-driven sparsification**: Most tuple pairs satisfy FDs; only violators need factor connections
2. **Attribute-separated factorization**: Multi-attribute FDs can be decomposed without significant accuracy loss
3. **Constraint-aware message scheduling**: FD violations form sparse, localized subgraphs that enable fast convergence

**Hypothesis**: Belief propagation on sparsified factor graphs tailored for FD-based repair can compute repair marginals orders of magnitude faster than Gibbs sampling while maintaining comparable accuracy, making uncertainty-aware holistic data cleaning practical for large-scale applications.

### 1.3 Proposed Contribution

We propose **CleanBP**, a system that makes belief propagation practical for holistic FD-based data repair through targeted sparsification:
1. **Violation-driven factor graph sparsification** that exploits FD structure to achieve linear graph size in the number of violations (not tuples)
2. **Attribute-separated factor decomposition** for multi-attribute FDs with theoretical approximation guarantees
3. **Constraint-aware inference** that achieves fast convergence on sparse, locally-connected graphs
4. **Scalability to millions of tuples** on commodity hardware without GPU acceleration

## 2. Proposed Approach

### 2.1 Problem Formalization

Given:
- A dirty dataset $D$ with cells $C = {c_1, c_2, ..., c_n}$
- A set of functional dependencies $\Sigma = \{X_1 \rightarrow Y_1, X_2 \rightarrow Y_2, ...\}$
- Optional: external signals (value frequencies, quality rules)

Find for each erroneous cell $c$:
- The marginal probability $P(T_c = v \mid D, \Sigma)$ for each candidate value $v$
- The maximum a posteriori (MAP) repair: $\arg\max_v P(T_c = v \mid D, \Sigma)$

### 2.2 Factor Graph Construction with FD-Specific Sparsification

**Standard approach (HoloClean)**: Constructs a dense factor graph where each cell is a random variable and factors encode constraints between all potentially-related tuples. This creates $O(n^2)$ factors for FDs.

**CleanBP's Violation-Driven Sparsification**:
1. **Identify violations**: For each FD $X \rightarrow Y$, find tuple pairs $(t_i, t_j)$ where $t_i[X] = t_j[X]$ but $t_i[Y] \neq t_j[Y]$
2. **Create violation-local factors**: Only create factors between cells involved in violations
3. **Attribute separation**: Decompose multi-attribute FD factors into single-attribute components

**Key property**: For a dataset with $m$ FD violations, CleanBP creates $O(m)$ factors instead of $O(n^2)$. Since $m \ll n^2$ in practice, this provides dramatic sparsification.

### 2.3 Belief Propagation with Constraint-Aware Scheduling

Instead of Gibbs sampling, CleanBP uses **loopy belief propagation** (LBP) on the sparsified factor graph:

1. **Message Passing**: Iteratively pass messages between variable nodes and factor nodes
2. **Convergence Scheduling**: Prioritize messages along violation chains; typically converges in <50 iterations
3. **Marginal Extraction**: Compute marginal probabilities from converged messages
4. **MAP Estimation**: Select the value with highest marginal for each cell

**Theoretical Guarantees**:
- For acyclic violation graphs (common after sparsification), LBP computes exact marginals
- For cyclic structures, approximation quality is bounded by graph girth
- Under data redundancy conditions, our sparsification preserves the MAP solution

### 2.4 Comparison with Prior BP-Based Methods

| Aspect | Chu et al. [2005] | ERACER [2010] | CleanBP |
|--------|-------------------|---------------|---------|
| Graph structure | Markov random fields | Relational dependency networks | Factor graphs with FD sparsification |
| Target domain | Sensor networks | Genealogical data | Relational databases with FDs |
| Constraint handling | None (pattern-based) | Shrinkage estimation | Violation-driven factors |
| Inference purpose | Imputation, outlier detection | Statistical inference | Holistic FD violation repair |
| Scalability | Limited | Moderate | Linear in violations |

**Key differences from ERACER**:
- ERACER uses relational dependency networks (directed, learned) while CleanBP uses factor graphs (undirected, constraint-driven)
- ERACER focuses on parameter tying across related tuples; CleanBP focuses on FD violation locality
- CleanBP introduces violation-driven sparsification specifically for FD repair

### 2.5 System Architecture

CleanBP consists of four components:

1. **FD Violation Detector**: Efficiently identifies all FD violations using hash-based partitioning
2. **Sparse Factor Graph Builder**: Constructs violation-local factor graphs with pruned domains
3. **BP Inference Engine**: Runs constraint-aware LBP with prioritized message scheduling
4. **Repair Extractor**: Computes marginals and extracts MAP repairs with uncertainty estimates

## 3. Related Work

### 3.1 Belief Propagation for Data Cleaning (Prior Work)

**Chu et al. [2005]** introduced the use of belief propagation for data cleaning in sensor networks. Their approach uses Markov random fields to model spatial correlations between sensor readings and requires multiple observations per node for estimation. CleanBP differs by focusing on relational FD constraints (not spatial correlations) and using violation-driven sparsification rather than dense grid structures.

**ERACER** [Mayfield et al., 2010] uses belief propagation with relational dependency networks for statistical inference and data cleaning. ERACER's approach uses shrinkage techniques to tie parameters across related tuples and handles genealogical and sensor network data. CleanBP differs in several key ways: (1) ERACER uses directed relational dependency networks while CleanBP uses undirected factor graphs, (2) ERACER focuses on statistical parameter estimation while CleanBP focuses on FD violation repair, (3) CleanBP introduces violation-driven sparsification specifically tailored for FD constraints, and (4) CleanBP handles holistic repair where multiple violations interact, whereas ERACER focuses on individual attribute inference.

### 3.2 Modern Probabilistic Data Cleaning

**HoloClean** [Rekatsinas et al., 2017] is the most closely related modern system. It uses factor graphs and Gibbs sampling for holistic repairs. While both use factor graphs, HoloClean relies on MCMC sampling which requires thousands of iterations to converge. CleanBP replaces sampling with belief propagation on sparsified graphs, trading exact inference for orders of magnitude speedup while maintaining comparable accuracy.

**BClean** [Qin et al., 2024] proposes Bayesian data cleaning using Bayesian network partitioning for approximate inference. BClean explicitly discusses belief propagation as an alternative to their approach but chooses BN partitioning to avoid erroneous propagation. CleanBP differs by focusing specifically on FD constraints and using violation-driven sparsification rather than BN partitioning. BClean will be a key baseline for comparison.

### 3.3 Deterministic Repair Methods

**Minimum Repair Computation** [Kolahi & Lakshmanan, 2009] showed that computing minimum repairs for FD violations is NP-hard and provided approximation algorithms. These approaches produce deterministic repairs without uncertainty quantification. CleanBP provides probabilistic repairs with uncertainty estimates.

**Consistent Query Answering** [Bertossi, 2019; Arenas et al., 1999] focuses on computing query answers true in all repairs. Our work complements this by efficiently computing repair distributions rather than just consistent answers.

### 3.4 Machine Learning-Based Cleaning

**Baran** [Mahdavi & Abedjan, 2020] uses transfer learning for error correction but does not model constraint-based dependencies. **Raha** [Mahdavi et al., 2019] generates error detection strategies using feature engineering. These approaches are orthogonal to CleanBP and can be combined with it.

## 4. Experiments

### 4.1 Datasets and Benchmarks

We will use standard data cleaning benchmarks:
- **REIN benchmark** [Abdelaal et al., 2023]: 14 datasets with realistic error profiles
- **Hospital** [Rekatsinas et al., 2017]: 100K records with FD violations
- **Flights**: 2M+ records for scalability testing
- **Adult Census** [Becker & Kohavi, 1996]: Standard ML benchmark with quality issues

### 4.2 Baseline Methods

1. **HoloClean** [Rekatsinas et al., 2017]: State-of-the-art probabilistic cleaning with Gibbs sampling
2. **BClean** [Qin et al., 2024]: Recent Bayesian cleaning with BN partitioning (discusses BP as alternative)
3. **Baran** [Mahdavi & Abedjan, 2020]: Transfer learning-based error correction
4. **Minimum Repair** [Kolahi & Lakshmanan, 2009]: Combinatorial deterministic approach
5. **ERACER-style BP** [Mayfield et al., 2010]: Re-implementation of BP with relational dependency networks for comparison

### 4.3 Evaluation Metrics

**Repair Quality**:
- Precision: Fraction of suggested repairs that are correct
- Recall: Fraction of actual errors detected and fixed
- F1-score: Harmonic mean of precision and recall

**Computational Efficiency**:
- Inference time: Wall-clock time to compute all repairs
- Convergence iterations: Number of BP iterations until convergence
- Memory usage: Peak RAM consumption
- Graph size: Number of factors created (vs. dense baseline)

**Uncertainty Calibration**:
- Expected Calibration Error (ECE): Measures how well marginals reflect true accuracy
- Brier score: Proper scoring rule for probabilistic predictions

### 4.4 Experimental Protocol

**Experiment 1: Quality vs. Speed Trade-off**
- Compare repair F1-scores between CleanBP, HoloClean, and BClean across datasets
- Measure wall-clock time for inference
- Vary dataset size (10K to 1M tuples) to analyze scalability

**Experiment 2: Ablation Study**
- Test impact of each sparsification technique separately
- Compare: (a) Dense factor graph, (b) Violation-driven sparsification only, (c) Attribute separation only, (d) Full CleanBP

**Experiment 3: Comparison with Prior BP Methods**
- Compare CleanBP against ERACER-style implementation
- Measure accuracy and speedup on FD-specific repair tasks
- Demonstrate benefits of violation-driven sparsification over RDNs

**Experiment 4: Uncertainty Quantification**
- Evaluate calibration of repair confidences
- Measure downstream ML performance when using uncertainty to defer uncertain repairs
- Compare with HoloClean's sampling-based marginals and BClean's partitioned inference

**Experiment 5: Constraint Complexity**
- Test on datasets with varying FD complexity
- Measure how inference time scales with number of violations (not tuples)
- Verify linear scaling property

### 4.5 Expected Results

We expect:
1. CleanBP achieves comparable F1 to HoloClean (within 5%) while being 10-50x faster
2. CleanBP outperforms ERACER-style BP on FD repair tasks due to violation-driven sparsification
3. CleanBP is competitive with BClean in accuracy while providing better uncertainty calibration
4. Violation-driven sparsification provides the largest speedup with minimal quality loss
5. Linear scalability with number of violations (sublinear in dataset size)
6. Well-calibrated uncertainty estimates (ECE < 0.1)

## 5. Success Criteria

### 5.1 Confirming the Hypothesis

The hypothesis is confirmed if:
- CleanBP runs inference in <10 minutes on Hospital dataset (vs. hours for HoloClean)
- F1-score is within 10% of HoloClean's performance
- Marginal probabilities are reasonably calibrated (ECE < 0.15)
- CleanBP demonstrates superior performance to ERACER-style BP on FD repair tasks
- Graph size scales linearly with number of violations (not quadratically with tuples)

### 5.2 Refuting the Hypothesis

The hypothesis is refuted if:
- Belief propagation fails to converge on realistic violation graphs
- Quality degradation exceeds 20% compared to sampling-based methods
- Violation-driven sparsification eliminates too many true repairs
- BClean's BN partitioning approach consistently outperforms CleanBP

### 5.3 Success Metrics for Publication

For this work to be publishable at a top venue (e.g., VLDB, SIGMOD, ICDE):
- Demonstrate significant speedup (≥10x) with minimal quality loss (<10% F1 degradation)
- Provide theoretical analysis of approximation guarantees for sparsification
- Show practical utility on real datasets with meaningful error rates
- Properly contextualize contribution as improving BP for FD repair, not introducing BP to data cleaning

## 6. Feasibility and Resource Requirements

### 6.1 Technical Feasibility

The proposed work is entirely algorithmic and analytical:
- Belief propagation is well-understood with many open-source implementations
- No deep learning or GPU compute required
- Standard data cleaning benchmarks are publicly available
- Prior BP-based methods (ERACER, Chu et al.) provide implementation guidance

### 6.2 Computational Resources

- **CPU**: 2 cores sufficient for belief propagation
- **RAM**: 128GB available; expect <32GB peak usage (sparsification reduces memory)
- **Time**: Each experiment runs in minutes to hours
- **Total estimated runtime**: <8 hours for all experiments

### 6.3 Implementation Plan

1. **Week 1**: Implement FD violation detection and sparse factor graph construction
2. **Week 2**: Implement belief propagation engine with constraint-aware scheduling
3. **Week 3**: Integration, baselines comparison (HoloClean, BClean, ERACER-style), evaluation

## 7. References

1. Abedjan, Z., et al. (2016). Detecting data errors: Where are we and what needs to be done? *PVLDB*, 9(12), 993-1004.

2. Abdelaal, M., Hammacher, C., & Schöning, H. (2023). REIN: A comprehensive benchmark framework for data cleaning methods in ML pipelines. *EDBT*, 499-511.

3. Arenas, M., Bertossi, L., & Chomicki, J. (1999). Consistent query answers in inconsistent databases. *PODS*, 68-79.

4. Becker, B., & Kohavi, R. (1996). Adult Census Income dataset. UCI ML Repository.

5. Bertossi, L. (2019). Database repairs and consistent query answering. *Synthesis Lectures on Data Management*, 11(1), 1-200.

6. Beskales, G., Soliman, M. A., & Ilyas, I. F. (2009). Efficient search for the top-k possible nearest neighbors in uncertain databases. *VLDB*, 3(1), 326-339.

7. Beskales, G., Soliman, M. A., & Ilyas, I. F. (2010). Effective search for top-k possible repairs in inconsistent databases. *ICDE*, 205-216.

8. Chu, F., Wang, Y., Parker, D. S., & Zaniolo, C. (2005). Data cleaning using belief propagation. *IQIS*, 99-104.

9. Kolahi, S., & Lakshmanan, L. V. (2009). On approximating optimum repairs for functional dependency violations. *ICDT*, 53-62.

10. Mahdavi, M., & Abedjan, Z. (2020). Baran: Effective error correction via a unified context representation and transfer learning. *PVLDB*, 13(12), 1948-1961.

11. Mahdavi, M., et al. (2019). Raha: A configuration-free error detection system. *SIGMOD*, 865-882.

12. Mayfield, C., Neville, J., & Prabhakar, S. (2010). ERACER: A database approach for statistical inference and data cleaning. *SIGMOD*, 75-86.

13. Qin, J., et al. (2024). BClean: A Bayesian data cleaning system. *ICDE*, 3407-3420.

14. Rekatsinas, T., Chu, X., Ilyas, I. F., & Ré, C. (2017). HoloClean: Holistic data repairs with probabilistic inference. *PVLDB*, 10(11), 1190-1201.
