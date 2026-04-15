# Research Proposal: SPICED
## Sample-Efficient Prior-Informed Causal Estimation via Directed Information

---

## 1. Introduction

### 1.1 Context and Problem Statement

Causal discovery—the task of inferring causal relationships from observational data—is a fundamental problem in machine learning with applications in healthcare, economics, biology, and the sciences. Despite significant advances, existing methods face a critical limitation: **they require large sample sizes to achieve reliable performance**. This severely restricts their applicability in domains where data collection is expensive or impossible, such as rare disease studies, personalized medicine, or analysis of emerging phenomena.

Current approaches fall into two broad categories, each with fundamental limitations:

1. **Constraint-based methods** (e.g., PC algorithm [Spirtes et al., 2000], FCI [Spirtes et al., 2000]) rely on conditional independence (CI) testing. While statistically consistent, CI tests have low power with small samples, and the number of tests grows exponentially with graph size, leading to error propagation [Maasch et al., 2024].

2. **Differentiable/score-based methods** (e.g., NOTEARS [Zheng et al., 2018], GOLEM [Ng et al., 2020]) reformulate causal discovery as continuous optimization. These scale to larger graphs but require large samples for accurate gradient estimates and lack strong finite-sample guarantees [Reisach et al., 2021].

Recent work has attempted to address small-sample settings through:
- Local search methods (LoSAM [2024])—limited to Additive Noise Models
- Argumentation-based approaches (ABA-PC [2024])—computationally expensive
- Quantum-enhanced methods (qPC [2025])—require quantum hardware
- Supervised skeleton learning (SPOT [2024])—requires pre-training on synthetic data

**The gap**: There is no method that combines (a) sample efficiency competitive with constraint-based methods, (b) scalability comparable to differentiable approaches, (c) applicability to general functional forms without parametric assumptions, and (d) no need for pre-training or specialized hardware.

### 1.2 Key Insight and Hypothesis

Our key insight is that **information-theoretic measures can be estimated more reliably than conditional independence tests with limited samples**, and **structural sparsity constraints can guide optimization to avoid local minima** while maintaining computational efficiency.

Specifically:
1. Directed information and transfer entropy measures capture causal dependencies more robustly than CI tests in small-sample regimes [Schreiber, 2000; Quinn et al., 2011]
2. Recent theoretical results show that causal discovery becomes fixed-parameter tractable (FPT) when parameterized by structural properties like the maximum number of edge-disjoint paths [Ganian et al., 2024]
3. These structural parameters can be estimated or bounded from data without requiring full causal discovery

**Central Hypothesis**: A hybrid algorithm that uses adaptive information-theoretic edge scoring combined with continuous optimization guided by structural sparsity constraints will achieve superior sample efficiency compared to both pure constraint-based and pure differentiable methods, while maintaining polynomial-time complexity for sparse graphs.

---

## 2. Proposed Approach

### 2.1 Overview

We propose **SPICED** (Sample-efficient Prior-Informed Causal Estimation via Directed Information), a three-phase algorithm:

**Phase 1: Information-Theoretic Skeleton Discovery**
- Estimate pairwise directed information (or transfer entropy for time series) between all variables
- Construct initial skeleton using statistically significant edges
- Apply sparsification based on conditional mutual information thresholds

**Phase 2: Structural Sparsity Constraint Extraction**
- Analyze the skeleton to identify structural parameters (edge-connectivity, feedback vertex set size)
- If structural parameters are bounded, extract constraints for the optimization phase
- Otherwise, apply divide-and-conquer decomposition

**Phase 3: Constrained Continuous Optimization**
- Formulate differentiable optimization with acyclicity constraint (NOTEARS-style)
- Incorporate structural constraints from Phase 2 as soft penalties
- Use information-theoretic scores to guide the optimization initialization

### 2.2 Technical Details

#### Phase 1: Information-Theoretic Skeleton Discovery

For variables $X$ and $Y$, we estimate directed information [Massey, 1990]:

$$I(X \rightarrow Y) = \sum_{t} I(X^{t}; Y^{t} | Y^{t-1})$$

For non-time-series data, we use conditional mutual information with conditioning on candidate parents.

**Key innovation**: We employ **k-nearest neighbor (k-NN) entropy estimators** [Kraskov et al., 2004] which have better small-sample properties than kernel-based methods. The estimator uses a fixed $k$ (typically $k=5$) and adapts to local density, making it more robust with limited samples.

Edge selection uses a novel **adaptive thresholding** procedure:
1. Compute null distribution via permutation testing (limited permutations for efficiency)
2. Select edges with p-value < $\alpha$ (adaptive based on graph size to control false discovery)
3. Iteratively refine by testing conditional independence given selected neighbors

#### Phase 2: Structural Constraint Extraction

We exploit recent theoretical results [Ganian et al., 2024] showing that causal discovery is FPT when parameterized by the maximum number of edge-disjoint paths between any pair of variables.

Algorithm:
1. Compute edge-connectivity of the skeleton graph
2. If connectivity $\leq k$ (a small constant), we have FPT guarantees
3. Extract **feedback vertex set** (FVS) approximation—nodes whose removal makes the graph acyclic
4. Use FVS to guide topological ordering constraints

For graphs with high connectivity, we apply **modular decomposition**:
- Decompose into 2-connected components
- Apply SPICED recursively to each component
- Merge results using separator nodes

#### Phase 3: Constrained Continuous Optimization

We formulate the optimization problem:

$$\min_{W} \mathcal{L}(W; X) + \lambda_1 \cdot h(W) + \lambda_2 \cdot r(W) + \lambda_3 \cdot c(W)$$

Where:
- $\mathcal{L}(W; X)$: Data fidelity term (least squares for linear, neural network loss for nonlinear)
- $h(W) = \text{tr}(e^{W \circ W}) - d$: Acyclicity constraint [Zheng et al., 2018]
- $r(W)$: Sparsity regularization (L1 or hierarchical)
- $c(W)$: **Structural constraint term** (novel contribution)

The structural constraint term $c(W)$ penalizes violations of constraints discovered in Phase 2:
- If topological ordering constraints from FVS: penalize edges that violate the partial order
- If edge-connectivity is bounded: penalize dense subgraphs that violate the bound

**Initialization**: Use information-theoretic scores from Phase 1 to warm-start the optimization, setting initial weights proportional to estimated causal strength.

### 2.3 Key Innovations

1. **Adaptive Information-Theoretic Scoring**: Uses k-NN entropy estimation which is more sample-efficient than CI testing or kernel methods for small $n$.

2. **Structural Sparsity Integration**: First method to explicitly incorporate FPT structural parameters (edge-connectivity, FVS) into differentiable causal discovery, providing theoretical guarantees for sparse graphs.

3. **Hybrid Three-Phase Architecture**: Combines the sample efficiency of information-theoretic methods with the scalability of continuous optimization, while avoiding the exponential worst-case of pure constraint-based approaches.

4. **No Pre-training Required**: Unlike SPOT [Ma et al., 2024], does not require supervised learning on synthetic data, making it applicable to novel domains.

---

## 3. Related Work

### 3.1 Constraint-Based Methods

The PC algorithm [Spirtes & Glymour, 1991] and its extension FCI [Spirtes et al., 2000] for latent confounders are the canonical constraint-based methods. These perform conditional independence tests to discover the causal skeleton and orient edges. While statistically consistent, they require large samples for reliable CI testing and have exponential worst-case complexity.

Recent improvements include:
- **MARVEL** [Mokhtarian et al., 2021]: Reduces number of CI tests but still exponential in worst case
- **ABA-PC** [Russo et al., 2024]: Uses argumentation to resolve inconsistencies; improves accuracy but computationally expensive
- **k-PC** [Kocaoglu, 2023]: Limits conditioning set size; heuristic without theoretical guarantees

**Difference**: SPICED avoids exponential CI testing by using information-theoretic screening followed by continuous optimization.

### 3.2 Differentiable Causal Discovery

NOTEARS [Zheng et al., 2018] reformulated DAG learning as continuous optimization using the acyclicity constraint $h(W) = \text{tr}(e^{W \circ W}) - d = 0$. Extensions include:
- **GOLEM** [Ng et al., 2020]: Likelihood-based objective, more robust to noise variance
- **DAGMA** [Bello et al., 2022]: Log-det acyclicity constraint
- **NOTEARS-MLP** [Zheng et al., 2020]: Nonlinear extensions using neural networks

**Limitations**: These methods require large samples for gradient accuracy, are sensitive to initialization, and can get stuck in local minima. SPICED addresses these via information-theoretic initialization and structural constraints.

### 3.3 Methods for Latent Confounders

FCI [Spirtes et al., 2000] and RFCI [Colombo et al., 2012] extend PC to latent confounders, producing Partial Ancestral Graphs (PAGs). Recent differentiable approaches:
- **SPOT** [Ma et al., 2024]: Uses supervised learning to estimate skeleton posterior; requires pre-training
- **ABIC** [Bhattacharya et al., 2021]: Differentiable MAG learning; limited to small graphs

**Difference**: SPICED handles latent confounders through information-theoretic measures (which detect dependencies even with confounding) and can optionally incorporate FCI-style orientation rules in post-processing.

### 3.4 Sample-Efficient Methods

- **LoSAM** [2024]: Polynomial-time local search for Additive Noise Models only
- **qPC** [Arai et al., 2025]: Quantum kernel methods; requires quantum hardware
- **CORE** [Sauter et al., 2024]: RL-based active discovery; requires intervention capabilities

**Difference**: SPICED is the first sample-efficient method for general functional forms without requiring specialized hardware or interventional data.

### 3.5 Complexity-Theoretic Results

Recent work [Ganian et al., IJCAI 2024] characterized when causal discovery is fixed-parameter tractable. They showed that parameterizing by the maximum number of edge-disjoint paths yields FPT algorithms, while treewidth does not.

**SPICED's contribution**: First practical algorithm to exploit these FPT results, incorporating structural parameters into a scalable differentiable framework.

---

## 4. Experiments

### 4.1 Experimental Setup

**Baselines**:
1. **PC**: Constraint-based algorithm (causal-learn implementation)
2. **FCI**: For latent confounders (causal-learn implementation)
3. **NOTEARS**: Differentiable baseline
4. **GOLEM**: Likelihood-based differentiable method
5. **SPOT** (if code available): Supervised skeleton learning
6. **LoSAM**: For ANM comparison
7. **GRaSP**: Greedy search baseline

**Datasets**:
- **Synthetic**: ER (Erdős-Rényi), SF (Scale-Free) graphs with $n \in \{10, 20, 30, 50\}$ nodes
- **Sample sizes**: $N \in \{50, 100, 200, 500, 1000, 5000\}$ (focus on small $N$)
- **Mechanisms**: Linear Gaussian, Linear non-Gaussian, Nonlinear (GP), ANM
- **Latent confounders**: 10-20% of variables

- **Real-world**:
  - Sachs protein signaling [Sachs et al., 2005] (853 samples, ground truth known)
  - Insurance benchmark [Binder et al., 1997]
  - Small biological datasets

**Metrics**:
- **Structural Hamming Distance (SHD)**: Between predicted and true DAG
- **True Positive Rate (TPR)** and **False Discovery Rate (FDR)**
- **SID** (Structural Intervention Distance): For causal effect estimation
- **Runtime**: Wall-clock time

### 4.2 Expected Results

**Hypothesis 1**: SPICED will achieve lower SHD than NOTEARS/GOLEM for $N < 500$, with comparable performance at large $N$.

*Rationale*: Information-theoretic initialization provides better starting point when gradients are noisy with small samples.

**Hypothesis 2**: SPICED will be significantly faster than PC/FCI for sparse graphs while maintaining comparable accuracy.

*Rationale*: Polynomial optimization vs. exponential CI testing.

**Hypothesis 3**: SPICED will outperform LoSAM on non-ANM data while matching it on ANMs.

*Rationale*: LoSAM assumes ANM; SPICED is model-agnostic.

**Hypothesis 4**: The structural constraint term $c(W)$ will improve convergence speed and reduce local minima for graphs with bounded edge-connectivity.

### 4.3 Ablation Studies

1. **Phase 1 ablation**: Compare k-NN vs. kernel-based MI estimation
2. **Phase 2 ablation**: Effect of structural constraints vs. unconstrained optimization
3. **Initialization ablation**: Random vs. information-theoretic initialization
4. **Parameter sensitivity**: $\lambda_1, \lambda_2, \lambda_3$ tuning

### 4.4 Computational Requirements

Given CPU-only constraint (2 cores, 128GB RAM):
- Synthetic experiments: Estimated 4-5 hours for full benchmark suite
- Real-world datasets: < 1 hour
- All experiments parallelizable across datasets

---

## 5. Success Criteria

### 5.1 Primary Success Criteria

1. **Sample Efficiency**: SPICED achieves statistically significantly lower SHD than NOTEARS for $N \leq 200$ on at least 3 out of 4 graph types (linear Gaussian, linear non-Gaussian, nonlinear, ANM).

2. **Scalability**: SPICED runs in < 5 minutes for 50-node graphs on 2 CPU cores.

3. **Accuracy**: On Sachs dataset, SPICED achieves SHD < 10 (comparable to or better than state-of-the-art).

### 5.2 Secondary Success Criteria

1. Structural constraints improve convergence vs. unconstrained baseline.
2. Information-theoretic initialization outperforms random initialization.
3. Method works without modification across different data types.

### 5.3 Failure Modes

- If information-theoretic estimation is unreliable for very small $N$ (< 50), method may fail to produce meaningful skeleton
- Dense graphs with high edge-connectivity may not benefit from structural constraints
- Nonlinear relationships with complex interactions may require more sophisticated entropy estimators

---

## 6. Theoretical Analysis

### 6.1 Computational Complexity

**Theorem** (Informal): For causal graphs where the skeleton has maximum edge-connectivity $k$, SPICED runs in time $O(f(k) \cdot n^{O(1)} + n^2 \cdot N \cdot \log N)$, where $f(k)$ is exponential in $k$ but polynomial in $n$ for fixed $k$.

*Proof sketch*: Phase 1 (information estimation) takes $O(n^2 \cdot N \cdot \log N)$ for k-NN MI estimation. Phase 2 (structural analysis) is FPT in $k$ [Ganian et al., 2024]. Phase 3 (optimization) is polynomial in $n$ with fixed iterations.

### 6.2 Consistency

Under standard faithfulness and causal Markov assumptions, as $N \rightarrow \infty$:
1. Phase 1 recovers the true skeleton
2. Phase 2 correctly identifies structural parameters
3. Phase 3 converges to the true DAG (following NOTEARS consistency proof)

### 6.3 Sample Complexity

For bounded-degree graphs with maximum in-degree $d$, the information-theoretic skeleton discovery requires $N = O(d \cdot \log n)$ samples for reliable edge detection, improving upon CI testing which requires $N = O(2^d)$ in worst case.

---

## 7. Broader Impact and Applications

### 7.1 Scientific Discovery

SPICED enables causal discovery in scientific domains with limited data:
- **Rare disease studies**: Identify causal biomarkers from small patient cohorts
- **Personalized medicine**: Individual-level causal networks from limited measurements
- **Ecology**: Causal relationships from short-term observational studies

### 7.2 Practical Applications

- **Healthcare resource planning**: Causal factors for treatment outcomes from limited hospital data
- **Policy evaluation**: Causal effects from pilot studies before full deployment
- **Scientific hypothesis generation**: Prioritize experiments based on causal uncertainty

### 7.3 Ethical Considerations

- Risk of false causal discovery with small samples requires careful uncertainty quantification
- Method should provide confidence estimates alongside discovered graphs
- Users should be educated about limitations in small-sample regimes

---

## 8. Timeline and Milestones

**Phase 1: Implementation** (3 hours)
- Implement k-NN entropy estimation
- Implement Phase 1 skeleton discovery
- Implement structural constraint extraction
- Integrate with NOTEARS-style optimization

**Phase 2: Synthetic Experiments** (3 hours)
- Generate benchmark datasets
- Run comparisons on synthetic data
- Perform ablation studies

**Phase 3: Real-World Evaluation** (1.5 hours)
- Sachs dataset experiments
- Insurance benchmark
- Analysis and visualization

**Phase 4: Documentation** (0.5 hours)
- Paper writing preparation
- Code documentation

---

## 9. Conclusion

SPICED addresses a critical gap in causal discovery: the need for sample-efficient, scalable methods that work without strong parametric assumptions. By combining information-theoretic skeleton discovery, structural sparsity constraints, and continuous optimization, we expect to achieve superior performance in small-sample regimes while maintaining computational efficiency. The method has potential for significant impact in scientific domains where data is scarce but causal understanding is crucial.

---

## References

1. Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.

2. Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *NeurIPS*, 31.

3. Ng, I., Ghassami, A., & Zhang, K. (2020). On the role of sparsity and DAG constraints for learning linear DAGs. *NeurIPS*, 33.

4. Ma, P., Ding, R., Fu, Q., et al. (2024). Scalable differentiable causal discovery in the presence of latent confounders with skeleton posterior. *KDD 2024*.

5. Ganian, R., Korchemna, V., & Szeider, S. (2024). Revisiting causal discovery from a complexity-theoretic perspective. *IJCAI 2024*.

6. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.

7. Colombo, D., Maathuis, M. H., Kalisch, M., & Richardson, T. S. (2012). Learning high-dimensional directed acyclic graphs with latent and selection variables. *Annals of Statistics*, 40(1), 294-321.

8. Bello, K., Aragam, B., & Ravikumar, P. (2022). DAGMA: DAG learning via m-matrices and accelerated gradient method. *NeurIPS*, 35.

9. Maasch, J. M., et al. (2024). Local discovery by partitioning: Polynomial-time causal discovery around exposure-outcome pairs. *NeurIPS* (2024).

10. Massey, J. (1990). Causality, feedback and directed information. *Proc. Int. Symp. on Info. Theory and its Applications*.

11. Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461.

12. Reisach, A., Seiler, C., & Weichwald, S. (2021). Beware of the simulated DAG! Varsortability in additive noise models. *NeurIPS*, 34.

13. Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. *Science*, 308(5721), 523-529.
