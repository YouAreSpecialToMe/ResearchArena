# Optimal Error Budgeting for Heterogeneous Sketch Pipelines in Approximate Stream Processing

## Introduction

### Context

Modern stream processing systems routinely compose multiple probabilistic data structures (sketches) in pipelines to answer complex queries over high-velocity data streams. For example, a network monitoring system might use a Bloom filter to check whether an IP address belongs to a watchlist, a Count-Min Sketch (CMS) to estimate the traffic frequency of matched IPs, and a HyperLogLog (HLL) to count the distinct heavy hitters. Each sketch introduces approximation error, and these errors interact as they propagate through the pipeline.

### Problem Statement

Current practice treats each sketch in a pipeline independently: practitioners size each structure using its standalone worst-case error bound, then compose the results with conservative union bounds (Bonferroni correction). This approach is wasteful because:

1. **Loose bounds**: Union bounds over independent sketch errors dramatically overestimate the true end-to-end error, leading to over-provisioning of memory.
2. **Uniform allocation**: Without understanding error interactions, systems allocate memory uniformly or by ad-hoc heuristics rather than directing resources to the stages where they matter most.
3. **Heterogeneous error types**: Different sketch types produce fundamentally different error distributions (one-sided vs. symmetric, additive vs. multiplicative), and these interact non-trivially when composed.

### Key Insight

When heterogeneous sketches are chained in a pipeline, their errors propagate through well-defined functional compositions. A Bloom filter's false positives create "phantom items" that inflate downstream frequency estimates; a CMS's overestimates shift the threshold for heavy-hitter detection, altering which items a downstream HLL counts. By analyzing these functional dependencies, we can derive **tight composition bounds** that are substantially tighter than naive product bounds, and use these to **optimally distribute a memory budget** across pipeline stages.

### Hypothesis

Given a fixed total memory budget and a pipeline of heterogeneous sketches, our composition-aware error analysis and optimal allocation algorithm will achieve significantly lower end-to-end error (or equivalently, use significantly less memory for the same error target) compared to: (a) uniform allocation, (b) independent worst-case sizing, and (c) proportional allocation based on standalone error curves.

## Proposed Approach

### Overview

We propose **SketchBudget**, a framework for analyzing and optimizing multi-stage approximate streaming pipelines composed of heterogeneous probabilistic data structures. The framework has three components:

1. **Error Propagation Algebra**: A formal model for how different error types (false positive rate, additive overestimate, relative error) compose through common pipeline operators (filter, threshold, aggregate).

2. **Tight Composition Bounds**: For canonical pipeline patterns, we derive closed-form expressions for end-to-end error as a function of per-stage parameters, proving these are tighter than naive bounds.

3. **Optimal Allocation Algorithm**: Given a pipeline DAG and total memory budget, we formulate and solve the constrained optimization problem of distributing memory across stages to minimize end-to-end error.

### Method Details

#### Error Propagation Algebra

We model a sketch pipeline as a directed acyclic graph (DAG) where each node is a sketch operator and edges represent data flow. Each node i has:
- An **error function** e_i(m_i) mapping allocated memory m_i to its standalone error parameter
- An **error type**: false positive rate (FPR), additive error (epsilon), or relative error (delta)
- A **propagation function** that describes how upstream errors affect this node's effective error

For example, consider a two-stage pipeline: Bloom filter (BF) with FPR p, followed by CMS with additive error epsilon on the filtered stream. The CMS processes N' = N_true + N_false items, where N_false ~ Binomial(N_neg, p). The effective additive error of the CMS becomes epsilon' = epsilon * (N_true + E[N_false]) / N_true, which depends on the BF's FPR.

We formalize three canonical composition patterns:
- **Filter-Estimate**: BF/Cuckoo filter -> CMS/Count-Sketch (false positives inflate frequency estimates)
- **Estimate-Threshold-Count**: CMS -> threshold -> HLL (overestimates create false heavy hitters inflating cardinality)
- **Estimate-Aggregate**: Multiple CMS -> aggregate function (errors combine through the aggregation)

#### Tight Composition Bounds

For each canonical pattern, we derive:
1. **Expected end-to-end error** as a closed-form function of per-stage parameters
2. **High-probability bounds** using concentration inequalities (Chernoff, McDiarmid)
3. **Comparison with naive bounds** showing the gap is Omega(k) for k-stage pipelines in the worst case

Key theoretical results:
- For Filter-Estimate pipelines, the tight bound accounts for the fact that phantom items from BF false positives have frequencies that are conditionally independent of the CMS hash collisions, yielding a tighter variance bound than treating the errors as correlated.
- For Estimate-Threshold-Count pipelines, we show that the error amplification depends critically on the frequency distribution near the threshold, providing distribution-aware bounds that can be orders of magnitude tighter than worst-case bounds for heavy-tailed distributions (common in practice).

#### Optimal Allocation Algorithm

Given composition bounds, the allocation problem becomes:

    minimize E_total(m_1, ..., m_k)
    subject to: sum(m_i) <= M (total budget)
                m_i >= m_min_i (minimum per stage)

Where E_total is the composed error function from our algebra. We analyze this optimization:
- For two-stage pipelines, we derive closed-form optimal allocations
- For general DAGs, E_total is typically non-convex but has exploitable structure (product of monotone decreasing functions). We develop a branch-and-bound algorithm with monotonicity-based pruning.
- We also propose a fast greedy heuristic: iteratively allocate the next unit of memory to the stage with the largest marginal error reduction, which we prove achieves a (1-1/e) approximation ratio for a natural subclass of error functions.

### Key Innovations

1. **First formal framework** for cross-type sketch error composition (prior work by Dobra et al. 2004 considered only homogeneous AMS sketches for parallel queries, not heterogeneous sequential pipelines)
2. **Distribution-aware bounds** that exploit knowledge of the data distribution near critical thresholds
3. **Practical allocation algorithm** with provable approximation guarantees
4. **Unified treatment** of FPR, additive, and multiplicative errors in a common algebra

## Related Work

### Probabilistic Data Structures (Foundational)

- **Bloom (1970)**: Introduced the Bloom filter for space-efficient set membership testing with allowable false positives. Our work uses Bloom filters as pipeline components and analyzes how their FPR propagates to downstream stages.
- **Cormode and Muthukrishnan (2005)**: Introduced the Count-Min Sketch for frequency estimation in data streams with one-sided additive error guarantees. We extend their error analysis to account for upstream filtering errors.
- **Flajolet, Fusy, Gandouet, and Meunier (2007)**: Introduced HyperLogLog for near-optimal cardinality estimation. We analyze how upstream errors (false heavy hitters) affect HLL accuracy.
- **Greenwald and Khanna (2001)**: Space-efficient quantile summaries for streaming data. Relevant as a potential pipeline component for threshold-based filtering.

### Multi-Query Sketch Optimization

- **Dobra, Garofalakis, Gehrke, and Rastogi (2004)**: Studied sketch-based multi-query optimization over data streams, showing that the space allocation problem for average error is NP-complete. Their work considers **parallel queries** sharing **homogeneous sketches** (all AMS/CMS), while we consider **sequential pipelines** of **heterogeneous sketch types** with error propagation analysis.
- **Dobra, Garofalakis, Gehrke, and Rastogi (2009)**: Extended their 2004 work to a full journal treatment with join coalescing heuristics. Still limited to homogeneous sketches and parallel query workloads.

### Learned and Adaptive Sketches

- **Kraska, Beutel, Chi, Dean, and Polyzotis (2018)**: Proposed learned index structures, sparking interest in ML-augmented data structures including Bloom filters and frequency estimators. Our work is complementary: learned sketches can be pipeline components whose error characteristics we analyze.
- **Mitzenmacher (2018)**: Formalized learned Bloom filters and the "sandwiching" optimization. Our framework can incorporate learned BF error models as drop-in replacements for standard BF error functions.
- **Dolera, Favaro, and Peluchetti (2023)**: Applied Bayesian nonparametric priors (Dirichlet process, Pitman-Yor process) to Count-Min Sketches. Their posterior-based estimates could serve as improved single-stage error models within our framework.
- **Zhu, Wei, Mun, and Athanassoulis (2025)**: Mnemosyne - workload-aware Bloom filter tuning for LSM trees. Addresses single-sketch adaptation, while we address multi-sketch pipeline optimization.

### How Our Work Differs

Prior work optimizes individual sketches or parallel queries over shared sketches. No existing work provides: (a) a formal model of error propagation through sequential pipelines of heterogeneous sketch types, (b) tight composition bounds exploiting cross-stage independence, or (c) an optimal allocation algorithm for heterogeneous pipelines. Our contribution fills this gap.

## Experiments

### Setup

- **Implementation**: Python with NumPy/SciPy. All sketches implemented from scratch for precise memory control, plus validation against existing libraries (datasketch, pdsa).
- **Hardware**: CPU-only (2 cores, 128GB RAM). All experiments are algorithmic/analytical and require no GPU.
- **Datasets**:
  - Synthetic: Zipfian streams with configurable skew (alpha = 0.5 to 2.0), uniform streams, and adversarial distributions
  - Real-world: CAIDA anonymized internet traces (publicly available), web access log datasets
  - Scale: 10M to 100M stream items per experiment

### Pipeline Configurations

We evaluate three canonical pipeline patterns:

1. **Pipeline P1 (Filter-Estimate)**: Bloom filter -> Count-Min Sketch
   - Query: "Estimate the frequency of items in set S"
   - BF filters items; CMS estimates frequencies of filtered items

2. **Pipeline P2 (Estimate-Threshold-Count)**: CMS -> Threshold -> HyperLogLog
   - Query: "Count distinct items with frequency above threshold T"
   - CMS estimates frequencies; threshold filters heavy hitters; HLL counts distinct heavy hitters

3. **Pipeline P3 (Filter-Estimate-Aggregate)**: BF -> CMS -> Aggregation
   - Query: "Estimate the total traffic from items in set S"
   - BF filters; CMS estimates per-item frequency; aggregation sums frequencies

### Allocation Strategies (Baselines)

- **Uniform**: Equal memory per stage
- **Independent**: Size each stage to achieve the same standalone error guarantee (ignoring composition)
- **Proportional**: Allocate memory proportional to each sketch type's standalone space requirement for a target error
- **SketchBudget (ours)**: Composition-aware optimal allocation

### Metrics

- **End-to-end error**: Measured as mean absolute error, mean relative error, or recall/precision depending on the pipeline
- **Memory efficiency**: Memory required to achieve a target end-to-end error level
- **Bound tightness**: Ratio of predicted (theoretical) error to observed (empirical) error

### Expected Results

1. **Tighter bounds**: Our composition bounds will be 2-10x tighter than naive product bounds, with the gap increasing with pipeline depth and data skew.
2. **Better allocation**: SketchBudget allocation will reduce end-to-end error by 20-50% compared to uniform allocation for the same total memory budget.
3. **Memory savings**: For a fixed accuracy target, SketchBudget will require 30-60% less total memory than independent sizing.
4. **Distribution sensitivity**: Gains will be largest for heavy-tailed (Zipfian) distributions, which are common in practice.

### Ablation Studies

- Vary total memory budget (10KB to 10MB) and plot error curves for each allocation strategy
- Vary pipeline depth (2 to 5 stages) and measure error amplification
- Vary data distribution (Zipfian skew parameter) and measure distribution-awareness benefit
- Vary stream length and measure convergence of empirical error to theoretical bounds
- Compare greedy heuristic vs. exact optimization on small instances

## Success Criteria

### Confirms hypothesis if:
1. Composition bounds are provably and empirically tighter than naive bounds for all tested pipeline configurations
2. SketchBudget allocation achieves >= 20% error reduction over the best baseline for at least 2 of 3 pipeline configurations
3. Memory savings of >= 25% for a fixed accuracy target across at least 2 of 3 pipeline configurations
4. The greedy allocation heuristic runs in under 1 second for pipelines with up to 10 stages

### Refutes hypothesis if:
1. Composition bounds are no tighter than naive bounds (errors are effectively independent and worst-case)
2. Uniform allocation is near-optimal (< 5% gap), suggesting error composition doesn't significantly favor non-uniform allocation
3. The optimal allocation is highly sensitive to unknown distribution parameters, making it impractical

## References

1. Bloom, B.H. (1970). "Space/Time Trade-offs in Hash Coding with Allowable Errors." Communications of the ACM, 13(7), 422-426.

2. Cormode, G. and Muthukrishnan, S. (2005). "An Improved Data Stream Summary: The Count-Min Sketch and its Applications." Journal of Algorithms, 55(1), 58-75.

3. Flajolet, P., Fusy, E., Gandouet, O., and Meunier, F. (2007). "HyperLogLog: The Analysis of a Near-Optimal Cardinality Estimation Algorithm." Proceedings of the Analysis of Algorithms (AofA), 127-146.

4. Greenwald, M. and Khanna, S. (2001). "Space-Efficient Online Computation of Quantile Summaries." Proceedings of ACM SIGMOD, 58-66.

5. Dobra, A., Garofalakis, M., Gehrke, J., and Rastogi, R. (2004). "Sketch-Based Multi-Query Processing over Data Streams." Proceedings of EDBT, 551-568.

6. Dobra, A., Garofalakis, M., Gehrke, J., and Rastogi, R. (2009). "Multi-Query Optimization for Sketch-Based Estimation." Information Systems, 34(6), 518-536.

7. Kraska, T., Beutel, A., Chi, E.H., Dean, J., and Polyzotis, N. (2018). "The Case for Learned Index Structures." Proceedings of ACM SIGMOD, 489-504.

8. Mitzenmacher, M. (2018). "A Model for Learned Bloom Filters, and Optimizing by Sandwiching." Proceedings of NeurIPS, 464-473.

9. Dolera, E., Favaro, S., and Peluchetti, S. (2023). "Learning-Augmented Count-Min Sketches via Bayesian Nonparametrics." Journal of Machine Learning Research, 24(12), 1-35.

10. Zhu, Z., Wei, Y., Mun, J.H., and Athanassoulis, M. (2025). "Mnemosyne: Dynamic Workload-Aware BF Tuning via Accurate Statistics in LSM Trees." Proceedings of the ACM on Management of Data (PACMMOD), 3(3).
