# E-Valued Causal Discovery: Constraint-Based Structure Learning with Anytime-Valid FDR Control

## Introduction

### Context and Problem Statement

Constraint-based causal discovery algorithms, led by the PC algorithm (Spirtes et al., 2000), are foundational tools for learning causal graphs from observational data. These algorithms rely on a series of conditional independence (CI) tests to iteratively remove edges from a complete graph and orient remaining edges, ultimately producing a completed partially directed acyclic graph (CPDAG). However, a critical statistical challenge has been largely overlooked: **the PC algorithm performs dozens to thousands of dependent CI tests using a single, fixed significance level, without rigorous control over the graph-level false discovery rate (FDR)**.

In practice, this means that the reported causal graph may contain a substantial fraction of spurious edges (false positives) or miss true edges (false negatives), with no principled way to quantify or control these errors. The few existing solutions—most notably the PC-p algorithm (Strobl et al., 2019), which computes edge-specific p-values and applies the Benjamini-Yekutieli (BY) FDR procedure—suffer from excessive conservatism because they must use the BY correction to handle the arbitrary dependence structure among CI tests, resulting in significant power loss.

### Key Insight

**E-values** (Wang and Ramdas, 2022) are a recently developed alternative to p-values for hypothesis testing. An e-value is a non-negative random variable with expectation at most 1 under the null hypothesis. Unlike p-values, e-values can be safely composed under arbitrary dependence: the **e-BH procedure** controls FDR at the nominal level without any correction factor, regardless of the dependence structure among tests. This property makes e-values ideally suited for the causal discovery setting, where CI tests are heavily dependent (they share the same data and their outcomes are logically constrained by the true causal structure).

### Hypothesis

We hypothesize that replacing p-value-based CI tests with e-value-based CI tests in constraint-based causal discovery, combined with the e-BH procedure for graph-level FDR control, will achieve:
1. **Tighter FDR control**: Nominal FDR levels without conservative corrections
2. **Higher power**: More true edges recovered at the same FDR level
3. **Anytime validity**: The ability to accumulate evidence across data splits, producing valid confidence measures at any stopping point
4. **Better calibration**: The actual FDR will match the nominal level more closely than existing methods

## Proposed Approach

### Overview

We propose **E-PC** (E-value PC), a modified constraint-based causal discovery algorithm that replaces traditional p-value-based CI testing with e-value-based CI testing and uses the e-BH procedure for principled graph-level FDR control. Our approach has three key components:

### Component 1: E-Value Conditional Independence Tests

We develop e-value versions of standard CI tests used in causal discovery:

**Linear case (Gaussian data):** For testing X ⊥ Y | Z in a linear Gaussian model, the standard approach uses the Fisher z-transformation of the partial correlation to produce a p-value. We convert this to an e-value using the **calibrator** approach: given a p-value p from a valid test, e = (1/p) * f(p) where f is a calibration function. Specifically, we use the universal calibrator e = (1/(p * e * log(1/p))) for p < 1/e, and e = 0 otherwise (Vovk and Wang, 2021). This produces a valid e-value from any valid p-value.

However, to maximize power, we also develop **direct e-value CI tests** based on the universal inference framework (Wasserman et al., 2020): compute the likelihood ratio between the unconstrained model and the model with the CI constraint X ⊥ Y | Z enforced via data splitting. The resulting likelihood ratio is a valid e-value by construction.

**Nonlinear case:** For nonparametric CI testing, we adapt the kernel-based approach. We split the data into two halves: use the first half to fit a regression model Y ~ f(X, Z), and compute the e-value as the likelihood ratio of the fitted model versus the null model Y ~ g(Z) on the second half. This split-likelihood-ratio approach yields valid e-values without distributional assumptions.

### Component 2: Sequential Evidence Accumulation

A unique advantage of e-values is their composability: if e₁, e₂, ..., eₖ are e-values for the same hypothesis computed on independent data, then their product e₁ * e₂ * ... * eₖ is also a valid e-value. This enables **sequential evidence accumulation**:

1. Split the dataset into K folds
2. For each CI query (X ⊥ Y | Z), compute an e-value on each fold
3. Multiply the fold-specific e-values to obtain a combined e-value

This sequential approach has two benefits:
- **Increased power**: Evidence accumulates multiplicatively across folds
- **Anytime validity**: After processing any number of folds, the current product is a valid e-value, so the algorithm can be stopped at any time with valid FDR guarantees

### Component 3: Graph-Level FDR Control via e-BH

After running the modified PC algorithm to produce edge-specific e-values, we apply the **e-BH procedure** (Wang and Ramdas, 2022) for graph-level FDR control:

1. For each edge (i, j) in the skeleton, compute an edge-specific e-value E_{ij} that measures the evidence for the edge's presence
2. Sort edges by decreasing e-value: E_{(1)} ≥ E_{(2)} ≥ ... ≥ E_{(m)}
3. Find the largest k such that E_{(k)} ≥ m/(k * q), where q is the target FDR level
4. Include all edges with rank ≤ k in the final graph

The key theorem (Wang and Ramdas, 2022) guarantees that this procedure controls FDR ≤ q under **arbitrary dependence** among the e-values—no correction needed.

### Algorithm: E-PC

```
Input: Data X, target FDR level q, number of folds K
Output: CPDAG G with FDR ≤ q

1. Split data into K folds D₁, ..., Dₖ
2. Initialize G as complete undirected graph
3. For conditioning set size d = 0, 1, 2, ...:
   a. For each adjacent pair (X, Y) in G:
      b. For each subset Z of adj(X)\{Y} with |Z| = d:
         c. For each fold k = 1, ..., K:
            - Compute e-value e_k for H₀: X ⊥ Y | Z on fold Dₖ
         d. Combined e-value: E_{X⊥Y|Z} = ∏ₖ eₖ
         e. Store max e-value for independence: E^ind_{XY} = max_Z E_{X⊥Y|Z}
   f. For each edge (X,Y): E^edge_{XY} = 1/E^ind_{XY}
      (high independence evidence → low edge evidence)
4. Apply e-BH at level q to {E^edge_{XY}} to select edges
5. Orient edges using standard PC orientation rules
6. Return G
```

### Key Innovations

1. **First application of e-values to causal graph structure learning**: While e-values have been applied to Gaussian graphical models (Zhao et al., 2025), the extension to directed causal graphs with orientation rules is non-trivial and novel.

2. **Anytime-valid causal graphs**: The sequential accumulation property means the algorithm can be stopped at any fold and still provides valid FDR control—a property no existing causal discovery method offers.

3. **Dependence-robust FDR control**: Unlike PC-p which requires the conservative BY correction (losing a factor of ~log(m) in power), E-PC achieves FDR control without any correction.

4. **Principled edge confidence scores**: Each edge carries a meaningful e-value that quantifies the evidence for its presence, enabling ranking of edges by reliability.

## Related Work

### Constraint-Based Causal Discovery

The PC algorithm (Spirtes et al., 2000) is the foundational constraint-based method, later refined by the PC-stable variant (Colombo and Maathuis, 2014) for order-independence and extended to handle latent variables via FCI (Spirtes et al., 2000). Recent work has focused on computational efficiency: E-CIT (Guan and Kuang, 2025) reduces CI test complexity via ensemble aggregation, while FMCIT uses flow matching for fast CI testing. Our work is orthogonal—we focus on statistical validity rather than computational speed.

### FDR Control in Causal Discovery

PC-p (Strobl et al., 2019) is the most directly related work. It computes edge-specific p-values by taking the maximum p-value across all CI tests for each edge, then applies the BY procedure for FDR control. Our approach differs in three ways: (1) we use e-values instead of p-values, eliminating the need for conservative BY corrections; (2) we support sequential evidence accumulation across data splits; (3) we provide anytime validity. Zhao et al. (2025) apply e-values for FDR control in Gaussian graphical models (undirected), but do not address causal graph discovery with orientation rules.

### Soft Constraint-Based Methods

DAGPA (Zhou et al., 2025) introduces differentiable d-separation scores for continuous relaxation of CI constraints, enabling gradient-based optimization. While DAGPA also moves beyond binary CI decisions, it takes a fundamentally different approach (continuous optimization) whereas we maintain the combinatorial constraint-based framework with improved statistical guarantees. ECCIT (Pan et al., 2026) addresses CI test calibration but does not provide graph-level FDR control.

### Robust Causal Discovery

MosaCD (Lyu et al., 2025) addresses cascading errors in constraint-based discovery using LLM priors and confidence-based propagation. Bootstrap aggregation methods (Debeire et al., 2024) provide per-edge confidence via resampling. Our approach provides a complementary, principled statistical framework for edge reliability.

### E-Values and Safe Testing

E-values were formalized by Vovk and Wang (2021) and Grünwald et al. (2024), with FDR control via the e-BH procedure established by Wang and Ramdas (2022). The comprehensive treatment by Ramdas and Wang (2024) covers the full theory. Our work brings this powerful statistical framework to causal discovery for the first time.

## Experiments

### Experimental Setup

All experiments run on CPU (2 cores, 128GB RAM) within an 8-hour budget. We use Python with the `causal-learn` library as a base implementation.

### Baselines

1. **PC** (standard): Fixed significance level α ∈ {0.01, 0.05, 0.1}
2. **PC-stable** (Colombo and Maathuis, 2014): Order-independent variant
3. **PC-p** (Strobl et al., 2019): p-value-based FDR control with BY correction
4. **GES** (Chickering, 2002): Score-based baseline (Greedy Equivalence Search)
5. **NOTEARS** (Zheng et al., 2018): Continuous optimization baseline

### Synthetic Data Experiments

**Graph structures**: Erdos-Renyi and scale-free DAGs with p ∈ {10, 20, 50, 100, 200} nodes and average degrees ∈ {2, 4, 6}.

**Data generation**: Linear Gaussian, linear non-Gaussian, and nonlinear additive noise models.

**Sample sizes**: n ∈ {100, 500, 1000, 5000} to study finite-sample behavior.

**Metrics**:
- True FDR vs. nominal FDR (calibration)
- True Positive Rate (power/recall)
- Structural Hamming Distance (SHD)
- F1 score for edge recovery
- Orientation accuracy

**Key experiments**:
1. **FDR calibration**: Compare actual FDR vs. target FDR across methods. We expect E-PC to be well-calibrated, PC-p to be conservative, and standard PC to be anti-conservative.
2. **Power analysis**: At matched FDR levels, compare the number of true edges recovered. We expect E-PC to recover more true edges than PC-p due to avoiding BY correction.
3. **Scalability**: Measure runtime as p increases. E-PC has overhead from data splitting but avoids the computational cost of BY correction on large p-value sets.
4. **Fold sensitivity**: Vary K (number of data splits) from 2 to 20 and measure the power-sample tradeoff.
5. **Anytime property**: Show that stopping after K' < K folds still yields valid FDR control.

### Semi-Synthetic and Real Data Experiments

1. **Sachs dataset** (Sachs et al., 2005): Protein signaling network with 11 nodes and known ground truth from interventional experiments. 853 observational samples.
2. **ALARM network**: Medical diagnosis network with 37 nodes, 46 edges.
3. **Insurance network**: 27 nodes, 52 edges.
4. **Asia network**: 8 nodes, 8 edges (small benchmark).

For each, we compare edge recovery and FDR calibration across methods.

### Ablation Studies

1. **Calibrator choice**: Compare universal calibrator vs. direct split-likelihood e-values
2. **Number of folds K**: Study the power-variance tradeoff
3. **CI test type**: Fisher-z vs. kernel-based e-value CI tests
4. **Graph density**: Performance across sparse to dense graphs
5. **Sample size sensitivity**: How E-PC degrades at small n

### Expected Results

1. E-PC achieves nominal FDR control (actual FDR ≈ target q) across all settings
2. PC-p is conservative (actual FDR << target q) due to BY correction, especially for large graphs
3. Standard PC has uncontrolled FDR that increases with graph size
4. E-PC recovers 10-30% more true edges than PC-p at the same FDR level
5. The anytime property holds: stopping early yields valid but less powerful results
6. Sequential accumulation across folds improves power over single-split approaches

## Success Criteria

### Confirming the hypothesis
- E-PC controls FDR within 10% of the nominal level across ≥80% of experimental configurations
- E-PC recovers significantly more true edges than PC-p at matched FDR levels (p < 0.05 in paired comparison)
- The anytime-valid property is empirically verified: FDR control holds at every stopping point

### Refuting the hypothesis
- If e-value CI tests have systematically lower power than p-value tests (making FDR control trivial but useless)
- If the data splitting required for e-values costs too much statistical power
- If PC-p's BY correction is not actually conservative in practice (making the improvement marginal)

## References

1. Spirtes, P., Glymour, C., and Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press, 2nd edition.

2. Colombo, D. and Maathuis, M. H. (2014). Order-independent constraint-based causal structure learning. *Journal of Machine Learning Research*, 15:3921-3962.

3. Wang, R. and Ramdas, A. (2022). False discovery rate control with e-values. *Journal of the Royal Statistical Society Series B*, 84(3):822-852.

4. Strobl, E. V., Spirtes, P. L., and Visweswaran, S. (2019). Estimating and controlling the false discovery rate of the PC algorithm using edge-specific p-values. *ACM Transactions on Intelligent Systems and Technology*, 10(5):1-37.

5. Ramdas, A. and Wang, R. (2024). Hypothesis testing with e-values. *Foundations and Trends in Statistics*, 1(1-2):1-390.

6. Vovk, V. and Wang, R. (2021). E-values: Calibration, combination, and applications. *Annals of Statistics*, 49(3):1736-1754.

7. Zhou, J., Wang, M., He, A., Zhou, Y., Olya, H., Kocaoglu, M., and Ribeiro, B. (2025). Differentiable constraint-based causal discovery. *Advances in Neural Information Processing Systems (NeurIPS)*.

8. Pan, M., de Mathelin, A., and Tansey, W. (2026). Empirically calibrated conditional independence tests. *arXiv preprint arXiv:2602.21036*.

9. Guan, Z. and Kuang, K. (2025). Efficient ensemble conditional independence test framework for causal discovery. *International Conference on Learning Representations (ICLR 2026)*.

10. Lyu, R., Turcan, A., Zhang, M. J., and Wilder, B. (2025). Improving constraint-based discovery with robust propagation and reliable LLM priors. *arXiv preprint arXiv:2509.23570*.

11. Zhao, S. et al. (2025). FDR control for high-dimensional graphical models via e-values. *Stat*, e70123.

12. Debeire, K. et al. (2024). Bootstrap aggregation and confidence measures to improve time series causal discovery. *Proceedings of the Third Conference on Causal Learning and Reasoning (CLeaR)*.

13. Chickering, D. M. (2002). Optimal structure identification with greedy search. *Journal of Machine Learning Research*, 3:507-554.

14. Zheng, X., Aragam, B., Ravikumar, P., and Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *Advances in Neural Information Processing Systems (NeurIPS)*.

15. Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., and Nolan, G. P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. *Science*, 308(5721):523-529.

16. Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4):1165-1188.

17. Wasserman, L., Ramdas, A., and Balakrishnan, S. (2020). Universal inference. *Proceedings of the National Academy of Sciences*, 117(29):16880-16890.

18. Grünwald, P., de Heide, R., and Koolen, W. M. (2024). Safe testing. *Journal of the Royal Statistical Society Series B*, 86(5):1091-1128.
