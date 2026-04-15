# Know Your Assumptions: Adaptive Algorithm Selection for Robust Causal Discovery via Data-Driven Diagnostics

## Introduction

### Context

Causal discovery from observational data is a fundamental problem in science and engineering. Given a dataset of jointly observed variables, the goal is to infer the underlying causal directed acyclic graph (DAG) that generated the data. Over the past three decades, a rich landscape of algorithms has emerged, spanning constraint-based methods (PC, FCI), score-based methods (GES, BOSS), and functional model-based methods (LiNGAM, ANM). Each algorithm family relies on distinct assumptions about the data-generating process: PC and GES assume faithfulness and causal sufficiency but are agnostic to functional form; LiNGAM requires linearity and non-Gaussianity; ANM requires additive noise with specific independence conditions.

### Problem Statement

A critical but underappreciated challenge in causal discovery is **assumption-method mismatch**: practitioners must choose an algorithm before knowing which assumptions hold in their data. When assumptions are satisfied, the chosen algorithm performs well; when they are violated, performance degrades unpredictably. Recent empirical work has confirmed that (1) hyperparameter and algorithm choices dramatically affect causal discovery outcomes (Constantinou et al., CLeaR 2024), (2) differentiable methods degrade severely under realistic assumption violations (Montagna et al., NeurIPS 2023), and (3) no single algorithm dominates across all data regimes (Hasan et al., 2024). Yet no existing framework systematically diagnoses which assumptions hold for a given dataset and uses this diagnosis to guide algorithm selection.

### Key Insight

Different assumptions can be empirically *tested* on the observed data before committing to an algorithm. Linearity can be assessed via nonlinearity tests on variable pairs; non-Gaussianity can be detected via normality tests on marginals and residuals; the additive noise model can be validated via residual independence tests; and faithfulness violations can be flagged by detecting near-zero partial correlations. These diagnostics, computed as a preprocessing step, provide a data-driven basis for selecting and weighting algorithms whose assumptions best match the data characteristics.

### Hypothesis

**An adaptive ensemble approach that diagnoses data characteristics and weights multiple causal discovery algorithms by assumption compatibility will produce more accurate causal graphs than any single algorithm or naive ensemble, particularly on datasets with heterogeneous causal mechanisms (mixtures of linear/nonlinear, Gaussian/non-Gaussian relationships). This advantage will hold not only on synthetic data but also on real-world benchmarks where the data-generating assumptions are unknown a priori.**

## Proposed Approach

### Overview

We propose **AACD (Assumption-Adaptive Causal Discovery)**, a framework with three stages:

1. **Diagnose**: Run a battery of statistical tests on the observed data to characterize which assumptions hold globally and locally (per variable pair).
2. **Discover**: Run multiple causal discovery algorithms from different families, each operating under its own assumptions.
3. **Aggregate**: Combine the outputs using assumption-diagnostic-informed weights, producing a final causal graph with calibrated edge-level confidence scores.

### Stage 1: Assumption Diagnostics

For each variable pair (X_i, X_j), we compute the following diagnostic scores:

**D1 -- Linearity Score.** Fit both a linear regression and a polynomial regression of degree 3 from X_i to X_j (and vice versa). The linearity score is 1 minus the normalized improvement in R^2 from linear to nonlinear. High scores indicate the relationship is approximately linear.

**D2 -- Non-Gaussianity Score.** Apply the Shapiro-Wilk test to the residuals of the best-fitting (linear or nonlinear) model. The score is 1 minus the p-value, so high scores indicate non-Gaussian residuals. Also compute marginal non-Gaussianity using excess kurtosis and skewness.

**D3 -- Additive Noise Model (ANM) Score.** Fit X_i = f(X_j) + N_j and test independence of residuals N_j from X_j using the Hilbert-Schmidt Independence Criterion (HSIC). We use 200 permutations on subsamples of min(n, 500) points to keep computation tractable. Do the same in the reverse direction. If one direction yields independent residuals and the other does not, the ANM assumption is satisfied and direction is identifiable. The ANM score captures the asymmetry. For sample sizes n < 500, we use the full sample and report a diagnostic confidence flag indicating reduced reliability.

**D4 -- Faithfulness Proximity Score.** Compute partial correlations for all variable pairs conditioning on subsets up to size k=2 (for p > 30, fall back to k=1). Flag pairs where the partial correlation is close to zero (|r| < epsilon) but not statistically significant at the chosen alpha level. High proximity scores indicate potential near-unfaithfulness, which can cause errors in constraint-based methods.

**D5 -- Homoscedasticity Score.** For each ordered pair (X_i, X_j), fit X_j = f(X_i) + N and test whether the variance of N depends on X_i (Breusch-Pagan test). Heteroscedasticity can provide additional identifiability but violates assumptions of some methods.

These diagnostics produce a profile matrix D of size (p x p x 5) for p variables, plus p-length vectors for marginal properties.

#### Diagnostic Reliability Under Small Samples

A critical concern is that diagnostic tests may be unreliable at small sample sizes (n < 500), where statistical tests have low power. We explicitly characterize this by:

1. **Diagnostic confidence scores**: For each diagnostic D_k, we compute a sample-size-dependent confidence weight c_k(n) = min(1, n / n_k^*), where n_k^* is the minimum effective sample size for diagnostic k (determined empirically on synthetic calibration data). When c_k(n) is low, the corresponding diagnostic contributes less to the weighting function, and the framework gracefully degrades toward equal weighting.

2. **Dedicated sample-size robustness experiment**: We systematically evaluate diagnostic accuracy (e.g., ability to correctly classify linear vs. nonlinear, Gaussian vs. non-Gaussian) as a function of sample size n in {200, 500, 1000, 2000, 5000} and trace how diagnostic errors propagate to AACD's final graph accuracy.

### Stage 2: Multi-Algorithm Discovery

Run the following algorithms on the dataset, chosen to represent the major assumption families:

| Algorithm | Key Assumptions | Family |
|-----------|----------------|--------|
| PC | Faithfulness, causal sufficiency | Constraint-based |
| GES | Faithfulness, causal sufficiency | Score-based |
| BOSS | Faithfulness, causal sufficiency | Permutation-based |
| DirectLiNGAM | Linearity, non-Gaussianity, causal sufficiency | Functional |
| ANM (pairwise) | Additive noise, restricted function class | Functional |

Each algorithm outputs a graph G_k (a DAG or CPDAG). For CPDAGs, we consider all edges in the equivalence class.

### Stage 3: Assumption-Adaptive Aggregation

For each potential edge (X_i -> X_j), we compute a weighted confidence score:

**Step 3a -- Algorithm-level assumption compatibility.** For each algorithm k, compute an assumption compatibility score w_k based on how well the global and local diagnostics match algorithm k's assumptions. For example:
- DirectLiNGAM's weight increases when the average linearity score is high AND the average non-Gaussianity score is high.
- PC's weight decreases when faithfulness proximity scores indicate near-violations.
- ANM's weight increases when ANM scores show clear directional asymmetry.

Formally, for algorithm k with assumption set A_k, the weight is:

w_k = prod_{a in A_k} phi_a(D) * c_a(n)

where phi_a(D) maps the relevant diagnostic scores to [0, 1] via a sigmoid function, and c_a(n) is the sample-size-dependent diagnostic confidence weight.

**Sigmoid parameter tuning.** The sigmoid functions phi_a(D) = 1 / (1 + exp(-s * (d - t))) have two parameters each: threshold t and steepness s. We perform a grid search over t in {0.3, 0.5, 0.7} and s in {5, 10} using held-out synthetic tuning datasets (30 datasets per data type, separate from test data). We select parameters that minimize average SHD on the tuning set. This tuning is performed once and the selected parameters are fixed for all test evaluations. Importantly, tuning data and test data use different random seeds and different graph structures to mitigate circularity concerns (see Section on addressing circularity).

**Step 3b -- Edge-level aggregation.** For each edge (i, j), the confidence score is:

C(i -> j) = sum_k w_k * I[G_k contains i -> j] / sum_k w_k

where I[.] is the indicator function.

**Step 3c -- Acyclicity enforcement.** The confidence matrix C may contain cycles. We enforce acyclicity using a principled two-step approach: (1) compute a topological ordering by solving a minimum feedback arc set (FAS) relaxation via the weighted Eades heuristic (which greedily assigns nodes to a linear ordering by comparing outgoing vs. incoming edge weights), then (2) remove edges that violate this ordering. We compare this against the simpler greedy lowest-confidence-edge-removal baseline and report the difference. For small graphs (p <= 20), we also compute the exact minimum-weight FAS via integer linear programming as a reference.

**Step 3d -- Threshold selection.** Edges with confidence above a threshold tau are included in the final graph. We select tau using the same held-out synthetic tuning data as sigmoid calibration, or report the full confidence-ranked edge list.

### Extension: Learned Weighting Function

As a more principled alternative to sigmoid-based weighting, we explore training a lightweight meta-learner (logistic regression) on synthetic calibration data. The meta-learner maps diagnostic feature vectors (D1-D5 values, sample size, graph density estimate) to per-algorithm weight vectors. This is trained on 500 synthetic datasets with known ground truth, where the optimal weight for each algorithm is computed retrospectively based on its actual performance. We compare this learned weighting against the sigmoid-based approach.

### Addressing Circularity Between Synthetic Tuning and Evaluation

A key concern is that tuning diagnostics and weighting functions on synthetic data could create circularity: if the synthetic data generator and the diagnostics share structural assumptions, AACD may appear to work on synthetic data but fail on messier real data. We address this through three mechanisms:

1. **Structural separation**: Tuning data uses different graph structures (Erdos-Renyi with p=15, degree=3) than test data (random DAGs with varied p and degree), ensuring the tuned parameters are not overfit to specific graph topologies.

2. **Real-world validation**: We evaluate AACD on 3 real-world benchmark datasets (Sachs, Alarm, Child) where the data characteristics are not controlled by the experimenter and the ground truth was established independently. This is the definitive test of whether AACD's diagnostics provide genuine signal beyond synthetic artifacts.

3. **Cross-validated tuning**: Within the synthetic tuning phase, we use 5-fold cross-validation to select sigmoid parameters, reporting both in-fold and out-of-fold performance to detect overfitting.

### Key Innovations

1. **Data-driven diagnostics before discovery**: Unlike existing ensemble approaches that aggregate without considering assumption fit -- including Saldanha (2020)'s ensemble methods, VISTA's structural voting (Sun et al., 2026), Bagged-PCMCI+'s bootstrap aggregation (Debeire et al., 2024), and Guo et al.'s (2021) data/algorithm ensemble -- AACD uses statistical tests to assess which algorithms' assumptions match the data and weights accordingly.

2. **Heterogeneous mechanism handling**: In real causal systems, different causal mechanisms may have different functional forms. AACD handles this because diagnostics are computed per variable pair, allowing different algorithms to be trusted for different parts of the graph.

3. **Edge-level calibrated confidence**: Rather than outputting a single binary graph, AACD produces confidence scores for each edge, enabling downstream risk-aware decision making.

4. **Graceful degradation under small samples**: The diagnostic confidence weighting ensures that when sample sizes are too small for reliable diagnostics, AACD falls back toward equal weighting rather than being misled by noisy diagnostic scores.

5. **Real-world validation**: Unlike prior ensemble approaches evaluated only on synthetic data, we validate on standard benchmarks (Sachs, Alarm, Child) where ground truth was established independently of the data generation process.

## Related Work

### Causal Discovery Algorithms
The foundational algorithms in our ensemble span three paradigms. **Constraint-based methods** like PC (Spirtes, Glymour, and Scheines, 2000) and FCI (Spirtes, 2001) use conditional independence tests to build the causal skeleton and orient edges. **Score-based methods** like GES (Chickering, 2002) search the space of DAGs by optimizing a score function (BIC, BDeu). **Permutation-based methods** like BOSS (Andrews et al., NeurIPS 2023) and GRaSP (Lam et al., UAI 2022) search over variable orderings. **Functional methods** like LiNGAM (Shimizu et al., JMLR 2006) and ANM (Hoyer et al., NeurIPS 2008) exploit asymmetries in the data-generating process.

### Ensemble and Aggregation Approaches

The most directly related prior work is **Saldanha (2020)**, who evaluated algorithm selection and ensemble methods for causal discovery. Saldanha proposed combining multiple causal discovery algorithms into a single prediction, evaluating accuracy, generalizability, and stability. However, Saldanha's ensemble methods combine algorithm outputs via voting-based aggregation without diagnosing which algorithms' assumptions match the data. **AACD differs fundamentally** by introducing data-driven assumption diagnostics that provide a principled, per-dataset basis for differential algorithm weighting.

**Mio et al. (ACM SAC 2025)** propose supervised ensemble-based causal DAG selection, using multilabel classification on DAG topologies to select the best DAG from a set of candidates. Their approach requires training a supervised model to distinguish good DAGs from bad ones based on structural features. AACD differs by weighting algorithms based on assumption compatibility rather than learning to classify DAG quality from topology alone.

**Guo, Huang, and Wang (2021)** develop scalable two-phase ensemble algorithms that combine data partitioning with algorithm ensembling, parallelized via Spark. Their focus is on scalability rather than assumption-aware weighting.

VISTA (Sun et al., 2026) decomposes the global problem into local Markov Blanket subgraphs and aggregates via weighted voting with exponential decay penalization. Bagged-PCMCI+ (Debeire et al., CLeaR 2024) applies bootstrap aggregation to a single algorithm for time series data. **Our work differs** in that we do not simply aggregate by voting or bootstrap stability -- we use data-driven assumption diagnostics to differentially weight algorithms based on their compatibility with the observed data.

### Assumption Diagnostics
Prakash et al. (2024) propose CDDR, a diagnostic tool for bivariate functional causal discovery that evaluates how assumption violations interact with sample size. However, CDDR is limited to bivariate analysis and is used for post-hoc evaluation, not for guiding algorithm selection. Montagna et al. (NeurIPS 2023) benchmark causal discovery methods under assumption violations, finding that score matching methods are surprisingly robust -- but they do not propose an adaptive framework. GaussDetect-LiNGAM (Ding and Zhang, 2025) diagnoses non-Gaussianity to decide whether LiNGAM is applicable, but only for bivariate cases within the LiNGAM family. **Our work is the first to propose a comprehensive, multivariate assumption diagnostic framework that guides multi-algorithm ensemble weighting for causal discovery.**

### Real-World Causal Discovery Benchmarks
The Sachs protein signaling dataset (Sachs et al., Science 2005) is a canonical benchmark with 11 phosphoproteins, 17 known causal edges, and ~5,400 flow cytometry observations from primary immune cells. The Alarm monitoring network (Beinlich et al., 1989) with 37 variables and 46 edges, and the Child network (Spiegelhalter et al., 1993) with 20 variables and 25 edges, provide additional standard benchmarks from the bnlearn repository with known ground-truth DAGs and well-characterized data distributions. The Tuebingen cause-effect pairs (Mooij et al., JMLR 2016) provide 108 real-world bivariate cause-effect pairs for pairwise evaluation. These benchmarks are critical because they test whether AACD's diagnostics provide genuine signal on data whose characteristics were not designed by the experimenter.

### Hyperparameter Sensitivity and Power Analysis
Constantinou et al. (CLeaR 2024) demonstrate that hyperparameter choices dramatically affect causal discovery outcomes. Kummerfeld and Rao (2025) characterize how PC-based methods err. Kummerfeld, Williams, and Ma (2024) develop the first power analysis framework for causal discovery. These findings motivate our approach: rather than relying on a single algorithm with potentially misspecified hyperparameters, we aggregate across algorithms and let the diagnostics arbitrate.

## Experiments

### Experimental Setup

**Synthetic Data Generation.** We generate synthetic datasets with controlled characteristics using structural equation models (SEMs):

1. **Homogeneous-Linear-Gaussian (HLG)**: All mechanisms are linear with Gaussian noise. This is the "easy" setting where PC and GES have well-understood guarantees.

2. **Homogeneous-Linear-NonGaussian (HLNG)**: All mechanisms are linear with non-Gaussian noise (Laplace, uniform). LiNGAM should excel here.

3. **Homogeneous-Nonlinear (HNL)**: All mechanisms are nonlinear (e.g., quadratic, sigmoid) with additive noise. ANM-based methods should be preferred.

4. **Heterogeneous-Mixed (HM)**: Different edges have different functional forms -- some linear-Gaussian, some linear-non-Gaussian, some nonlinear. Each edge is independently assigned from 4 mechanism types with equal probability (25% each).

5. **Semi-Heterogeneous (SH)**: A more realistic heterogeneity pattern where 70% of edges follow a dominant mechanism type (e.g., linear-Gaussian) with 30% exceptions (e.g., nonlinear or non-Gaussian). This models real-world systems where most relationships share a common form but a minority deviate. We test all 4 dominant-mechanism variants.

6. **Near-Unfaithful (NU)**: Include near-canceling paths that create near-zero partial correlations, violating faithfulness. Constraint-based methods should struggle.

**Graph Parameters.** We use a focused design to stay within computational budget:
- Number of variables: p in {10, 20} for the full grid; p = 50 for a targeted subset (HM and SH data types only, n = 1000)
- Graph density: expected degree = 3 (fixed to reduce combinatorial explosion)
- Sample size: n in {500, 1000, 5000}

**Repetitions.** 10 random seeds per configuration, with Wilcoxon signed-rank tests for statistical comparisons aggregated across configurations.

**Tuning Data.** A separate set of 30 synthetic datasets per data type (6 types x 30 = 180 total) generated with p=15, degree=3, n=1000 using different graph structures (Erdos-Renyi) than test data (random DAGs). Never used in test evaluation.

### Real-World Benchmark Evaluation

A critical addition addressing the concern that AACD's advantage may be an artifact of synthetic data design:

**Sachs protein signaling network** (Sachs et al., Science 2005). 11 phosphoproteins and phospholipids measured via flow cytometry in ~5,400 primary immune system cells. Ground truth: 17 directed edges established from prior biological literature. The data contains a mix of linear and nonlinear relationships with non-Gaussian marginals, making it a natural testbed for AACD's diagnostics. We use the observational subset (853 samples from the unperturbed condition) following the standard protocol in the causal discovery literature.

**Alarm monitoring network** (Beinlich et al., 1989). 37 variables, 46 directed edges, modeling an anesthesia monitoring system. Data is discrete/categorical, so we use a continuous relaxation (sample from the Bayesian network with Gaussian noise) following standard practice, or use discretized versions of PC/GES. This tests AACD at moderate scale.

**Child network** (Spiegelhalter et al., 1993). 20 variables, 25 directed edges. Similar protocol to Alarm. This provides a mid-sized benchmark with known ground truth.

**Tuebingen cause-effect pairs** (Mooij et al., JMLR 2016). 108 real-world bivariate cause-effect pairs from diverse domains. While bivariate, this evaluates whether AACD's diagnostics (particularly D1, D2, D3) correctly identify the appropriate algorithm for individual causal relationships.

For real-world benchmarks, we report both SHD and F1 against the known ground truth, and compare AACD's diagnostic profile against the actual data characteristics (as a sanity check that diagnostics are measuring something real).

### Baselines

1. **Individual algorithms**: PC, GES, BOSS, DirectLiNGAM, ANM (each with default hyperparameters)
2. **Oracle selection**: Best individual algorithm per dataset (upper bound on selection)
3. **Naive ensemble**: Majority voting across all algorithms (equal weights)
4. **Stability-weighted ensemble**: Bootstrap each algorithm 10 times (reduced from 20 for computational budget), weight by stability
5. **AACD-sigmoid (ours)**: Assumption-adaptive weighted ensemble with tuned sigmoid weighting
6. **AACD-learned (ours)**: Assumption-adaptive weighted ensemble with meta-learned weighting

### Metrics

- **Structural Hamming Distance (SHD)**: Number of edge additions, deletions, and reversals needed to transform the inferred graph into the true graph.
- **F1 score**: Harmonic mean of precision and recall for edge presence (ignoring direction).
- **Orientation accuracy**: Among correctly identified edges, the fraction with correct orientation.
- **Calibration**: For AACD's confidence scores, measure calibration (do edges with confidence 0.8 appear in the true graph ~80% of the time?).

### Core Experiments

**Experiment 1: Main comparison across synthetic data regimes.** Compare all methods across 6 data types x 2 graph sizes {10, 20} x 3 sample sizes {500, 1000, 5000} = 36 configurations, plus 2 targeted p=50 configurations (HM and SH at n=1000) = 38 total configurations, 10 seeds each (380 total runs). Report SHD, F1, and orientation accuracy. Aggregate results by data type.

**Experiment 2: Real-world benchmark evaluation.** Apply all methods to Sachs, Child, and a subset of Tuebingen pairs. Report SHD and F1 against known ground truth. This is the definitive test of whether AACD's diagnostics transfer beyond synthetic data. Also report AACD's diagnostic profile for each dataset (e.g., "Sachs: 65% nonlinear edges, 80% non-Gaussian residuals") as interpretability output.

**Experiment 3: Sample-size robustness of diagnostics.** For a fixed graph (p=20, degree=3), evaluate: (a) diagnostic classification accuracy as a function of n in {200, 500, 1000, 2000, 5000}; (b) AACD's SHD with and without diagnostic confidence weighting c_a(n) at each sample size.

**Experiment 4: Semi-heterogeneous vs. fully-heterogeneous.** Compare AACD's advantage on the semi-heterogeneous (SH, 70/30 split) vs. fully-heterogeneous (HM, 25/25/25/25 split) data types. This tests whether AACD's advantage is an artifact of uniform mixing or persists under more realistic heterogeneity patterns.

### Ablation Studies

1. **Diagnostic importance**: Remove each diagnostic (D1-D5) one at a time and measure performance degradation.
2. **Number of algorithms**: Start with 2 algorithms and add more; at what point do additional algorithms help?
3. **Diagnostic computation cost**: Profile the wall-clock time spent on diagnostics vs. discovery algorithms.
4. **Sigmoid vs. learned weighting**: Compare hand-tuned sigmoid, grid-search-tuned sigmoid, and meta-learned weighting functions.
5. **Diagnostic confidence weighting**: Compare AACD with and without the sample-size-dependent confidence weights c_a(n).
6. **Acyclicity enforcement**: Compare Eades heuristic vs. greedy edge removal vs. exact ILP (for p <= 20).

### Expected Results

1. On **homogeneous** datasets, AACD should match or slightly exceed the best individual algorithm.
2. On **heterogeneous-mixed** and **semi-heterogeneous** datasets, AACD should significantly outperform all individual algorithms and naive ensemble.
3. On **real-world benchmarks** (Sachs, Child), AACD should outperform naive ensemble and match or exceed the best individual algorithm, validating that diagnostics provide genuine signal beyond synthetic artifacts.
4. **Semi-heterogeneous vs. heterogeneous**: AACD's advantage should persist (though possibly reduced) on the more realistic 70/30 split compared to the 25/25/25/25 split.
5. **Sample-size robustness**: At n >= 1000, diagnostics should be reliable and AACD should clearly outperform baselines. At n = 500, confidence weighting should prevent degradation below naive ensemble.
6. **Ablation** should show D1 (linearity) and D2 (non-Gaussianity) as the most important diagnostics.
7. **Acyclicity enforcement**: Eades heuristic should match or slightly outperform greedy removal; exact ILP should serve as a reference showing both are near-optimal.

### Computational Feasibility

All experiments run on CPU only (2 cores, 128GB RAM). Revised runtime budget:

- **Experiment 1 (synthetic)**: 38 configs x 10 seeds x 6 methods = 2,280 method-runs. Average ~2 min per method-run for p=10-20, ~10 min for p=50. Estimated: ~3-4 hours.
- **Experiment 2 (real-world)**: 3 benchmark datasets x 6 methods. Estimated: ~30 min.
- **Experiment 3 (sample-size)**: 5 sample sizes x 10 seeds x 6 methods. Estimated: ~30 min.
- **Experiment 4 (SH vs HM)**: Subset of Experiment 1, no additional time.
- **Ablations**: ~1.5 hours.
- **Tuning**: Sigmoid grid search on 180 datasets, ~30 min.
- **Total estimated: ~6-7 hours on 2 CPU cores.**

Fallback: If ANM's HSIC permutation tests are too slow at p=50, we reduce permutations to 100 or skip ANM for p=50 runs, reporting this limitation.

We will use the `causal-learn` Python library for PC, GES, DirectLiNGAM, and ANM, and the `gcastle` or `py-tetrad` wrapper for BOSS. All diagnostics use standard `scipy` and `scikit-learn` functions.

## Success Criteria

The hypothesis is confirmed if:
1. AACD achieves lower SHD and higher F1 than all individual algorithms on heterogeneous-mixed data, with statistical significance (Wilcoxon signed-rank test, p < 0.05).
2. AACD achieves lower SHD than naive ensemble and stability-weighted ensemble on heterogeneous-mixed data.
3. **On at least 2 of 3 real-world benchmarks**, AACD matches or outperforms the best individual algorithm and outperforms naive ensemble.
4. AACD's edge confidence scores are reasonably well-calibrated (calibration error < 0.15).
5. Ablation confirms that assumption diagnostics (not just algorithm diversity) drive the improvement.
6. AACD with confidence weighting never performs worse than naive ensemble, even at n=500.
7. AACD's advantage persists on semi-heterogeneous (70/30) data, not just fully-heterogeneous (25/25/25/25).

The hypothesis is refuted if:
1. Naive ensemble (equal weights) performs as well as AACD on both synthetic and real-world data.
2. A single algorithm consistently dominates regardless of data characteristics.
3. Diagnostic tests are too noisy at realistic sample sizes (n >= 1000) to provide useful signal.
4. AACD outperforms on synthetic data but fails on real-world benchmarks, indicating the diagnostics are artifacts of synthetic data design.

## References

1. Spirtes, P., Glymour, C., and Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.
2. Chickering, D.M. (2002). Optimal structure identification with greedy search. *JMLR*, 3, 507-554.
3. Shimizu, S., Hoyer, P.O., Hyvarinen, A., and Kerminen, A. (2006). A linear non-Gaussian acyclic model for causal discovery. *JMLR*, 7, 2003-2030.
4. Hoyer, P.O., Janzing, D., Mooij, J.M., Peters, J., and Scholkopf, B. (2008). Nonlinear causal discovery with additive noise models. *NeurIPS*.
5. Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D.A., and Nolan, G.P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. *Science*, 308(5721), 523-529.
6. Beinlich, I.A., Suermondt, H.J., Chavez, R.M., and Cooper, G.F. (1989). The ALARM monitoring system: A case study with two probabilistic inference techniques for belief networks. *AIME*.
7. Spiegelhalter, D.J., Dawid, A.P., Lauritzen, S.L., and Cowell, R.G. (1993). Bayesian analysis in expert systems. *Statistical Science*, 8(3), 219-247.
8. Mooij, J.M., Peters, J., Janzing, D., Zscheischler, J., and Scholkopf, B. (2016). Distinguishing cause from effect using observational data: methods and benchmarks. *JMLR*, 17(32), 1-102.
9. Saldanha, E. (2020). Evaluation of Algorithm Selection and Ensemble Methods for Causal Discovery. *CMU Technical Report*.
10. Guo, P., Huang, Y., and Wang, J. (2021). Scalable and flexible two-phase ensemble algorithms for causality discovery. *Big Data Research*, 26, 100252.
11. Andrews, B., Ramsey, J., Sanchez-Romero, R., Camchong, J., and Kummerfeld, E. (2023). Fast scalable and accurate discovery of DAGs using the best order score search and grow-shrink trees. *NeurIPS*.
12. Lam, W.-Y., Andrews, B., and Ramsey, J. (2022). Greedy relaxations of the sparsest permutation algorithm. *UAI*.
13. Montagna, F., Mastakouri, A.A., Eulig, E., Noceti, N., Rosasco, L., Janzing, D., Aragam, B., and Locatello, F. (2023). Assumption violations in causal discovery and the robustness of score matching. *NeurIPS*.
14. Constantinou, A.C., et al. (2024). Hyperparameters in structure learning. *CLeaR*.
15. Prakash, S., Xia, F., and Erosheva, E. (2024). A diagnostic tool for functional causal discovery. *arXiv:2406.07787*.
16. Kummerfeld, E., Williams, L., and Ma, S. (2024). Power analysis for causal discovery. *International Journal of Data Science and Analytics*, 17, 289-304.
17. Ding, Z. and Zhang, X.-P. (2025). GaussDetect-LiNGAM: Causal direction identification without Gaussianity test. *arXiv:2512.03428*.
18. Mio, C., Lin, J., Damiani, E., and Gianini, G. (2025). Supervised ensemble-based causal DAG selection. *ACM SAC*.
19. Kummerfeld, E. and Rao, A. (2025). How PC-based methods err. *arXiv:2502.14719*.
20. Sun, H., Tian, P., Zhou, Z., Zhang, J., Li, P., and Liu, A.L. (2026). Efficient causal structure learning via modular subgraph integration. *arXiv:2601.21014*.
21. Debeire, K., Runge, J., Gerhardus, A., and Eyring, V. (2024). Bootstrap aggregation and confidence measures to improve time series causal discovery. *CLeaR*.
22. Hasan, U., et al. (2024). A comprehensive review of causal discovery algorithms. *arXiv:2407.13054*.
23. Eades, P., Lin, X., and Smyth, W.F. (1993). A fast and effective heuristic for the feedback arc set problem. *Information Processing Letters*, 47(6), 319-323.
