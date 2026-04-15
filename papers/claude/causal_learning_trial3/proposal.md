# Assumption-Diagnostic Ensemble Causal Discovery: Reconciling Algorithm Disagreements via Data-Driven Assumption Profiling

## 1. Introduction

### Context and Motivation

Causal discovery---the task of inferring causal structure from observational data---is a cornerstone of scientific reasoning and data-driven decision-making. Over the past two decades, a rich ecosystem of causal discovery algorithms has emerged, spanning constraint-based methods (PC, FCI), score-based methods (GES, FGES), continuous optimization approaches (NOTEARS), and functional causal model methods (LiNGAM, DirectLiNGAM). Each algorithm family makes distinct assumptions about the data-generating process: faithfulness, causal sufficiency, linearity, Gaussianity, or specific noise models.

A critical challenge confronts practitioners: **when applied to the same dataset, different causal discovery algorithms frequently produce conflicting causal graphs**. This disagreement is not merely an inconvenience---it reflects fundamental uncertainty about which assumptions hold for the data at hand. Currently, practitioners must either (a) commit to a single algorithm and hope its assumptions are satisfied, (b) apply naive ensemble strategies like majority voting that ignore *why* algorithms disagree, or (c) manually inspect results and apply domain knowledge to reconcile conflicts.

### Problem Statement

No existing framework systematically exploits the *pattern of disagreement* across structurally different causal discovery algorithms as a diagnostic signal to both (1) infer which causal assumptions are likely satisfied or violated in a given dataset, and (2) use this diagnosis to produce a principled, assumption-aware reconciled causal graph with calibrated per-edge confidence.

### Key Insight

Algorithm disagreements are not random---they are **diagnostic**. Each causal discovery algorithm makes a specific subset of assumptions (faithfulness, sufficiency, linearity, non-Gaussianity, etc.). When two algorithms that differ in exactly one assumption disagree on a specific edge, this disagreement provides direct evidence about whether that distinguishing assumption holds locally. By systematically analyzing disagreement patterns across a diverse portfolio of algorithms, we can construct an **assumption profile** for each edge in the graph, and use this profile to weight algorithms appropriately in a principled reconciliation.

### Hypothesis

We hypothesize that an assumption-diagnostic ensemble approach that (1) runs a portfolio of causal discovery algorithms with complementary assumptions, (2) computes per-edge statistical diagnostics of assumption satisfaction, and (3) reconciles disagreements using assumption-weighted combination, will outperform both individual algorithms and naive ensemble methods (majority voting, union, intersection) across diverse data-generating conditions, while providing interpretable diagnostics about which assumptions hold.

## 2. Proposed Approach

### 2.1 Overview

We propose **ADECD** (Assumption-Diagnostic Ensemble Causal Discovery), a framework with three components:

1. **Algorithm Portfolio Execution**: Run a diverse set of causal discovery algorithms that collectively span the major assumption families.
2. **Per-Edge Assumption Profiling**: For each candidate edge, compute statistical tests that assess whether key assumptions (linearity, Gaussianity, faithfulness, sufficiency) hold locally.
3. **Diagnostic Reconciliation**: Combine algorithm outputs using weights derived from the assumption profile, producing a reconciled graph with calibrated confidence scores and interpretable diagnostics.

### 2.2 Algorithm Portfolio

We select algorithms to maximize coverage of the assumption space:

| Algorithm | Faithfulness | Sufficiency | Linearity | Non-Gaussianity | Identifiability |
|-----------|-------------|-------------|-----------|-----------------|-----------------|
| PC        | Required    | Required    | Agnostic  | Agnostic        | Up to MEC       |
| FCI       | Required    | Not req.    | Agnostic  | Agnostic        | Up to PAG       |
| GES       | Required    | Required    | Agnostic  | Agnostic        | Up to MEC       |
| NOTEARS   | Not req.    | Required    | Required  | Agnostic        | Full (linear)   |
| LiNGAM    | Not req.    | Required    | Required  | Required        | Full            |
| DirectLiNGAM | Not req. | Required   | Required  | Required        | Full            |
| CAM       | Not req.    | Required    | Not req.  | Agnostic        | Full (additive) |

This portfolio ensures that for any pair of algorithms differing in one assumption, their disagreement is informative about that assumption.

### 2.3 Per-Edge Assumption Profiling

For each candidate edge (i, j) that appears in at least one algorithm's output, we compute the following diagnostic features:

**Linearity Assessment:**
- Fit linear regression of X_j on X_i (and vice versa) controlling for other candidate parents
- Compute the HSIC (Hilbert-Schmidt Independence Criterion) test between residuals and the predictor using 500 permutations (increased from a naive 100 to ensure adequate statistical power; runtime remains under 0.5s per edge for n <= 5000)
- A significant HSIC indicates nonlinearity, suggesting linear methods (NOTEARS, LiNGAM) may be unreliable for this edge

**Gaussianity Assessment:**
- Apply the Anderson-Darling test to regression residuals
- Compute excess kurtosis and skewness of residuals
- Non-Gaussian residuals validate the use of LiNGAM-family methods for orientation

**Faithfulness Assessment (Scalable Variant):**
- For graphs with p <= 20 nodes: compute the partial correlation between X_i and X_j given all subsets of potential confounders up to size min(3, p-2)
- For graphs with p > 20 nodes: restrict conditioning sets to subsets of the Markov blanket of X_i and X_j (estimated from the union of algorithm outputs), with a maximum conditioning set size of 4. This reduces the number of conditioning sets from O(C(p,3)) to O(C(|MB|,4)) where |MB| is typically 5-10, making the computation tractable even for 100-node graphs
- Flag edges where the partial correlation is near-zero (below a threshold tau = 0.05) but not exactly zero---these are near-faithfulness violations
- Report the fraction of conditioning sets that yield near-zero partial correlations as the faithfulness violation score

**Sufficiency Assessment:**
- Compare PC output (assumes sufficiency) with FCI output (allows latent confounders)
- If FCI produces a bidirected edge (X_i <-> X_j) where PC produces a directed edge, this indicates a latent confounder

**Effect Strength:**
- Partial correlation magnitude
- Mutual information between X_i and X_j given candidate parents
- These provide a measure of signal strength independent of distributional assumptions

### 2.4 Diagnostic Reconciliation

For each candidate edge e = (i, j), we compute a reconciled confidence score:

$$\text{conf}(e) = \sum_{k=1}^{K} w_k(e) \cdot \mathbb{1}[e \in G_k]$$

where $G_k$ is the graph from algorithm $k$, and $w_k(e)$ is the assumption-profile-derived weight for algorithm $k$ on edge $e$:

$$w_k(e) = \frac{\exp(\alpha_k(e))}{\sum_{k'} \exp(\alpha_{k'}(e))}$$

The algorithm-edge affinity $\alpha_k(e)$ is computed as:

$$\alpha_k(e) = \sum_{a \in \text{Assumptions}(k)} s_a(e) \cdot \beta_a$$

where $s_a(e)$ is the diagnostic score for assumption $a$ at edge $e$ (higher means the assumption is more likely satisfied), and $\beta_a$ are learnable parameters.

**Adaptive Calibration of beta Parameters (Addressing Transfer Concerns):**

Rather than relying solely on synthetic-to-real transfer, we employ a two-stage calibration strategy:

1. **Pre-training on synthetic data**: Learn initial $\beta_a^{(0)}$ parameters on a diverse training set of synthetic SEMs spanning multiple graph sizes, noise types, and assumption profiles. This provides a reasonable initialization.

2. **Adaptive fine-tuning on target data via bootstrap**: Generate B=50 bootstrap resamples of the target dataset. For each resample, run the full algorithm portfolio and compute assumption diagnostics. Use the cross-resample agreement as a self-supervised signal: edges that are consistently present across bootstrap resamples under diverse algorithm outputs are treated as pseudo-ground-truth positives. Optimize $\beta_a$ to maximize the log-likelihood of these pseudo-labels:

$$\beta_a^* = \arg\max_{\beta} \sum_{b=1}^{B} \sum_{e} \left[ \hat{y}_e^{(b)} \log(\text{conf}_\beta(e)) + (1 - \hat{y}_e^{(b)}) \log(1 - \text{conf}_\beta(e)) \right]$$

where $\hat{y}_e^{(b)}$ is the bootstrap stability indicator for edge $e$ in resample $b$. Since $\beta$ has only 4 parameters (one per assumption), this optimization is fast and avoids overfitting.

This adaptive approach mitigates the synthetic-to-real gap by leveraging the structure of the target data itself, while the synthetic pre-training provides a principled initialization that prevents degenerate solutions.

### 2.5 Interpretable Diagnostics

Beyond the reconciled graph, ADECD outputs for each edge:
- An **assumption profile** showing which assumptions are estimated to hold locally
- A **disagreement summary** showing which algorithms include/exclude the edge and why
- A **confidence interval** derived from the assumption-weighted combination

This interpretability is a key advantage over black-box ensemble methods.

## 3. Related Work

### Foundational Causal Discovery Algorithms

The PC algorithm (Spirtes, Glymour, & Scheines, 2000) introduced constraint-based causal discovery using conditional independence tests, producing a completed partially directed acyclic graph (CPDAG) under faithfulness and causal sufficiency assumptions. GES (Chickering, 2002) offered a score-based alternative with provable optimality guarantees. NOTEARS (Zheng et al., 2018) reformulated structure learning as continuous optimization, while LiNGAM (Shimizu et al., 2006) exploited non-Gaussianity for full identifiability. FCI (Spirtes et al., 2000) extended constraint-based methods to allow latent confounders.

### Robustness and Assumption Violations

Montagna et al. (NeurIPS 2023) provided the first systematic benchmark of causal discovery algorithms under assumption violations, finding that score-matching methods are surprisingly robust. However, their work is purely evaluative---they do not propose a method to diagnose violations or reconcile algorithms. Prakash et al. (2024) introduced CDDR, a diagnostic tool for functional causal discovery, but it is limited to bivariate settings. The recent dcFCI (Ribeiro & Heider, 2025) jointly addresses latent confounding, unfaithfulness, and mixed data, but as a single algorithm rather than an ensemble framework.

### Ensemble Approaches in Causal Discovery

**Saldanha (2020)** is the most directly related prior work. She developed a causal ensemble method that combines the outputs of multiple causal discovery algorithms into a single prediction and evaluated it for accuracy, generalizability, and stability across different graph structures. Her approach explores both algorithm selection (choosing the best single algorithm for a dataset) and ensemble combination (aggregating multiple outputs). However, Saldanha's ensemble uses uniform or learned-global weights across all edges---it does not perform *per-edge* assumption profiling to diagnose *why* algorithms disagree on specific edges, which is the core innovation of ADECD.

**Guo, Huang, & Wang (2021)** proposed a scalable two-phase ensemble framework for causality discovery that uses data partitioning and ensemble techniques to handle large datasets. Their first phase partitions data and runs algorithms in parallel; the second phase aggregates results via voting. While this improves scalability, the aggregation remains assumption-agnostic---edges are combined by frequency rather than by diagnosing which algorithm is most trustworthy for each specific edge.

**Dai, Li, & Zhou (2004)** introduced ensembling for MML-based causal discovery, proposing six ensemble algorithms including weighted voting. Their weighted voting without seeding outperformed individual learners. However, their work is restricted to a single algorithm family (MML) and the ensemble diversity comes from random initialization rather than from algorithms with fundamentally different assumptions.

Additional related ensemble work includes: bootstrap aggregation methods (Debeire et al., 2024) that resample data for a single algorithm, capturing only sampling variability; supervised ensemble-based DAG selection (Trentin et al., 2025) that uses multilabel classification on graph topology features to select the best DAG, achieving average distance from ground truth only 10% larger than the oracle; and LLM-based multi-agent approaches (ASoT, 2025) that use language model reasoning rather than statistical diagnostics.

### How ADECD Differs from Prior Ensemble Work

The key distinction of ADECD from all prior ensemble approaches is **per-edge assumption profiling**. Saldanha (2020) learns global algorithm weights; Guo et al. (2021) use frequency-based voting; Dai et al. (2004) ensemble within a single algorithm family; Trentin et al. (2025) select a single best DAG. ADECD instead:
1. Formally maps each algorithm to its specific structural assumptions
2. Uses per-edge statistical tests to construct *local* assumption profiles that vary across edges in the same graph
3. Derives algorithm weights that are edge-specific based on how well local data properties match each algorithm's requirements
4. Provides interpretable diagnostics explaining *why* each edge is included or excluded, rather than treating the ensemble as a black box

This per-edge, assumption-aware design means ADECD can assign high weight to LiNGAM for a non-Gaussian edge while simultaneously trusting PC for a Gaussian edge in the same graph---something no prior ensemble method supports.

## 4. Experiments

### 4.1 Experimental Setup

**Synthetic Data Generation:**
We generate data from structural equation models (SEMs) with known ground truth, varying:
- **Graph structure**: Erdos-Renyi and scale-free DAGs, 10--50 nodes, edge density 1--3 edges per node
- **Linearity**: linear SEMs, nonlinear SEMs (quadratic, sigmoid), and mixed (some edges linear, some nonlinear)
- **Noise distribution**: Gaussian, Laplace, uniform, and mixtures
- **Faithfulness**: exact faithful, near-unfaithful (path cancellations with coefficient ratios < 1.2)
- **Confounders**: causally sufficient, 10% latent confounders, 30% latent confounders
- **Sample size**: 500, 1000, 5000

This yields a grid of ~200 distinct settings, each repeated with 5 random seeds, for a total of ~1,000 experiments. We estimate ~4 hours total runtime on 2 CPU cores based on profiling individual algorithm runtimes on 50-node graphs.

**Real-World Benchmarks:**
- **Sachs et al. (2005)**: Protein signaling network (11 nodes, 17 edges, 7,466 samples)
- **Asia (Lauritzen & Spiegelhalter, 1988)**: Medical diagnosis network (8 nodes, 8 edges)
- **Alarm (Beinlich et al., 1989)**: Medical monitoring network (37 nodes, 46 edges)
- **Child (Spiegelhalter et al., 1993)**: Medical diagnosis network (20 nodes, 25 edges)

### 4.2 Baselines

- **Individual algorithms**: PC, FCI, GES, NOTEARS, LiNGAM, DirectLiNGAM, CAM
- **Naive ensembles**: Majority voting, union graph, intersection graph
- **Saldanha ensemble**: Global-weight ensemble following Saldanha (2020)
- **Bootstrap aggregation**: Bootstrap + PC, Bootstrap + GES (Debeire et al., 2024)
- **Oracle ensemble**: Majority voting with oracle knowledge of which assumptions hold (upper bound)

### 4.3 Metrics

- **Structural Hamming Distance (SHD)**: Number of edge additions, deletions, and reversals needed to match ground truth
- **F1 Score**: Harmonic mean of precision and recall on edges
- **Precision and Recall**: Separately for edge presence and orientation
- **Calibration**: How well confidence scores predict actual edge correctness (Brier score, reliability diagram)
- **Assumption Profile Accuracy**: How accurately the diagnostic framework identifies which assumptions are satisfied (measured against ground truth data-generating conditions)

### 4.4 Ablation Studies

1. **Portfolio size**: Performance vs. number of algorithms in the ensemble (2, 3, 5, 7)
2. **Diagnostic feature importance**: Leave-one-out analysis of each assumption test
3. **Sample size sensitivity**: How performance degrades with smaller samples
4. **Transfer vs. adaptive calibration**: Compare (a) synthetic-only $\beta$, (b) bootstrap-adapted $\beta$, and (c) oracle $\beta$ tuned on test ground truth, to quantify the value of adaptive calibration
5. **Faithfulness approximation quality**: Compare exact enumeration vs. Markov-blanket-restricted conditioning sets on graphs where exact computation is feasible (p <= 20)

### 4.5 Expected Results

We ground our expectations in prior ensemble causal discovery results. Saldanha (2020) reported that her ensemble improved over the median individual algorithm but gains over the *best* algorithm were modest (typically 5-15% SHD improvement in favorable conditions). Trentin et al. (2025) achieved distance from ground truth only 10% larger than an oracle. Based on this:

1. ADECD should outperform individual algorithms by **5--15% in SHD** on average across diverse settings, with the largest gains (up to 20%) in mixed-assumption settings where no single algorithm dominates
2. ADECD should outperform naive majority voting by **5--10% in F1**, with larger gains in settings with heterogeneous assumption violations across edges
3. ADECD should outperform global-weight ensembles (Saldanha-style) by **3--8% in F1**, demonstrating the value of per-edge assumption profiling over global weighting
4. The assumption profile should achieve >75% accuracy in detecting linearity, Gaussianity, and sufficiency violations
5. Confidence scores should be reasonably calibrated (Brier score < 0.20)
6. On real-world benchmarks, ADECD should match or exceed the best individual algorithm without requiring prior knowledge of which algorithm to use

## 5. Success Criteria

### Primary Success Criteria
1. **Accuracy improvement**: ADECD achieves statistically significantly lower SHD than the best individual algorithm across at least 50% of tested settings
2. **Robustness**: ADECD never performs worse than the median individual algorithm in any setting (i.e., it is robust to not knowing which assumptions hold)
3. **Calibration**: Per-edge confidence scores are calibrated (observed accuracy within 15% of predicted confidence across all deciles)

### Secondary Success Criteria
4. **Diagnostic accuracy**: The assumption profile correctly identifies the dominant assumption violation in >65% of cases
5. **Per-edge advantage**: ADECD with per-edge weighting outperforms global-weight ensemble (ablation) in at least 60% of settings
6. **Interpretability**: Case studies demonstrate that the diagnostic output provides actionable insights for practitioners
7. **Efficiency**: Total runtime is no more than 5x the runtime of the slowest individual algorithm

### Refutation Criteria
The hypothesis would be refuted if:
- Naive majority voting consistently matches or outperforms assumption-weighted reconciliation (suggesting disagreement patterns are uninformative)
- Per-edge assumption profiling has near-random accuracy (suggesting local statistical tests cannot reliably assess assumptions)
- The framework only improves performance in contrived settings but not on realistic benchmarks
- Global-weight ensembles match per-edge weighting (suggesting local assumption variation across edges is not exploitable)

## 6. References

1. Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.

2. Chickering, D. M. (2002). Optimal structure identification with greedy search. *Journal of Machine Learning Research*, 3, 507--554.

3. Shimizu, S., Hoyer, P. O., Hyvarinen, A., & Kerminen, A. J. (2006). A linear non-Gaussian acyclic model for causal discovery. *Journal of Machine Learning Research*, 7, 2003--2030.

4. Shimizu, S., Inazumi, T., Sogawa, Y., Hyvarinen, A., Kawahara, Y., Washio, T., Hoyer, P. O., & Bollen, K. (2011). DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model. *Journal of Machine Learning Research*, 12, 1225--1248.

5. Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *Advances in Neural Information Processing Systems*, 31.

6. Montagna, F., Mastakouri, A. A., Eulig, E., Noceti, N., Rosasco, L., Janzing, D., Aragam, B., & Locatello, F. (2023). Assumption violations in causal discovery and the robustness of score matching. *Advances in Neural Information Processing Systems*, 36.

7. Prakash, S., Xia, F., & Erosheva, E. (2024). A diagnostic tool for functional causal discovery. *arXiv preprint arXiv:2406.07787*.

8. Niu, W., Gao, Z., Song, L., & Li, L. (2024). Comprehensive review and empirical evaluation of causal discovery algorithms for numerical data. *arXiv preprint arXiv:2407.13054*.

9. Debeire, K., et al. (2024). Bootstrap aggregation and confidence measures to improve time series causal discovery. *Proceedings of the Conference on Causal Learning and Reasoning (CLeaR)*.

10. Ribeiro, A. H. & Heider, D. (2025). dcFCI: Robust causal discovery under latent confounding, unfaithfulness, and mixed data. *arXiv preprint arXiv:2505.06542*.

11. Brouillard, P., Squires, C., Wahl, J., Kording, K. P., Sachs, K., Drouin, A., & Sridhar, D. (2024). The landscape of causal discovery data: Grounding causal discovery in real-world applications. *arXiv preprint arXiv:2412.01953*.

12. Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. *Science*, 308(5721), 523--529.

13. Peters, J., Buhlmann, P., & Meinshausen, N. (2016). Causal inference by using invariant prediction: identification and confidence intervals. *Journal of the Royal Statistical Society: Series B*, 78(5), 947--1012.

14. Buhlmann, P., Peters, J., & Ernest, J. (2014). CAM: Causal additive models, high-dimensional order search and penalized regression. *Annals of Statistics*, 42(6), 2526--2556.

15. Kaddour, J., Lynch, A., Liu, Q., Kusner, M. J., & Silva, R. (2022). Causal machine learning: A survey and open problems. *arXiv preprint arXiv:2206.15475*.

16. Saldanha, E. (2020). Evaluation of algorithm selection and ensemble methods for causal discovery. *Pacific Northwest National Laboratory (PNNL) Technical Report*.

17. Guo, P., Huang, Y., & Wang, J. (2021). Scalable and flexible two-phase ensemble algorithms for causality discovery. *Big Data Research*, 26, 100252.

18. Dai, H., Li, G., & Zhou, Z.-H. (2004). Ensembling MML causal discovery. *Proceedings of the Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)*, 260--271.

19. Trentin, E., et al. (2025). Supervised ensemble-based causal DAG selection. *Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing (SAC)*.
