# Error Amplification in Entity Resolution Pipelines: A Formal Analysis of Stage-Wise Quality Propagation

## Introduction

### Context
Entity resolution (ER) -- the task of identifying records across or within data sources that refer to the same real-world entity -- is typically implemented as a multi-stage pipeline: **blocking** (filtering candidate pairs), **matching** (classifying pairs as match/non-match), and **clustering** (grouping matched records into entity clusters). Each stage depends on the output of the previous one, creating a sequential dependency chain where errors propagate forward.

### Problem Statement
Current ER research overwhelmingly evaluates stages in isolation -- reporting blocking recall (pairs completeness), matching F1, or clustering metrics separately. This creates a critical blind spot: practitioners tune individual stages without understanding how per-stage improvements translate to end-to-end quality gains. When a pipeline produces poor results, there is no principled way to determine whether to invest in better blocking, better matching, or better clustering.

### Key Insight
Errors at different pipeline stages have fundamentally **asymmetric propagation behavior**. Blocking errors (missed true pairs) are *irrecoverable* -- a true match excluded from the candidate set can never be recovered by downstream matching or clustering. In contrast, matching false negatives can be partially compensated by transitive closure during clustering: if A matches B and B matches C, the cluster {A,B,C} is formed even if the A-C pair was misclassified. Meanwhile, matching false positives can be amplified by clustering when connected components merge distinct entities. This asymmetry means that the marginal value of improving one stage can be orders of magnitude higher than improving another, depending on the current per-stage quality profile.

### Hypothesis
End-to-end ER quality is a non-linear function of per-stage quality metrics, with blocking recall serving as a hard upper bound on end-to-end recall. A formal error propagation model can accurately predict end-to-end quality from per-stage measurements (R^2 >= 0.85) and identify the *bottleneck stage* where improvement effort yields the greatest end-to-end gain. Targeted improvement at the bottleneck stage will yield 1.5-3x more end-to-end F1 improvement per unit of effort compared to uniform improvement across all stages.

## Proposed Approach

### Overview
We propose **PipeER** (Pipeline Error analysis for Entity Resolution), a framework for modeling, measuring, and optimizing error propagation across ER pipeline stages. PipeER has three components:

1. **Error Propagation Model (EPM):** A formal model expressing end-to-end precision, recall, and F1 as functions of per-stage quality metrics.
2. **Error Amplification Factors (EAFs):** Per-stage sensitivity coefficients quantifying the marginal impact of improving each stage on end-to-end quality.
3. **Stage-Optimal Allocation (SOA):** An algorithm that optimally allocates improvement effort across stages to maximize end-to-end F1.

### Method Details

#### 1. Error Propagation Model

We model the ER pipeline as a composition of three stages, where each stage transforms its input and passes the result downstream.

**Stage 1 -- Blocking.** Given the set of all record pairs P = {(r_i, r_j) | i < j}, blocking produces a candidate set C subset of P. Let M subset of P be the true match set. Define:
- Pairs Completeness: PC = |C intersection M| / |M| (blocking recall)
- Reduction Ratio: RR = 1 - |C| / |P|

**Stage 2 -- Matching.** Given candidates C, matching classifies each pair as match or non-match. Let D subset of C be pairs classified as matches. Define:
- Matching Precision: MP = |D intersection M| / |D|
- Matching Recall (conditional on candidates): MR = |D intersection C intersection M| / |C intersection M|

**Stage 3 -- Clustering.** Given predicted matches D, clustering groups records into entity clusters via graph analysis. Define:
- Cluster Precision: CP = fraction of within-cluster pairs that are true matches
- Cluster Recall: CR = fraction of true matches co-clustered

**End-to-End Quality.** We derive closed-form expressions:
- **Recall:** R_e2e = PC * MR * CR_adj, where CR_adj captures the transitive closure effect. Critically, **R_e2e <= PC** -- end-to-end recall can never exceed blocking recall.
- **Precision:** P_e2e is a function of MP, the clustering method, and the topology of false positives in the similarity graph (isolated false positives are filtered by clustering; connected false positives merge into incorrect clusters).
- **F1:** F1_e2e = 2 * P_e2e * R_e2e / (P_e2e + R_e2e)

The key theoretical contributions are: (a) a proof that R_e2e <= PC, establishing blocking as the hard recall bottleneck; (b) a formal characterization of when clustering helps (sparse true-match graphs where transitive closure recovers missed edges) vs. when it hurts (dense false-positive graphs where connected components merge distinct entities); (c) closed-form bounds on end-to-end F1 as a function of per-stage metrics.

#### 2. Error Amplification Factors

For each stage s in {blocking, matching, clustering}, define the Error Amplification Factor:

EAF_s = dF1_e2e / dQ_s

where Q_s is the primary quality metric at stage s (PC for blocking, F1 for matching, etc.). The stage with the highest EAF is the **bottleneck** -- improving it yields the greatest end-to-end gain.

We draw an analogy to **Amdahl's Law** in parallel computing: just as Amdahl's Law identifies the sequential bottleneck that limits parallel speedup, our EAFs identify the pipeline stage that limits end-to-end ER quality. The analogy is precise: blocking acts as a "serial fraction" that bounds achievable recall regardless of how much matching or clustering improves.

We compute EAFs both analytically (by differentiating the EPM) and empirically (by controlled stage degradation experiments), validating that they agree.

#### 3. Stage-Optimal Allocation (SOA)

Given a total improvement budget B, improvement cost functions c_s(delta_Q_s) for each stage, and the EAFs, we formulate:

maximize F1_e2e(Q_1 + dQ_1, Q_2 + dQ_2, Q_3 + dQ_3)
subject to: sum c_s(dQ_s) <= B, dQ_s >= 0

We prove that when improvement costs are linear and the EPM is concave in each Q_s, the optimal strategy reduces to greedy allocation to the highest-EAF stage. When costs are convex (diminishing returns), we provide a projected gradient descent algorithm that converges to the global optimum under mild assumptions. The algorithm complexity is O(n_stages * n_iterations), making it negligible compared to the ER pipeline itself.

### Key Innovations

1. **First formal model** of error propagation across all three ER pipeline stages (blocking, matching, clustering), with closed-form expressions for end-to-end quality from per-stage metrics.
2. **Amdahl's Law analogy for ER:** Error Amplification Factors as a principled metric for identifying pipeline bottlenecks, providing the same role for ER that Amdahl's Law provides for parallel computing.
3. **Transitive closure analysis:** Formal characterization of clustering's dual role -- recovering missed matches via transitivity vs. propagating false matches through connected components.
4. **Stage-optimal allocation algorithm** with provable optimality guarantees under convex cost assumptions.
5. **Framing as an empirical study with formal grounding:** The contributions are primarily the formal model and its empirical validation, honestly positioned as a framework rather than a novel algorithm.

## Related Work

### End-to-End ER Surveys
Christophides et al. (2020) provide a comprehensive survey of ER for big data, covering all pipeline stages independently. Binette and Steorts (2022) present a unified treatment of ER from a statistical perspective, emphasizing the importance of propagating uncertainty through all stages. However, neither work formally models how per-stage errors compound into end-to-end quality or provides optimization frameworks for stage-wise improvement allocation. Our EPM fills this gap with closed-form propagation equations.

### Blocking Evaluation
Papadakis et al. (2020) survey blocking and filtering techniques, establishing standard metrics (pairs completeness, reduction ratio). Their evaluation framework treats blocking in isolation from downstream stages. Our work extends this by formally quantifying blocking's *downstream impact* on matching and clustering quality, proving that blocking recall is a hard upper bound on end-to-end recall.

### Clustering Evaluation
Hassanzadeh et al. (2009) present a framework for evaluating clustering algorithms in duplicate detection. They define cluster-level precision and recall metrics and show that clustering method choice significantly affects ER quality. Our work builds on their metrics but places clustering within the full pipeline context, formally analyzing how matching quality (especially false positive topology) determines clustering behavior.

### ER Benchmark Difficulty
Papadakis, Kirielle, Christen, and Palpanas (2023) critically re-evaluate ER benchmark datasets using linearity and complexity measures, showing many benchmarks are too easy for deep learning methods. Their work characterizes dataset-level difficulty but does not study how pipeline stage interactions vary across datasets of different difficulty levels. Our analysis complements theirs by showing that the bottleneck stage depends on dataset characteristics -- easier datasets tend to have high blocking recall, making matching the bottleneck, while harder datasets are blocked by blocking itself.

### Pipeline Optimization for ML
Siddiqi et al. (2024) present SAGA (SIGMOD 2024), a framework for optimizing data cleaning pipelines for ML applications by searching over pipeline configurations. SAGA treats the pipeline as a black box and optimizes via Auto-ML-style search. Our work differs fundamentally: we provide a *white-box* formal model of error propagation specific to ER pipelines, enabling analytical bottleneck identification without expensive black-box search.

### Probabilistic ER
Fellegi and Sunter (1969) introduced the foundational probabilistic model for record linkage. Their model addresses single-stage classification decisions; our framework extends the analysis to the full multi-stage pipeline, studying how classification errors interact with blocking and clustering.

### Deep Learning for ER
Mudgal et al. (2018, DeepMatcher) and Li et al. (2020, Ditto) show that deep learning improves matching quality on textual and dirty data. Our framework is method-agnostic and analyzes how improvements at the matching stage (from any method) translate to end-to-end gains, providing guidance on whether investing in better matching models is worthwhile vs. improving blocking.

### Progressive ER
Galhotra et al. (2021) propose progressive blocking that uses partial ER output in a feedback loop to refine blocking. This work recognizes the interaction between blocking and matching but focuses on a specific algorithmic solution rather than providing a general formal framework for understanding stage interactions.

## Experiments

### Datasets
We use six standard ER benchmark datasets from the Magellan/DeepMatcher data repositories, spanning structured, dirty, and textual categories:

| Dataset | Domain | Category | Records | True Matches |
|---------|--------|----------|---------|--------------|
| DBLP-ACM | Bibliographic | Structured | 2,616 + 2,294 | 2,224 |
| DBLP-Scholar | Bibliographic | Structured | 2,616 + 64,263 | 5,347 |
| Amazon-Google | Products | Structured | 1,363 + 3,226 | 1,300 |
| Walmart-Amazon | Products | Dirty | 2,554 + 22,074 | 1,154 |
| Abt-Buy | Products | Textual | 1,081 + 1,092 | 1,097 |
| Fodors-Zagats | Restaurants | Structured | 533 + 331 | 112 |

### ER Methods (Matching Stage)
We test four matching methods spanning the methodology spectrum:
1. **Rule-based:** TF-IDF cosine similarity with threshold tuning
2. **Classical ML:** Random forest on hand-crafted similarity features (via py_entitymatching/Magellan)
3. **Active learning:** Dedupe library with Fellegi-Sunter model
4. **Pre-trained embeddings:** Sentence-BERT similarity (lightweight, CPU-compatible; uses a small distilled model that runs efficiently on CPU)

### Blocking Methods
We test three blocking approaches:
1. **Standard blocking:** Single-attribute blocking key
2. **Sorted neighborhood:** Sliding window on sorted keys
3. **Token blocking:** Index on attribute tokens with meta-blocking pruning

### Clustering Methods
We test three clustering approaches:
1. **Connected components:** Transitive closure of predicted matches
2. **Center clustering:** Select a representative center record per cluster
3. **Merge-center clustering:** Iterative merge of most-similar clusters

### Experimental Protocol

**Experiment 1: Baseline Per-Stage Quality Measurement.**
Run the full pipeline on all datasets with all method combinations (3 blocking x 4 matching x 3 clustering = 36 configurations per dataset). Record quality metrics at each stage independently: PC and RR for blocking, precision/recall/F1 for matching (conditional on candidate set), cluster precision/recall/F1 for clustering. This establishes the baseline quality profile.
- 6 datasets x 36 configs = 216 pipeline runs
- Each run repeated with 5 random seeds (for train/test splits in ML methods)

**Experiment 2: Controlled Stage Degradation.**
For each stage independently, artificially degrade its quality while keeping other stages at their best configuration:
- *Blocking degradation:* Randomly remove x% of true pairs from the candidate set (x in {5, 10, 20, 30, 50})
- *Matching degradation:* Randomly flip x% of correct matching decisions (both FP and FN)
- *Clustering degradation:* Randomly split x% of correct clusters or merge x% of adjacent incorrect clusters
Measure end-to-end F1 after each degradation. This produces the **error amplification curves** showing how per-stage degradation maps to end-to-end quality loss.
- 6 datasets x 3 stages x 5 degradation levels = 90 configurations
- Best pipeline configuration used for each dataset
- Repeated with 5 random seeds

**Experiment 3: EPM Validation.**
Compare the EPM's predicted end-to-end F1 (computed analytically from per-stage metrics) against the actual measured F1 across all configurations from Experiments 1 and 2. Report RMSE and R^2 of the model's predictions. If R^2 >= 0.85, the model is validated.

**Experiment 4: Bottleneck Identification and Analysis.**
Compute EAFs for each stage on each dataset using both the analytical EPM and empirical degradation curves. Analyze:
(a) Is blocking always the bottleneck?
(b) Under what conditions does the bottleneck shift to matching or clustering?
(c) How do dataset characteristics (difficulty, size, dirtiness) affect the bottleneck stage?
(d) Do the analytical and empirical EAFs agree?

**Experiment 5: Stage-Optimal Allocation Validation.**
Compare three improvement strategies across datasets:
- **Uniform:** Allocate equal improvement effort to all stages
- **Bottleneck-first:** Allocate all effort to the highest-EAF stage
- **SOA (ours):** Gradient-based optimal allocation using the EPM
We simulate "improvement effort" by interpolating between a degraded and optimal configuration at each stage. Measure end-to-end F1 improvement per unit of effort.

### Metrics
- **Per-stage:** Pairs Completeness, Reduction Ratio, Matching Precision/Recall/F1, Cluster Precision/Recall/F1
- **End-to-end:** Precision, Recall, F1 (pairwise, entity-level)
- **Model accuracy:** RMSE, R^2, Mean Absolute Error between predicted and actual F1
- **Efficiency:** F1 improvement per unit of allocated effort
- All experiments with 5 random seeds; report mean +/- standard deviation

### Expected Results

1. **Model accuracy:** The EPM predicts end-to-end F1 with R^2 >= 0.85 across all datasets and configurations.
2. **Hard recall bound:** R_e2e <= PC confirmed empirically on all datasets (theoretical guarantee validated).
3. **Asymmetric amplification:** EAFs differ by at least 2x between the highest and lowest stages on >= 4/6 datasets. Blocking typically has the highest EAF, especially on harder datasets.
4. **Bottleneck shifts by difficulty:** On easy structured datasets (DBLP-ACM, Fodors-Zagats) where blocking recall is near-perfect, the bottleneck shifts to matching. On harder/dirty datasets (Walmart-Amazon, Abt-Buy), blocking is the clear bottleneck.
5. **Clustering dual role:** Connected-component clustering amplifies both true matches (via transitivity) and false matches (via error propagation), showing a precision-recall tradeoff that depends on the false-positive density.
6. **SOA outperforms:** The SOA allocation achieves 1.5-3x higher F1 improvement per unit effort compared to uniform allocation on >= 4/6 datasets.

### Computational Feasibility
- Each full pipeline run on the largest dataset (DBLP-Scholar, ~64K records) takes < 2 minutes on CPU with standard blocking
- Experiment 1: 216 configs x 5 seeds x 2 min = ~36 hours worst case, but most datasets are much smaller; estimated ~4 hours with parallelization on 2 cores
- Experiment 2: 90 configs x 5 seeds x 1 min = ~7.5 hours, estimated ~2 hours with parallelization
- Experiments 3-5: Analytical computation, negligible time
- Sentence-BERT embedding computation: pre-computed once per dataset (~10 min total)
- **Total: ~6-7 hours on 2 CPU cores**, within the 8-hour constraint

## Success Criteria

### Primary (confirms hypothesis)
1. The EPM predicts end-to-end F1 with R^2 >= 0.85 across all datasets and configurations
2. EAFs differ by at least 2x between the highest and lowest stages on >= 4/6 datasets
3. SOA allocation outperforms uniform allocation by >= 20% in F1 improvement per unit effort on >= 4/6 datasets

### Secondary (strengthens contribution)
4. The hard recall bound R_e2e <= PC holds on all datasets (theoretical validation)
5. The bottleneck stage predictably shifts based on dataset difficulty
6. Analytical and empirical EAFs agree within 15% relative error

### Negative result (refutes hypothesis)
- Per-stage qualities are approximately independent (errors don't compound)
- The EPM fails to predict end-to-end quality (R^2 < 0.5)
- EAFs are approximately equal across stages (no bottleneck structure)

## References

1. Fellegi, I. P. and Sunter, A. B. "A Theory for Record Linkage." Journal of the American Statistical Association, 64(328):1183-1210, 1969.

2. Hassanzadeh, O., Chiang, F., Lee, H. C., and Miller, R. J. "Framework for Evaluating Clustering Algorithms in Duplicate Detection." Proceedings of the VLDB Endowment, 2(1):1282-1293, 2009.

3. Christen, P. Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection. Springer, 2012.

4. Konda, P., Das, S., Suganthan G. C., P., Doan, A., Ardalan, A., Ballard, J. R., Li, H., Panahi, F., Zhang, H., Naughton, J. F., Prasad, S., Krishnan, G., Deep, R., and Raghavendra, V. "Magellan: Toward Building Entity Matching Management Systems." Proceedings of the VLDB Endowment, 9(12):1197-1208, 2016.

5. Mudgal, S., Li, H., Rekatsinas, T., Doan, A., Park, Y., Krishnan, G., Deep, R., Arcaute, E., and Raghavendra, V. "Deep Learning for Entity Matching: A Design Space Exploration." Proceedings of the 2018 International Conference on Management of Data (SIGMOD), 19-34, 2018.

6. Papadakis, G., Skoutas, D., Thanos, E., and Palpanas, T. "Blocking and Filtering Techniques for Entity Resolution: A Survey." ACM Computing Surveys, 53(2):31:1-31:42, 2020.

7. Christophides, V., Efthymiou, V., Palpanas, T., Papadakis, G., and Stefanidis, K. "An Overview of End-to-End Entity Resolution for Big Data." ACM Computing Surveys, 53(6):127:1-127:42, 2020.

8. Li, Y., Li, J., Suhara, Y., Doan, A., and Tan, W.-C. "Deep Entity Matching with Pre-Trained Language Models." Proceedings of the VLDB Endowment, 14(1):50-60, 2020.

9. Galhotra, S., Firmani, D., Saha, B., and Srivastava, D. "Efficient and Effective ER with Progressive Blocking." The VLDB Journal, 30(4):537-557, 2021.

10. Binette, O. and Steorts, R. C. "(Almost) All of Entity Resolution." Science Advances, 8(12):eabi8021, 2022.

11. Papadakis, G., Kirielle, N., Christen, P., and Palpanas, T. "A Critical Re-evaluation of Benchmark Datasets for (Deep) Learning-Based Matching Algorithms." arXiv preprint arXiv:2307.01231, 2023.

12. Siddiqi, S., Kern, R., and Boehm, M. "SAGA: A Scalable Framework for Optimizing Data Cleaning Pipelines for Machine Learning Applications." Proceedings of the ACM on Management of Data (SIGMOD), 1(4):258:1-258:26, 2024.
