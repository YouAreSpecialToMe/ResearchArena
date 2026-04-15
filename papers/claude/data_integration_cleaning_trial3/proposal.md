# Characterizing and Exploiting Operator Interaction Effects in Data Cleaning Pipelines

## Introduction

### Context

Data cleaning is a critical step in modern data science workflows, consuming up to 80% of data scientists' time. A typical data cleaning pipeline composes multiple operators—missing value imputation, outlier detection and removal, deduplication, format standardization, constraint-based repair, and type coercion—applied sequentially to transform dirty data into a form suitable for downstream analysis. The quality of the final output depends not only on which operators are included but crucially on the *order* in which they are applied.

### Problem Statement

Despite significant progress in automated data cleaning pipeline construction—systems like AlphaClean (Krishnan & Wu, 2019), Learn2Clean (Berti-Équille, 2019), SAGA (Siddiqi et al., 2024), and DiffPrep (Li et al., 2023) can search for effective pipeline configurations—the field lacks a fundamental understanding of *why* certain operator orderings outperform others. Current approaches treat the pipeline search space as opaque: they either enumerate permutations exhaustively, use reinforcement learning to explore blindly, or require differentiable relaxations. None of them characterize the structural properties of operator interactions that determine when ordering matters and when it does not.

This is analogous to the state of query processing before relational algebra: without understanding algebraic equivalences between operator sequences, query optimizers could not efficiently prune the plan search space. We argue that data cleaning pipeline optimization faces the same fundamental bottleneck.

### Key Insight

Data cleaning operators exhibit measurable *interaction effects*: the output quality achieved by applying operator A followed by operator B can differ substantially from applying B followed by A, and the combined effect can be greater than (synergistic) or less than (antagonistic) the sum of individual effects. These interactions arise from concrete structural reasons: for example, imputing missing values before outlier detection may introduce synthetic outliers, while detecting outliers first may bias imputation by treating valid extreme values as outliers.

### Hypothesis

**We hypothesize that pairwise interaction effects between data cleaning operators are (1) systematic and dataset-characteristic-dependent rather than random, (2) predictable from lightweight dataset features without running the full pipeline, and (3) sufficient to guide pipeline construction that matches or exceeds the quality of exhaustive search while requiring only a fraction of the computational cost.**

## Proposed Approach

### Overview

We propose a three-part research contribution:

1. **Interaction Characterization Framework**: A formal framework for measuring and categorizing operator interaction effects in data cleaning pipelines, inspired by factorial experimental design from statistics (specifically, ANOVA-style interaction analysis).

2. **Systematic Empirical Study**: A large-scale empirical study measuring pairwise and higher-order interaction effects between 8 common data cleaning operator categories across 18 real-world datasets with diverse error profiles.

3. **Interaction-Aware Pipeline Optimizer (IAPO)**: An efficient pipeline construction algorithm that uses learned interaction profiles to prune the search space, achieving near-optimal pipeline quality in sublinear time relative to exhaustive search.

### Method Details

#### Part 1: Interaction Characterization Framework

We define the following formal concepts:

**Operator Effect**: For a cleaning operator $O_i$ applied to dataset $D$, the *main effect* $\text{ME}(O_i, D)$ is the change in data quality metric $Q$ when $O_i$ is applied: $\text{ME}(O_i, D) = Q(O_i(D)) - Q(D)$.

**Pairwise Interaction Effect**: For operators $O_i$ and $O_j$, the *interaction effect* measures the non-additive component of their combined application:
$$\text{IE}(O_i, O_j, D) = Q(O_i(O_j(D))) - Q(O_j(D)) - \text{ME}(O_i, D)$$

This captures how much $O_i$'s effectiveness changes when applied after $O_j$ versus on clean data.

**Order Sensitivity**: The *order sensitivity* of a pair measures asymmetry:
$$\text{OS}(O_i, O_j, D) = |Q(O_i(O_j(D))) - Q(O_j(O_i(D)))|$$

Pairs with $\text{OS} \approx 0$ are approximately commutative; high $\text{OS}$ indicates strong order-dependence.

**Interaction Categories**: Based on these metrics, we classify operator pairs as:
- **Commutative**: $\text{OS} \approx 0$ (order doesn't matter)
- **Synergistic**: $\text{IE} > 0$ (one enables the other)
- **Antagonistic**: $\text{IE} < 0$ (one degrades the other's effectiveness)
- **Order-Critical**: High $\text{OS}$ (specific order is strongly preferred)
- **Independent**: Both $\text{IE} \approx 0$ and $\text{OS} \approx 0$ (no interaction)

We also define *idempotency* (applying an operator twice yields no additional benefit) and *absorption* (one operator subsumes another's effect) as special cases.

#### Part 2: Systematic Empirical Study

**Operators Studied** (8 categories, each with 2-3 implementations):
1. Missing value imputation (mean/median, KNN, iterative/MICE)
2. Outlier detection and removal (IQR, Z-score, isolation forest)
3. Duplicate detection and removal (exact, fuzzy/Jaccard, sorted-neighborhood)
4. Format standardization (regex-based, rule-based)
5. Type coercion and casting (strict, lenient)
6. Constraint-based repair (FD-based, denial constraint-based)
7. Value normalization (min-max, z-score, robust scaling)
8. Encoding of categorical variables (one-hot, ordinal, target encoding)

**Datasets** (18 datasets): We assemble a diverse benchmark from three sources:
- **From CleanML** (Li et al., 2021): 10 datasets with naturally occurring errors—Adult/USCensus, Credit, EEG, Marketing, Titanic, Movie, Company, Restaurant, Sensor, Airbnb
- **From REIN** (Abdelaal et al., 2023): 4 datasets spanning healthcare (Cardiotocography), finance (German Credit), industrial telemetry (Steel Plates Faults), and general tabular (Anneal)
- **From OpenML**: 4 additional datasets selected for diverse error profiles—Hepatitis (ID 55, high missing rate ~48%), Labor (ID 4, small with missing values), Soybean (ID 42, multi-class with inconsistencies), and Vote (ID 56, political survey with missing values)

Each dataset has naturally occurring errors (not injected synthetically) covering missing values, outliers, duplicates, and inconsistencies. This 18-dataset collection provides sufficient diversity for the interaction predictor to identify meaningful patterns across dataset characteristics.

**Experimental Protocol**:
- For each dataset, measure all pairwise interaction effects ($8 \times 7 = 56$ ordered pairs per operator implementation)
- Evaluate using multiple quality metrics: downstream ML model accuracy (classification, regression), data quality scores (completeness, consistency, accuracy), and constraint satisfaction rates
- Analyze interaction patterns across datasets to identify systematic trends
- Compute interaction stability: do the same operator pairs consistently synergize/antagonize across datasets with similar characteristics?

**Dataset Characteristics Analyzed**: We measure lightweight features of each dataset (percentage of missing values, outlier ratio, duplication rate, constraint violation rate, column type distribution, dataset size) and correlate them with observed interaction patterns.

#### Part 3: Interaction-Aware Pipeline Optimizer (IAPO)

Using insights from Parts 1 and 2, we build a pipeline optimizer with a two-tier prediction strategy designed to work reliably even with a modest number of training datasets:

1. **Profiles the input dataset** by computing lightweight features (O(n) scan): missing rate per column, outlier ratio, duplication rate, constraint violation rate, column type distribution, dataset size
2. **Predicts interaction effects** using a hybrid approach:
   - **Tier 1 — Rule-based heuristics** derived from the empirical study (e.g., "if missing_rate > 0.2, imputation-outlier interaction is likely antagonistic"; "if duplication_rate > 0.05, deduplication before constraint repair is synergistic"). These rules are interpretable, require no training data, and serve as strong defaults.
   - **Tier 2 — Dataset-similarity-weighted lookup**: For operator pairs not covered by rules, IAPO computes the cosine similarity between the new dataset's feature vector and each training dataset's features, then returns the similarity-weighted average of observed interaction effects. This avoids the underdetermined regression problem that arises with few training points and many features.
3. **Constructs an interaction graph** where nodes are operators and directed edges encode predicted synergy/antagonism and order preferences
4. **Generates candidate pipelines** using topological sorting on the interaction graph, prioritizing strongly synergistic orderings and avoiding antagonistic sequences
5. **Evaluates top-K candidates** (typically K=5-10 vs. n!=40320 for 8 operators) and returns the best pipeline

**Fallback Strategy**: When prediction confidence is low—defined as (a) no training dataset has cosine similarity > 0.5 to the input, or (b) the similarity-weighted predictions have high variance (coefficient of variation > 1.0)—IAPO automatically increases K (up to K=20) or falls back to a greedy beam search that evaluates operators incrementally, adding the best next operator at each step. This ensures robust performance even for out-of-distribution datasets.

**Complexity Reduction**: For $n$ operators, exhaustive search requires $O(n!)$ pipeline evaluations. IAPO requires $O(n^2)$ interaction predictions + $O(K)$ pipeline evaluations, where $K \ll n!$. The fallback greedy search requires $O(n^2)$ evaluations in the worst case, still far below $O(n!)$.

### Key Innovations

1. **First systematic empirical characterization** of cleaning operator interaction effects across diverse datasets
2. **Formal interaction framework** with clear metrics (interaction effect, order sensitivity) enabling quantitative comparison
3. **Practical pipeline optimization** that leverages structural understanding rather than brute-force search
4. **Transferable interaction profiles**: once interaction patterns are characterized for a class of datasets, they can be reused without re-measurement

## Related Work

### Automated Data Cleaning Pipeline Construction

**AlphaClean** (Krishnan & Wu, 2019) formulates pipeline construction as a search problem over a space of cleaning transformations. It notes that conditional assignments are non-commutative but does not systematically study interaction effects; instead, it uses asynchronous search to find good orderings. Our work provides the formal framework that could explain and improve AlphaClean's search.

**Learn2Clean** (Berti-Équille, 2019) uses Q-learning to find optimal preprocessing sequences. It treats the pipeline as a black box, learning through trial and error without characterizing why certain orderings work. Our interaction analysis could provide the inductive bias that accelerates RL exploration.

**SAGA** (Siddiqi, Kern, & Boehm, SIGMOD 2024, Best Paper Honorable Mention) optimizes cleaning pipelines at scale, using pruning by monotonicity and generating top-K effective pipelines. Our work is complementary: interaction effects provide a richer set of pruning properties beyond monotonicity.

**DiffPrep** (Li, Chen, Chu, & Rong, SIGMOD 2023) uses differentiable relaxation to search over preprocessing pipelines. It requires a differentiable downstream model and treats operators as differentiable modules. Our approach is model-agnostic and provides interpretable interaction characterizations.

### Data Cleaning Benchmarks

**CleanML** (Li et al., 2021) benchmarks the impact of individual cleaning methods on ML models but evaluates operators independently, not in composition. Our work directly addresses this gap by studying how composed operators interact.

**REIN** (Abdelaal et al., 2023) provides a comprehensive benchmark of 38 cleaning methods in ML pipelines, evaluating detection and repair individually. Our contribution extends this to studying pairwise and higher-order compositions.

**"Automatic Data Repair: Are We Ready to Deploy?"** (Ni et al., PVLDB 2024) evaluates 12 repair algorithms individually under different error rates and types. Our work studies what happens when multiple repair approaches are composed.

### Formal Foundations

**Núñez-Corrales et al. (2020)** propose an algebraic approach to data transformations in cleaning using homotopy type theory, focusing on provenance tracking. Their work is theoretical and does not include empirical validation of algebraic properties. Our work provides the missing empirical grounding.

**"An Algebraic Approach Towards Data Cleaning"** (Brüggemann et al., 2013) uses information algebra for data cleaning but focuses on association rules, not operator composition.

### Reinforcement Learning for Data Cleaning

**RLclean** (Peng et al., Information Sciences 2024) integrates detection and repair in an RL framework, iterating between them. Our interaction analysis could explain when this iterative approach helps (synergistic detection-repair interactions) vs. when it doesn't.

**CleanSurvival** (2025) applies RL to preprocessing for survival analysis. Like Learn2Clean, it treats operator interactions as a black box.

## Experiments

### Planned Setup

All experiments run on CPU (2 cores, 128GB RAM). Data cleaning operators are implemented in Python using scikit-learn, pandas, and custom implementations. The experimental framework is designed for reproducibility with fixed random seeds and statistical significance testing.

### Benchmarks and Datasets

We use 18 datasets from established benchmarks:
- **From CleanML**: 10 datasets with naturally occurring errors (Adult/USCensus, Credit, EEG, Marketing, Titanic, Movie, Company, Restaurant, Sensor, Airbnb)
- **From REIN**: 4 datasets spanning healthcare (Cardiotocography), finance (German Credit), industrial telemetry (Steel Plates Faults), and general tabular (Anneal)
- **From OpenML**: 4 datasets with diverse error profiles (Hepatitis, Labor, Soybean, Vote)

### Metrics

1. **Intrinsic quality**: Completeness, consistency, accuracy (measured against ground truth where available)
2. **Downstream ML quality**: Classification accuracy/F1, regression RMSE, across 5 ML models (logistic regression, random forest, gradient boosted trees, SVM, KNN)
3. **Pipeline search efficiency**: Number of pipeline evaluations needed to find a pipeline within X% of optimal quality

### Expected Results

Based on our preliminary analysis and the literature, we expect:

1. **Most operator pairs are NOT commutative**: We predict that 60-70% of operator pairs will exhibit statistically significant order effects ($\text{OS} > 0.01$ in normalized quality metrics).

2. **Imputation-outlier interactions are strongly order-dependent**: Imputing before outlier detection should often be antagonistic (synthetic outliers), while the reverse should be synergistic.

3. **Deduplication-constraint repair interactions are synergistic**: Removing duplicates before constraint repair reduces the repair search space and avoids conflicting repairs.

4. **Interaction patterns are predictable from dataset characteristics**: Datasets with high missing value rates should show stronger imputation interaction effects; datasets with many constraints should show stronger repair ordering effects.

5. **IAPO achieves 90%+ of optimal quality with <5% of the search cost**: The interaction graph should effectively prune the vast majority of suboptimal orderings.

### Ablation Studies

- **Interaction granularity**: Pairwise only vs. including 3-way interactions
- **Number of dataset features**: Minimal (3 features) vs. comprehensive (15+ features) for interaction prediction
- **Transfer across domains**: Train interaction predictor on one domain, test on another
- **Sensitivity to operator implementations**: Do interaction patterns hold across different implementations of the same operator category?

## Success Criteria

The hypothesis is **confirmed** if:
1. At least 50% of operator pairs show statistically significant (p < 0.05) non-zero interaction effects across multiple datasets
2. The rule-based heuristics correctly predict the sign (synergistic/antagonistic) of interaction effects for at least 70% of operator pairs on held-out datasets, and the similarity-weighted lookup achieves rank correlation (Spearman's rho) > 0.5 with actual interaction magnitudes in leave-one-out cross-validation
3. IAPO finds pipelines within 95% of exhaustive search quality using at most 10% of the search cost

The hypothesis is **refuted** if:
1. Most operator pairs commute (order rarely matters), making interaction analysis unnecessary
2. Interaction effects are entirely dataset-specific with no transferable patterns (rule accuracy < 50%, rank correlation < 0.2)
3. IAPO offers no meaningful speedup over random search baselines

## References

1. Krishnan, S., & Wu, E. (2019). AlphaClean: Automatic Generation of Data Cleaning Pipelines. arXiv:1904.11827.

2. Berti-Équille, L. (2019). Learn2Clean: Optimizing the Sequence of Tasks for Web Data Preparation. In Proceedings of The Web Conference (WWW) 2019.

3. Siddiqi, S., Kern, R., & Boehm, M. (2024). SAGA: A Scalable Framework for Optimizing Data Cleaning Pipelines for Machine Learning Applications. In Proceedings of the ACM on Management of Data (SIGMOD 2024).

4. Li, P., Chen, Z., Chu, X., & Rong, K. (2023). DiffPrep: Differentiable Data Preprocessing Pipeline Search for Learning over Tabular Data. In Proceedings of the ACM on Management of Data (SIGMOD 2023).

5. Li, P., Rao, X. S., Blase, J., Zhang, Y., Chu, X., & Zhang, C. (2021). CleanML: A Benchmark for Joint Data Cleaning and Machine Learning [Experiments and Analysis]. arXiv:1904.09483.

6. Abdelaal, M., Hammacher, C., & Schöning, H. (2023). REIN: A Comprehensive Benchmark Framework for Data Cleaning Methods in ML Pipelines. In Proceedings of EDBT 2023.

7. Ni, W., Miao, X., Zhao, X., Wu, Y., Liang, S., & Yin, J. (2024). Automatic Data Repair: Are We Ready to Deploy? In Proceedings of the VLDB Endowment, 17(10), 2617-2630.

8. Núñez-Corrales, S., Li, L., & Ludäscher, B. (2020). A First-Principles Algebraic Approach to Data Transformations in Data Cleaning: Understanding Provenance from the Ground Up. In 12th International Workshop on Theory and Practice of Provenance (TaPP 2020).

9. Mahdavi, M., & Abedjan, Z. (2019). Raha: A Configuration-Free Error Detection System. In Proceedings of the 2019 International Conference on Management of Data (SIGMOD 2019).

10. Mahdavi, M., & Abedjan, Z. (2020). Baran: Effective Error Correction via a Unified Context Representation and Transfer Learning. In Proceedings of the VLDB Endowment, 13(11).

11. Peng, J., et al. (2024). RLclean: An Unsupervised Integrated Data Cleaning Framework Based on Deep Reinforcement Learning. Information Sciences, 682, 121281.

12. Li, L., Fang, L., Ludäscher, B., & Torvik, V. I. (2024). AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark. In Findings of EMNLP 2025.

13. Chu, X., Ilyas, I. F., Krishnan, S., & Wang, J. (2016). Data Cleaning: Overview and Emerging Challenges. In Proceedings of the 2016 International Conference on Management of Data (SIGMOD 2016).

14. Rekatsinas, T., Chu, X., Ilyas, I. F., & Ré, C. (2017). HoloClean: Holistic Data Repairs with Probabilistic Inference. In Proceedings of the VLDB Endowment, 10(11).

15. Miao, X., Zhao, Y., An, B., Gao, J., & Deng, Y. (2024). Relational Data Cleaning Meets Artificial Intelligence: A Survey. Data Science and Engineering.
