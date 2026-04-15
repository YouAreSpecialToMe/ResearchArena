# Masked-Child Surrogate Calibration for Safe EC Prefix Decisions

## Introduction

Recent enzyme-function predictors still operate mainly as closed-set EC classifiers: they are trained to commit to one of the currently known EC-4 leaves. In practice, that is exactly where they fail. Under a known EC-3 parent, new or newly supported EC-4 children continue to appear over time, and a deployed model can do more harm by over-specializing into the wrong known child than by stopping at the correct prefix such as `2.7.1.-`.

This proposal therefore makes a deliberately narrow claim. The paper is not about solving open-world enzyme annotation in general. It is about a **decision-layer heuristic** for hierarchical EC models: can a leakage-controlled masked-child surrogate be used to calibrate when a fixed scorer should stop at an internal EC node rather than descend to an unsupported child prediction?

The key insight is that the practically important novelty pattern is often not arbitrary out-of-distribution biology, but a more specific label-space shift: the EC-3 parent is familiar while the correct EC-4 child is absent from the current vocabulary. We simulate that shift by masking observed children during calibration, rebuilding every parent-local feature that could leak the removed child, and using the resulting score distribution as a surrogate for "known parent, missing child" events. The scientific question is not whether this surrogate is perfect, but whether it transfers to temporally future child-emergence events better than generic hierarchical selective rules.

The central hypothesis is empirical and falsifiable: **masked-child surrogate calibration yields safer stop/go decisions than generic confidence, energy, OpenMax, or hierarchical selective baselines, and its parent-level gains transfer at least partially to real future-child events under known parents**. If that transfer is weak, the project still contributes a careful leakage-controlled benchmark of surrogate calibration for hierarchical EC prediction, but not a practical method claim.

## Proposed Approach

### Overview

The method, **MCC-EC**, is a post hoc decision layer on top of a fixed hierarchical scorer:

1. cache frozen protein embeddings
2. train lightweight child scorers at each internal EC node
3. calibrate a standard known-child acceptance threshold
4. calibrate a masked-child surrogate threshold under leakage-controlled child removal
5. descend only when both checks pass; otherwise emit the current EC prefix

The contribution is intentionally at the **decision layer**, not in backbone representation learning.

### Base hierarchical scorer

Each enzyme sequence `x` is encoded once with a frozen protein LM such as `esm2_t30_150M_UR50D`, giving embedding `z(x)`.

For each internal node `v`, we train a cheap child scorer over `c in children(v)` using parent-local features derived from cached embeddings:

- cosine similarity to child prototypes
- margin between the top two children
- local k-NN vote mass restricted to node `v`
- local density under retained child supports
- optional high-identity retrieval feature if already available cheaply

The default model is multinomial logistic regression. This makes the empirical claim cleaner: any improvement comes from calibration, not from a stronger backbone.

### Known-child acceptance test

At node `v`, let `c_hat_v(x)` be the highest-scoring child. Using ordinary calibration examples whose true child is represented in training, define nonconformity

`A_v(x, y) = 1 - s_v(x, y)`.

Let `q_v` be the split-conformal quantile at target error `alpha_v`. We allow descent only if

`A_v(x, c_hat_v(x)) <= q_v`.

This guards against descending when even an observed child looks atypical.

### Masked-child surrogate calibration

The novel component is a second threshold targeting the narrower failure mode "parent known, true child missing."

For each eligible parent `v`:

1. choose one calibration child per parent per seed from children with at least `m_child = 25` training examples
2. remove that child from the parent-local label set
3. rebuild every parent-local support structure without removed-child examples
4. refit only the parent-local logistic layer on the rebuilt features
5. score held-out examples from the removed child against the retained siblings

The rebuild is strict because leakage is the main methodological risk. For every masked child we recompute:

- child prototypes
- node-restricted k-NN indices or exact-neighbor caches
- vote-mass features
- density features
- any identity-based lookup pools

From masked-child calibration examples define the surrogate score

`B_v(x) = max_c s_v(x, c)`.

Let `t_v` be the empirical `(1 - beta_v)` quantile of `B_v` on masked-child examples. We descend only if

`B_v(x) > t_v`.

If the conformal known-child test passes but the masked-child surrogate test fails, the system stops at the current prefix and can optionally attach `open_child_under_parent`.

### Combined inference rule

Starting at the root, the model repeats:

1. score children at the current node
2. test known-child conformity
3. test masked-child surrogate safety
4. descend only if both pass

Outputs are:

- an EC-4 leaf when all levels pass
- a safe prefix such as `3.2.1.-`
- a safe prefix plus `open_child_under_parent`

### Distinctive methodological angle: transfer diagnostics

The paper's genuinely distinctive element is not just the threshold itself, but the **parent-level transfer study**:

- does masked-child gain on parent `v` predict future-child gain on the same parent?
- does masked-child calibration transfer better than generic hierarchical selective thresholds?
- which parent statistics predict transfer, such as sibling spread, prototype compactness, or nearest-sibling overlap?

This makes the paper useful even if headline gains are modest, because it tells us when surrogate calibration is trustworthy and when it is not.

## Related Work

### Enzyme-function prediction and evaluation

**EC-Bench** defines modern evaluation tasks for enzyme commission prediction and clarifies the distinction between exact label prediction and coarser placement. We borrow its evaluation mindset but focus on safe stopping under a known hierarchy rather than proposing a new benchmark.

**ProFAB** shows why realistic splits matter for protein function annotation. That motivates our use of temporally future child-emergence events rather than only random holdout.

### Hierarchical EC models

**GloEC** and **HIT-EC** already show that hierarchy-aware EC models outperform naive flat classifiers in closed-set settings. Our method does not compete as a new architecture; it assumes such a scorer already exists and asks when it should stop.

**Hierarchical Contrastive Learning for Enzyme Function Prediction** learns hierarchy-aware representations and reports gains on unseen EC numbers. That is still a representation-learning contribution, whereas MCC-EC is a post hoc stopping rule.

### Closest recent unseen-function paper

The most important recent paper for novelty positioning is **How Not to be Seen: Predicting Unseen Enzyme Functions using Contrastive Learning** (Ma, Joshi, Friedberg, Li; bioRxiv preprint created February 26, 2026). It presents **EnzPlacer**, a contrastive model that places proteins with unseen EC-4 labels at the EC-3, EC-2, or EC-1 level.

That paper is close in task motivation but still materially different from MCC-EC:

- EnzPlacer is a **new predictive model** trained to place unseen functions into coarser known EC space.
- MCC-EC is a **decision-layer calibration heuristic** on top of an existing hierarchical scorer.
- EnzPlacer targets coarse placement quality directly.
- MCC-EC targets **whether to descend** under a known parent, with explicit leakage-controlled surrogate calibration and transfer diagnostics to future-child events.

Because of EnzPlacer, this proposal should not claim novelty for "predict unseen enzyme functions via higher-level EC outputs." The novelty claim must be restricted to **masked-child surrogate calibration as an empirical stop/go heuristic and benchmarked transfer analysis**.

### Selective prediction and conformal methods

**Hierarchical Selective Classification** provides the right generic baseline family: confidence-thresholded internal-node prediction under coverage constraints. Our work differs by constructing a protein-specific surrogate for the missing-child regime and by testing its transfer to real temporal child-emergence events.

**Functional protein mining with conformal guarantees** shows that conformal methods can be useful in protein decision pipelines. We adopt standard conformal calibration for represented children, but the new piece is the surrogate calibration layer for the missing-child case.

### Positioning

The proposal should be positioned as one of two papers, depending on the results:

- if transfer is strong: a new empirical decision-layer heuristic for safe EC prefix decisions under future-child emergence
- if transfer is weak: a careful leakage-controlled benchmark study showing the limits of masked-child surrogate calibration

Either way, the claim is narrower and more defensible than a general open-world enzyme annotation method.

## Experiments

### Dataset and split

Primary data:

- Swiss-Prot enzymes with complete EC-4 annotations
- train/validation snapshot: Swiss-Prot `2018_02`
- future evaluation snapshot: Swiss-Prot `2023_01`

Main deployment-faithful parent selection:

- choose EC-3 parents using the 2018 snapshot only
- require at least 4 observed EC-4 children
- require at least 25 sequences per retained child
- keep at most 12 parents with the largest training support

This is narrower than the previous draft and is more credible for the compute budget.

### Evaluation settings

1. **Temporal future-child test**
   Evaluate 2023 proteins whose EC-3 parent existed in 2018 but whose correct EC-4 child is absent or unsupported in the 2018 label set. This is the headline experiment.

2. **Masked-child surrogate test**
   For the same training-selected parents, mask one child per parent per seed and evaluate safe stopping on those removed-child examples. This is the development proxy, not the final claim.

3. **Parent-level transfer diagnostic**
   For each parent, compute the gain of MCC-EC over each baseline on the masked-child test and on the temporal future-child test, then report Pearson/Spearman correlation and a simple regression against parent statistics such as sibling dispersion and support imbalance.

4. **Parent-level transferability stratification**
   Divide parents into easy-transfer and hard-transfer strata based on pre-registered statistics computed from the 2018 snapshot alone. Test whether MCC-EC is especially useful in one regime. Existing EC selective-prediction papers do not provide this diagnostic.

### Baselines

All baselines share the same frozen embeddings and parent set.

1. **Forced hierarchical classifier**
   Always predict the deepest EC-4 leaf.

2. **Flat EC-4 classifier with abstention**
   Tests whether hierarchy matters after adding stopping.

3. **HSC-style hierarchical selective rule**
   Confidence-thresholded deepest-valid ancestor from the same node scores.

4. **Energy threshold**
   Parent-local energy-based stopping.

5. **Parent-OpenMax**
   Tail-fitted unknown-child score at each parent.

6. **Max-probability stopping**
   Weak baseline only.

### Metrics

Primary metrics:

- catastrophic overspecialization rate on future-child events
- correct-safe-prefix rate
- hierarchical loss at matched coverage
- AUROC for detecting `open_child_under_parent`

Secondary metrics:

- selective EC-4 F1
- prefix accuracy at EC-1/2/3/4
- parent-level calibration error
- runtime and memory

### Matched-coverage protocol

To avoid test tuning, thresholds are selected only on the validation split. We report results at three pre-registered target coverages: `{0.60, 0.80, 0.95}`. This is sufficient to show the tradeoff while keeping the runtime manageable.

### Success conditions for novelty

The paper's novelty depends less on raw average gain and more on showing one of the following:

- masked-child calibration transfers to future-child events better than generic selective rules
- transfer can be predicted from parent-level diagnostics not previously reported

If neither holds, the work becomes an informative benchmark paper rather than a new method paper.

### Feasibility and compute budget

The experiment matrix is intentionally lean:

- one shared embedding pass
- at most 12 EC-3 parents
- 2 random seeds
- 3 coverage targets
- 1 masked child per parent per seed
- no full-tree refits, only parent-local logistic retraining

Budget estimate:

- 2.0 hours: embedding extraction and caching
- 1.0 hour: parent-local feature building and base scorer training
- 1.5 hours: validation calibration for all methods
- 1.5 hours: temporal future-child evaluation
- 1.0 hour: masked-child surrogate evaluation
- 0.75 hour: transfer diagnostics and plots
- 0.25 hour: buffer

This is much more plausible on one RTX A6000 than the previous design.

## Success Criteria

The proposal is successful if most of the following hold:

1. On the temporal future-child benchmark, MCC-EC reduces catastrophic overspecialization relative to both forced prediction and the HSC-style baseline at matched coverage.
2. On the masked-child benchmark, MCC-EC improves correct-safe-prefix rate over energy and OpenMax baselines.
3. Parent-level masked-child gains correlate positively with future-child gains, and that correlation is at least as strong as for generic hierarchical selective baselines.
4. Parent-level transfer diagnostics identify at least one interpretable regime where the surrogate is reliably helpful.
5. The full study completes within the stated 8-hour budget.

The main hypothesis is refuted if masked-child calibration does not beat generic stopping rules on future-child events and its synthetic gains fail to transfer at the parent level. In that case the paper should be reframed explicitly as a leakage-controlled benchmark study of surrogate calibration limits.

## References

1. Samet Ozdilek, A., Atakan, A., Ozsari, G., Acar, A., Atalay, M. V., Dogan, T., and Rifaioğlu, A. S. ProFAB: open protein functional annotation benchmark. *Briefings in Bioinformatics*, 24(2):bbac627, 2023.
2. Davoudi, S., Henry, C. S., Miller, C. S., and Banaei-Kashani, F. EC-Bench: a benchmark for enzyme commission number prediction. *Bioinformatics Advances*, 6(1), 2026.
3. Huang, Y., Lin, Y., Lan, W., Huang, C., and Zhong, C. GloEC: a hierarchical-aware global model for predicting enzyme function. *Briefings in Bioinformatics*, 25(5), 2024.
4. Yim, S., Hwang, D., Kim, K., and Han, S. Hierarchical Contrastive Learning for Enzyme Function Prediction. *ICML Workshop on Machine Learning for Life and Material Sciences*, 2024.
5. Dumontet, L., Han, S.-R., Lee, J. H., Oh, T.-J., and Kang, M. Trustworthy prediction of enzyme commission numbers using a hierarchical interpretable transformer. *Nature Communications*, 17(1), 2026.
6. Goren, S., Galil, I., and El-Yaniv, R. Hierarchical Selective Classification. *Advances in Neural Information Processing Systems 37*, 2024.
7. Boger, R. S., Chithrananda, S., Angelopoulos, A. N., Yoon, P. H., Jordan, M. I., and Doudna, J. A. Functional protein mining with conformal guarantees. *Nature Communications*, 16(1), 2025.
8. Zhang, Z., Lu, J., Chenthamarakshan, V., Lozano, A., Das, P., and Tang, J. ProtIR: Iterative Refinement between Retrievers and Predictors for Protein Function Annotation. *arXiv preprint arXiv:2402.07955*, 2024.
9. Yang, Y., Jerger, A., Feng, S., Wang, Z., Brasfield, C., Cheung, M. S., Zucker, J., and Guan, Q. Improved enzyme functional annotation prediction using contrastive learning with structural inference. *Communications Biology*, 7(1), 2024.
10. de Crécy-Lagard, V., Dias, R., Sexson, N., Friedberg, I., Yuan, Y., and Swairjo, M. A. Limitations of current machine learning models in predicting enzymatic functions for uncharacterized proteins. *G3: Genes, Genomes, Genetics*, 15(10), 2025.
11. Ma, X., Joshi, P., Friedberg, I., and Li, Q. How Not to be Seen: Predicting Unseen Enzyme Functions using Contrastive Learning. *bioRxiv preprint*, created February 26, 2026.
