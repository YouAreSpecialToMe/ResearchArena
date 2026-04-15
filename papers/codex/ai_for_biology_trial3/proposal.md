# SPARE-Gain: A Low-Compute Benchmark of Baseline-Relative Routing for Unseen-Perturbation Pseudobulk Prediction

## Introduction
Recent perturbation-response papers have made broad "better model" claims harder to defend. Systema showed that common metrics can reward systematic perturbed-versus-control variation rather than perturbation-specific generalization, while multiple comparison studies found that simple baselines remain competitive on held-out perturbations. In parallel, the modeling space is already crowded: scGen, CPA, GEARS, AttentionPert, BioDSNN, MechPert, PerturbNet, and PRESCRIBE collectively cover most obvious representation-learning directions for perturbation prediction.

That landscape suggests a narrower and more credible contribution. This project is not positioned as a new perturbation-prediction method family. It is a **benchmarked low-compute routing study** asking a specific question: when a frozen cheap pseudobulk baseline is available, can we decide when to apply a lightweight residual correction more reliably than standard uncertainty filtering or non-conformal gain prediction?

The key methodological hypothesis is intentionally falsifiable. If a conformal gain gate does not beat matched-acceptance uncertainty thresholds, classifier gates, and non-conformal gain regression, then the paper should be presented as a negative result showing that standard uncertainty filtering is sufficient. That is still publishable as a benchmarked systems result if the evaluation is careful and biologically interpretable.

## Proposed Approach
### Problem setup
For perturbation descriptor \(x\), let \(y(x)\) be the true pseudobulk response, \(\hat y_0(x)\) the frozen baseline prediction, and \(\hat y_1(x)\) the baseline plus residual correction. The routing task is to decide whether to output \(\hat y_0(x)\) or \(\hat y_1(x)\) before seeing the held-out response.

The **primary gain definition** is deliberately simple:

\[
g_{\mathrm{RMSE}}(x) = \mathrm{RMSE}(\hat y_0(x), y(x)) - \mathrm{RMSE}(\hat y_1(x), y(x)).
\]

Positive gain means the correction helped. Pearson-delta gain, DE-gene RMSE gain, and pathway-level gains are secondary analyses, not the primary target.

### Base predictors
The study uses only low-compute components that fit the stated hardware budget.

1. **Frozen cheap baseline**
   Train two inexpensive pseudobulk predictors on training perturbations only:
   - an additive or nearest-effect baseline built from controls and seen perturbation averages;
   - a descriptor-to-expression ridge regressor using leakage-free perturbation features.

   Validation selects the stronger one once. That model is frozen and becomes the reference baseline for every router.

2. **Lightweight residual corrector**
   Train a small residual model on a training-defined responsive panel of at most 256 genes:

   \[
   r(x) = y(x) - \hat y_0(x).
   \]

   Inputs are restricted to pre-response information:
   - frozen baseline prediction on the panel;
   - control pseudobulk on the panel;
   - perturbation descriptors built from hashed target identity, Norman pair-composition features, and simple co-target graph statistics;
   - retrieval-derived prototype residuals from training perturbations only;
   - novelty features such as neighbor distance, density, and cross-view disagreement.

   The residual model is intentionally modest: a shallow MLP or small gradient-boosted regressor. No novelty is claimed here. In the executed runs, curated GO and pathway descriptors were not restored locally, so the feature scope is narrower than originally proposed and the biological claims must stay aligned with that train-derived feature set.

### Routing policies to benchmark
The contribution lives in the router comparison, not in the predictor architecture.

1. **Always baseline**
   Never apply correction.

2. **Always correct**
   Always apply the residual model.

3. **Classifier gate**
   Predict \(\mathbb{1}[g_{\mathrm{RMSE}}(x) > 0]\) from routing features.

4. **Uncertainty-threshold gate**
   Accept correction when predictive variance, ensemble disagreement, or conformal interval width is below threshold.

5. **Non-conformal gain regressor**
   Predict \(E[g_{\mathrm{RMSE}}(x)\mid z]\) and accept when the predicted mean is positive.

6. **Conformal gain lower-bound gate**
   Fit a lower-quantile regressor \(q_\alpha(z)\) for gain and conformally calibrate it on held-out perturbations to obtain a lower bound \(\tilde q_\alpha(z)\). Apply correction only when

   \[
   \tilde q_\alpha(z) > 0.
   \]

### Why baseline-relative gain certification is different
The paper should be precise here. Standard selective regression and reject-option methods abstain when a single predictor looks unreliable. SPARE-Gain asks a different decision question: **is there enough evidence that switching away from a named frozen baseline will improve perturbation-level error?**

That difference matters because a correction can be high-uncertainty yet still beneficial relative to baseline, or low-uncertainty yet not worth deploying if its expected improvement is negligible. The benchmark therefore studies certification of **baseline-relative improvement**, not just confidence in one model.

## Related Work
### Perturbation prediction
- **scGen** introduced latent perturbation arithmetic for single-cell response prediction.
- **CPA** modeled compositional perturbation effects across drugs, doses, and cell contexts.
- **GEARS**, **AttentionPert**, **BioDSNN**, and **MechPert** pushed unseen genetic perturbation prediction with stronger architectural priors.
- **PerturbNet** modeled full single-cell response distributions across chemical and genetic perturbations.
- **PRESCRIBE** is the closest biology paper in spirit because it studies uncertainty-aware filtering of single-cell perturbation predictions.

These papers motivate the task but do not isolate the low-compute routing question over a frozen baseline. SPARE-Gain only claims a contribution if routing policies can be benchmarked in a way that uncertainty filtering alone cannot match.

### Benchmark and evaluation framing
- **Systema** motivates hard-subset analysis and skepticism toward aggregate metrics dominated by systematic variation.
- Recent comparison studies reporting strong simple baselines justify the frozen-baseline-first design.

### Methodological work outside biology
- **Regression with reject option and application to kNN** formalizes abstention in regression under fixed rejection rate.
- **SelectiveNet** studies learned reject-option models with integrated selective prediction.
- **Conformalized Selective Regression** is particularly relevant because it applies conformal prediction directly to selective regression and improves coverage-quality tradeoffs.
- **Learn then Test** and **Conformal Risk Control** show how calibration can target explicit risk constraints rather than raw confidence scores.
- **Learning to defer** papers provide the closest routing analogy: the system decides whether to use one decision-maker or defer to another.

Relative to this literature, SPARE-Gain is narrower than a general theory paper and less ambitious than a new perturbation backbone. Its novel angle is only the benchmarked question of whether **conformal certification of baseline-relative gain** yields better deployment decisions than uncertainty thresholds or non-conformal routing under the same compute budget.

## Experiments
### Datasets
Use public genetic perturbation datasets already common in unseen-perturbation evaluation:
- Adamson Perturb-seq
- Replogle K562 perturbations
- Norman combinatorial perturbations

All analyses are pseudobulk-first. The proposal explicitly acknowledges that pseudobulk discards cell-state heterogeneity, so scientific claims must be limited to aggregate response prediction and routing.

### Splits and calibration
Splits are defined at the perturbation identity level, never at the cell level. For each dataset:
- `train`: fit baseline and residual models
- `route-dev`: fit router baselines and gain models
- `calibration`: conformal calibration only
- `test`: final evaluation

The paper will report the **exact calibration perturbation count per dataset** in the main table. If any dataset yields fewer than 20 calibration perturbations after the canonical split, conformal claims on that dataset will be labeled underpowered and supplemented with a cross-fit sensitivity analysis rather than treated as decisive evidence.

In practice this caveat matters for Adamson and Replogle under the canonical split, so any apparent conformal advantage there should be treated as suggestive rather than decisive.

### Primary endpoint
The primary endpoint is:
- **mean perturbation-level RMSE gain on accepted perturbations at fixed acceptance rate**

The main operating point will be 40% acceptance, with 20% and 60% as sensitivity points. This is simpler and less noisy than a composite gain target. Selective-risk curves over the full acceptance range are secondary summaries.

### Secondary metrics
- all-gene RMSE
- top-DE-gene RMSE
- Pearson delta
- accepted-set mean gain and median gain
- worst-quartile accepted-set gain by descriptor novelty
- calibration of predicted gain versus observed gain
- all-gene conformal coverage and interval width

### Biological interpretation metrics
Accepted-set gains should mean more than a small numeric RMSE change. For each accepted perturbation, report:
- recovery of top responsive genes by overlap or rank correlation
- sign accuracy on the largest-magnitude differential-expression genes
- pathway-level response agreement using training-derived response-module score correlation

If routing improves RMSE but not these biological summaries, the paper should say that the practical biological significance is limited.

### Core falsification tests
1. **Matched-acceptance comparison**
   Compare the conformal gate against classifier, uncertainty-threshold, and non-conformal gain gates at the same acceptance rates.

2. **Hard-subset evaluation**
   Report results by descriptor novelty quartile when the frozen training-only bins are populated and, for Norman, by `0-seen`, `1-seen`, and `2-seen` combinations. If held-out perturbations collapse into a single novelty bin, state explicitly that the quartile analysis is not informative.

3. **Collapse-to-uncertainty test**
   Measure whether the conformal gate's accepted set is materially different from simple uncertainty filtering. If not, the method claim collapses.

4. **Gain calibration test**
   Check whether predicted lower bounds track realized baseline-relative gain, not just model uncertainty.

5. **Biological value test**
   Test whether accepted perturbations show improved pathway and marker recovery, not merely score improvements.

### Sensitivity analyses
The proposal pre-specifies three sensitivities requested by the feedback:
- **conformal split sensitivity**: vary the route-dev/calibration partition and compare one fixed split versus cross-fit calibration
- **quantile level sensitivity**: vary the lower-quantile target, e.g. \(\alpha \in \{0.1, 0.2, 0.3\}\)
- **gain definition sensitivity**: repeat routing with all-gene RMSE gain, top-DE RMSE gain, and Pearson-delta gain

These analyses are not optional extras. They determine whether the method is robust or merely tuned to one gain definition.

### Compute feasibility
The study is scoped to the available hardware:
- 1x RTX A6000 48GB
- 60GB RAM
- 4 CPU cores
- roughly 8 total experiment hours

To fit that budget:
- preprocess each dataset once
- use one canonical split per dataset for the full benchmark
- run three seeds only for the frozen best baseline and the final conformal router
- keep the responsive panel at 256 genes
- avoid deep end-to-end reproductions as core evidence

External deep models such as GEARS, CPA, AttentionPert, BioDSNN, or PerturbNet may be cited as contextual prior work, but the main claim must stand without reproducing them.

## Success Criteria
### Evidence supporting the hypothesis
The benchmark supports the gain-certification hypothesis if:
1. the conformal gate beats uncertainty-threshold and non-conformal gain baselines at matched acceptance rates on the primary endpoint in at least 2 of 3 datasets;
2. accepted-set gains remain positive on the hardest novelty quartile or Norman `0-seen` subset in at least 2 datasets;
3. accepted perturbations also improve at least one biological interpretation metric, not just RMSE;
4. results are directionally stable across conformal split, quantile, and gain-definition sensitivities.

### Evidence refuting the hypothesis
The hypothesis is refuted if:
1. uncertainty-threshold or non-conformal gain routing matches the conformal gate at the same acceptance rates;
2. accepted-set gains disappear after hard-subset stratification;
3. biological interpretation metrics do not improve despite score changes;
4. calibration is too data-hungry or too conservative to be useful under realistic split sizes.

If refuted, the paper should be reframed as a negative benchmark result: low-compute perturbation routing collapses to standard uncertainty filtering.

## References
1. Lotfollahi, M., Wolf, F. A., and Theis, F. J. scGen predicts single-cell perturbation responses. *Nature Methods*, 2019.
2. Lotfollahi, M. et al. Predicting cellular responses to complex perturbations in high-throughput screens. *Nature Biotechnology*, 2023.
3. Roohani, Y., Huang, K., and Leskovec, J. GEARS: Predicting transcriptional outcomes of novel multigene perturbations. *bioRxiv/2024 release*.
4. Bai, D. et al. AttentionPert: accurately modeling multiplexed genetic perturbations with multi-scale effects. *bioRxiv*, 2024.
5. Tan, Y. et al. BioDSNN: a dual-stream neural network with hybrid biological knowledge integration for multi-gene perturbation response prediction. *bioRxiv*, 2024.
6. Shi, Y. et al. MechPert: Mechanistic Consensus as an Inductive Bias for Unseen Perturbation Prediction. *bioRxiv*, 2025.
7. Yu, H. et al. PerturbNet predicts single-cell responses to unseen chemical and genetic perturbations. *bioRxiv*, 2025.
8. Cheng, J. et al. PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation. *arXiv*, 2025.
9. Vinas Torne, R. et al. Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation. *bioRxiv*, 2025.
10. Sokol, A., Moniz, N., and Chawla, N. Conformalized Selective Regression. *arXiv:2402.16300*, 2024.
11. Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., and Schuster, T. Conformal Risk Control. *ICLR*, 2024.
12. Angelopoulos, A. N., Bates, S., Candes, E. J., Jordan, M. I., and Lei, L. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. *arXiv:2110.01052*, 2022.
13. Denis, C., Hebiri, M., and Zaoui, A. Regression with reject option and application to kNN. *NeurIPS*, 2020.
14. Mozannar, H. and Sontag, D. Consistent Estimators for Learning to Defer to an Expert. *ICML*, 2020.
15. Geifman, Y. and El-Yaniv, R. SelectiveNet: A Deep Neural Network with an Integrated Reject Option. *ICML*, 2019.
