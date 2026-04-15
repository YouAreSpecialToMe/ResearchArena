# Does Systema-Style Perturbed-Reference Residualization Help as a Training Target for Unseen Single-Cell Perturbation Prediction? A Pre-Registered Benchmark Note

## Introduction

Recent benchmark papers have made two points that sharply narrow what is worth claiming in single-cell perturbation prediction. First, **Systema** argues that standard evaluation can over-credit models that recover broad systematic differences between perturbed and control cells rather than perturbation-specific effects. Second, **Kernfeld et al.** and later benchmark studies show that train-mean, train-median, and other simple predictors are often unexpectedly hard to beat. Under that literature, another small predictor is not itself a convincing contribution.

This project is therefore framed explicitly as a **benchmark note**, not a new method paper. The single pre-registered question is: **if evaluation should discount shared perturbation structure using a perturbed reference, is that same perturbed-reference view also a better training target for unseen-perturbation prediction?**

The intended contribution is modest but useful. We are not claiming novelty in retrieval, graph features, or decoder design. We are testing whether **Systema-style perturbed-reference residualization is useful as a training target** under strong simple baselines, fixed lightweight features, and an eight-hour compute budget that forces methodological discipline.

## Proposed Approach

### Benchmark question and target definition

For dataset \(d\), split \(s\), and held-out perturbation \(p\), let \(x_p \in \mathbb{R}^G\) be the perturbation centroid and let
\[
\mu^{\mathrm{train}}_{\mathrm{pert}}(d,s)=\frac{1}{|T_{\mathrm{train}}(d,s)|}\sum_{q \in T_{\mathrm{train}}(d,s)} x_q
\]
be the train-only mean centroid over training perturbations.

The **pre-registered residual target** is
\[
r_{\mathrm{sys}}(p;d,s)=x_p-\mu^{\mathrm{train}}_{\mathrm{pert}}(d,s).
\]

Residualized models predict \(\hat{r}_{\mathrm{sys}}\) and reconstruct
\[
\hat{x}_p=\mu^{\mathrm{train}}_{\mathrm{pert}}(d,s)+\hat{r}_{\mathrm{sys}}(p;d,s).
\]

The core comparison is not retrieval versus no retrieval. It is:

- full-target training on \(x_p\)
- versus Systema-style residual-target training on \(r_{\mathrm{sys}}\)

Retrieval is a bounded secondary question evaluated only after the residual target is fixed.

### Exact candidate model set

To keep the benchmark internally consistent, the candidate set is fixed in advance and will not be expanded in the experiment stage:

1. `Train Perturbed Mean`
2. `Train Perturbed Median`
3. `Non-residualized Ridge`
4. `Residualized Ridge`
5. `Residualized PLS`
6. `Residualized Linear Embedding`
7. `Retrieval-only Residual kNN`
8. `ReSRP-Linear`
9. `ReSRP-MLP`

`Residualized MLP` is **not** part of the primary baseline ladder. `Residualized Random Forest` is also excluded to protect the runtime budget and keep the benchmark focused on the target-definition question rather than on broad baseline fishing.

The **best residualized non-retrieval baseline** is selected only from:

- `Residualized Ridge`
- `Residualized PLS`
- `Residualized Linear Embedding`

This fixed set is the same in the proposal, `idea.json`, and the next-stage execution plan.

### Why use a fixed STRING perturbation descriptor

The perturbation descriptor is deliberately simple: a **STRING v12 high-confidence target-gene embedding** computed once from the training gene universe. The rationale is not that STRING is optimal. The rationale is that the benchmark needs a descriptor with three properties:

1. It is available for unseen target genes without retraining a large encoder.
2. It encodes biologically meaningful proximity rather than arbitrary IDs, which is necessary for unseen-perturbation generalization.
3. It is cheap enough to support repeated seeded runs and ablations within eight hours.

This choice is aligned with papers such as **TxPert**, which show that biological graph structure can be useful for out-of-distribution perturbation prediction, while avoiding the compute and implementation variance of large pretrained encoders.

### Feature-sufficiency contingency

A negative residualization result would be hard to interpret if STRING features were too weak. To separate those possibilities, the proposal pre-registers a **feature-sufficiency contingency analysis**:

- report STRING coverage diagnostics for each dataset: fraction of held-out perturbations with a nonzero STRING embedding and fraction with at least one connected training neighbor
- compare `Residualized Ridge` using true STRING embeddings against the same model using:
  - randomly permuted gene embeddings
  - a degree-only scalar feature baseline

If true STRING features do not beat these degraded descriptors, the paper will treat any null result about residualization as **inconclusive due to weak perturbation features**, not as evidence against residualization itself.

### Retrieval framing

Retrieval is included only as a secondary add-on because novelty here is limited. `Retrieval-only Residual kNN`, `ReSRP-Linear`, and `ReSRP-MLP` test whether neighbor residual summaries improve over the strongest residualized non-retrieval baseline once the target is fixed. Any retrieval claim must clear a higher bar than the residualization claim.

## Related Work

**Viñas Torné et al., Systema (2025)** is the direct conceptual precursor. Systema argues for perturbed-reference evaluation to isolate perturbation-specific biology from systematic variation. This proposal does not claim a new biological principle; it asks the narrower follow-up question of whether the same perturbed-reference idea is useful at **training time**.

**Kernfeld et al., A comparison of computational methods for expression forecasting (2025)** motivates the benchmark style. Their central result is cautionary: simple baselines remain competitive, so stronger claims require tighter benchmark design rather than another lightly modified model.

**Wei et al., Benchmarking algorithms for generalizable single-cell perturbation response prediction (2026)** provides processed benchmark assets and standardized unseen-perturbation evaluation settings, making a controlled lightweight study feasible.

**Csendes et al., Benchmarking foundation cell models for post-perturbation RNA-seq prediction (2025)** further supports the benchmark framing by showing that larger pretrained models do not reliably dominate trivial baselines.

**PT-RAG (2026)** is relevant mainly to limit novelty claims. Retrieval-augmented perturbation prediction is already active, so retrieval is not presented as a new idea here.

**TxPert (2025)** and **C3TL (2026)** motivate using compact biological priors and efficient architectures. The present project differs by studying **target definition**, not by proposing a new transfer or retrieval mechanism.

## Experiments

### Datasets

Primary datasets:

- Adamson
- Replogle K562 Essential

No third dataset is in scope for the required study. Optional expansions are intentionally removed from the main proposal to preserve feasibility.

### Splits and seeds

- Use the processed perturbation-generalization assets from Wei et al.
- Use exactly three seeds: `{11, 23, 37}`.
- Compute HVG selection, centroid construction, train-only references, PCA bases, and retrieval indices using training data only for each dataset-seed pair.

### Output space

- Predict perturbation centroids, not cell-level distributions.
- Restrict to at most 2,000 HVGs with forced inclusion of perturbed target genes.
- Fit a 64-dimensional PCA basis on training residuals only.
- Train all learned residual predictors in this PCA space.

### Primary and secondary endpoints

The study uses directly interpretable primary comparisons rather than a rank-only endpoint.

**Primary endpoint**

- difference in **perturbed-reference Pearson correlation** between a residualized model and its comparator on held-out perturbations

**Co-primary guardrail**

- difference in reconstructed-centroid **RMSE**

These are reported for:

1. the residualization question:
   `Residualized Ridge` versus `Non-residualized Ridge`
2. the retrieval question:
   best `ReSRP` variant versus the selected best residualized non-retrieval baseline

**Secondary support metrics**

- nearest-centroid top-1 accuracy
- median rank of the true perturbation
- composite rank across the four metrics above
- perturbed-reference Pearson on top 100 high-variance genes
- runtime and peak memory

The composite rank is retained only as a **secondary descriptive summary**, not as the primary inferential target.

### Statistical analysis

Inference is done at the dataset level, not via fixed-effects meta-analysis across seeds.

For each dataset:

- compute each primary metric on each seed separately
- bootstrap held-out perturbations within each seed
- average the bootstrap metric difference across the three seeds inside each bootstrap draw
- report the resulting 95% confidence interval for the seed-averaged difference

The conclusions must hold on **both primary datasets**. This avoids treating seed variation as if it were a study-level meta-analysis problem.

### Minimal ablation set

To fit the eight-hour budget, the required ablations are reduced to the ones that bear directly on interpretation:

1. **Full-target versus residual-target training** on the same compact MLP architecture, to test whether residualization helps when capacity is held fixed.
2. **No retrieval versus retrieval** for the retrieval MLP, to test whether neighbor residual summaries add signal.
3. **Feature-sufficiency contingency** using permuted and degree-only descriptors with `Residualized Ridge`.

One secondary sensitivity analysis is retained only if time remains:

- control-referenced residual target instead of perturbed-reference residual target

### Feasibility under the eight-hour budget

The proposal is intentionally lightweight:

- two datasets only
- three seeds only
- nine main models total
- no Random Forest
- no GEARS in the required path
- no third dataset
- no broad hyperparameter sweeps beyond small fixed grids
- sub-100k-parameter MLPs only

The expected runtime budget is:

- preprocessing and cached feature construction: about 60 minutes
- baseline ladder across two datasets and three seeds: about 90 minutes
- `ReSRP-Linear` and `ReSRP-MLP` runs: about 100 minutes
- required ablations: about 90 minutes
- bootstrap aggregation, tables, and figures: about 40 minutes
- contingency buffer: about 60 minutes

Total expected runtime: about **7.3 hours** on one RTX A6000 plus 4 CPU cores.

## Success Criteria

The benchmark supports the **residualization claim** only if:

1. `Residualized Ridge` improves perturbed-reference Pearson over `Non-residualized Ridge` on both primary datasets.
2. The same comparison does not worsen RMSE materially; the seed-averaged RMSE difference must be non-positive or negligible.
3. At least one residualized non-retrieval baseline also beats `Train Perturbed Mean` on perturbed-reference Pearson on both datasets.

The benchmark supports the stronger **retrieval add-on claim** only if:

1. the best `ReSRP` variant beats the selected best residualized non-retrieval baseline on perturbed-reference Pearson on both datasets
2. the same model does not worsen RMSE
3. `Retrieval-only Residual kNN` does not match the winning `ReSRP` variant

Interpretation is pre-registered as follows:

- if residualization helps but retrieval does not, the paper is a **negative result on retrieval** but a positive benchmark result on the training-target question
- if retrieval appears neutral and STRING features fail the feature-sufficiency contingency, the paper reports the residualization test as **inconclusive**
- if residualization fails despite informative STRING features, the main hypothesis is refuted

## References

Csendes, G., Sanz, G., Szalay, K. Z. & Szalai, B. Benchmarking foundation cell models for post-perturbation RNA-seq prediction. *BMC Genomics* (2025). https://doi.org/10.1186/s12864-025-11600-2

Di Francesco, A. G., Rubbi, A. & Liò, P. Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation. *arXiv* 2603.07233 (2026). https://arxiv.org/abs/2603.07233

Kernfeld, E., Yang, Y., Weinstock, J. S., Battle, A. & Cahan, P. A comparison of computational methods for expression forecasting. *Genome Biology* 26, 388 (2025). https://doi.org/10.1186/s13059-025-03840-y

Scholkemper, M. & Mukherjee, S. Causal Cellular Context Transfer Learning (C3TL): An Efficient Architecture for Prediction of Unseen Perturbation Effects. *arXiv* 2603.13051 (2026). https://arxiv.org/abs/2603.13051

Viñas Torné, R., Wiatrak, M., Piran, Z., Fan, S., Jiang, L., Teichmann, S. A., Nitzan, M. & Brbić, M. Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation. *Nature Biotechnology* (2025). https://doi.org/10.1038/s41587-025-02777-8

Wei, Z. et al. Benchmarking algorithms for generalizable single-cell perturbation response prediction. *Nature Methods* (2026). https://doi.org/10.1038/s41592-025-02980-0

Wenkel, F. et al. TxPert: Leveraging Biochemical Relationships for Out-of-Distribution Transcriptomic Perturbation Prediction. *arXiv* 2505.14919 (2025). https://arxiv.org/abs/2505.14919
