# Benchmarking Weakly Supervised Factor-Localized SAEs on Frozen Vision Features

## Introduction

Recent sparse autoencoder work already covers most of the relevant methodological ingredients for structure-aware dictionary learning: transformation-aware dictionaries, explicitly equivariant SAEs, and stability-oriented variants such as Archetypal SAE. The defensible contribution here is therefore not a new SAE family. It is a benchmark and protocol paper that isolates one narrower question:

**When exact single-factor counterfactual pairs are available, does weak supervision through a fixed latent partition produce cleaner factor-local responses than recent matched SAE baselines on frozen vision features?**

The paper is framed as a pre-registered evaluation study. Its contributions are:

1. A leakage-free protocol for testing factor-local counterfactual responses in SAE latents.
2. A fair evaluation recipe for both partitioned and unpartitioned baselines, including invariant and residual pseudo-blocks for non-partitioned models.
3. A nuisance-pair construction that is explicitly factor-preserving for dSprites and Shapes3D.
4. A small backbone-sensitivity check so conclusions are not reducible to a single frozen encoder.

The strongest claim is modest by design: under a controlled benchmark, weak supervision plus fixed partitioning can be tested fairly against recent SAE baselines, and may or may not improve factor localization in frozen pretrained features.

**Primary hypothesis.** On dataset and backbone combinations where the frozen features linearly retain the benchmark factors, fixed-partition methods will achieve higher held-out target-factor change concentration than matched unpartitioned baselines. Any extra benefit of adding nuisance-invariance regularization on top of partitioning should be smaller and is treated as a secondary question.

## Proposed Approach

### Benchmark framing

The proposal contributes a protocol, not a new representation-learning paradigm. `FB-OSAE` is included as one benchmark condition among several closely related baselines. Positive results would support a claim about benchmarking weak supervision plus fixed partitioning under a pre-registered test, not a broad claim that `FB-OSAE` is a new state-of-the-art SAE method.

### Representation setting

Each image `x` is mapped to a frozen pooled feature vector `h(x) in R^d`.

Primary backbone:

- `DINOv2-small`

Secondary sensitivity backbone:

- `OpenCLIP ViT-B/32` on dSprites only, with no retuning

The second backbone is intentionally small. Its purpose is not to establish cross-backbone generalization exhaustively, but to test whether the main ranking is entirely an artifact of one frozen encoder.

### Datasets and factors

**dSprites**

- factors: shape, scale, rotation, x-position, y-position
- exact single-factor counterfactual pairs can be generated from the labeled factor grid

**Shapes3D**

- factors: object hue, wall hue, floor hue, scale, shape, orientation
- exact single-factor counterfactual pairs can be generated from the labeled factor grid

Data are split by factor tuples before pair construction. All training, validation, and test pairs are generated strictly within split, preventing overlap leakage through shared tuples.

### Factor-preserving nuisance pairs

The earlier proposal used spatial and photometric transforms that could alter labeled benchmark factors. That is removed.

Nuisance pairs are now defined as two independently corrupted views of the **same canonical image**:

\[
t_1(x), t_2(x), \quad \text{with identical ground-truth factor tuple as } x.
\]

Each view samples one corruption pipeline from a pre-registered set of sensor-style perturbations:

- additive Gaussian noise with `sigma in [0.01, 0.05]`
- Gaussian blur with kernel sigma in `[0.4, 1.0]`
- JPEG compression with quality in `[50, 95]`
- random dead-pixel masking of `0.5%` to `2%` of pixels

These corruptions do not change the underlying labeled tuple on either dataset. They are therefore compatible with factor-local evaluation for all retained factors. No translation, crop-resize, rotation, or color jitter is used in the nuisance objective.

### Model family

All methods learn a shallow SAE over cached frozen features:

\[
\hat h(x) = D(E(h(x))).
\]

Shared settings across methods:

- same frozen features
- same optimizer and number of steps
- same total latent width
- same TopK sparsity target
- same train/validation/test tuples and pair-construction code

The latent width is `4d`. Partitioned models split the code into:

- invariant block: `20%`
- factor blocks: `60%` total, divided equally across known factors
- residual block: `20%`

This gives one invariant block, one block per factor, and one residual block.

### Methods compared

Five matched-capacity methods are trained.

1. `TopK-SAE`: plain sparse autoencoder.
2. `RA-SAE`: relaxed-archetypal sparse autoencoder, included as the strongest structured baseline.
3. `Orbit-SAE`: unpartitioned SAE with nuisance-invariance regularization on the full code.
4. `FB-SAE`: fixed partition plus factor-supervised counterfactual loss, but no nuisance regularization.
5. `FB-OSAE`: fixed partition plus factor-supervised counterfactual loss plus nuisance regularization on the invariant block.

The key comparison is not whether `FB-OSAE` is “novel,” but whether fixed partitioning and weak supervision outperform recent unpartitioned alternatives under a common held-out localization test.

### Partitioned objective

For partitioned models,

\[
z = [z_{\mathrm{inv}}, z_1, \dots, z_F, z_{\mathrm{res}}].
\]

The loss is

\[
\mathcal{L} =
\mathcal{L}_{\mathrm{rec}}
+ \lambda_{\mathrm{sp}} \mathcal{L}_{\mathrm{TopK}}
+ \lambda_{\mathrm{nuis}} \mathcal{L}_{\mathrm{inv\_nuis}}
+ \lambda_{\mathrm{cf}} \mathcal{L}_{\mathrm{cf\_block}}.
\]

`L_rec` is feature-space mean-squared error. `L_TopK` is the shared sparsity mechanism.

For a nuisance pair `(t_1(x), t_2(x))`,

\[
\mathcal{L}_{\mathrm{inv\_nuis}}
= \|z_{\mathrm{inv}}(t_1(x)) - z_{\mathrm{inv}}(t_2(x))\|_2^2.
\]

For a single-factor counterfactual pair `(x, x^{(f)})`, define normalized block change

\[
\Delta_b(x, x^{(f)}) = \frac{1}{|b|}\sum_{u \in b}|z_u(x) - z_u(x^{(f)})|.
\]

Then

\[
\mathcal{L}_{\mathrm{cf\_block}}
= \Delta_{\mathrm{inv}}
+ \frac{1}{F-1}\sum_{f' \neq f}\Delta_{f'}
+ \Delta_{\mathrm{res}}
+ \max(0, m - \Delta_f).
\]

This encourages the changed factor to dominate its own block while discouraging spillover into invariant, non-target, and residual blocks.

`FB-SAE` sets `lambda_nuis = 0`. `FB-OSAE` uses the full objective.

### Unpartitioned baselines and pseudo-block evaluation

Evaluation for unpartitioned baselines is fully specified in advance.

Let `W_f` be the factor-block width and `W_inv` the invariant-block width used by the partitioned models for the same dataset and backbone.

For each unpartitioned model, using **training split pairs only**:

1. Compute each unit’s factor-selectivity score

\[
S_{u,f} =
\mathbb{E}_{(x, x^{(f)})}\left[
\frac{|z_u(x) - z_u(x^{(f)})|}{\sum_j |z_j(x) - z_j(x^{(f)})| + \epsilon}
\right].
\]

2. Build disjoint factor pseudo-blocks greedily.
   Process factors in descending order of validation probe strength for that dataset-backbone pair. For factor `f`, assign the top unassigned `W_f` units ranked by `S_{u,f}` to pseudo-block `B_f`.

3. Define a nuisance-response score on training nuisance pairs

\[
N_u =
\mathbb{E}_{(t_1(x), t_2(x))}
\left[
\frac{|z_u(t_1(x)) - z_u(t_2(x))|}{\sum_j |z_j(t_1(x)) - z_j(t_2(x))| + \epsilon}
\right].
\]

4. Among the remaining units, assign the `W_inv` units with the smallest `N_u` to the invariant pseudo-block `B_inv`.

5. Define the residual pseudo-block `B_res` as **all remaining units** after factor and invariant assignment.

These pseudo-blocks are frozen before validation and test evaluation. No test information is used for block construction.

This resolves the denominator issue in TFCC for unpartitioned baselines: every model has exactly one invariant block, `F` factor blocks, and one residual block at evaluation time.

### Metrics

**Primary metric: Target-Factor Change Concentration (TFCC)**

\[
\mathrm{TFCC}(x, x^{(f)}) =
\frac{\Delta_f}{\Delta_{\mathrm{inv}} + \sum_{k=1}^{F}\Delta_k + \Delta_{\mathrm{res}}}.
\]

TFCC is computed using native blocks for partitioned models and frozen pseudo-blocks for unpartitioned models.

**Secondary metric: Target-Block Accuracy (TBA)**

Fraction of single-factor pairs for which the target factor block has the largest normalized change.

**Nuisance leakage metrics**

- invariant-block nuisance change: mean `Delta_inv` on nuisance pairs for partitioned models
- pseudo-invariant nuisance change: mean `Delta_inv` on nuisance pairs for unpartitioned models

**Reconstruction and sparsity controls**

- feature-space MSE
- fraction of variance explained
- realized mean `L0`

### Backbone admissibility checks

The study explicitly separates backbone mismatch from SAE behavior.

Before any SAE training, train-split linear probes are fit from frozen features to each ground-truth factor, with validation and test evaluation on held-out tuples. Results are reported for every dataset-backbone pair.

Interpretation rule:

- aggregate localization claims are made only for factors that are recoverable by the frozen backbone with materially above-chance held-out probe performance
- weak-probe factors are reported separately as backbone limitations, not counted as evidence against the SAE design

This keeps a poor frozen representation from being misread as a failure of the latent partitioning protocol.

### Hyperparameter selection

Hyperparameters are chosen once on dSprites with `DINOv2-small`, then frozen for all other runs.

Pilot grid:

- `lambda_nuis in {0.1, 0.3}` for orbit-regularized models
- `lambda_cf in {0.3, 1.0}` for partitioned models
- RA-SAE relaxation coefficient in `{0.01, 0.05}`

Fixed settings:

- latent width `= 4d`
- TopK active units `= 32` for `d <= 512`, else `64`
- Adam learning rate `= 1e-3`
- batch size `= 2048`
- maximum epochs `= 20`
- early stopping patience `= 3`

Selection rule:

1. Discard settings whose validation variance explained is below `95%` of the best setting in that method family.
2. Rank remaining settings by validation TFCC.
3. Break ties by lower realized mean `L0`, then lower validation MSE.

No dataset-specific or backbone-specific retuning is allowed after the dSprites pilot.

## Related Work

### Transformation-aware and equivariant SAEs

Group Crosscoders and Equivariant Sparse Autoencoders already show that sparse dictionaries can be aligned to symmetry structure. This proposal does not compete on that axis. It uses their existence as the main reason to narrow the contribution from “new method” to “controlled benchmark.”

### Stability-oriented SAE baselines

Archetypal SAE argues that plain SAEs can be unstable and that more structured dictionaries improve reproducibility and interpretability. That makes `RA-SAE` a required baseline here; otherwise any apparent gain from fixed partitioning could just reflect comparison to an unnecessarily weak SAE.

### Disentanglement benchmarks with known factors

beta-VAE, Understanding Disentangling in beta-VAE, Eastwood and Williams, FactorVAE, and Locatello et al. establish why dSprites and Shapes3D are useful controlled testbeds and why claims about factor structure need careful evaluation. This proposal uses those datasets only as intervention-ready benchmarks, not as evidence of unsupervised factor discovery.

### Intervention-style interpretability benchmarks

RAVEL’s key lesson is methodological: interpretability methods should be compared with pre-registered intervention tests instead of qualitative cherry-picking. This proposal imports that principle into frozen vision features by making exact single-factor counterfactual pairs the central evaluation object.

## Experiments

### Main comparisons

Primary comparisons on `DINOv2-small`:

1. `FB-SAE` vs `TopK-SAE`, `RA-SAE`, and `Orbit-SAE`
2. `FB-OSAE` vs `TopK-SAE`, `RA-SAE`, and `Orbit-SAE`
3. `FB-OSAE` vs `FB-SAE`

The first two comparisons test the benchmark’s main question: does weak supervision plus fixed partitioning improve held-out localization? The third tests whether nuisance regularization adds anything beyond partitioning.

### Controls

**Permutation null**

- for `FB-SAE` and `FB-OSAE`, randomly permute factor-block identities at evaluation time
- TBA should collapse toward `1/F`

**Backbone sensitivity**

- rerun the five methods on dSprites with `OpenCLIP ViT-B/32`
- use the same hyperparameters chosen on `DINOv2-small`
- use one seed per method

This is not part of the primary success claim, but it tests whether the main conclusion is entirely backbone-specific.

## Concrete Run Matrix

**Pilot stage: 8 runs**

- dSprites
- `DINOv2-small`
- seed `11`
- 2 settings each for `RA-SAE`, `Orbit-SAE`, `FB-SAE`, `FB-OSAE`

**Primary benchmark stage: 30 runs**

- 2 datasets
- 5 methods
- 3 seeds
- `DINOv2-small`

**Secondary backbone sensitivity stage: 5 runs**

- dSprites
- `OpenCLIP ViT-B/32`
- 5 methods
- 1 seed

Total training runs: **43**

Permutation tests, pseudo-block assignment, probe fitting, and plotting are evaluation-only.

## Feasibility Under the 8-Hour Budget

The project remains feasible because it trains only shallow SAEs on cached frozen features.

Dataset caps:

- dSprites: `12,000` train, `2,000` validation, `2,000` test
- Shapes3D: `12,000` train, `2,000` validation, `2,000` test

Estimated wall-clock:

- `0.8h` for tuple splits, counterfactual-pair generation, and nuisance-pair generation
- `1.1h` to `1.4h` for frozen feature extraction and linear probes across the primary and secondary backbones
- `0.5h` to `0.8h` for the 8 pilot runs
- `3.0h` to `3.8h` for the 30 primary runs at roughly `6` to `7.5` minutes each
- `0.4h` to `0.7h` for the 5 secondary-backbone runs
- `0.8h` to `1.0h` for pseudo-block construction, held-out evaluation, permutation nulls, and tables
- `0.5h` contingency

Total estimated budget: `7.1h` to `8.0h`

If pilot timing exceeds budget, the fallback is to reduce both datasets uniformly to `10,000` train examples while keeping the same run matrix and protocol. The study does not expand to additional sweeps or additional datasets.

## Success Criteria

Because this is a benchmark paper, success is defined first at the protocol level and second at the method-comparison level.

**Protocol success**

1. The nuisance-pair construction is label-preserving for all evaluated factors.
2. The pseudo-block recipe yields fully specified invariant, factor, and residual blocks for every unpartitioned baseline without test leakage.
3. Probe tables make backbone-factor recoverability explicit before any SAE comparison.

**Empirical success**

1. On `DINOv2-small`, at least one partitioned method (`FB-SAE` or `FB-OSAE`) improves mean held-out TFCC over all three unpartitioned baselines by at least `0.05` absolute on both datasets for probe-admissible factors.
2. The same partitioned method improves TBA over `Orbit-SAE` by at least `5` percentage points on both datasets.
3. If `FB-OSAE` is claimed to help beyond partitioning, it must beat `FB-SAE` on TFCC or TBA on both datasets without reducing variance explained below `95%` of `FB-SAE`.
4. On the dSprites `OpenCLIP` sensitivity check, the ranking between partitioned and unpartitioned families should not fully reverse.
5. Under block-label permutation, partitioned-model TBA should fall near chance.

**Refutation conditions**

- partitioned methods do not beat unpartitioned baselines under the leakage-free protocol
- any apparent gain disappears once residual and invariant pseudo-blocks are included correctly
- `FB-OSAE` offers no measurable advantage over `FB-SAE`
- results depend entirely on a single backbone while reversing on the sensitivity check
- frozen features fail to expose the benchmark factors, making any localization claim uninterpretable

## References

Christopher P. Burgess, Irina Higgins, Arka Pal, Loic Matthey, Nick Watters, Guillaume Desjardins, and Alexander Lerchner. 2018. Understanding Disentangling in beta-VAE. arXiv.

Cian Eastwood and Christopher K. I. Williams. 2018. A Framework for the Quantitative Evaluation of Disentangled Representations. ICLR Workshop.

Ege Erdogan and Ana Lucic. 2025. Group Equivariance Meets Mechanistic Interpretability: Equivariant Sparse Autoencoders. NeurIPS Mechanistic Interpretability and UniReps Workshops.

Thomas Fel, Ekdeep Singh Lubana, Jacob S. Prince, Matthew Kowal, Victor Boutin, Isabel Papadimitriou, Binxu Wang, Martin Wattenberg, Demba Ba, and Talia Konkle. 2025. Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models. arXiv.

Liv Gorton. 2024. Group Crosscoders for Mechanistic Analysis of Symmetry. arXiv.

Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. 2017. beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR.

Jing Huang, Zhengxuan Wu, Christopher Potts, Mor Geva, and Atticus Geiger. 2024. RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations. ACL.

Hyunjik Kim and Andriy Mnih. 2018. Disentangling by Factorising. ICML.

Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schoelkopf, and Olivier Bachem. 2019. Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. ICML.

Maxime Oquab, Timothee Darcet, Theo Moutakanni, Hugo Vo, Marc Szafraniec, Marie-Anne Lachaux, et al. 2023. DINOv2: Learning Robust Visual Features without Supervision. arXiv.

Mateusz Pach, Shyamgopal Karthik, Quentin Bouniot, Serge Belongie, and Zeynep Akata. 2025. Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models. NeurIPS.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. ICML.
