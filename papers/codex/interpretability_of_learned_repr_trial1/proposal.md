# Do Shared Decoders Improve Prototype-Edit Reusability on Frozen CLIP Features?

## Introduction

Recent sparse-autoencoder work has made broad novelty claims in this area hard to defend. CLIP and other vision-language models already have SAE-based concept discovery and steering results; shared-latent sparse spaces across modalities also now exist. That makes this paper viable only as a tightly scoped empirical benchmark, not as a new method paper.

This proposal therefore asks one narrow question: under compute-matched training on the same frozen CLIP vision features, does a shared decoder improve reusable prototype edits compared with vanilla SAE and SSAE? "Reusable" is the key word. The goal is not to show that a model can reconstruct one paired difference, but that an attribute prototype estimated from training pairs transfers to unseen examples with higher target specificity and lower collateral drift.

The claim is limited to frozen-feature editability. Positive results would support a practical benchmark conclusion: sharing a dictionary between activation reconstruction and paired-difference reconstruction is a useful inductive bias for prototype reuse in this regime. Negative results are equally publishable because they would show that SSAE already captures the available benefit and the extra structure is unnecessary.

The study is sized for one RTX A6000 and an approximately 8-hour wall-clock budget. All models train on cached vectors. The proposal uses frozen CLIP ViT-B/32 pooled pre-projection vision features for the main benchmark and adds a lightweight frozen DINOv2-S/14 evaluation bridge so the main conclusion is not judged only inside CLIP geometry.

## Proposed Approach

### Benchmark framing

This is an empirical benchmark paper with one controlled architectural variant:

- `SAE`: sparse reconstruction of single CLIP features.
- `SSAE`: sparse reconstruction of paired CLIP-feature differences.
- `ASD`: anchor-plus-shift training with one shared decoder, one anchor encoder, and one shift encoder.

All three methods use the same latent width, optimizer family, training-step budget, and edit-scale normalization. The main comparison is not "can a new model work?" but "does decoder sharing help prototype-edit reusability beyond the strongest nearby baseline, SSAE, when training data and compute are matched?"

### Representation and parameterization

Let `h(x) in R^d` be a frozen CLIP pooled pre-projection feature. ASD learns:

- anchor codes `z(x) in R_+^k`;
- signed shift codes `s(x, x') in R^k`;
- one shared decoder `W in R^{d x k}`.

Its losses are

`L_anchor = ||h - W z||_2^2`

`L_shift = ||(h' - h) - W s||_2^2`

`L_tie = ||s - stopgrad(z' - z)||_1`

`L_sparse = ||z||_1 + ||s||_1`

and

`L = L_anchor + lambda_shift L_shift + lambda_tie L_tie + lambda_sparse L_sparse`.

The point of `L_tie` is not to claim a new identifiability theorem. It only enforces that the same decoder basis is useful for both single-example reconstruction and edit transport. One ablation, `ASD-no-tie`, tests whether any gain comes from decoder sharing alone rather than from explicit anchor/shift agreement.

### Method-matched edit construction

For a held-out source-target pair `(x, x')`, edits are constructed in the fairest native way for each method:

- `SAE`: `v = W (z(x') - z(x))`
- `SSAE`: `v = W s(x, x')`
- `ASD`: `v = W s(x, x')`

Prototype edits are the primary endpoint. For each attribute or factor `a`, estimate one reusable edit from training pairs only:

- `SAE`: `v_a = W mean[z(x') - z(x)]`
- `SSAE`: `v_a = W mean[s(x, x')]`
- `ASD`: `v_a = W mean[s(x, x')]`

where the mean is over training pairs that flip `a` while keeping nuisance variation small. These prototypes are then applied to disjoint held-out sources. This makes the benchmark about reuse rather than pair memorization.

### Edit-scale normalization

All decoded edits are normalized before application:

`v_norm = v / (||v||_2 + eps)`.

For each attribute `a`, define a method-independent scale

`alpha_a = median ||h(x') - h(x)||_2`

over training pairs for `a`. The applied edit is

`h_edit(gamma) = h(x) + gamma alpha_a v_norm`

with `gamma in {0.5, 1.0, 1.5}`. Reporting all methods under the same scale removes the common failure mode where a method appears better only because it emits larger raw vectors.

## Data and Pair Construction

### Controlled benchmark: 3D Shapes instead of dSprites

dSprites is a poor fit for this question because pooled CLIP features may not preserve its exact binary factors well, making negative results hard to interpret. The controlled dataset is therefore `3D Shapes`, which has richer rendered visual structure and exact factor labels while remaining lightweight.

To stay within budget and avoid testing factors CLIP barely encodes, the study first runs a one-time factor-retention screen on the training split using fixed linear probes on frozen CLIP features. The benchmark then keeps the three most predictable factors among shape, object hue, scale, and floor hue. Pairs are exact one-factor flips with all other labeled factors held constant.

This dataset provides the cleanest answer to the locality question: when applying a factor prototype, does the intended factor change while the non-target factors remain stable?

### Real benchmark: CelebA subset

Use a reduced CelebA subset with non-sensitive visible attributes:

- `Smiling`
- `Eyeglasses`
- `Wearing_Hat`

If hat coverage is too low after filtering, replace it with `Blond_Hair` and record the swap explicitly.

For each target attribute, source-target pairs are mined with label constraints plus neighborhood filtering. Crucially, neighborhood search for pair mining is done in frozen `DINOv2-S/14` feature space rather than CLIP space. CLIP remains the only training representation; DINO is used only to reduce leakage from evaluating and mining entirely inside the same geometry.

Pair eligibility requires:

- target label flips;
- non-target target-set labels remain unchanged;
- DINO-space source-target distance below an attribute-specific percentile threshold.

This yields compute-feasible, semantically close pairs without making CLIP geometry both the training substrate and the pair-mining oracle.

## Related Work

### Direct novelty threats

**Sparse Shift Autoencoders for Identifying Concepts from Large Language Model Activations** (Joshi, Dittadi, Lachapelle, and Sridhar, arXiv:2502.12179v2, revised February 27, 2026) is the main baseline. It already shows why difference-based sparse coding can be more identifiable than activation-only SAE training. The present paper does not claim to supersede SSAE conceptually; it only asks whether a shared decoder improves prototype-edit reuse on frozen CLIP features under the same compute budget.

**Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models** (Pach, Karthik, Bouniot, Belongie, and Akata, NeurIPS 2025) is a direct threat to any claim that CLIP/VLM sparse features can be interpreted or used for steering. It already studies CLIP-like VLM representations, monosemanticity, and intervention effects. The proposed paper differs only in comparing activation-only, shift-only, and shared-decoder training for reusable prototypes under matched constraints.

**Steering CLIP's vision transformer with sparse autoencoders** (Joseph et al., 2025) shows that CLIP sparse features can support interventions. That sharply narrows the scope here: this proposal is not "SAEs can steer CLIP," but "does shared decoding help prototype reuse versus SSAE on the same CLIP features?"

**Visual Sparse Steering: Improving Zero-shot Image Classification with Sparsity Guided Steering Vectors** (Chatzoudis et al., 2025) and PASS reduce novelty around sparse steering in vision even further. Those papers optimize downstream zero-shot classification gains. The proposed study does not. It instead benchmarks edit locality and transferability of one reusable prototype vector.

### Shared-latent threats

**SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability** (Nasiri-Sarvi, Rivaz, and Hosseini, TMLR 2026; arXiv revised March 6, 2026) and **LUCID-SAE: Learning Unified Vision-Language Sparse Codes for Interpretable Concept Discovery** (Gu et al., arXiv:2602.07311, February 7, 2026) largely eliminate any novelty claim around unified sparse spaces. Both papers already pursue shared or aligned vision-language sparse codes. The present proposal is much narrower: one frozen CLIP vision feature space, no multimodal latent-sharing claim, and no attempt to learn a generally aligned concept dictionary.

### Cautionary framing

**Sparse Autoencoders Do Not Find Canonical Units of Analysis** (Leask et al., 2025) argues against overclaiming monosemantic feature recovery. **How Reliable are Causal Probing Interventions?** (Canby et al., 2024) provides a better framing: interventions should be judged by target completeness, non-target selectivity, and their tradeoff. **Stable and Steerable Sparse Autoencoders with Weight Regularization** (Jedryszek and Crook, 2026) further motivates reporting uncertainty and cross-seed variability instead of single-run wins.

Taken together, the literature leaves room for only one modest contribution: a careful benchmark of whether shared decoding helps reusable sparse edits once the comparison is narrowed to compute-matched CLIP-feature editing and judged with uncertainty-aware metrics.

## Experiments

### Setup

- Frozen main encoder: `CLIP ViT-B/32` image tower, pooled pre-projection features
- Frozen auxiliary encoder: `DINOv2-S/14` pooled features for pair mining and one transfer evaluation only
- Datasets: `3D Shapes` and reduced `CelebA`
- Primary methods: `SAE`, `SSAE`, `ASD`
- Reduced ablation: `ASD-no-tie`
- Random seeds: `{11, 22, 33}` for every primary method on every dataset; `ASD-no-tie` runs with seed `11` first and extends to the remaining seeds only if time remains

The three primary methods are compute matched by latent width, training-step budget, and optimizer family. The paper will report both parameter count and measured GPU minutes per run.

### Metrics

#### Primary endpoint: prototype-edit reusability

For each attribute or factor:

1. estimate a prototype edit on training pairs only;
2. apply it to held-out source examples;
3. score target change and non-target drift with fixed readouts;
4. summarize reusability as reliability, the harmonic mean of target-change and non-target preservation.

The main table reports this metric for all attributes and both datasets. Pair-specific inferred edits are secondary diagnostics only.

#### Controlled benchmark metrics

On `3D Shapes`:

- single-feature reconstruction MSE
- pair-delta reconstruction MSE
- mean active latent count
- exact-factor prototype locality
- nuisance-factor drift after editing

Because labels are exact, the strongest evidence here is simple: a factor prototype should change the intended factor far more than the others.

#### Real-benchmark metrics

On `CelebA`:

- target-change under fixed attribute probes
- non-target preservation under the same probes
- reliability
- pair coverage after filtering
- active-latent count and prototype sparsity

### Evaluation less coupled to CLIP geometry

The proposal adds one explicit auxiliary evaluation that is not judged directly in CLIP space.

First, cache frozen `DINOv2-S/14` features for the same images. Fit a ridge map `A` from training CLIP features to training DINO features. This map is learned once and never fine-tuned on edited examples. After editing in CLIP space, evaluate `A h_edit` with fixed DINO-space linear probes trained on untouched training images.

This auxiliary score is intentionally modest in claim. It does not prove semantic faithfulness. It tests whether edits that look good in CLIP also transfer through an independently learned representation bridge to a second vision geometry. If ASD beats SAE only inside CLIP but not after CLIP-to-DINO transfer, the paper should say so directly.

### Statistical analysis

The paper predefines uncertainty-aware decision rules instead of thresholding seed means.

- Report per-seed results for seeds `{11, 22, 33}`.
- For each dataset-attribute pair, compute a paired bootstrap 95% confidence interval over held-out source examples for the difference `ASD - SSAE` and `ASD - SAE` on prototype reliability.
- For the aggregate claim, average attribute-level effects within each seed, then run a paired permutation test across the three seed-level differences. Because `n=3` seeds is small, this p-value is secondary; the primary decision signal is the effect size plus the bootstrap interval.
- Also report a hierarchical bootstrap CI that resamples seeds and test instances together for the overall reliability gap.

### Runtime budget

The budget remains inside the stated 8-hour limit:

- CLIP and DINO feature caching on reduced datasets: `2.0` to `2.5` GPU hours
- pair mining, factor screening, and probe fitting: `0.5` to `0.8` CPU/GPU hours
- primary matrix `3 methods x 2 datasets x 3 seeds = 18 runs`: `2.2` to `2.8` GPU hours
- minimum `ASD-no-tie` ablation: `0.2` to `0.4` GPU hour
- evaluation, bootstrap CIs, transfer audit, and plots: `0.8` to `1.2` hours

Total expected wall-clock demand is roughly `5.7` to `7.7` hours, leaving a small contingency margin.

## Success Criteria

The hypothesis is:

`ASD improves prototype-edit reusability over vanilla SAE and provides a measurable advantage over SSAE under compute-matched training on the same frozen CLIP features.`

This hypothesis is supported only if all of the following hold:

1. On `CelebA`, the mean `ASD - SSAE` prototype-reliability gap is positive, the hierarchical 95% bootstrap CI excludes `0`, and the direction is the same in at least `2/3` seeds.
2. On `3D Shapes`, ASD improves exact-factor locality over both baselines by at least `5` absolute points, with paired bootstrap 95% CIs excluding `0` for at least two of the retained factors.
3. The advantage is not due only to edit magnitude; the ranking is stable across at least two `gamma` values.
4. The DINO-transfer auxiliary evaluation shows the same direction of effect as the CLIP-space primary metric on at least one dataset, indicating the gain is not purely an artifact of scoring inside CLIP geometry.

The hypothesis is weakened if ASD only beats vanilla SAE but is statistically indistinguishable from SSAE. It is rejected for this scope if SSAE matches or exceeds ASD on prototype reliability and locality once uncertainty is reported.

## Ethics and Risk Statement

The CelebA study excludes identity-sensitive targets such as gender and age. All claims are restricted to edits of frozen representations used for measurement. The paper does not present an image-generation or identity-manipulation system, and it should not claim strong causal control of visual concepts.

## References

1. Marc Canby, Adam Davies, Chirag Rastogi, and Julia Hockenmaier. *How Reliable are Causal Probing Interventions?* arXiv:2408.15510, 2024.
2. Shruti Joshi, Andrea Dittadi, Sebastien Lachapelle, and Dhanya Sridhar. *Sparse Shift Autoencoders for Identifying Concepts from Large Language Model Activations.* arXiv:2502.12179v2, revised February 27, 2026.
3. Patrick Leask, Bart Bussmann, Michael Pearce, Joseph Bloom, Curt Tigges, Noura Al Moubayed, Lee Sharkey, and Neel Nanda. *Sparse Autoencoders Do Not Find Canonical Units of Analysis.* arXiv:2502.04878, 2025.
4. Vladimir Zaigrajew, Hubert Baniecki, and Przemyslaw Biecek. *Interpreting CLIP with Hierarchical Sparse Autoencoders.* arXiv:2502.20578, 2025.
5. Mateusz Pach, Shyamgopal Karthik, Quentin Bouniot, Serge Belongie, and Zeynep Akata. *Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models.* NeurIPS 2025 / arXiv:2504.02821v3, 2025.
6. Sonia Joseph, Praneet Suresh, Ethan Goldfarb, Lorenz Hufe, Yossi Gandelsman, Robert Graham, Danilo Bzdok, Wojciech Samek, and Blake Aaron Richards. *Steering CLIP's vision transformer with sparse autoencoders.* arXiv:2504.08729, 2025.
7. Gerasimos Chatzoudis, Zhuowei Li, Gemma E. Moran, Hao Wang, and Dimitris N. Metaxas. *Visual Sparse Steering: Improving Zero-shot Image Classification with Sparsity Guided Steering Vectors.* arXiv:2506.01247, 2025.
8. Ali Nasiri-Sarvi, Hassan Rivaz, and Mahdi S. Hosseini. *SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability.* TMLR 2026 / arXiv:2507.06265v2, revised March 6, 2026.
9. Piotr Jedryszek and Oliver M. Crook. *Stable and Steerable Sparse Autoencoders with Weight Regularization.* arXiv:2603.04198, 2026.
10. Difei Gu, Yunhe Gao, Gerasimos Chatzoudis, Zihan Dong, Guoning Zhang, Bangwei Guo, Yang Zhou, Mu Zhou, and Dimitris Metaxas. *LUCID-SAE: Learning Unified Vision-Language Sparse Codes for Interpretable Concept Discovery.* arXiv:2602.07311, 2026.
