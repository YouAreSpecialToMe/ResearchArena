# Pair-Supervised Regularization for Selective Counterfactual Edits in Frozen Vision SAEs

## Introduction

Recent 2025 work has already established most of the broad claims that would otherwise make a vision-SAE paper sound novel. Standard vision SAEs on frozen backbones already support causal interventions and hypothesis testing; Archetypal SAE already changes the SAE objective to improve geometry and stability; Concept-SAE and related papers already add supervision for grounded causal probing; MP-SAE already shows that stronger sparse-coding procedures can change the apparent quality of learned features. A credible contribution therefore cannot be "we make vision SAEs interpretable" or "we add supervision to learn better concepts."

This proposal makes a narrower claim: **pair-supervised regularization can improve the selectivity of counterfactual latent edits in a frozen vision SAE**. The study asks whether an SAE trained on frozen DINOv2 features can be nudged so that exact single-factor counterfactual pairs produce concentrated, repeatable latent differences while nuisance-preserving pairs stay stable. If that works, then the same simple edit rule should cause larger target-factor changes and smaller off-target changes than comparable SAE baselines.

The significance claim is intentionally limited. The core evidence comes from Shapes3D, where exact counterfactual pairs are available. That makes the paper a controlled mechanism study of counterfactual edit selectivity, not a broad claim about real-world concept learning. Any non-synthetic transfer result is treated as secondary and is only discussed if it survives a strict pair-quality audit.

## Proposed Approach

### Scope and feasibility

The project is scoped for one RTX A6000, 60 GB RAM, 4 CPU cores, and roughly 8 total experimental hours:

- one frozen backbone: DINOv2 ViT-S/14
- one primary representation: final-layer CLS token
- one primary benchmark: Shapes3D
- three main methods: vanilla SAE, Archetypal SAE, pair-supervised SAE
- one confirmatory method: MP-SAE, one seed
- one small tokenization sensitivity pilot: final CLS, penultimate CLS, mean-pooled final patch tokens
- one optional audited transfer check: CelebA only if pair quality is demonstrably high

All backbone features are cached once. No backbone finetuning, no large-scale retraining, and no claim beyond this frozen-backbone setting are required.

### Method overview

For a frozen activation `h`, the SAE produces sparse code `z` and reconstruction `h_hat`. Training mixes unpaired activations with two pair types:

- nuisance-preserving pairs `(x, x+)`
- exact single-factor counterfactual pairs `(x, x_cf^k)` where only factor `k` changes

The training objective is

`L = L_rec + lambda_s L_sparse + lambda_i L_inv + lambda_c L_conc + lambda_a L_align + lambda_o L_sep`

where:

- `L_rec`: reconstruction MSE on frozen features
- `L_sparse`: standard sparsity penalty
- `L_inv`: latent stability for nuisance-preserving pairs
- `L_conc`: concentration of counterfactual latent differences
- `L_align`: same-factor agreement of signed difference directions
- `L_sep`: weak separation between average signed difference directions of different factors

The contribution is not a new sparse-coding family. It is a regularization study about how an otherwise standard frozen vision SAE should behave on paired perturbations.

### Nuisance-preserving pair regularization

The invariance term is deliberately conservative. On Shapes3D, nuisance pairs come from small photometric perturbations and mild crop-resize perturbations applied to the same image. A perturbation family is only used if frozen factor heads preserve all ground-truth factors above a high agreement threshold on a held-out audit set. If crop-based positives fail the audit, the invariance term falls back to photometric pairs only.

This matters because the paper should not quietly assume that DINOv2 treats augmentations as nuisance-free. The invariance term is only valid if the pair construction is validated quantitatively.

### Counterfactual-difference regularization

For a single-factor pair, define the signed latent difference

`d = z_cf - z`

and the absolute difference

`delta = |d|`.

The concentration term promotes selective change:

`p_i = delta_i / (sum_j delta_j + eps)`

`C(delta) = (m * sum_i p_i^2 - 1) / (m - 1)`

`L_conc = 1 - C(delta)`

To make edits repeatable rather than merely sparse, each factor `k` also maintains a running signed centroid `mu_k = E[d | factor = k]`. The alignment term encourages factor-`k` deltas to point in a consistent direction:

`L_align = 1 - cos(d, mu_k)`.

Finally, `L_sep` penalizes large cosine overlap between different `mu_k` vectors so that all factors do not collapse onto the same edit direction.

### Why this is not subsumed by existing methods

The proposal is only interesting if it is clearly not covered already.

- **Not subsumed by Interpretable and Testable Vision Features via Sparse Autoencoders**: that paper shows standard vision SAEs can already support causal editing. This proposal does not claim first causal editing; it asks whether paired supervision during SAE training improves edit selectivity under the same downstream editing pipeline.
- **Not subsumed by Archetypal SAE**: Archetypal SAE targets global dictionary stability and geometry. Here the supervision is on how codes move under validated nuisance pairs and exact counterfactual pairs. A method can have a stable dictionary yet still yield diffuse, off-target edits.
- **Not subsumed by Concept-SAE**: Concept-SAE uses semantic supervision to learn grounded concept tokens for causal probing and localization. This proposal does not learn grounded concept names, masks, or tokens. It only uses pair labels saying "same except nuisance" or "different in exactly one factor" to regularize delta structure.
- **Not subsumed by MP-SAE**: MP-SAE could simply be a stronger sparse-coding baseline. That is exactly why it appears as a confirmatory baseline. If MP-SAE erases the gain, the paper's claim weakens sharply.

## Related Work

### Vision SAEs and causal editing

**Interpretable and Testable Vision Features via Sparse Autoencoders** (Stevens et al., 2025) is the main baseline-setting paper because it demonstrates editable sparse features in frozen vision representations. The present proposal should be read as a follow-up question: once editable features exist, can pair supervision make those edits more selective?

**Causal Interpretation of Sparse Autoencoder Features in Vision** (Han et al., 2025) is adjacent but different. It studies whether feature activations are causally explained by image regions and patch interventions. That improves explanation fidelity for existing features; this proposal changes SAE training to improve low-off-target feature-space edits.

### Modified SAE objectives

**Archetypal SAE** (Fel et al., 2025) and **From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit** (Costa et al., 2025) are the two most important modified-objective baselines. Archetypal SAE addresses instability and identifiability through geometry. MP-SAE changes the sparse-coding procedure itself. The proposal differs by targeting pair-conditioned code behavior rather than dictionary geometry or pursuit dynamics.

**Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry** (Hindupur et al., 2025) is a useful cautionary reference: SAE behavior depends strongly on geometric assumptions in the data. That motivates using exact-factor synthetic pairs and avoiding broader claims.

**Sparse Autoencoders Do Not Find Canonical Units of Analysis** (Leask et al., 2025) also motivates the study design. If latent units are not canonical, the relevant question is not whether a single neuron equals a single concept, but whether the representation supports reliable and selective interventions under a shared evaluation rule.

### Supervised and concept-grounded SAEs

**Concept-SAE: Active Causal Probing of Visual Model Behavior** (Ding et al., 2025), **SPARC** (Nasiri-Sarvi et al., 2025), and **SAEmnesia** (Cassano et al., 2025) show that supervision can reshape SAE structure for grounded probing, cross-model alignment, or concept erasure. Those papers narrow the novelty space: adding supervision alone is not novel. The distinctive claim here is pair supervision aimed specifically at counterfactual edit selectivity in a frozen vision backbone.

### Evaluation context

Older interpretability benchmarks such as **Network Dissection** (Bau et al., 2017), **TCAV** (Kim et al., 2018), and **Concept Bottleneck Models** (Koh et al., 2020) motivate intervention-based evaluation rather than purely qualitative feature inspection. More recent evaluation-focused papers such as **Probing the Representational Power of Sparse Autoencoders in Vision Models** (Olson et al., 2025) and **Evaluating Adversarial Robustness of Concept Representations in Sparse Autoencoders** (Li et al., 2025) reinforce that vision SAE quality should be judged by downstream behavior, not by monosemanticity rhetoric alone.

## Experiments

### Datasets

**Shapes3D** is the primary and only claim-bearing benchmark. It provides exact latent factors and exact single-factor counterfactuals, which are necessary for a controlled edit-selectivity study.

**CelebA** is optional and explicitly non-claim-bearing unless it passes a strict pair-quality audit. Candidate pairs must:

- differ on exactly one selected attribute and match on the remaining evaluation attributes
- satisfy a DINOv2 nearest-neighbor similarity threshold
- achieve a reported pair-purity rate high enough to justify interpretation

If this audit fails, CelebA is omitted from the final claim rather than used as weak external validation.

### Methods and runs

Main Shapes3D runs:

- vanilla SAE, 3 seeds
- Archetypal SAE, 3 seeds
- pair-supervised SAE, 3 seeds

Confirmatory Shapes3D run:

- MP-SAE, 1 seed

Ablations, 1 seed each:

- no `L_inv`
- no `L_conc`
- no `L_align` or `L_sep`

Representation sensitivity pilot, 1 seed each on a reduced split:

- final-layer CLS
- penultimate-layer CLS
- final-layer mean-pooled patch tokens

### Symmetric model selection and checkpointing

The previous draft risked unfairness by giving the proposed method a selectivity-aware selection rule. The revised protocol removes that asymmetry.

- All methods use the same train/validation/test split.
- All methods are compared at matched SAE width and matched target sparsity budgets.
- Hyperparameter sweeps only tune reconstruction/sparsity tradeoffs needed to hit those budgets.
- Early stopping and final checkpoint selection use the same validation criterion for every method: best reconstruction among checkpoints whose mean active-latent count falls inside the target sparsity window.
- Intervention metrics are never used for early stopping or checkpoint selection.

Primary comparisons are reported at two matched sparsity budgets. Reconstruction-selectivity tradeoff curves are also shown so the result is not an artifact of one operating point.

### Shared edit-construction rules

To avoid conflating representation quality with one brittle editing heuristic, every method is tested under two shared edit rules.

**Rule A: sparse association edit**

For factor `k` and latent `j`, compute

`s_{k,j} = E[|z_j(x) - z_j(x_cf)| | factor = k] - E[|z_j(x) - z_j(x_cf)| | factor != k]`.

Select a fixed `K` highest-scoring latents and apply the mean signed shift over training pairs on only those latents.

**Rule B: dense centroid edit**

Use the full signed centroid

`mu_k = E[z_cf - z | factor = k]`

with no top-`K` truncation. Apply a scalar multiple chosen from a small global grid shared across methods on a calibration split.

If gains appear only under Rule A and disappear under Rule B, the paper can only claim compatibility with a sparse edit heuristic, not a more robust representational advantage.

### Primary evaluation

The paper should prioritize intervention outcomes over proxy interpretability metrics.

1. **Downstream prediction-shift test**

Train small frozen-head predictors on cached DINOv2 features for each Shapes3D factor. After editing `h`, evaluate:

- target-factor success: prediction moves to the true counterfactual value
- off-target preservation: untouched factors remain unchanged
- selective intervention score: target improvement minus average off-target change

2. **Counterfactual consistency test**

For each held-out pair `(x, x_cf^k)`, edit `x` and compare the edited feature to the true counterfactual feature:

- cosine distance to `h_cf`
- rank of `h_cf` among same-factor candidate counterfactuals
- consistency rate against distractor counterfactuals that change other factors

### Secondary diagnostics

- normalized reconstruction error
- mean active latent count
- nuisance-pair latent drift
- delta concentration
- within-factor signed-delta cosine similarity
- number of latents carrying 90% of delta mass
- linear factor decoding from `delta`

These are useful for mechanism analysis, but they do not constitute the main success claim by themselves.

### Statistical treatment

With 3 seeds for the main methods, significance language must be restrained.

- Report per-seed results and bootstrap confidence intervals over held-out pairs.
- Emphasize effect sizes and consistency across seeds, not fragile null-hypothesis claims.
- Treat CelebA as descriptive unless the pair-quality audit is passed and the directional result survives there as well.

## Success Criteria

The hypothesis is supported if all of the following hold on Shapes3D:

1. The pair-supervised SAE beats both vanilla SAE and Archetypal SAE on at least one primary intervention metric while matching or improving off-target preservation.
2. The gain appears under both shared edit rules, or at minimum under Rule A with a clear statement that the evidence is edit-rule-dependent.
3. Reconstruction stays within 5% relative of the best matched-budget baseline and the latent code does not collapse.
4. The result is not eliminated by the one-seed MP-SAE confirmatory run.

The hypothesis is weakened or refuted if any of the following happen:

1. Gains appear only in proxy diagnostics and not in prediction-shift or counterfactual-consistency tests.
2. Archetypal SAE or MP-SAE matches the proposed method once model selection and edit construction are made symmetric.
3. The nuisance-pair audit fails, making the invariance signal untrustworthy.
4. The result depends entirely on the top-`K` edit rule.
5. No non-synthetic result survives the pair-quality audit, in which case the paper must present itself strictly as a synthetic mechanism study.

## References

1. David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. *Network Dissection: Quantifying Interpretability of Deep Visual Representations*. CVPR, 2017. arXiv:1704.05796.
2. Chris Burgess and Hyunjik Kim. *3D Shapes Dataset*. Dataset repository, 2018. https://github.com/google-deepmind/3d-shapes.
3. Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and Rory Sayres. *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)*. ICML, 2018. arXiv:1711.11279.
4. Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. *Concept Bottleneck Models*. ICML, 2020. arXiv:2007.04612.
5. Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. *Deep Learning Face Attributes in the Wild*. ICCV, 2015. arXiv:1411.7766.
6. Maxime Oquab, Timothée Darcet, Théodore Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Michael Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Quentin Deforme, Vivek Sharma, Gabriel Synnaeve, Hu Xu, Ross Girshick, and Hervé Jégou. *DINOv2: Learning Robust Visual Features without Supervision*. TMLR, 2024. arXiv:2304.07193.
7. Samuel Stevens, Wei-Lun Chao, Tanya Berger-Wolf, and Yu Su. *Interpretable and Testable Vision Features via Sparse Autoencoders*. arXiv, 2025. arXiv:2502.06755.
8. Thomas Fel, Ekdeep Singh Lubana, Jacob S. Prince, Matthew Kowal, Victor Boutin, Isabel Papadimitriou, Binxu Wang, Martin Wattenberg, Demba Ba, and Talia Konkle. *Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models*. arXiv, 2025. arXiv:2502.12892.
9. Patrick Leask, Bart Bussmann, Michael Pearce, Joseph Bloom, Curt Tigges, Noura Al Moubayed, Lee Sharkey, and Neel Nanda. *Sparse Autoencoders Do Not Find Canonical Units of Analysis*. arXiv, 2025. arXiv:2502.04878.
10. Sai Sumedh R. Hindupur, Ekdeep Singh Lubana, Thomas Fel, and Demba Ba. *Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry*. arXiv, 2025. arXiv:2503.01822.
11. Aaron J. Li, Suraj Srinivas, Usha Bhalla, and Himabindu Lakkaraju. *Evaluating Adversarial Robustness of Concept Representations in Sparse Autoencoders*. arXiv, 2025. arXiv:2505.16004.
12. Valérie Costa, Thomas Fel, Ekdeep Singh Lubana, Bahareh Tolooshams, and Demba Ba. *From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit*. NeurIPS, 2025. arXiv:2506.03093.
13. Ali Nasiri-Sarvi, Hassan Rivaz, and Mahdi S. Hosseini. *SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability*. arXiv, 2025. arXiv:2507.06265.
14. Matthew Lyle Olson, Musashi Hinck, Neale Ratzlaff, Changbai Li, Phillip Howard, Vasudev Lal, and Shao-Yen Tseng. *Probing the Representational Power of Sparse Autoencoders in Vision Models*. ICCV Workshops, 2025. arXiv:2508.11277.
15. Sangyu Han, Yearim Kim, and Nojun Kwak. *Causal Interpretation of Sparse Autoencoder Features in Vision*. arXiv, 2025. arXiv:2509.00749.
16. Enrico Cassano, Riccardo Renzulli, Marco Nurisso, Mirko Zaffaroni, Alan Perotti, and Marco Grangetto. *SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders*. arXiv, 2025. arXiv:2509.21379.
17. Jianrong Ding, Muxi Chen, Chenchen Zhao, and Qiang Xu. *Concept-SAE: Active Causal Probing of Visual Model Behavior*. arXiv, 2025. arXiv:2509.22015.
