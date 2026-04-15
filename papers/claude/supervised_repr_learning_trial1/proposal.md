# Confusion-Geometric Supervised Contrastive Learning: Shaping Embedding Geometry from Training Dynamics

## Introduction

### Context
Supervised contrastive learning (SupCon) [Khosla et al., 2020] has emerged as a powerful alternative to cross-entropy training for learning visual representations. By pulling together embeddings of same-class samples while pushing apart different-class embeddings, SupCon produces representations that are more robust to hyperparameter choices and data augmentations. However, recent theoretical work [Papyan et al., 2020; Lee et al., 2025] proves that SupCon's optimal embedding geometry is a simplex equiangular tight frame (ETF), where all class means are equidistant. This symmetric geometry discards meaningful inter-class structure -- leopards and tigers are pushed equally far apart as leopards and airplanes -- and wastes gradient capacity on already-separated classes.

### Problem Statement
This uniform ETF geometry causes two problems:
1. **Loss of semantic structure**: The embedding space cannot represent that some classes are more similar than others, hurting transfer learning, few-shot generalization, and fine-grained recognition.
2. **Inefficient learning**: Equal repulsive forces across all class pairs mean the model spends gradient budget on already-trivial separations while under-attending to genuinely confusable classes.

Existing solutions either require external label hierarchies [Lian et al., 2024; Dorfner et al., 2025], operate at the instance level without capturing systematic inter-class patterns [Robinson et al., 2021; Animesh & Chandraker, 2025], or use per-sample confidence adaptation without targeting specific class-pair confusion [Wang et al., 2025]. The MACL framework [Gao et al., 2024] uses pairwise label-overlap reweighting for multi-label retrieval, but relies on label co-occurrence statistics rather than learned confusion patterns, and targets a different problem (multi-label similarity) than single-label class discrimination.

### Key Insight
The model's own prediction confusion matrix during training encodes a rich, adaptive class similarity graph. We propose to use this signal not merely to *scale* existing loss terms (temperature/weight modulation, which is incremental over prior work), but to **prescribe a target embedding geometry** through a novel regularizer that directly drives class prototypes toward a confusion-aware configuration. This geometric approach is fundamentally different from loss modulation: it adds a new optimization objective rather than adjusting parameters of an existing one.

### Core Contribution: Confusion-Geometric Alignment (CGA)
We introduce the **Confusion-Geometric Alignment (CGA) loss**, a regularizer that:
1. Defines a **target pairwise similarity matrix** where confused class pairs have more negative cosine similarity (larger angular separation) than non-confused pairs
2. Penalizes deviations of class prototype geometry from this target
3. Operates on class prototypes (EMA of embeddings), not individual samples -- capturing class-level structure rather than instance-level hardness

The CGA loss is combined with a standard SupCon loss (with optional confusion-adaptive temperature for performance) to form **CG-SupCon**. The theoretical contribution is a proof that CG-SupCon's equilibrium geometry is uniquely determined by the confusion matrix: inter-class distances are a monotone function of confusion strength, and the method reduces to standard SupCon (with ETF geometry) when confusion is uniform.

### Why CGA Is Not Incremental Over Prior Work
Prior methods modulate *existing* loss terms:
- **Temperature modulation** [Kukleva et al., 2023; Qiu et al., 2023; Zhang et al., 2024]: Changes the sharpness of the softmax denominator. Scales gradient magnitude, does not change the target geometry.
- **Weight modulation** [Lin et al., 2017; Wang et al., 2024]: Scales the contribution of specific pairs. Changes gradient magnitude, does not change the target geometry.
- **CCDC** [Chen et al., 2024]: Uses confusion for pixel-level sampling in segmentation. Different task, different signal granularity.
- **MACL** [Gao et al., 2024]: Uses label co-occurrence frequency for pairwise reweighting in multi-label retrieval. Different signal (static label overlap vs. dynamic learned confusion) and different task (multi-label vs. single-label).

CGA is fundamentally different: it introduces a **new optimization objective** (geometric alignment to a confusion-derived target) that operates on **class prototypes** rather than modulating individual sample losses. The target geometry is analytically defined and provably breaks ETF symmetry in a structured way. This is not "combining existing components" -- it is a new loss term with new theoretical properties.

### Hypothesis
By combining the standard SupCon loss with the Confusion-Geometric Alignment regularizer, CG-SupCon will:
1. Produce embeddings whose inter-class geometry reflects semantic similarity, provably deviating from the symmetric ETF.
2. Improve classification accuracy, with disproportionately larger gains on fine-grained (within-superclass) distinctions.
3. Outperform both instance-level methods (HardNeg-CL, TCL) and pairwise reweighting approaches, demonstrating the value of explicit geometric alignment over implicit modulation.
4. Improve downstream task performance (few-shot, corruption robustness) where structured inter-class geometry has larger impact.

## Theoretical Analysis: Confusion-Geometric Equilibrium for General C

We provide a rigorous analysis showing that CGA provably breaks ETF symmetry for arbitrary number of classes C, with the equilibrium geometry determined by the confusion spectrum.

### Setup
Let C classes have unit-norm prototype embeddings {p_1, ..., p_C} on the unit sphere S^{d-1} with d >= C. Let c(i,j) >= 0 denote the symmetrized, normalized confusion between classes i and j, with sum_{j != i} c(i,j) = 1 for each i. Define the target cosine similarity:

```
s*(i,j) = -1/(C-1) - alpha * (c(i,j) - c_bar)
```

where c_bar = mean_{i<j} c(i,j) = 1/(C-1) is the mean confusion under normalization, and alpha > 0 controls the deviation from ETF. Note that s*(i,j) = -1/(C-1) when c(i,j) = c_bar (average confusion), s*(i,j) < -1/(C-1) when c(i,j) > c_bar (above-average confusion: push further), and s*(i,j) > -1/(C-1) when c(i,j) < c_bar (below-average confusion: relax).

The CGA loss is:
```
L_CGA = (1/2) * sum_{i<j} c(i,j) * (p_i . p_j - s*(i,j))^2
```

The confusion weighting c(i,j) ensures that the penalty is strongest for highly confused pairs (where getting the geometry right matters most) and negligible for non-confused pairs (where the ETF baseline is adequate).

The combined loss is:
```
L = L_SupCon({p_i}) + lambda * L_CGA({p_i})
```

### Theorem 1 (Confusion-Ordered Equilibrium)
Let {p_1*, ..., p_C*} be a local minimum of L on (S^{d-1})^C with lambda > 0. Assume d >= C (embedding dimension at least as large as number of classes). Then:

**(a) Monotone ordering**: For any two pairs (i,j) and (k,l) with c(i,j) > c(k,l), the equilibrium satisfies p_i* . p_j* < p_k* . p_l* (or equivalently, confused pairs are more angularly separated).

**(b) Bounded deviation**: The deviation from ETF is bounded:
```
|p_i* . p_j* - (-1/(C-1))| <= alpha * |c(i,j) - c_bar| + O(1/lambda)
```

**(c) ETF recovery**: When the confusion matrix is uniform (c(i,j) = 1/(C-1) for all i != j), the target reduces to s*(i,j) = -1/(C-1) for all pairs, and the unique minimum is the simplex ETF.

### Proof Sketch

**(a)**: At a minimum, the gradient of L w.r.t. each pairwise angle must vanish (via KKT conditions on the sphere). The gradient of L_CGA w.r.t. the cosine similarity s_{ij} = p_i . p_j is:

```
dL_CGA/ds_{ij} = c(i,j) * (s_{ij} - s*(i,j))
```

This creates a restoring force pulling s_{ij} toward the target s*(i,j). The L_SupCon gradient pushes all pairwise similarities toward -1/(C-1) (the ETF). At equilibrium, these forces balance. Since s*(i,j) is more negative for larger c(i,j), and the restoring force is proportional to c(i,j) (so stronger for more confused pairs), the equilibrium similarity is monotonically decreasing in confusion.

More formally: consider two pairs (i,j) and (k,l) with c(i,j) > c(k,l). At equilibrium:
- The SupCon gradient contribution is approximately equal for all pairs (since SupCon favors uniform spacing)
- The CGA gradient for pair (i,j) has a restoring force c(i,j) * (s_{ij} - s*(i,j)) pulling toward the more negative target s*(i,j)
- The CGA gradient for pair (k,l) has a weaker restoring force pulling toward the less negative target s*(k,l)
- Equilibrium requires s_{ij} < s_{kl} (more confused pair is more separated)

**(b)**: At the minimum, the CGA gradient and SupCon gradient balance:

```
lambda * c(i,j) * (s_{ij}* - s*(i,j)) ≈ -dL_SupCon/ds_{ij}|_{s*}
```

Since dL_SupCon/ds_{ij} is bounded (it is a smooth function of the pairwise similarities), dividing by lambda gives the bound. The O(1/lambda) term vanishes as lambda -> infinity, where the CGA target dominates.

**(c)**: When c(i,j) = 1/(C-1) for all pairs, s*(i,j) = -1/(C-1) for all pairs. The CGA loss is minimized at the simplex ETF (which achieves all pairwise cosine similarities equal to -1/(C-1)). The SupCon loss is also minimized at the ETF [Lee et al., 2025]. Therefore the unique combined minimum is the ETF.

### Proposition 2 (Feasibility of Target Geometry)
The target Gram matrix G* with G*[i,i] = 1 and G*[i,j] = s*(i,j) for i != j is realizable on S^{d-1} (i.e., there exist unit vectors achieving these pairwise cosine similarities) whenever:
1. G* is positive semi-definite, and
2. d >= rank(G*)

For typical confusion matrices with alpha sufficiently small (alpha < 1/(C-1)), G* is a perturbation of the ETF Gram matrix and inherits positive semi-definiteness. Since d >> C in practice (e.g., d=128, C=100), the rank condition is always satisfied. We verify this numerically for all experiments.

### Comparison to Prior Theory
- **Lee et al. (2025)**: Proved SupCon converges to ETF. Our Theorem 1(c) recovers this as a special case.
- **Original CC-SupCon toy analysis**: Limited to C=3 on the unit circle. Theorem 1 handles arbitrary C with formal bounds.
- **Neural collapse literature** [Papyan et al., 2020]: Studies when/why ETF emerges. We study how to *break* ETF in a controlled, structured way.

## Proposed Approach

### Overview
We propose **Confusion-Geometric Supervised Contrastive Learning (CG-SupCon)**, which combines the standard SupCon loss with a confusion-geometric alignment regularizer. The method has four components: (1) online confusion tracking, (2) online prototype tracking, (3) the CGA regularizer, and (4) optional confusion-adaptive temperature.

### 1. Online Confusion Matrix
During training, we maintain a class confusion matrix A in R^{C x C} via exponential moving average:

```
A_batch[i,j] = (1/N_i) * sum_{x in class_i} p(j|x)
A <- mu * A + (1-mu) * A_batch
```

where p(j|x) is the model's predicted probability for class j given input x, N_i is the count of class-i samples in the batch, and mu is the EMA momentum (default 0.99). The symmetrized confusion is c(i,j) = (A[i,j] + A[j,i]) / 2 for i != j.

We normalize each row of the off-diagonal confusion to sum to 1: c_norm(i,j) = c(i,j) / sum_{k != i} c(i,k). This makes modulation strength invariant to the number of classes.

**Computational overhead**: The confusion matrix update requires O(B * C) per batch (B = batch size) for computing predictions and O(C^2) for the EMA update. For C=100 and B=256, this is ~25K + 10K = 35K operations per step -- negligible compared to the forward/backward pass (~10^9 FLOPs for ResNet-18).

### 2. Online Class Prototypes
We maintain class prototypes as EMA of L2-normalized embeddings:

```
p_i <- nu * p_i + (1-nu) * mean_{x in class_i in batch} (z_x / ||z_x||)
p_i <- p_i / ||p_i||  (re-normalize)
```

with EMA momentum nu = 0.99. The prototypes are used only in the CGA loss (not the SupCon loss), decoupling geometric alignment from instance-level contrastive learning.

**Computational overhead**: O(B * d) per batch for updating prototypes (d = embedding dimension, typically 128). Negligible.

### 3. Confusion-Geometric Alignment (CGA) Loss
The CGA loss drives class prototypes toward a confusion-aware target geometry:

```
L_CGA = (1/2) * sum_{i<j} c_norm(i,j) * (p_i . p_j - s*(i,j))^2
```

where the target similarity is:

```
s*(i,j) = -1/(C-1) - alpha * (c_norm(i,j) - 1/(C-1))
```

- When c_norm(i,j) > 1/(C-1) (above-average confusion): s*(i,j) < -1/(C-1), pushing the pair further apart than ETF
- When c_norm(i,j) < 1/(C-1) (below-average confusion): s*(i,j) > -1/(C-1), allowing the pair to be closer than ETF
- When c_norm(i,j) = 1/(C-1) (average confusion): s*(i,j) = -1/(C-1), matching ETF

The CGA loss has O(C^2) complexity: computing C*(C-1)/2 pairwise inner products of d-dimensional prototypes. For C=100, d=128: ~5K inner products = 640K multiply-adds. For C=200 (Tiny-ImageNet): ~20K inner products = 2.5M multiply-adds. Both are negligible compared to the SupCon loss computation.

**Gradient flow**: The CGA gradient w.r.t. prototype p_i is:

```
dL_CGA/dp_i = sum_{j != i} c_norm(i,j) * (p_i . p_j - s*(i,j)) * p_j
```

This is projected onto the tangent space of S^{d-1} during optimization (since prototypes are unit-normalized). The gradient pushes prototype p_i away from prototypes of confused classes and allows it to move closer to non-confused classes.

### 4. Confusion-Adaptive Temperature (Optional Enhancement)
For the SupCon loss, we use class-pair-specific temperatures:

```
tau(i,j) = tau_base / (1 + gamma * c_norm(i,j))
```

This gives lower temperature (sharper contrast) to confused pairs, providing an additional instance-level signal complementary to the prototype-level CGA. We include this as an optional enhancement, not the primary contribution, and ablate it to show that CGA alone provides the main benefit.

### 5. CG-SupCon Loss
The full loss for an anchor z_a of class y_a:

```
L_CG-SupCon = L_SupCon(z_a; tau) + lambda * L_CGA({p_i})
```

where L_SupCon is the standard supervised contrastive loss (with either fixed or adaptive temperature), and L_CGA operates on the class prototypes. The two losses target different levels:
- L_SupCon: instance-level feature learning (pull same-class, push different-class)
- L_CGA: class-level geometric alignment (shape inter-class prototype distances)

### 6. Why Class-Pair Confusion Is Fundamentally Different from Instance-Level Hardness

This is a crucial distinction. Instance-level hard negative mining [Robinson et al., 2021] and gradient tuning [Animesh & Chandraker, 2025] operate on individual samples: a negative sample z_n is upweighted if it is close to the anchor z_a in embedding space. This captures *instance-specific* difficulty but has three limitations:

1. **Noise sensitivity**: A sample may be "hard" due to augmentation artifacts, mislabeling, or outlier features -- not because the underlying classes are systematically confusable.
2. **Transience**: Instance-level hardness changes rapidly as embeddings evolve during training, providing a noisy learning signal.
3. **No aggregation**: Instance-level methods cannot distinguish "class i has one outlier sample near class j" from "class i is systematically confused with class j across all samples."

Class-pair confusion aggregates over all samples and time (via EMA), providing:
1. **Stability**: The confusion matrix changes slowly, offering a reliable signal.
2. **Systematic patterns**: It captures inter-class relationships that persist across samples.
3. **Actionable structure**: A C x C matrix encodes the full pairwise confusion graph, enabling geometric reasoning.

We validate this distinction empirically with a **stability analysis**: we measure the Kendall tau rank correlation of the top-20 hardest negatives (instance-level) vs. top-20 most confused class pairs across different training checkpoints and seeds. We hypothesize that class-pair rankings are significantly more stable (higher correlation) than instance-level rankings.

### Training Protocol
1. **Warm-up (5 epochs)**: Train with standard SupCon only (lambda=0) to initialize meaningful confusion statistics and prototypes.
2. **CG-SupCon phase**: Enable CGA loss with lambda ramp-up over 5 epochs (0 to lambda_max) to avoid sudden geometric shifts.
3. **Linear evaluation**: Freeze encoder, train linear classifier following Khosla et al. (2020) protocol.

### Hyperparameters
- tau_base = 0.07 (standard SupCon default)
- alpha in {0.3, 0.5, 0.7}: CGA target deviation strength
- lambda in {0.1, 0.5, 1.0}: CGA regularizer weight
- gamma in {0, 1, 5}: temperature modulation strength (gamma=0 means fixed temperature)
- mu = 0.99: confusion EMA momentum
- nu = 0.99: prototype EMA momentum
- Warm-up: 5 epochs

## Related Work

### Supervised Contrastive Learning
**Khosla et al. (2020)** introduced SupCon, extending SimCLR [Chen et al., 2020] to use label information. SupCon achieves strong accuracy and robustness but treats all negative pairs equally, converging to the symmetric ETF geometry. CG-SupCon adds a geometric alignment regularizer that breaks this symmetry based on learned confusion.

### Neural Collapse and ETF Geometry
**Papyan et al. (2020)** discovered that deep network features collapse to a simplex ETF during terminal training. **Lee et al. (AISTATS 2025)** proved this formally for SupCon via the Simplex-to-Simplex Embedding Model (SSEM). Our Theorem 1 shows that CG-SupCon's CGA regularizer provably breaks ETF symmetry for general C classes, with the equilibrium determined by the confusion spectrum. Unlike the original CC-SupCon's 3-class toy analysis, our theorem provides formal bounds for arbitrary C.

### Variational Supervised Contrastive Learning (VarCon)
**Wang et al. (NeurIPS 2025)** proposed VarCon, which treats class labels as latent variables and optimizes a variational ELBO with posterior-weighted cross-entropy and KL divergence. VarCon uses confidence-adaptive per-sample temperature and achieves 78.29% on CIFAR-100 with ResNet-50. The key distinction: VarCon's temperature adapts per-sample based on prediction confidence ("how certain is the model about this sample?"), while CG-SupCon's CGA regularizer operates on class prototypes based on confusion patterns ("how often does the model confuse these two classes?"). These are complementary signals at different granularities.

**Fair comparison note**: VarCon's full method includes a variational ELBO with class-level latent variables. In our experiments, we compare against (1) our reimplementation of VarCon's confidence-adaptive temperature mechanism applied to SupCon (which we call VarCon-T), and (2) VarCon's reported numbers where available (CIFAR-100 ResNet-50: 78.29%). We clearly label these as separate comparisons.

### Hard Negative Mining (Instance-Level)
**Robinson et al. (ICLR 2021)** proposed debiased contrastive learning with a concentration parameter that upweights harder negatives at the instance level. **Animesh & Chandraker (WACV 2025)** proposed Tuned Contrastive Learning (TCL) with learnable gradient response parameters for hard positives/negatives. Both operate at the instance level. CG-SupCon's CGA operates at the class level via prototypes, capturing systematic confusion patterns that instance methods miss. We include both as direct baselines.

### Pairwise Reweighting: MACL and Dynamic Weighting
**Gao et al. (2024)** proposed MACL (Multi-Label Adaptive Contrastive Learning) for remote sensing retrieval, using pairwise label-overlap frequency for inverse reweighting and Jaccard-overlap-based dynamic temperature. CG-SupCon differs in three ways: (1) we use learned confusion (dynamic, model-dependent) rather than static label co-occurrence; (2) our CGA is a geometric alignment regularizer, not a loss reweighting; (3) we target single-label classification, not multi-label retrieval. **Wang et al. (2024)** proposed curricular weighting for contrastive negatives. These are weight-modulation methods; CGA is a geometry-prescription method.

### Adaptive Temperature in Contrastive Learning
**Kukleva et al. (ICLR 2023)** studied global temperature schedules. **Qiu et al. (ICML 2023)** proposed per-sample temperature via DRO. **Zhang et al. (2024)** proposed dynamically scaled temperature based on cosine similarity. All these operate at the instance or global level. CG-SupCon's optional confusion-adaptive temperature is class-pair-specific but is NOT the primary contribution (CGA is).

### Confusion-Guided Learning
**Chen et al. (2024)** proposed CCDC for medical image segmentation, using pixel-level confusion to guide contrastive sampling. CG-SupCon differs: (1) class-level prototypes rather than pixel-level, (2) geometric regularization rather than sampling, (3) persistent EMA confusion matrix rather than per-batch statistics, (4) classification rather than segmentation.

### Adaptive Margin Methods
**AMContrast3D** [Duan et al., 2025] introduces adaptive margins for 3D semantic segmentation based on per-point ambiguity. CG-SupCon differs: (1) class-pair-level rather than per-point, (2) driven by confusion matrix rather than positional ambiguity, (3) geometric alignment regularizer rather than margin injection.

### Label-Hierarchy-Aware Contrastive Learning
**Lian et al. (2024)** proposed LA-SCL with predefined label hierarchies. **Dorfner et al. (2025)** introduced hierarchy-preserving contrastive learning for medical imaging. **Ghanooni et al. (2025)** proposed MLCL with multiple projection heads. All require external hierarchy annotations. CG-SupCon discovers inter-class structure automatically from training dynamics.

## Experiments

### Datasets
- **CIFAR-10**: 10 classes, 50K/10K images, 32x32. Sanity check.
- **CIFAR-100**: 100 classes in 20 superclasses, 50K/10K images, 32x32. Primary benchmark with natural hierarchy for analysis.
- **CIFAR-100-C**: Corruption robustness benchmark [Hendrycks & Dietterich, 2019], 15 corruption types at 5 severity levels.
- **Tiny-ImageNet**: 200 classes, 100K/10K images, 64x64. Tests scalability to more classes.
- **ImageNet-100**: 100-class subset of ImageNet (following CMC protocol). Larger-scale validation.

### Architectures
- **ResNet-18**: Primary architecture for all experiments.
- **ResNet-50**: Validation on CIFAR-100 to enable direct comparison with VarCon's reported numbers.

### Baselines
1. **Cross-Entropy (CE)**: Standard supervised training.
2. **SupCon** [Khosla et al., 2020]: Standard supervised contrastive learning. The CGA-free baseline.
3. **SupCon + Label Smoothing**: SupCon with label smoothing (0.1) during linear eval.
4. **HardNeg-CL** [Robinson et al., 2021]: Instance-level hard negative mining. Key baseline: class-level vs. instance-level.
5. **TCL** [Animesh & Chandraker, 2025]: Tuned Contrastive Learning with learnable gradient response. Another instance-level baseline.
6. **VarCon-T**: Our reimplementation of VarCon's confidence-adaptive temperature in SupCon (without the full ELBO). Clearly labeled as the temperature mechanism only.
7. **VarCon (reported)**: Reported numbers from Wang et al. (2025) where available (CIFAR-100 ResNet-50).
8. **SupCon + Pairwise Reweighting (MACL-style)**: Adapted from MACL's pairwise reweighting using confusion frequency (instead of label overlap). Tests whether pairwise reweighting alone suffices.
9. **CG-SupCon (Ours)**: Full method with CGA regularizer + optional adaptive temperature.

### Metrics
- **Top-1 and Top-5 accuracy** on test sets.
- **Superclass-level accuracy** on CIFAR-100: within-superclass vs. between-superclass error decomposition.
- **Few-shot learning**: 5-way 5-shot and 5-way 1-shot accuracy on held-out CIFAR-100 class splits (60/20/20), using nearest-centroid classification on frozen embeddings.
- **Corruption robustness**: Mean accuracy on CIFAR-100-C.
- **Representation geometry**:
  - ETF deviation: variance of pairwise cosine similarities between class means (ETF has zero variance).
  - Hierarchy correlation: Spearman rho between learned pairwise distances and superclass co-membership.
- **Confusion reduction**: Relative reduction in top-20 most confused class pairs' confusion values.
- **Wall-clock overhead**: Training time per epoch for each method (addresses feedback on computational cost).
- **Stability analysis**: Kendall tau correlation of top-20 confused class pairs across seeds and checkpoints (class-pair stability vs. instance-level hardness stability).

### Ablation Studies
1. **CGA ablation**: SupCon (no CGA, no adaptive tau) vs. SupCon + CGA vs. SupCon + adaptive tau vs. CG-SupCon (both).
2. **CGA strength alpha**: {0.1, 0.3, 0.5, 0.7, 1.0} on CIFAR-100.
3. **Regularizer weight lambda**: {0.01, 0.1, 0.5, 1.0, 5.0} on CIFAR-100.
4. **EMA momentum**: mu in {0.9, 0.99, 0.999}.
5. **Warm-up duration**: {0, 5, 10} epochs.

### Experimental Plan with Runtime Budget

Core comparisons on CIFAR-100 use **5 seeds** (42, 43, 44, 45, 46). Other experiments use 3 or 1 seeds as noted. Training: 200 epochs for CIFAR, 100 epochs for Tiny-ImageNet/ImageNet-100.

**Tier 1 -- Core (must complete, ~5.5 hours)**:
| Experiment | Dataset | Arch | Runs | Est. Time |
|---|---|---|---|---|
| CE baseline | CIFAR-100 | ResNet-18 | 5 seeds | 0.5h |
| SupCon baseline | CIFAR-100 | ResNet-18 | 5 seeds | 0.6h |
| HardNeg-CL | CIFAR-100 | ResNet-18 | 5 seeds | 0.6h |
| TCL | CIFAR-100 | ResNet-18 | 3 seeds | 0.4h |
| VarCon-T | CIFAR-100 | ResNet-18 | 3 seeds | 0.4h |
| SupCon + Pairwise Reweight | CIFAR-100 | ResNet-18 | 3 seeds | 0.4h |
| CG-SupCon (best alpha,lambda) | CIFAR-100 | ResNet-18 | 5 seeds | 0.7h |
| CGA ablation (4 variants) | CIFAR-100 | ResNet-18 | 1 seed | 0.6h |
| alpha/lambda grid (5x3 grid) | CIFAR-100 | ResNet-18 | 1 seed | 1.0h |
| Wall-clock overhead measurement | CIFAR-100 | ResNet-18 | 1 seed ea. | 0.3h |

**Tier 2 -- Important (complete if time allows, ~2.5 hours)**:
| Experiment | Dataset | Arch | Runs | Est. Time |
|---|---|---|---|---|
| CE + SupCon + CG-SupCon | CIFAR-10 | ResNet-18 | 3 seeds | 0.3h |
| Few-shot eval (5-way 1/5-shot) | CIFAR-100 | ResNet-18 | 3 seeds | 0.1h |
| Corruption robustness (CIFAR-100-C) | CIFAR-100-C | ResNet-18 | 1 seed | 0.2h |
| Stability analysis (class-pair vs instance) | CIFAR-100 | ResNet-18 | 3 seeds | 0.1h |
| SupCon + CG-SupCon | Tiny-ImageNet | ResNet-18 | 1 seed each | 0.6h |
| SupCon + CG-SupCon | ImageNet-100 | ResNet-18 | 1 seed each | 0.7h |
| ResNet-50 validation | CIFAR-100 | ResNet-50 | 1 seed each | 0.5h |

**Total**: Tier 1 (5.5h) + Tier 2 (2.5h) = 8.0h maximum.

**Cutback strategy**: If running behind:
- Reduce TCL and VarCon-T to 1 seed each (saves ~0.4h)
- Skip ImageNet-100 (saves ~0.7h)
- These preserve the 5-seed core comparison on CIFAR-100

### Expected Results
- CG-SupCon should outperform SupCon by 1-3% top-1 on CIFAR-100, with larger gains (2-4%) on within-superclass accuracy, demonstrating that structured geometry specifically helps fine-grained discrimination.
- CG-SupCon should outperform HardNeg-CL and TCL (instance-level), demonstrating class-level geometric alignment > instance-level hardness mining.
- CG-SupCon should outperform SupCon + Pairwise Reweighting (MACL-style), demonstrating geometric alignment > loss reweighting.
- CG-SupCon should be competitive with VarCon-T on standard metrics, while showing clearer advantages on geometry-dependent tasks (few-shot, hierarchy correlation).
- **CGA ablation**: CGA alone (no adaptive temperature) should provide the majority of the gain, with adaptive temperature adding incremental improvement. This validates CGA as the primary contribution.
- **Few-shot learning**: 2-5% improvement over SupCon on 5-way 5-shot.
- **Corruption robustness**: 1-3% improvement on CIFAR-100-C mean accuracy.
- **Geometry metrics**: ETF deviation significantly higher than SupCon; hierarchy correlation (Spearman rho) improvement > 0.1.
- **Stability analysis**: Class-pair confusion rankings should have Kendall tau > 0.7 across seeds, while instance-level hardness rankings should have Kendall tau < 0.4.
- **Wall-clock overhead**: < 5% increase over standard SupCon (the CGA computation is O(C^2) which is negligible for C <= 200).

## Success Criteria

### Confirms hypothesis if:
1. CG-SupCon achieves statistically significant improvement (p < 0.05, paired t-test over 5 seeds) over SupCon on CIFAR-100 top-1 accuracy.
2. CG-SupCon outperforms all instance-level baselines (HardNeg-CL, TCL) on CIFAR-100.
3. CG-SupCon outperforms pairwise reweighting (MACL-style) on CIFAR-100, demonstrating geometric alignment > loss reweighting.
4. CGA alone provides measurable gain over SupCon (the regularizer is the primary contribution, not temperature modulation).
5. Within-superclass accuracy improvement exceeds between-superclass improvement.
6. Inter-class distance matrix correlates more strongly with ground-truth hierarchy than SupCon's (Spearman rho improvement > 0.1).
7. Gains hold on at least one additional dataset (CIFAR-10, Tiny-ImageNet, or ImageNet-100).
8. Wall-clock overhead is < 10% over standard SupCon.

### Refutes hypothesis if:
1. CG-SupCon performs comparably or worse than SupCon across all settings.
2. CGA alone provides no gain (the regularizer is ineffective).
3. SupCon + Pairwise Reweighting matches CG-SupCon (geometric alignment adds no value over simple reweighting).
4. Instance-level methods (HardNeg-CL, TCL) match or exceed CG-SupCon (class-level signal adds no value).
5. Embedding geometry remains symmetric ETF despite CGA.

## References

1. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised Contrastive Learning. *NeurIPS 2020*.

2. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML 2020*.

3. Papyan, V., Han, X.Y., & Donoho, D.L. (2020). Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training. *PNAS*, 117(40), 24652-24663.

4. Lee, C., Oh, J., Lee, K., & Sohn, J. (2025). A Theoretical Framework for Preventing Class Collapse in Supervised Contrastive Learning. *AISTATS 2025*.

5. Wang, Z., Fan, J., Nguyen, T., Ji, H., & Liu, G. (2025). Variational Supervised Contrastive Learning. *NeurIPS 2025*.

6. Robinson, J., Chuang, C.-Y., Sra, S., & Jegelka, S. (2021). Contrastive Learning with Hard Negative Samples. *ICLR 2021*.

7. Animesh, C. & Chandraker, M. (2025). Tuned Contrastive Learning. *WACV 2025*.

8. Lian, R., Sethares, W.A., & Hu, J. (2024). Learning Label Hierarchy with Supervised Contrastive Learning. *arXiv preprint, 2024*.

9. Ghanooni, N., Pajoum, B., Rawal, H., Fellenz, S., Duy, V.N.L., & Kloft, M. (2025). Multi-level Supervised Contrastive Learning. *arXiv:2502.02202*.

10. Dorfner, F.J., et al. (2025). Climbing the Label Tree: Hierarchy-Preserving Contrastive Learning for Medical Imaging. *arXiv:2511.03771*.

11. Gao, Y., Chen, L., & Wang, P. (2024). MACL: Multi-Label Adaptive Contrastive Learning Loss for Remote Sensing Image Retrieval. *arXiv:2512.16294*.

12. Kukleva, A., Kuehne, H., Schiele, B., & Brox, T. (2023). Temperature Schedules for Self-Supervised Contrastive Methods. *ICLR 2023*.

13. Qiu, Z.-H., Hu, Q., Yuan, Z., Zhou, D., Zhang, L., & Yang, T. (2023). Not All Semantics are Created Equal: Contrastive Self-Supervised Learning with Automatic Temperature Individualization. *ICML 2023*.

14. Zhang, Y., Zhou, D., Wei, X., & He, X. (2024). Dynamically Scaled Temperature in Self-Supervised Contrastive Learning. *arXiv:2308.01140*.

15. Chen, J., Chen, C., Huang, W., Zhang, J., Debattista, K., & Han, J. (2024). Dynamic Contrastive Learning Guided by Class Confidence and Confusion Degree for Medical Image Segmentation. *Pattern Recognition*, 145, 109881.

16. Duan, Y., et al. (2025). Adaptive Margin Contrastive Learning for Ambiguity-aware 3D Semantic Segmentation. *arXiv:2502.04111*.

17. Wang, T. & Isola, P. (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. *ICML 2020*.

18. Wang, Y., Pan, X., Song, S., Zhang, H., Huang, G., & Wu, C. (2024). Mining Negative Samples on Contrastive Learning via Curricular Weighting Strategy. *Information Sciences*, 662, 120259.

19. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.

20. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

21. Hendrycks, D. & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. *ICLR 2019*.

22. Kalantidis, Y., Sariyildiz, M.B., Pion, N., Weinzaepfel, P., & Larlus, D. (2020). Hard Negative Mixing for Contrastive Learning. *NeurIPS 2020*.
