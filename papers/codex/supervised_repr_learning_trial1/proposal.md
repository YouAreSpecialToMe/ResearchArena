# Points or Spans? A Benchmark-and-Analysis Study of Representative Families in Supervised Contrastive Learning

## Introduction

Supervised contrastive learning (SupCon) is a strong supervised representation-learning baseline, but several papers now show that it can collapse latent subclasses inside a class, which hurts coarse-to-fine transfer and robustness. That general failure mode is no longer novel. The open question is narrower: once we accept that one representative per class can be too restrictive, should we model a class with multiple points or with low-rank spans?

This proposal is explicitly a benchmark-and-analysis paper, not a new objective-family paper. Its central claim is deliberately limited:

> Under matched supervised-contrastive training, low-rank span representatives help over point representatives only when within-class structure is primarily directional and low-rank rather than cluster-like.

That question is not already answered by the closest prior work. PSC and MPSC show that multiple point representatives can help, but they do not isolate points versus spans under matched training. CLOP studies orthonormal prototypes and class-level subspaces to prevent collapse, but it is about class separation and collapse prevention rather than explicit within-class multi-mode modeling. Perfectly Balanced shows that subclass-preserving SupCon matters, but it changes the objective with weighted InfoNCE and an autoencoder, so it does not identify whether the missing ingredient is point representatives, span representatives, or the auxiliary objective. SBCL introduces dynamic subclass clustering for long-tailed recognition, again combining a different task setting with a different training recipe rather than isolating representative family choice.

The paper therefore contributes three things:

1. A controlled benchmark that compares SupCon, PSC, MPSC, a CLOP-style class-control baseline, and span representatives inside one matched training template.
2. A theory-guided prediction that links the points-versus-spans outcome to whether latent subclass variation survives normalization as directional low-rank energy or mostly as mean separation.
3. A scope that is credible on one RTX A6000 in about eight hours by focusing on one primary real benchmark, one synthetic benchmark, and only minimal sanity checks.

If spans do not beat MPSC under the predicted regime, that is still publishable evidence for the benchmark question: it would show that point representatives are already sufficient in the small-scale supervised setting studied here.

## Proposed Approach

### Common representative-family framework

Let `h_i = f_theta(x_i) in R^d` be the pre-normalized embedding and `z_i = h_i / ||h_i||_2` the normalized embedding used by contrastive learning. For each class `y` and representative index `k in {1, ..., K}`, define a score `s_{y,k}(z)` and class logit

`ell_y(z) = tau_c log sum_k exp(s_{y,k}(z) / tau_c)`.

Class probabilities are computed for all classes, not only the ground-truth class:

`pi_y(z) = exp(ell_y(z)) / sum_{y'} exp(ell_{y'}(z))`.

This gives a well-defined classification term for negative classes under every representative family. The training loss is

`L = L_ce(pi, y) + lambda_supcon L_wsupcon + lambda_reg L_reg`.

The weighted SupCon part uses only same-label assignment agreement to modulate positive pairs:

`q_{i,k} = softmax_k(s_{y_i,k}(z_i) / tau_a),  w_{ij} = q_i^T q_j`.

Negatives in SupCon remain all samples from other labels, exactly as in the standard loss. The representative family only changes how `q_i` is formed for same-label samples and how class logits are scored for all classes.

Compared methods:

- `SupCon`: no persistent representatives; `w_{ij}=1` for same-label pairs and `ell_y` comes from a linear classifier head.
- `PSC`: one point per class, `s_{y,1}(z)=z^T p_y`.
- `MPSC`: multiple points per class, `s_{y,k}(z)=z^T p_{y,k}`.
- `CLOP-style control`: one orthonormal point or class-level target per class, included as a collapse-prevention control rather than a within-class multi-mode model.
- `Span`: multiple rank-`r` projectors per class, `s_{y,k}(z)=z^T P_{y,k} z = ||U_{y,k}^T z||_2^2`, where `P_{y,k}=U_{y,k}U_{y,k}^T`.

The span model uses only mild regularization:

- `L_div`: `sum_y sum_{k<k'} ||U_{y,k}^T U_{y,k'}||_F^2` to discourage duplicate spans.
- `L_cov`: a weak entropy or coverage penalty that prevents permanent dormant spans without forcing equal occupancy.

There is no mandatory occupancy balancing, no repeated reset mechanism, and no method-specific auxiliary autoencoder. That is intentional: the study aims to isolate representative family, not the effect of stronger subclass-discovery heuristics.

### Why prior work does not already answer the question

The literature gap needs to be precise.

- `PSC/MPSC` already occupy the multi-prototype supervised-contrastive space, but they answer a memory-efficiency and long-tailed feature-learning question. They do not compare multi-point versus multi-span representatives under matched losses, matched representative counts, and matched backbone/training schedules.
- `CLOP` already shows that orthonormal prototypes or subspaces can prevent collapse, but it studies class-level separation with one target per class or class-level subspaces. It does not test whether multiple low-rank spans beat multiple points for preserving latent subclasses.
- `Perfectly Balanced` already studies coarse-to-fine transfer and subclass preservation, but it changes the mechanism through weighted class-conditional InfoNCE and a class-conditional autoencoder. It therefore cannot attribute gains to points versus spans.
- `SBCL` already uses subclass-aware contrastive structure, but in a long-tailed regime with dynamic clustering of head classes into pseudo-subclasses. It does not hold the training template fixed and vary only representative family.

The paper's novelty is therefore not "multiple representatives" or "subspaces in contrastive learning." It is the controlled, theory-guided comparison of points versus spans.

### Theory-guided prediction with normalized embeddings

The previous draft used a latent Gaussian/subspace model on unnormalized features but did not explain why it still says something about normalized contrastive embeddings. The revised proposal makes that approximation explicit.

Assume that before normalization, subclass `m` of class `y` generates

`h = mu_{y,m} + U^*_{y,m} a + eps`,

with `U^*_{y,m} in R^{d x r_*}` orthonormal, `a ~ N(0, diag(lambda_1, ..., lambda_{r_*}))`, and `eps ~ N(0, sigma^2 I)`.

Contrastive learning uses `z = h / ||h||_2`. Under norm concentration, i.e. when `||h||_2` has small relative variance inside a subclass, we may write `||h||_2 approx rho_{y,m}` and obtain the first-order approximation

`z approx h / rho_{y,m}`.

Then

`E[z] approx mu_{y,m} / rho_{y,m}`,

and

`E[zz^T] approx (mu_{y,m}mu_{y,m}^T + U^* diag(lambda) U^{*T} + sigma^2 I) / rho_{y,m}^2`.

This does not claim an exact normalized-Gaussian theory. It claims that, when subclass norms are concentrated, normalization mostly rescales both mean information and directional covariance information without changing which signal source dominates.

For a point representative `p` with `||p||_2=1`, the expected score is approximately

`E[s_point(z;p)] = E[z^T p] approx (mu_{y,m}^T p) / rho_{y,m}`.

For a span projector `P` of rank `r`,

`E[s_span(z;P)] = E[z^T P z] approx (||P mu_{y,m}||_2^2 + tr(P U^* diag(lambda) U^{*T}) + sigma^2 tr(P)) / rho_{y,m}^2`.

So the ordering between representative families depends on where subclass signal lives after normalization:

- If subclasses differ mainly in means `mu_{y,m}`, point representatives should be competitive.
- If subclasses share similar means but differ in a few dominant directions, the span score captures the directional energy term `tr(P U^* diag(lambda) U^{*T})` that points cannot.
- If variation is nearly isotropic, all rank-`r` projectors collect about the same `r sigma^2 / rho^2`, so spans should lose their advantage.

This is the paper's actual prediction: spans help over points when low-rank directional variation survives normalization strongly enough to matter more than subclass-mean separation.

## Related Work

### Supervised contrastive learning and class collapse

Khosla et al. introduced SupCon as the base training objective. Xue et al. analyzed why simplicity bias can drive supervised contrastive learning toward class collapse. Lee et al. provided a recent theoretical framework for preventing class collapse. These papers justify why subclass-preserving evaluation matters, but none compare representative families.

### Point representatives: PSC and MPSC

Wang et al. introduced PSC in a hybrid long-tailed classification framework and explicitly discussed multi-prototype extensions. This is the closest prior art, and it removes any claim that "multiple supervised representatives" is novel. The present paper instead asks whether, under matched training, replacing those points by low-rank spans changes the regime in which subclass structure is retained.

### Class-level subspace controls: CLOP

Li et al. showed that orthonormal prototypes and subspaces can prevent collapse in contrastive learning. That makes CLOP an essential control baseline. But its design goal is different: prevent class-level collapse through orthogonal class targets. It does not benchmark multiple within-class spans against multiple within-class points.

### Subclass-preserving objectives: Perfectly Balanced and SBCL

Perfectly Balanced shows that SupCon's subclass collapse can be mitigated by controlling class-conditional spread and breaking permutation invariance. SBCL shows that subclass-aware structure helps in long-tailed recognition by dynamically clustering head classes. These papers demonstrate that subclass preservation matters, but they do not isolate the representative family question because they change the objective, task regime, or pseudo-labeling mechanism simultaneously.

## Experiments

### Primary benchmark question

Under matched supervised-contrastive training, when do low-rank spans help over point representatives?

### Benchmarks

The study uses one synthetic benchmark and one primary real benchmark.

1. `Synthetic latent-mode benchmark`
   A feature-space benchmark with `C=10` classes and `M=2` latent subclasses per class. Each subclass is generated from the normalized low-rank model above. Two regimes are mandatory:
   - `Directional-low-rank`: weak mean separation, strong low-rank directional energy.
   - `Mean-separated / isotropic`: stronger mean separation or nearly isotropic within-class variation.

   This benchmark directly tests the theory prediction.

2. `CIFAR-100 coarse-to-fine transfer`
   Train the encoder using the 20 coarse labels, freeze it, then evaluate on the 100 fine labels with linear probe and `20`-NN. This is the main real-data benchmark because it directly measures whether latent fine-grained structure is preserved.

3. `Fine-label sanity check`
   A small sanity check on standard CIFAR-100 fine-label training is run only for `SupCon`, `MPSC`, and `Span` to confirm that any transfer gain is not bought by a large drop in ordinary discriminative quality. This is not treated as a full benchmark in the idea stage.

### Mandatory methods

The primary coarse-to-fine benchmark uses:

- Cross-Entropy
- SupCon
- PSC
- MPSC
- CLOP-style control
- Span

The synthetic benchmark uses the same set so the representative-family story stays aligned across theory and practice.

### Metrics

Synthetic:

- AMI for subclass recovery
- recovered-mode accuracy
- principal-angle error between learned and true latent spans
- gap between directional-low-rank and isotropic regimes

Real:

- coarse-to-fine linear-probe accuracy
- coarse-to-fine `20`-NN accuracy
- coarse-label training accuracy
- within-class effective rank on fine labels after coarse-label training

Sanity:

- fine-label top-1 accuracy for `SupCon`, `MPSC`, and `Span`

### Compute-constrained protocol

The resource story is intentionally reduced to fit one RTX A6000 and about eight hours total.

- Backbone: `ResNet-18`
- Embedding dimension: `128`
- Projection head: `2`-layer MLP
- Mixed precision throughout
- Batch size: `256`
- Coarse-to-fine training: `30` epochs
- Fine-label sanity runs: `20` epochs
- Synthetic benchmark: small MLP or direct feature optimization

Seed policy:

- `3` seeds for every result in the synthetic benchmark
- `3` seeds for every method on the primary CIFAR-100 coarse-to-fine benchmark
- `1` seed each for the fine-label sanity runs

Expected run budget:

- `6 methods x 3 seeds = 18` primary real runs at roughly `12-15` minutes each on CIFAR-100 ResNet-18 with AMP: about `3.6-4.5` GPU-hours.
- `6 methods x 2 synthetic regimes x 3 seeds = 36` synthetic runs at roughly `1-2` minutes each: about `0.6-1.2` GPU-hours.
- `3` fine-label sanity runs at roughly `12-15` minutes each: about `0.6-0.75` GPU-hours.
- One shared pilot sweep for temperatures and `lambda_reg` on a single seed: about `1` GPU-hour.

Total expected budget: about `5.8-7.5` GPU-hours, leaving limited but plausible slack for smoke tests and CPU overhead.

### Minimal ablations

Only one ablation is mandatory in the idea stage:

- `Span rank`: `r=1` versus `r=2`, one seed each on the synthetic directional-low-rank regime.

No broader grid over `K`, resets, or balancing heuristics is part of the acceptance-critical package.

## Success Criteria

The hypothesis is supported if:

1. On the synthetic directional-low-rank regime, `Span` beats `MPSC` on at least two recovery metrics, including either AMI or principal-angle error.
2. On the synthetic mean-separated or isotropic regime, that advantage shrinks materially or disappears.
3. On CIFAR-100 coarse-to-fine transfer, `Span` improves over `SupCon` by at least `0.5` point on linear probe or `20`-NN accuracy.
4. On the same transfer metric, `Span` is no worse than the better of `PSC` and `MPSC` by more than `0.3` point.
5. On the fine-label sanity check, `Span` stays within `0.5` point of `SupCon`.
6. The main result does not rely on occupancy forcing, repeated resets, or another auxiliary objective that changes the benchmark question.

The hypothesis is weakened if `Span` only beats `SupCon` but not `MPSC`, or if the synthetic regime distinction is absent. It is refuted if `MPSC` matches or beats `Span` in the directional-low-rank regime on both synthetic recovery and coarse-to-fine transfer.

## References

- Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. "Supervised Contrastive Learning." Advances in Neural Information Processing Systems 33, 2020.
- Peng Wang, Kai Han, Xiu-Shen Wei, Lei Zhang, and Lei Wang. "Contrastive Learning Based Hybrid Networks for Long-Tailed Image Classification." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.
- Huanran Li, Manh Nguyen, and Daniel Pimentel-Alarcón. "Preventing Collapse in Contrastive Learning with Orthonormal Prototypes." arXiv:2403.18699, 2024.
- Mayee Chen, Daniel Y. Fu, Avanika Narayan, Michael Zhang, Zhao Song, Kayvon Fatahalian, and Christopher Ré. "Perfectly Balanced: Improving Transfer and Robustness of Supervised Contrastive Learning." Proceedings of the 39th International Conference on Machine Learning, PMLR 162, 2022.
- Yihao Xue, Siddharth Joshi, Eric Gan, Pin-Yu Chen, and Baharan Mirzasoleiman. "Which Features are Learnt by Contrastive Learning? On the Role of Simplicity Bias in Class Collapse and Feature Suppression." Proceedings of the 40th International Conference on Machine Learning, PMLR 202, 2023.
- Chungpa Lee, Jeongheon Oh, Kibok Lee, and Jae-yong Sohn. "A Theoretical Framework for Preventing Class Collapse in Supervised Contrastive Learning." arXiv:2503.08203, 2025.
- Chengkai Hou, Jieyu Zhang, Haonan Wang, and Tianyi Zhou. "Subclass-balancing Contrastive Learning for Long-tailed Recognition." Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.
