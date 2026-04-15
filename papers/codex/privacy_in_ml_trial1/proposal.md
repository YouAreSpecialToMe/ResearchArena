# ASPIRE-Unlearn: Privacy-Targeted Basis Selection Under Independent Auditing for Approximate Unlearning

## Introduction

Approximate machine unlearning is attractive because exact retraining is often too expensive, but recent privacy work shows that deletion can create a second harm: after the forget set is removed, some retained users become easier to identify. Chen et al. showed that approximate unlearning can worsen privacy leakage for retained data, Carlini et al. explained why relative memorization can expose new members after partial deletion, and Naderloui et al. argued that common unlearning evaluations still under-measure privacy risk because attack calibration and target selection are often too weak.

This proposal therefore makes a deliberately narrow claim. It does **not** introduce a new unlearning family. It studies a specific extension of GU/OrthoGrad-style retain-projection unlearning:

1. choose the protected retain basis from a retained slice predicted to be privacy-vulnerable rather than from retain geometry globally; and
2. evaluate the method under an independently held-out audit protocol in which stopping and final privacy reporting use different shards.

The target contribution is empirical and methodological: determine whether **privacy-targeted basis selection under independent auditing** improves retained-user privacy beyond untargeted retain projection at matched forgetting quality.

**Problem statement.** Existing retain-projection methods protect average retain geometry. They do not directly ask whether the retained examples most exposed to post-unlearning membership inference should define the protected subspace, nor whether any apparent gain survives a protocol that never reuses the final audit shard for stopping.

**Key insight.** Retained privacy degradation is likely concentrated in a small subset of retained examples whose gradients align strongly with forget-induced update directions. If that is true, then a low-rank basis built from that vulnerable retained slice should be more privacy-aligned than a global retain basis of similar rank.

**Hypothesis.** Relative to short-horizon gradient unlearning and to GU-/OrthoGrad-style untargeted retain protection, a privacy-targeted protected basis built from a disjoint selector split will lower retained-set LiRA leakage at comparable forgetting quality, and the gain will still appear when checkpoints are selected only on `R_audit_A` and reported only on untouched `R_audit_B`.

## Proposed Approach

### Positioning and novelty boundary

ASPIRE-Unlearn should be reviewed as a narrow extension to projection-based unlearning, not as a broad method family. The novelty boundary is:

- projection is not new;
- per-sample orthogonalization is not new;
- low-rank retain protection is not new;
- the proposed contribution is **which retained examples define the protected basis**, and **how the privacy claim is audited**.

This is more than a selector heuristic on top of GU because the selector changes the actual optimization constraint. GU protects a basis estimated from retain geometry globally. ASPIRE protects a basis estimated only from the retained slice predicted to be privacy-vulnerable. If both methods use the same update budget, the same checkpoint rule, and the same rank cap, any difference is attributable to basis construction rather than to extra compute or looser auditing.

### Dataset-specific split protocol

The protocol is redesigned to make the attack-only pool and matched non-member pools concrete.

#### CIFAR-10

- Start from the canonical `50,000` training images and `10,000` test images.
- Reserve `15,000` training images as a fully disjoint `A_pool` used only for LiRA shadow-model training.
- Use the remaining `35,000` training images as `D_target`, the only pool from which target-model train, forget, retain, selector, and audit splits are drawn.
- Split the canonical test set into:
  - `V_pool = 4,000` images used only to form matched non-member pools `V_audit_A` and `V_audit_B`;
  - `T_eval = 6,000` images used only for test accuracy and calibration reporting.

#### Purchase100

- Because there is no universally fixed public train/test protocol for the privacy study needed here, make one stratified split of the full dataset into:
  - `D_target = 70%` for the target model only;
  - `A_pool = 10%` for LiRA shadow-model training only;
  - `V_pool = 10%` for matched non-member pools only;
  - `T_eval = 10%` for utility evaluation only.

#### Target-model retained split

Within `D_target`, define the forget set first. From the remaining retained examples, create:

- `R_train = 80%`
- `R_select = 10%`
- `R_audit_A = 5%`
- `R_audit_B = 5%`

using class-stratified sampling.

For each dataset and seed, create matched non-member pools by sampling from `V_pool` with the same size and class histogram as the corresponding retained audit shard:

- `V_audit_A` matches `R_audit_A`
- `V_audit_B` matches `R_audit_B`

No example may appear in more than one of `A_pool`, `V_pool`, `T_eval`, `D_target`, `R_select`, `R_audit_A`, or `R_audit_B`.

### 1. Vulnerability score

Let `g_f(i)` be the forget-example gradient of the base unlearning loss on a restricted parameter scope: classifier head for Purchase100, and final residual block plus classifier for CIFAR-10. Estimate the forget covariance

`C_f = (1/m) sum_i (g_f(i) - mean(g_f))(g_f(i) - mean(g_f))^T`.

For each retained example `x` in `R_select`, compute

`s(x) = g_r(x)^T C_f g_r(x)`.

This score is fixed before experiments and does not use any membership-attack outputs. It measures first-order sensitivity of a retained example to forget-dominated update directions.

### 2. Privacy-targeted basis selection

Rank `R_select` by `s(x)` and keep the top `q` fraction, with `q` fixed before the full sweep. The default is `q = 25%`, but the paper will report one matched-size ablation to test whether the gain comes from the vulnerability score rather than from simply using a smaller subset.

Using only the selected examples, form

`F_sel = sum_i w_i g_r(x_i) g_r(x_i)^T`, where `w_i proportional to s(x_i)`.

Let `U_priv` be the smallest-rank eigenspace whose cumulative weighted energy exceeds `tau = 0.90`, with rank capped at `16` for Purchase100 and `32` for CIFAR-10.

The method distinction is now explicit:

- **GU**: protect a global retain basis.
- **OrthoGrad**: orthogonalize each update against current retain gradients.
- **ASPIRE**: protect a static low-rank basis built only from the retained slice predicted to be privacy-vulnerable.

### 3. Projected unlearning update

Given a base unlearning update `u_t`, apply

`u_t' = u_t - lambda P_{U_priv} u_t`

with `lambda = 1.0`.

Add a weak retained anchor on the selected slice only:

`L_anchor = mean_i KL(p_theta0(x_i) || p_theta(x_i))`.

The anchor is intentionally weak and explicitly secondary. The core claim is about the protected basis, not about adding another retain loss.

### 4. Independent audit rule

Checkpoint selection is pre-registered and common to all approximate methods:

1. define a forgetting band from retrain-on-retain oracles on calibration seeds `[7, 17]`;
2. consider only checkpoints that lie inside that band;
3. among eligible checkpoints, choose the earliest checkpoint minimizing retained-member LiRA `TPR@1%FPR` on `R_audit_A` versus `V_audit_A`.

`R_audit_B` and `V_audit_B` are never used for:

- stopping;
- threshold fitting;
- selector construction;
- hyperparameter choice;
- final model choice.

This is the protocol contribution: the final privacy number is produced on a shard that was never involved in method tuning or checkpointing.

### 5. LiRA protocol

The paper uses standard Gaussian LiRA as the primary privacy evaluation for retained membership inference after unlearning.

Implementation details:

- fixed attack score: true-class log-loss for Purchase100, and mean true-class log-loss over `4` deterministic augmentations for CIFAR-10;
- `8` shadow models per dataset trained only on `A_pool`;
- Gaussian in/out distributions fit only from shadow-model scores;
- `1%` FPR threshold calibrated only from shadow non-member scores.

To avoid overclaiming, the paper will make **no headline claim at `0.1%` FPR**. If reported at all, `TPR@0.1%FPR` will be appendix-only and explicitly marked exploratory.

## Related Work

### Privacy failures after unlearning

Chen et al., *When Machine Unlearning Jeopardizes Privacy* (2021), showed that approximate unlearning can increase membership leakage. Carlini et al., *The Privacy Onion Effect: Memorization is Relative* (2022), explained why removing one memorization layer can expose another. Naderloui et al., *Rectifying Privacy and Efficacy Measurements in Machine Unlearning* (2025), argued that stronger auditing and target selection are needed because weak protocols can understate privacy failures.

ASPIRE is motivated by this line of work but is defense-side and narrowly scoped to retained privacy under approximate unlearning.

### Closest basis-protection methods

Huang et al., *Unified Gradient-Based Machine Unlearning with Remain Geometry Enhancement* (2024), and Zhou et al., *Geometric-disentangelment Unlearning* (2025), are the closest global retain-geometry baselines. Shamsian et al., *Go Beyond Your Means: Unlearning with Per-Sample Gradient Orthogonalization* (2025), are also very close: OrthoGrad orthogonalizes the forget update against retain gradients and is designed for limited-retain settings.

These papers define the true novelty boundary. ASPIRE differs only in the basis source:

- versus GU: global retain basis vs privacy-targeted retained slice;
- versus OrthoGrad: per-batch orthogonalization vs static low-rank basis from a disjoint selector split;
- versus remain-geometry methods broadly: retain-utility protection vs retained-privacy-targeted protection.

### Privacy-aware evaluation and defenses

Maheri et al., *WARP: Weight Teleportation for Attack-Resilient Unlearning Protocols* (2025/2026), show that privacy-aware protocol design and defense-side interventions matter for approximate unlearning. ASPIRE is much smaller in scope: it does not introduce teleportation or a new attack model, but it adopts the lesson that privacy claims should survive a clean audit pipeline.

AMUN and related privacy-aware unlearning papers are relevant as defense-side context, but they do not study the specific question here: whether a protected retain basis should be built from examples predicted to be retained-membership vulnerable.

### Membership inference methodology

Carlini et al., *Membership Inference Attacks From First Principles* (2022), established LiRA and emphasized low-FPR evaluation. That is the correct attack family here because the paper's contribution is not a new attack, but a defense-side change evaluated under a stronger audit protocol.

## Experiments

### Scope

The experimental matrix is intentionally small enough for `1x RTX A6000`, `60 GB` RAM, `4` CPU cores, and an `~8` hour cap.

### Datasets and models

- **Purchase100** with a 2-layer MLP
- **CIFAR-10** with ResNet-18

These are appropriate because they support repeated unlearning, shadow training, and independent auditing within budget.

### Methods

- Base short-horizon gradient unlearning
- Base + **GU-style global retain projection**
- Base + **OrthoGrad-style orthogonalization**
- Base + **ASPIRE-Unlearn**
- Retrain-on-retain oracle for forgetting-band calibration

### Deletion regimes

1. **Primary regime on both datasets:** random `1%` point deletion
2. **Stress regime on CIFAR-10 only:** high-loss `5%` deletion, where forget examples are the highest-loss target-training points under the pretrained model

### Seeds

- Main target seeds: `[7, 17, 27]`
- Retrain calibration seeds: `[7, 17]`

The seed counts are fixed across methods.

### Metrics

Primary:

- retained-member LiRA `TPR@1%FPR` on `R_audit_B` versus `V_audit_B`

Secondary:

- retained-member LiRA AUC on `R_audit_B`
- forget accuracy and forget log-loss increase
- test accuracy on `T_eval`
- retained accuracy on `R_audit_B`
- runtime overhead and peak memory

### Critical ablations

The ablations are designed to directly test the narrow novelty claim.

1. **Targeted-score vs matched-size untargeted subset.** Replace the top-`s(x)` selector with a uniform class-stratified subset from `R_select` of the same size, keep the same weighting rule, the same rank cap, and the same audit rule. If ASPIRE still wins, the gain is tied to the vulnerability score rather than to using any reduced-rank subset.
2. **No-anchor ablation.** Set the anchor weight to zero while keeping the targeted basis unchanged. If performance stays similar, the effect comes from basis construction rather than an auxiliary loss.

Both ablations run only on the hardest regime, CIFAR-10 high-loss `5%` deletion, to stay within budget.

### Expected outcomes

Expected positive result:

- ASPIRE beats the base method on retained privacy at matched forgetting on both datasets.
- ASPIRE beats at least one of GU or OrthoGrad on retained privacy in the main settings.
- The largest gain appears in the CIFAR-10 stress regime.

Useful negative result:

- If ASPIRE only helps on `R_audit_A` but not on `R_audit_B`, the paper still makes a publishable protocol point: the privacy-targeted basis does not survive independent auditing.

### Runtime budget

Planned total runtime on one RTX A6000:

- target and retrain training: about `150` minutes
- `8` LiRA shadow models across the two datasets: about `90` minutes
- core unlearning matrix: about `150` minutes
- CIFAR-10 stress-regime ablations: about `45` minutes
- LiRA fitting, bootstrap intervals, and export: about `30` minutes

Total planned time: about **465 minutes** (`7.75` hours).

This is tight but feasible because the models are small, the benchmark set is narrow, and the ablations are restricted to one stress regime.

## Success Criteria

The hypothesis is supported if all of the following hold on final `R_audit_B` results:

1. ASPIRE lowers retained-member LiRA `TPR@1%FPR` relative to the base method on both random-deletion settings while staying inside the retrain-calibrated forgetting band.
2. ASPIRE beats at least one of GU or OrthoGrad on both random-deletion settings and is not more than `5%` worse than the stronger of them on either setting.
3. On CIFAR-10 high-loss `5%` deletion, ASPIRE shows its largest relative privacy gain.
4. The targeted-score ablation performs worse than full ASPIRE, showing that the gain is not explained by any reduced-rank retained subset.
5. Runtime stays within `1.35x` of the base method.

The hypothesis is weakened if gains appear only on `R_audit_A` or disappear once `R_audit_B` is used as the only final reporting shard. It is refuted if GU or OrthoGrad match ASPIRE once the same independent audit protocol is enforced, or if the matched-size untargeted-subset ablation performs comparably to the full targeted selector.

## References

Chen, M., Zhang, Z., Wang, T., Backes, M., Humbert, M., & Zhang, Y. (2021). *When Machine Unlearning Jeopardizes Privacy*. arXiv:2005.02205.

Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022). *Membership Inference Attacks From First Principles*. IEEE Symposium on Security and Privacy.

Carlini, N., Jagielski, M., Zhang, C., Papernot, N., Terzis, A., & Tramer, F. (2022). *The Privacy Onion Effect: Memorization is Relative*. arXiv:2206.10469.

Huang, Z., Cheng, X., Zheng, J., Wang, H., He, Z., Li, T., & Huang, X. (2024). *Unified Gradient-Based Machine Unlearning with Remain Geometry Enhancement*. arXiv:2409.19732.

Shamsian, A., Shaar, E., Navon, A., Chechik, G., & Fetaya, E. (2025). *Go Beyond Your Means: Unlearning with Per-Sample Gradient Orthogonalization*. arXiv:2503.02312.

Naderloui, N., Yan, S., Wang, B., Fu, J., Wang, W. H., Liu, W., & Hong, Y. (2025). *Rectifying Privacy and Efficacy Measurements in Machine Unlearning: A New Inference Attack Perspective*. arXiv:2506.13009.

Maheri, M. M., Cadet, X., Chin, P., & Haddadi, H. (2025). *WARP: Weight Teleportation for Attack-Resilient Unlearning Protocols*. arXiv:2512.00272.

Ebrahimpour-Boroojeny, A., Sundaram, H., & Chandrasekaran, V. (2025). *AMUN: Adversarial Machine UNlearning*. arXiv:2503.00917.

Zhou, D., Zhang, Y., Wei, T., Qiu, R., Yang, K., Lin, X., Qian, C., He, J., Tong, H., Zhai, C., Ji, H., & Zhang, H. (2025). *Geometric-disentangelment Unlearning*. arXiv:2511.17100.

Fang, K., Tao, Q., Liu, J., Xiao, Y., Ye, Q., Sun, J., & Hu, H. (2026). *Machine Unlearning in Low-Dimensional Feature Subspace*. arXiv:2601.22456.
