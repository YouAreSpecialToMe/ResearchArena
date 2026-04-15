# Adaptive Prototype Granularity in Frozen-Feature Contrastive Adaptation

## Introduction

This proposal is intentionally framed as a **careful incremental empirical study**, not as a strongly novel learning algorithm. The central question is narrow and testable: **when adapting a frozen vision backbone with lightweight supervised heads, does allowing the number of same-class prototypes to vary by class help enough to justify the added complexity?**

That question remains relevant because nearby literatures pull in different directions:

- **Supervised Contrastive Learning** improves discrimination but can compress within-class structure that matters for transfer.
- **Perfectly Balanced** argues that preserving latent subclass structure, not just class spread, can matter for transfer and robustness.
- **Contrastive Adapters** shows that head-only adaptation on frozen CLIP features is a strong regime for group robustness.
- **SBCL** shows subclass-aware supervision can help in end-to-end long-tailed learning.
- **CoPA** and adjacent prototype-based adaptation papers show that prototype-image mismatch matters in transfer settings, including regimes that start from pretrained backbones.
- **Hyperspherical and vMF classifier work** already establishes that cosine-normalized features are naturally modeled on the unit sphere.

Taken together, these papers weaken any claim that adaptive spherical prototypes are fundamentally new. What they do **not** yet settle is the specific empirical question studied here: **under a frozen CLIP backbone, matched head capacity, and fixed compute, is adaptive per-class prototype count selection materially better than class-level contrastive adaptation, fixed-`K` prototype heads, or simpler spread-preserving alternatives?**

After targeted searches over arXiv and Semantic Scholar for combinations of terms including "frozen CLIP prototype adaptation", "cross-domain finetuning prototypes", "adaptive subclass contrastive", "hyperspherical prototype classifier", and "von Mises-Fisher classifier", I did not find an exact prior paper centered on this controlled comparison. The closest papers each cover only part of the setup:

- **Contrastive Adapters for Foundation Model Group Robustness** matches the frozen-feature robustness regime but uses class-level objectives.
- **Perfectly Balanced** studies latent subclass preservation, but in end-to-end supervised contrastive learning rather than frozen-feature head adaptation.
- **Subclass-balancing Contrastive Learning for Long-tailed Recognition** uses subclass-aware learning, but in a different end-to-end long-tailed regime and without a frozen-backbone model-selection question.
- **Mind the Gap Between Prototypes and Images in Cross-domain Finetuning (CoPA)** studies prototype adaptation in transfer, but not adaptive subclass count selection inside a frozen CLIP contrastive-adapter setup.
- **Hyperspherical Prototype Networks** and **Towards Calibrated Hyper-Sphere Representation via Distribution Overlap Coefficient for Long-tailed Learning** show that hyperspherical / vMF-style prototype classifiers are established tools, so the use of vMF here should be presented as an appropriate modeling choice rather than the source of novelty.

The claim is therefore modest: this paper aims to provide a **well-controlled empirical answer** about adaptive prototype granularity in a practically important frozen-feature regime.

## Proposed Approach

### Study goal

Freeze one encoder, cache features once, and train only lightweight heads. Hold backbone, features, optimizer family, adapter width, training budget, and prototype parameterization fixed. Change only one substantive factor: **whether prototype count is fixed or selected adaptively per class**.

### Base model

Use a frozen **CLIP ViT-B/16** image encoder. For each cached feature `h_i`, train a small adapter `g` and normalize:

`z_i = normalize(g(h_i))`

Prototype-based heads maintain `K_c` unit-norm prototypes for class `c`:

`P_c = {p_{c,1}, ..., p_{c,K_c}}`

Within the true class, examples use soft prototype assignments:

`q_{i,k} = softmax_k(cos(z_i, p_{c,k}) / tau_p)`

The same prototype parameterization is used for both fixed-`K` and adaptive methods so that the comparison stays mechanism-matched.

### Training objective

The main prototype family uses:

`L = L_ce + lambda_con L_supcon + lambda_align L_align + lambda_occ L_occ`

where:

- `L_ce` is class cross-entropy,
- `L_supcon` is supervised contrastive loss on cached augmented views,
- `L_align` pulls features toward their soft-assigned same-class prototype,
- `L_occ` discourages dead or severely imbalanced prototypes.

The non-contrastive prototype baseline removes `L_supcon`. This matters because any gain from the adaptive method should not be attributable merely to adding extra contrastive supervision.

### Adaptive selection rule

The adaptive method uses a **conservative split/merge rule on unit-normalized features**. The purpose is not to claim a new vMF method, but to use a selection criterion aligned with cosine geometry.

For each class:

- a single prototype corresponds to one vMF component,
- a split proposal replaces one prototype by two child components initialized with spherical `k`-means,
- a merge proposal replaces two nearby same-class prototypes by one vMF component fit to their union.

Proposal acceptance is based on held-out class-conditional penalized vMF likelihood:

`score = log p_vMF(Z_val^c | model) - 0.5 * d_eff * log n_val^c`

To stay stable and cheap, concentration is shared within a class or clipped to a narrow range. Structure updates are conservative:

1. Propose at most one split per class at each update round.
2. Propose merges only for same-class prototype pairs with small angular separation.
3. Accept a proposal only if the penalized validation score improves and occupancy constraints are satisfied.

Occupancy floors are pre-registered:

- `n_child >= max(12, 0.08 * n_c)` on CUB,
- `n_child >= max(10, 0.10 * n_c)` on Waterbirds.

Dataset-specific caps prevent overfitting:

- `K_c <= 2` on Waterbirds,
- `K_c <= 3` on CUB.

### Waterbirds low-sample fallback

The earlier version relied on a CUB-tuned threshold for small-sample Waterbirds decisions. That is too bespoke. The revised primary rule is simpler and threshold-free.

If a Waterbirds class has fewer than `40` held-out validation examples for model selection:

- restrict that class to `K_c in {1, 2}`,
- evaluate split acceptance by averaging penalized vMF improvement across `5` stratified bootstrap resamples of the class training set,
- accept the split only if the **mean improvement is positive** and at least `4/5` resamples agree on the preferred decision.

This makes the main rule independent of an externally tuned threshold. To ensure the overall claim does not hinge on this choice, the proposal pre-registers a small **sensitivity analysis** on Waterbirds only: rerun the adaptive method with two stricter margins, `delta in {0.0025, 0.005}` penalized log-likelihood per validation example, and report whether conclusions change.

### What the contribution is and is not

The contribution is:

- a controlled frozen-feature comparison,
- a hypothesis-driven test of adaptive prototype granularity,
- a spherical model-selection rule appropriate for cosine-normalized embeddings,
- a robustness analysis that checks whether conclusions survive simple fallback variations.

The contribution is **not**:

- a new backbone,
- a new contrastive objective,
- a fundamentally new prototype classifier,
- a claim that adaptive vMF prototypes are novel in general.

## Related Work

### Supervised contrastive learning and transfer-relevant structure

**Supervised Contrastive Learning** (Khosla et al., 2020) is the foundation for the contrastive baselines used here. It provides strong class supervision but does not explicitly preserve within-class multimodality.

**The Power of Contrast for Feature Learning: A Theoretical Analysis** (Ji et al., 2023) motivates the problem: contrastive objectives can improve source discrimination while discarding information useful for downstream transfer. This paper tests whether, in a frozen-feature regime, lightweight subclass-aware heads recover some of that information.

### Latent subclass preservation

**Perfectly Balanced: Improving Transfer and Robustness of Supervised Contrastive Learning** (Chen et al., 2022) is the nearest conceptual predecessor. It argues that preserving latent subclass information matters beyond preserving overall class spread. However, its setting differs materially:

- it is end-to-end rather than frozen-backbone,
- it studies modified supervised contrastive training rather than head-only adaptation,
- it does not ask the narrow fixed-`K` versus adaptive-`K` question under matched cached features.

For fairness, this proposal compares against a **PB-inspired frozen-feature spread-preserving baseline**, not against a claimed full reproduction of Perfectly Balanced.

### Frozen-feature robustness adaptation

**Contrastive Adapters for Foundation Model Group Robustness** (Zhang and Re, 2022) is the most direct baseline because it studies frozen CLIP adaptation for robustness. This proposal differs only in the representation of same-class structure: class-level contrastive supervision versus multi-prototype heads with either fixed or adaptive granularity.

### Subclass-aware end-to-end learning

**Subclass-balancing Contrastive Learning for Long-tailed Recognition** (Hou et al., 2023) shows that subclass-aware relabeling can help. But SBCL is not already the answer to this proposal's question:

- it is designed for long-tailed end-to-end training,
- subclass machinery co-evolves with the representation,
- the core question is balancing head and tail subclasses, not selecting per-class prototype count in a frozen-feature head.

This makes SBCL an important methodological neighbor, but not a direct solution to the present setting.

### Frozen-backbone and prototype adaptation neighbors

**Mind the Gap Between Prototypes and Images in Cross-domain Finetuning (CoPA)** (Tian et al., 2024) is a close adjacent paper because it studies prototype-image mismatch in transfer from pretrained models. It weakens any claim that prototype adaptation around pretrained representations is unexplored. But CoPA does not already cover this paper's main setting:

- it is framed as cross-domain finetuning rather than a frozen CLIP head-only comparison,
- it does not isolate adaptive subclass count selection as the single variable under study,
- it does not compare fixed versus adaptive prototype granularity under matched frozen-feature compute.

This paper should therefore be positioned as a **narrow empirical follow-up in a different adaptation regime**, not as opening an untouched area.

### Hyperspherical and vMF prototype classifiers

**Hyperspherical Prototype Networks** (Mettes et al., 2019) and **Towards Calibrated Hyper-Sphere Representation via Distribution Overlap Coefficient for Long-tailed Learning** (Zhu et al., 2022) show that prototype classifiers on normalized features, including vMF-like reasoning on the sphere, are established. These works matter because they remove any novelty claim around "using spherical prototypes" or "using vMF-compatible scoring."

Their relevance here is narrower:

- they justify modeling normalized features on the sphere,
- they do not study frozen CLIP group-robustness adaptation,
- they do not answer whether adaptive per-class prototype count is better than fixed `K` under a controlled frozen-feature budget.

So vMF is used here as the **correct geometric tool**, not as a novel method component.

### Robustness under spurious correlations

**Learning Robust Classifiers with Self-Guided Spurious Correlation Mitigation** (Zheng et al., 2024) represents a different line of work on subgroup robustness. It improves robustness by mitigating spurious cues rather than by modeling same-class multimodality with adaptive prototypes. Including it helps position the proposal as a modest representation-side study rather than a comprehensive robustness method.

## Experiments

### Datasets

The core study uses two required datasets and one tightly scoped optional extension:

1. **Waterbirds**
   - primary metric: worst-group accuracy,
   - secondary metrics: average accuracy, macro-F1.
2. **CUB-200-2011**
   - primary metrics: top-1 accuracy and macro-F1,
   - secondary metric: class-balanced accuracy.
3. **Optional robustness extension if time remains: CelebA**
   - evaluate only the strongest 4 methods for **1 seed**,
   - primary metric: worst-group accuracy,
   - role: check whether any Waterbirds conclusion transfers to a second robustness benchmark without expanding the main claim.

The main paper claim will rely on Waterbirds and CUB only. The optional CelebA run is explicitly framed as a bounded external robustness check, not a required pillar of the proposal.

### Backbone and cached features

- Backbone: **CLIP ViT-B/16** only.
- Cache one standard feature per image and two lightweight augmented views for contrastive training.
- No backbone finetuning.

This keeps the study feasible on a single RTX A6000 and ensures the paper is about head adaptation rather than large-scale training.

### Main baselines

The comparison set is intentionally compact and mechanism-matched:

1. Linear probe on frozen CLIP features.
2. Cross-entropy adapter.
3. Contrastive Adapters-style class-level contrastive adapter.
4. Fixed-`K=2` contrastive prototype adapter.
5. Fixed-`K=2` non-contrastive prototype head.
6. PB-inspired frozen-feature spread-preserving baseline.
7. Adaptive prototype-count selection with penalized vMF split/merge.

### Diagnostics

Beyond accuracy, report:

- selected `K_c` per class,
- prototype occupancies and occupancy entropy,
- accepted split and merge counts,
- validation score gains for accepted structure changes,
- on Waterbirds and optional CelebA, alignment between discovered subclasses and observed groups.

These diagnostics are necessary because a positive result is not convincing unless the adaptive mechanism is actually active and stable.

### Ablations

Keep the ablation set minimal:

1. Adaptive `K_c` versus fixed `K=2`.
2. Penalized vMF selection versus deterministic grow-to-cap control.
3. Waterbirds fallback primary rule versus stricter margin sensitivity settings.

This isolates the real contribution while keeping the total runtime within budget.

### Seed policy and compute budget

The revised proposal uses **3 seeds everywhere for the main experiments** to remove the earlier inconsistency. Cached-feature training is cheap enough that this remains feasible.

Planned budget on one RTX A6000:

- dataset setup plus CLIP feature extraction for Waterbirds and CUB: `~1.5 to 2.0` hours,
- pilot tuning on one shared validation split: `~0.5` hour,
- main methods on Waterbirds and CUB with **3 seeds each**: `~4.0 to 4.5` hours,
- targeted ablations and Waterbirds sensitivity analysis: `~0.75` hour,
- aggregation and plotting: `~0.25` hour.

Total planned time: `~7.0 to 8.0` hours.

Optional CelebA extension is run **only if** the main grid finishes early or feature caching is already available. If not, it is omitted without affecting the main claim.

## Success Criteria

### Positive outcome

- On Waterbirds, the adaptive method improves worst-group accuracy over both the class-level contrastive adapter and fixed-`K=2` prototype contrastive baseline by roughly `1.5` absolute points or more.
- On CUB, the adaptive method improves at least one primary metric over the best fixed-`K` baseline.
- At least a nontrivial minority of classes retain `K_c > 1` in repeated runs, with occupancy floors satisfied and similar split decisions across seeds.
- Waterbirds conclusions remain qualitatively unchanged under the pre-registered stricter fallback margins.

### Negative but informative outcome

- Adaptive selection collapses back to `K_c = 1` for most classes, or matches fixed-`K` within variation on both datasets.
- This would support the claim that, under a frozen-feature budget, extra prototype adaptivity adds little beyond existing class-level or fixed-prototype adapters.

### Refutation conditions

The hypothesis is weakened or refuted if:

- fixed-`K` matches or beats adaptive selection consistently,
- the non-contrastive prototype baseline explains the gains,
- the PB-inspired spread baseline closes the gap,
- selected subclass counts are unstable across seeds,
- the Waterbirds result changes materially under the simple fallback sensitivity analysis.

## References

- Chen, M. F., Fu, D. Y., Narayan, A., Zhang, M., Song, Z., Fatahalian, K., and Re, C. 2022. *Perfectly Balanced: Improving Transfer and Robustness of Supervised Contrastive Learning*. ICML.
- Hou, C., Zhang, J., Wang, H., and Zhou, T. 2023. *Subclass-balancing Contrastive Learning for Long-tailed Recognition*. ICCV.
- Ji, W., Deng, Z., Nakada, R., Zou, J., and Zhang, L. 2023. *The Power of Contrast for Feature Learning: A Theoretical Analysis*. JMLR.
- Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., and Krishnan, D. 2020. *Supervised Contrastive Learning*. NeurIPS.
- Mettes, P., van der Pol, E., and Snoek, C. G. M. 2019. *Hyperspherical Prototype Networks*. NeurIPS.
- Tian, H., Liu, F., Zhou, Z., Liu, T., Zhang, C., and Han, B. 2024. *Mind the Gap Between Prototypes and Images in Cross-domain Finetuning*. NeurIPS.
- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. 2021. *Learning Transferable Visual Models From Natural Language Supervision*. ICML.
- Zhang, M. and Re, C. 2022. *Contrastive Adapters for Foundation Model Group Robustness*. ICML.
- Zheng, G., Ye, W., and Zhang, A. 2024. *Learning Robust Classifiers with Self-Guided Spurious Correlation Mitigation*. IJCAI.
- Wang, H., Fu, S., He, X., Fang, H., Liu, Z., and Hu, H. 2022. *Towards Calibrated Hyper-Sphere Representation via Distribution Overlap Coefficient for Long-tailed Learning*. ECCV.
