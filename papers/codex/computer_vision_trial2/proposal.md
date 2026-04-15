# FOCUS: Object-Centric Evidence as an Update-Safety Signal for Realistic Online VLM Adaptation

## Introduction

Realistic online test-time adaptation (TTA) for CLIP-style vision-language models is fragile because the model must decide, on the fly and without labels, which test samples are safe to write into its running adaptation state. In the realistic stream setting of Zanella et al. (2025), early pseudo-label mistakes can corrupt later predictions through temporally correlated updates, especially on context-biased benchmarks such as Waterbirds and CounterAnimal.

This proposal makes a deliberately narrow claim. **FOCUS is not a new debiasing method, not a new TTA family, and not a general reliability framework.** It is a study of whether **object-centric evidence is a better update-safety signal than existing lightweight reliability scores for StatA-style online adaptation**. The intended contribution is therefore empirical and mechanistic: when a sample's foreground supports the current class prediction but its background does not, that sample may be safer to use for online state updates.

The hypothesis is specific and falsifiable: **at the same update acceptance rate, an object-centric score built from full-image, foreground, and background views predicts harmful StatA updates better than entropy, max-softmax, and a recent reliability-style consistency score.** If true, this should reduce adaptation drift on context-heavy streams without claiming a new general-purpose TTA algorithm.

## Proposed Approach

### Overview

The base adaptation method remains **StatA** from Zanella et al. (2025). FOCUS only changes the decision of whether StatA may update on the current sample.

For each test image `x`, FOCUS constructs three inexpensive views:

1. the original image `x_full`;
2. a foreground-focused view `x_fg`;
3. a complementary background view `x_bg`.

All views are encoded by the same frozen CLIP backbone. In the actual executed stack this is Hugging Face `transformers` CLIP, specifically `openai/clip-vit-base-patch16`, running under Python `3.12`. Let `k = argmax z_full`, where `z_full`, `z_fg`, and `z_bg` are the class logits of the three views. FOCUS computes:

- `a(x) = 1[argmax z_full = argmax z_fg]`, full/foreground agreement;
- `f(x) = p_fg(k)`, foreground support for the predicted class;
- `b(x) = max softmax(z_bg)`, background dominance.

The default safety score is

`s_focus(x) = a(x) * (f(x) - lambda * b(x) - tau)`.

StatA updates only when `s_focus(x) > 0`. Prediction still comes from the base StatA pipeline. A soft-weighted variant,

`w(x) = sigmoid(alpha * s_focus(x))`,

is retained as one optional ablation, but hard gating is the main method because it directly matches the update-safety question.

### Why this is scoped to update safety rather than general prediction improvement

StatA is attractive in realistic online adaptation because it updates running statistics rather than performing unstable gradient steps on every test sample. But this also means a harmful accepted sample has persistent downstream impact. FOCUS does not try to solve single-image debiasing in general. It asks whether object-centric evidence is especially useful for the narrower question: **should this sample be allowed to modify the online state?**

That distinction matters for novelty. Prior work already studies reliability-based filtering, segmentation-based debiasing, and spatially aware TTA. The new question here is whether those object/background cues carry unique signal for **future update harm** under realistic StatA streams.

### Foreground and background views

The main pipeline uses only cheap mask generation:

- CLIP `ViT-B/16` attention rollout from the Hugging Face `openai/clip-vit-base-patch16` vision encoder;
- thresholding by normalized saliency mass;
- largest connected component cleanup;
- one masked-or-cropped foreground view;
- one complementary background view.

Because using the same backbone for both saliency and classification creates a circularity concern, the study includes one small diagnostic on 128 validation images comparing CLIP-rollout masks against an independent frozen saliency source such as DINOv2 attention or generic spectral saliency. This diagnostic is not part of the main method. It is only a sanity check on whether the mask signal is completely self-confirming.

### Matched-rate baselines and simple-variant controls

The central reviewer objection is that FOCUS might win only because it is conservative. The evaluation therefore compares methods at the **same update acceptance rate**.

Mandatory matched-rate gates for StatA:

- `Entropy gate`: lowest prediction entropy samples update.
- `MaxProb gate`: highest max-softmax samples update.
- `CER gate`: a ReTA-style consistency/reliability score using the same light augmentations but applied only as an update gate for StatA.
- `FOCUS gate`: the proposed object-centric score.

Mandatory simple-variant controls:

- agreement-only: `a(x)`;
- foreground-only: `f(x)`;
- background-only: `-b(x)`;
- no-background variant: `a(x) * f(x)`.

If FOCUS only beats plain StatA, that is weak evidence. If it still beats these matched-rate gates and simple object-centric variants, then the gain is more plausibly due to the combined object-centric safety signal rather than generic filtering.

## Related Work

### Realistic online VLM adaptation

**Realistic Test-Time Adaptation of Vision-Language Models** (Zanella et al., 2025) is the primary setting paper. It argues that VLM TTA should be evaluated under realistic online streams with variable effective class sets and temporal correlation, and it introduces **StatA** as a strong method in that regime. FOCUS is explicitly a wrapper around this setting.

**A Lost Opportunity for Vision-Language Models** (Döbler et al., 2024) is important because it shows online VLM TTA is highly sensitive to evaluation setup and that many simple adaptations are brittle. This argues against broad claims and supports the narrow update-safety framing.

**The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models** (Sheng et al., 2025) further cautions that VLM TTA gains often weaken under stronger comparisons and trustworthiness-oriented analysis. That paper motivates the matched-rate comparisons and the emphasis on update precision rather than top-line accuracy alone.

### Reliability-oriented VLM TTA

**ReTA: Advancing Reliable Test-Time Adaptation of Vision-Language Models under Visual Variations** (Liang et al., 2025) is the closest reliability baseline. It already argues that entropy is unreliable under shift and introduces consistency-aware reliability mechanisms. This sharply limits the novelty claim: FOCUS cannot claim to discover that selective updating matters. The difference is that FOCUS tests whether **object-centric evidence outperforms an existing reliability-style score as an update gate for StatA at the same acceptance rate**.

**Adaptive Cache Enhancement for Test-Time Adaptation of Vision-Language Models** (Nguyen et al., 2025) and **Mitigating Cache Noise in Test-Time Adaptation for Large Vision-Language Models** (Zhai et al., 2025) further establish that conservative pseudo-label selection and noise-aware filtering are already active directions. FOCUS is narrower than both: it does not redesign the cache, and it does not claim a superior general reliability principle. It proposes one specific spatial cue for update safety.

**Bayesian Test-Time Adaptation for Vision-Language Models** (Zhou et al., 2025) is also relevant because it treats uncertainty estimation as the route to safer adaptation. FOCUS differs by using object/background evidence rather than probabilistic posterior modeling and by targeting an online update-acceptance decision for StatA.

**Fair Context Learning for Evidence-Balanced Test-Time Adaptation in Vision-Language Models** (Yun et al., 2026) is a critical adjacent paper. It argues that entropy-minimizing prompt adaptation can amplify shared-evidence bias and proposes fairness-driven calibration of text contexts. This paper makes the overlap with context-aware reliability concerns even clearer. FOCUS therefore should not be framed as the first method to address evidence imbalance or spurious context in VLM TTA. Its claim is narrower: **for realistic online streams, object-centric evidence may be a better write-safety signal for StatA than standard reliability filters.**

### Object-centric and spatially aware test-time methods

**SegDebias: Test-Time Bias Mitigation for ViT-Based CLIP via Segmentation** (Wu and Cai, 2025) is the closest object-centric prior art. It already uses foreground/background separation to debias CLIP predictions at test time. This rules out any novelty claim about decomposition itself. FOCUS differs only in using cheap object-centric evidence to gate online updates rather than to define a new prediction rule.

**GS-Bias: Global-Spatial Bias Learner for Single-Image Test-Time Adaptation of Vision-Language Models** (Huang et al., 2025) is another important adjacent method because it learns spatial bias corrections during TTA. This further narrows the novelty window: spatial cues in VLM TTA are not new. FOCUS is instead about whether a lightweight spatial cue helps determine update safety in the realistic online setting.

**Robust Context-Aware Object Recognition** (Janouskova et al., 2025), **A Sober Look at the Robustness of CLIPs to Spurious Features** (Wang et al., 2024), and **Task Bias in Vision-Language Models** (Menon et al., 2022) motivate the problem itself by showing that CLIP-style models frequently rely on contextual or shortcut evidence. FOCUS converts that concern into a narrow update-selection test rather than a broad robustness claim.

## Experiments

### Core experimental question

The mandatory question is not "does FOCUS produce the best overall TTA accuracy?" It is:

**Does object-centric evidence identify safe StatA updates better than equally conservative non-object-centric gates?**

Everything in the experimental design is chosen to answer that question under the one-GPU, eight-hour budget.

### Mandatory benchmark

The mandatory benchmark is intentionally small:

- Waterbirds;
- CounterAnimal.

These two datasets are enough to test the central claim because both are context-sensitive and realistic failures of shortcut-driven pseudo-labeling are plausible. A single-seed ImageNet-R sanity run is optional and only included if runtime remains after the mandatory study.

Common setup:

- Hugging Face `openai/clip-vit-base-patch16` only;
- realistic online stream generation following Zanella et al. (2025);
- `2` seeds from the start for every mandatory experiment;
- one fixed hyperparameter setting selected on validation streams, then frozen;
- Waterbirds loaded from local WILDS when available and otherwise from the documented Hugging Face mirror;
- CounterAnimal loaded from the Hugging Face mirror, using provided validation/test splits when present and otherwise deterministic class-balanced validation/test subsets.

### Simplified harmful-update analysis

The original long-horizon counterfactual analysis is too expensive and too fragile for the stated budget. It is replaced by a sparse, local counterfactual protocol.

For each validation stream:

- use the first 256 samples only;
- probe every 4th step, yielding 64 candidate update points;
- at each probe, clone the current StatA state once;
- branch into `update` and `skip`;
- compare average labeled accuracy over the next `W = 16` stream samples.

Define a harmful update if:

- the update branch performs worse than the skip branch by at least `0.25` percentage points over that short horizon.

This produces a direct but affordable label for update harm. In parallel, report a cheaper proxy label:

- accepted pseudo-label is incorrect.

The paper's main mechanistic claim requires improvement on the direct sparse counterfactual label; the proxy is auxiliary.

### Mandatory baselines

Prediction baselines:

- zero-shot CLIP;
- StatA.

Matched-rate StatA update-gating baselines:

- StatA + entropy gate;
- StatA + max-softmax gate;
- StatA + ReTA-style CER gate;
- FOCUS-StatA.

Object-centric comparison baselines:

- agreement-only gate;
- foreground-only gate;
- background-only gate.

Reference object-centric prior-art baseline:

- SegDebias on Waterbirds and CounterAnimal only if its released code and segmentation requirements are lightweight enough; otherwise Waterbirds-only with that limitation stated explicitly.

This baseline suite is stronger than the original one because it tests whether the gain survives comparison to both standard confidence filtering and an existing reliability-style gate under the same acceptance-rate budget.

### Hyperparameters and acceptance-rate control

To block threshold-tuning objections, every gate is evaluated at the same target acceptance rates, for example `{0.3, 0.5, 0.7}`. Thresholds are chosen on short validation streams and then frozen.

FOCUS tunes only:

- `lambda`;
- `tau`;
- hard vs soft gate.

The ReTA-style CER gate tunes only its own threshold under the same rate targets. The final comparison reports either:

- best common acceptance rate shared by all gates; or
- average rank across the three fixed rates.

All headline aggregation uses only seeds `7` and `17`. Seed `27` is reserved for a confirmation-only rerun at the single frozen common rate for plain StatA, entropy, CER, and FOCUS, and is reported separately rather than averaged into the main tables.

### Metrics

Primary metrics:

- average online accuracy across each stream;
- worst-group accuracy on Waterbirds;
- class-balanced accuracy where applicable.

Update-safety metrics:

- AUROC and AUPRC for harmful-update prediction on the sparse counterfactual labels;
- precision among accepted updates;
- accuracy difference between accepted and rejected samples;
- cumulative drift indicators of StatA's running state.

Efficiency metrics:

- per-image latency;
- extra CLIP forward-pass cost;
- total runtime.

### Compute envelope

The proposal is scoped to fit one RTX A6000 and roughly eight hours total:

- mask sanity and sparse harmful-update pilot: `1.5` hours;
- feature and view caching for Waterbirds and CounterAnimal: `2.0` hours;
- mandatory baselines over both datasets and `2` seeds: `3.5` hours;
- simple-variant ablations and analysis: `1.0` hour.

If runtime is tight, the drop order is fixed:

1. remove the optional ImageNet-R sanity run;
2. remove the soft-gate ablation;
3. reduce the number of fixed acceptance rates from three to two;
4. keep both mandatory datasets and both seeds unchanged.

## Success Criteria

The hypothesis is supported if:

1. FOCUS has better harmful-update AUROC/AUPRC than entropy, max-softmax, and the ReTA-style CER gate at matched acceptance rates on the sparse counterfactual analysis.
2. FOCUS-StatA improves over plain StatA and entropy-gated StatA on both Waterbirds and CounterAnimal, with the strongest gains on worst-group or context-heavy slices.
3. No single simple variant such as agreement-only or foreground-only reproduces the full gain.
4. Runtime stays within the stated one-GPU budget without requiring heavy segmentation in the main pipeline.

The hypothesis is weakened or refuted if:

1. cheap masks fail the sanity check or are so noisy that FOCUS does not outperform non-object-centric gates on harmful-update prediction;
2. gains disappear once acceptance rate is matched against entropy, max-softmax, or the ReTA-style CER gate;
3. one simple object-centric variant performs the same as the full score, implying the proposed combination is unnecessary;
4. the extra view construction cost makes the method impractical for realistic online use.

## References

1. Maxime Zanella, Clément Fuchs, Christophe De Vleeschouwer, and Ismail Ben Ayed. *Realistic Test-Time Adaptation of Vision-Language Models*. arXiv:2501.03729, 2025. https://arxiv.org/abs/2501.03729
2. Mario Döbler, Robert A. Marsden, Tobias Raichle, and Bin Yang. *A Lost Opportunity for Vision-Language Models: A Comparative Study of Online Test-Time Adaptation for Vision-Language Models*. arXiv:2405.14977, 2024. https://arxiv.org/abs/2405.14977
3. Lijun Sheng, Jian Liang, Ran He, Zilei Wang, and Tieniu Tan. *The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models*. arXiv:2506.24000, 2025. https://arxiv.org/abs/2506.24000
4. Yiwen Liang, Hui Chen, Yizhe Xiong, Zihan Zhou, Mengyao Lyu, Zijia Lin, Shuaicheng Niu, Sicheng Zhao, Jungong Han, and Guiguang Ding. *Advancing Reliable Test-Time Adaptation of Vision-Language Models under Visual Variations*. arXiv:2507.09500, 2025. https://arxiv.org/abs/2507.09500
5. Khanh-Binh Nguyen, Phuoc-Nguyen Bui, Hyunseung Choo, and Duc Thanh Nguyen. *Adaptive Cache Enhancement for Test-Time Adaptation of Vision-Language Models*. arXiv:2508.07570, 2025. https://arxiv.org/abs/2508.07570
6. Haotian Zhai, Xinyu Chen, Can Zhang, Tianming Sha, and Ruirui Li. *Mitigating Cache Noise in Test-Time Adaptation for Large Vision-Language Models*. arXiv:2503.18334, 2025. https://arxiv.org/abs/2503.18334
7. Lihua Zhou, Mao Ye, Shuaifeng Li, Nianxin Li, Xiatian Zhu, Lei Deng, Hongbin Liu, and Zhen Lei. *Bayesian Test-Time Adaptation for Vision-Language Models*. arXiv:2503.09248, 2025. https://arxiv.org/abs/2503.09248
8. Sanggeon Yun, Ryozo Masukawa, SungHeon Jeong, Wenjun Huang, Hanning Chen, and Mohsen Imani. *Fair Context Learning for Evidence-Balanced Test-Time Adaptation in Vision-Language Models*. arXiv:2602.07027, 2026. https://arxiv.org/abs/2602.07027
9. Fangyu Wu and Yujun Cai. *SegDebias: Test-Time Bias Mitigation for ViT-Based CLIP via Segmentation*. arXiv:2511.00523, 2025. https://arxiv.org/abs/2511.00523
10. Zhaohong Huang, Yuxin Zhang, Jingjing Xie, Fei Chao, and Rongrong Ji. *GS-Bias: Global-Spatial Bias Learner for Single-Image Test-Time Adaptation of Vision-Language Models*. arXiv:2507.11969, 2025. https://arxiv.org/abs/2507.11969
11. Klara Janouskova, Cristian Gavrus, and Jiri Matas. *Robust Context-Aware Object Recognition*. arXiv:2510.00618, 2025. https://arxiv.org/abs/2510.00618
12. Qizhou Wang, Yong Lin, Yongqiang Chen, Ludwig Schmidt, Bo Han, and Tong Zhang. *A Sober Look at the Robustness of CLIPs to Spurious Features*. arXiv:2403.11497, 2024. https://arxiv.org/abs/2403.11497
13. Sachit Menon, Ishaan Preetam Chandratreya, and Carl Vondrick. *Task Bias in Vision-Language Models*. arXiv:2212.04412, 2022. https://arxiv.org/abs/2212.04412
