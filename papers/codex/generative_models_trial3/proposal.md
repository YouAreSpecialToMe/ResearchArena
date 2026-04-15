# ParaDG: Disagreement-Gated Paraphrase Blending for Robust Text-to-Image Diffusion

## Introduction

Text-to-image diffusion models remain sensitive to harmless prompt rewording. Equivalent phrasings can change object counts, attribute bindings, or spatial relations even when the intended scene is unchanged. This has been documented from two directions: compositional benchmarks such as T2I-CompBench++ and Structured Semantic Alignment expose binding and relation failures, while MetaLogic shows that logically equivalent prompts can yield different generations.

This proposal makes a narrow claim. **ParaDG is not a new general guidance family.** It is a **robustness-oriented extension of AAPB-style multi-prompt blending** specialized to semantically equivalent prompts. The two design choices are:

1. only audited paraphrases of the same prompt are allowed as auxiliary prompts;
2. blending strength is increased only when those paraphrases disagree.

The core hypothesis is that disagreement across verified paraphrases is a direct indicator of wording brittleness. If that signal is used selectively during denoising, Stable Diffusion v1.5 should become more faithful to the intended scene and more invariant across equivalent prompts, without requiring finetuning or large-scale experiments.

## Proposed Approach

### Overview

For each prompt, ParaDG constructs a small paraphrase set containing the original prompt and up to two controlled rewrites. At selected denoising steps, the same latent is evaluated under each prompt. ParaDG then:

1. computes a paraphrase consensus prediction;
2. measures disagreement across paraphrase-conditioned predictions;
3. applies a correction to the original-prompt prediction only when disagreement is high.

The method is training-free and targeted to `runwayml/stable-diffusion-v1-5` in `diffusers`.

### A Single Reproducible Slot Schema

The previous draft used too much bespoke machinery. The revised design uses one fixed prompt schema for the entire study and selects prompts that fit it:

`{count_1, object_1, attribute_1, relation, count_2, object_2, attribute_2}`

Not every field is required for every prompt, but all prompts must be representable as:

- one or two objects;
- optional count for each object;
- optional bound attribute for each object;
- optional directed relation between the two objects.

This keeps the study focused on the benchmark categories where SD1.5 commonly fails and where semantic equivalence can be audited with low ambiguity: attribute binding, relations, and numeracy.

### Pre-Registered Paraphrase Audit

To reduce implementation risk, ParaDG does **not** depend on free-form LLM paraphrase generation plus post hoc parsing. Instead, each approved paraphrase is created from one of a small number of controlled rewrite templates that preserve the slot schema exactly.

Allowed rewrite operations:

1. attribute-order swaps that preserve noun attachment;
2. count-preserving numeral rewrites;
3. active/passive or clause-order rewrites that preserve directed relations;
4. determiner or connective rewrites that do not alter slot fields.

Acceptance rules are pre-registered:

1. the paraphrase must preserve every populated slot exactly;
2. no new object, attribute, count, or relation may appear;
3. relation direction must be unchanged for asymmetric relations;
4. style-only additions are disallowed;
5. each accepted paraphrase is checked against a fixed checklist.

Audit protocol:

- Every accepted held-out paraphrase receives a one-pass checklist audit.
- A 48-pair subset is double-annotated and reported with Cohen's kappa.
- The full accepted paraphrase table is saved as an artifact for the experiment stage.

If pilot-time coverage falls below target, the study falls back to one paraphrase per prompt rather than relaxing the audit.

### Disagreement-Gated Paraphrase Blending

For prompt set `P = {p0, p1, ...}`, where `p0` is the original prompt, compute the conditional noise prediction `eps_t(pj)` at guided timestep `t`. Let

`eps_bar_t = (1 / |P|) sum_j eps_t(pj)`.

ParaDG updates the original-prompt prediction by

`eps_para_t = eps_t(p0) + alpha_t * g_t * (eps_bar_t - eps_t(p0))`.

Here:

- `alpha_t` is a fixed timestep-adaptive blend schedule chosen on the pilot and frozen before held-out runs;
- `g_t` is a disagreement gate derived from paraphrase variance.

The gate is deliberately simple:

`g_t = min(1, c * ||Var_j[eps_t(pj)]||_1)`

with scalar `c` set on the pilot. This keeps the main method close to AAPB-style adaptive blending while making the blending signal robustness-specific.

### Token-Group Slot Weighting

The previous draft used slot-graph extraction and slot-attention alignment. The revised version uses only the fixed slot schema and exact token groups associated with those slots. For each slot field, ParaDG averages cross-attention over the tokens that realize that field in each paraphrase and computes a slot disagreement weight.

This weight is used only to redistribute the already computed gated update across slot token groups. It does not introduce extra learned components or external models.

### Sparse-Time Execution

To fit the compute budget, extra prompt evaluations are used on only 8 of 30 DDIM steps:

- denoising steps: `30`
- guided steps: `[2, 4, 6, 9, 12, 16, 20, 24]`
- seeds: `[11, 22, 33]`

With at most three prompts per sample, this keeps the overhead close to a sparse 2x-2.3x multiplier rather than a full ensemble at every step.

## Related Work

Latent Diffusion Models provide the SD1.5 backbone used here. T2I-CompBench++, Structured Semantic Alignment, and MetaLogic motivate the scientific target: prompt-faithful image generation should be robust to equivalent wording, especially for binding, relations, and numeracy.

Among guidance methods, Attend-and-Excite, MaskDiffusion, and Self-Coherence Guidance improve alignment from a single prompt. These methods limit any claim that attention repair itself is new. On the multi-prompt side, Adaptive Auxiliary Prompt Blending (AAPB) is the closest prior work and is the strongest novelty constraint: it already performs test-time prompt blending during diffusion. ParaDG therefore positions itself only as a **paraphrase-restricted, disagreement-gated specialization** of that idea for wording robustness.

Prompt optimization methods such as NeuroPrompts search for stronger prompts, but they do not aim to make several correct prompts agree. MetaLogic and SSA are also close, but they are evaluation-oriented rather than inference-time correction methods.

## Experiments

### Fixed Setup

- Backbone: `runwayml/stable-diffusion-v1-5`
- Resolution: `512x512`
- Scheduler: DDIM
- Denoising steps: `30`
- CFG scale: `7.5`
- Seeds: `[11, 22, 33]`
- Hardware: 1x RTX A6000 48GB, 60GB RAM, 4 CPU cores

No training or large-scale VLM evaluation is required.

### Prompt Matrix

- 96 prompts total from T2I-CompBench++ categories of attribute binding, relations, and numeracy
- 12 pilot prompts for threshold setting, runtime measurement, and audit calibration
- 84 held-out prompts for the main study
- target overlap subset of 36 held-out prompts with at least one approved paraphrase

The reduced fallback matrix is pre-registered:

- 60 held-out prompts for original-prompt faithfulness
- 24 overlap prompts for robustness

### Baselines

Main held-out set:

1. Vanilla SD1.5.
2. Static paraphrase consensus: same paraphrase set and same sparse guided steps, but constant blend strength and no disagreement gate.
3. Adaptive paraphrase blend without gating: same paraphrase set, same timestep-adaptive `alpha_t`, but `g_t = 1` always.
4. ParaDG.

Overlap robustness subset:

1. Vanilla SD1.5.
2. Static paraphrase consensus.
3. Adaptive paraphrase blend without gating.
4. ParaDG.
5. Faithful AAPB baseline.
6. One single-prompt faithfulness baseline.

The added adaptive ungated baseline is critical: it tests whether any improvement comes from disagreement gating rather than from paraphrase-conditioned adaptive blending alone.

### Operational Baseline Fidelity Rules

The strongest comparison is AAPB, so fidelity is defined in advance.

**AAPB on SD1.5 counts as faithful only if:**

1. it uses exactly two prompt conditions per sample;
2. the blending coefficient follows the paper's adaptive rule rather than a new heuristic we introduce;
3. the blend is global, not slot-gated;
4. the same scheduler, step count, resolution, and seeds are used as ParaDG.

For the robustness subset, the auxiliary prompt is an approved paraphrase when compatible with the implementation. If faithful AAPB code is unavailable or cannot be reproduced on SD1.5 within the pilot budget, the paper reports that failure explicitly and treats AAPB as a documented reproduction attempt rather than silently replacing it.

The required single-prompt comparison is:

1. Self-Coherence Guidance only if it can be ported to SD1.5 without changing its core mechanism.
2. Otherwise MaskDiffusion is used as the pre-registered fallback single-prompt baseline.

### Metrics

Primary metrics:

- T2I-CompBench++ category metrics on the held-out set
- Paraphrase Robustness Score (PRS): fraction of paraphrase pairs whose slot judgments agree under the same seed

Secondary metrics:

- CLIPScore
- ImageReward
- DINO-based image consistency across paraphrases
- LPIPS and DINO dispersion across seeds
- runtime and peak memory

### Pre-Registered Audit Protocol

The evaluation uses the same slot schema that defines the prompts. For each prompt, the study generates a deterministic checklist of slot questions:

- object present?
- attribute correctly bound?
- count correct?
- directed relation correct?

Automation is used only for large-scale coverage, and the study explicitly validates it.

- Human subset: 72 prompt-seed-method cases, double annotated.
- Reliability reports: human-human agreement and automated-human agreement.
- If automated agreement is weak, the paper downgrades automated PRS claims and bases the main robustness conclusion on the human subset only.

This makes a failed evaluator validation informative rather than a reason to weaken the comparison post hoc.

### Pilot Gate

The full study runs only if the 12-prompt pilot satisfies both conditions:

1. approved paraphrase coverage is at least 70% overall and at least 60% in each category;
2. projected end-to-end runtime is at most 7 GPU hours.

Otherwise the reduced matrix activates automatically.

### Compute Feasibility

The planned study is intentionally small. Using 84 held-out prompts, 3 seeds, and 4 main methods yields 1,008 main-set generations. The 36-prompt overlap subset adds 216 generations for AAPB and the single-prompt baseline. Including the pilot, total volume stays near 1,300 images at 512x512 with no finetuning. On one RTX A6000, that is realistic within the 8-hour budget.

## Success Criteria

The proposal is supported if all of the following hold:

1. ParaDG beats vanilla SD1.5 and both ungated paraphrase baselines on at least two of the three main held-out categories.
2. ParaDG improves PRS over vanilla SD1.5 and both ungated paraphrase baselines on the overlap subset.
3. ParaDG beats faithful AAPB on PRS, or matches it on PRS while exceeding it on at least one compositional category.
4. Human adjudication does not overturn the robustness conclusion.
5. Gains are not explained by severe diversity collapse, defined as more than a 20% LPIPS drop versus vanilla without matching semantic gains.

The claim is weakened if:

1. gains appear only against static consensus, not against adaptive ungated blending;
2. paraphrase coverage is too low in numeracy or relations;
3. automated and human slot judgments disagree substantially.

The claim is refuted if ParaDG fails to beat the adaptive ungated paraphrase baseline, or if robustness gains disappear under human adjudication.

## References

Chefer, Hila, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. 2023. "Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models." ACM Transactions on Graphics 42(4).

Feng, Qianyu, Yulei Sui, and Hongyu Zhang. 2023. "Uncovering Limitations in Text-to-Image Generation: A Contrastive Approach with Structured Semantic Alignment." Findings of the Association for Computational Linguistics: EMNLP 2023.

Huang, Kaiyi, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. 2025. "T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation." IEEE Transactions on Pattern Analysis and Machine Intelligence.

Lee, Kwanyoung, SeungJu Cha, Yebin Ahn, Hyunwoo Oh, Sungho Koh, and Dong-Jin Kim. 2026. "Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation." arXiv preprint arXiv:2603.19158. Accepted to CVPR 2026.

Rombach, Robin, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. 2022. "High-Resolution Image Synthesis with Latent Diffusion Models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Rosenman, Shachar, Vasudev Lal, and Phillip Howard. 2024. "NeuroPrompts: An Adaptive Framework to Optimize Prompts for Text-to-Image Generation." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision.

Shen, Yifan, Yangyang Shu, Hye-young Paik, and Yulei Sui. 2025. "MetaLogic: Robustness Evaluation of Text-to-Image Models via Logically Equivalent Prompts." International Conference on Formal Engineering Methods.

Wang, Shulei, Wang Lin, Hai Huang, Hanting Wang, Sihang Cai, WenKang Han, Tao Jin, Jingyuan Chen, Jiacheng Sun, Jieming Zhu, and Zhou Zhao. 2025. "Towards Transformer-Based Aligned Generation with Self-Coherence Guidance." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Zhou, Yupeng, Daquan Zhou, Zuo-Liang Zhu, Yaxing Wang, Qibin Hou, and Jiashi Feng. 2023. "MaskDiffusion: Boosting Text-to-Image Consistency with Conditional Mask." arXiv preprint arXiv:2309.04399.
