# When Assignment Matters: A Grounded Study of Compositional Text-to-Image Reranking

## Introduction

Compositional text-to-image failures are now well documented: requested objects disappear, colors swap across entities, and left/right relations invert even when the image is otherwise photorealistic. Benchmarks such as GenEval, T2I-CompBench, and the newer CompAlign benchmark show that these errors remain common for modern diffusion models.

This proposal is intentionally narrow. It is not a new generation method, not a denoising-time controller, and not a claim that DAAM is a generally reliable localization method. Instead, the paper is framed as an **incremental but rigorous assignment-analysis study**: under a fixed SDXL best-of-4 candidate budget, does better prompt-slot-to-region assignment produce better post-hoc reranking?

The motivation is that the closest reranking and evaluator baselines already do much of what a reviewer would expect. LogicRank already composes logical evidence across prompt atoms. Crop-structured inference methods already show that adding simple structure at inference helps compositional scoring. Newer evaluator work such as CompQuest and T2I-FineEval already decomposes prompts into atomic checks and queries image regions or question-answer pairs. The remaining defensible gap is narrower: **existing methods do not isolate whether final reranking errors come primarily from wrong region assignment rather than from the downstream verifier itself.**

The paper therefore asks a falsifiable question:

**If we keep generation fixed and compare rerankers on the same SDXL candidate cache, does DAAM-assisted assignment improve reranking only when it measurably improves slot assignment accuracy over detector-only and crop-only alternatives?**

If the answer is no, the contribution is still a useful negative result about the limits of generator-native attention traces for post-hoc verification. If the answer is yes, the contribution is a carefully controlled grounded verifier study rather than a new compositional generation paradigm.

## Proposed Approach

### Overview

The method, which I will still call **Assign-and-Verify (A&V)** for convenience, is a training-free reranker operating on a fixed cache of SDXL candidates. For each prompt-seed pair:

1. generate four SDXL candidates once,
2. parse the prompt into object groups and atomic constraints,
3. detect candidate boxes with GroundingDINO-Tiny,
4. capture and cache lightweight DAAM phrase traces from the same SDXL run,
5. solve explicit slot-to-box assignment with `null` nodes and exchangeable repeated-object slots,
6. score attributes and relations on the assigned regions, and
7. rerank the four candidates with a shared soft conjunction over atom scores.

The generator itself is untouched. There is no repair loop, no reward model, and no extra diffusion pass after candidate generation.

### Prompt Representation

The parser supports the same atom families used in the main evaluation:

- `object_groups`: noun phrases with optional attributes and requested counts
- `count_atoms`: exact cardinality requests
- `attribute_atoms`: color or simple adjectival attributes linked to one group
- `relation_atoms`: `left of`, `right of`, `above`, `below`

Prompts whose semantics are not cleanly grounded with 2D boxes are excluded from the main test set. This is important because the paper’s main claim is about assignment quality, so hidden semantic ambiguity would contaminate the core metric.

### Repeated Objects And Null Semantics

Repeated identical objects are modeled explicitly. For `two red apples`, the parser creates one group `G_apple` with count `2` and two exchangeable demand units `apple#1` and `apple#2`. The units exist only for matching. During evaluation they are collapsed back into an unordered matched set.

Each demand unit can match either:

- one real box, or
- one explicit `null` node

Each real box has capacity `1`. This makes missing-object and under-count failures explicit rather than forcing a verifier to hallucinate correspondences.

For a group `g` with requested count `c_g` and matched distinct non-null boxes `m_g`, the group count score is:

`count_score(g) = exp(-|m_g - c_g|) * exp(-0.5 * extra_box_penalty(g))`

where `extra_box_penalty(g)` counts high-confidence leftover detections of the same noun when the prompt requests an exact count.

### DAAM Trace Capture And Caching For SDXL

Feasibility depends on not storing raw SDXL attention tensors. The proposal therefore uses an **online aggregation** scheme:

- hook SDXL cross-attention only for the object and attribute token spans that appear in the parsed prompt,
- keep only the final `10` denoising steps,
- average across heads and across all upsampling blocks during generation,
- accumulate directly into one `64 x 64` latent-space heatmap per phrase per step,
- save only the final phrase-step arrays as compressed `fp16` `npz` files.

This avoids saving full per-layer attention volumes. With roughly `4-6` tracked phrases per prompt and `10` saved steps, storage is about `0.4-0.6 MB` per image. For the full `2400`-image main cache, DAAM traces occupy roughly `1.0-1.5 GB`, which is realistic on local disk and can be reused across all rerankers and ablations.

For each phrase heatmap, A&V later:

- upsamples once to image space,
- normalizes the map to sum to `1`,
- computes attention mass inside each candidate box,
- caches that scalar box-level feature for all downstream scoring.

This makes the DAAM contribution cheap at reranking time; the expensive part is paid once during candidate generation.

### Region Proposals And Assignment

For each generated image:

- run `GroundingDINO-Tiny` on each noun phrase,
- keep at most the top `4` post-NMS boxes per phrase,
- compute for slot `s` and box `b`:

`compat(s, b) = z(det_conf(s, b)) + z(attn_mass(s, b)) + z(size_prior(s, b))`

where:

- `det_conf` is detector confidence,
- `attn_mass` is normalized DAAM mass inside the box,
- `size_prior` weakly penalizes boxes smaller than `1%` or larger than `60%` of the image,
- `z(.)` is computed across candidate boxes for that phrase within the image.

Assignment is solved as a small capacitated min-cost matching problem with explicit `null` nodes. For every slot or demand unit, the pipeline caches:

- matched box or `null`,
- runner-up box,
- assignment margin,
- grouped matched set for repeated objects.

These cached outputs are central to the paper because they let the study measure whether DAAM changes assignment itself rather than only the final image score.

### Atomic Verification

The verifier is deliberately simple because the paper’s novelty claim should sit in assignment analysis, not in a large verification stack.

#### Existence and count

- unique-slot existence depends on whether the final assignment is non-null and on assignment margin
- repeated-object groups use the explicit matched-set count score

#### Attribute binding

Given assigned crop `c_s` for slot `s`, compute a positive phrase `p+` and a set of in-prompt counterfactual negatives `N(s)`. Example: from `red cube and blue sphere`, negatives for `red cube` include `blue cube` and `red sphere`.

The attribute score is:

`attr_score(s) = sigmoid((sim(c_s, p+) - max_{n in N(s)} sim(c_s, n)) / tau_attr)`

with `sim` given by SigLIP image-text similarity and `tau_attr` fixed on the dev set.

#### Relations

For `left of`, `right of`, `above`, and `below`, relation scores use assigned box geometry:

`rel_score(a, b, r) = sigmoid(margin_r(box_a, box_b) / tau_r)`

where `tau_r` is fixed on the dev set.

### Aggregation

All structured rerankers use the same aggregation:

`score(image) = exp(mean_j log(eps + atom_score_j))`

with `eps = 1e-4`.

Sharing the same aggregation across methods matters because it ensures the comparison isolates assignment quality and counterfactual verification rather than trivial score-combination tricks.

## Related Work

### Positioning

The proposal should be positioned as a post-generation verifier study that sits between compositional benchmarks and generation-time binding methods. The closest works are already strong, so the contribution needs to be stated conservatively.

### Comparison against nearest prior work

| Method | Main role | Crop or region selection | Logical or atomic composition | Explicit slot assignment | Generator-native grounding | Post-hoc verification |
| --- | --- | --- | --- | --- | --- | --- |
| LogicRank (Deiseroth et al., 2022) | Reranking | Uses structured visual evidence but not assignment as the main object of analysis | Yes | Implicit / limited | No | Yes |
| Adding simple structure at inference (Miranda et al., 2025) | Structured scoring baseline | Yes, crop-structured | Yes | Partial, via crop matching | No | Yes |
| T2I-FineEval (Hosseini et al., 2025) | Evaluator | Yes, detector-derived entity and relation components | Yes, question decomposition | Partial, question-component matching | No | Yes |
| CompQuest in CompAlign (Wan and Chang, 2025) | Evaluator and alignment signal | Mostly whole-image with atomic sub-questions | Yes | No | No | Yes |
| A&V (this proposal) | Reranking analysis study | Yes, detector boxes | Yes | **Yes, primary measured variable** | **Yes, via DAAM** | Yes |

This table is the central novelty claim. A&V is not meaningfully more novel than these works unless explicit assignment accuracy becomes the primary measured quantity and DAAM yields a measurable separation from detector-only and crop-only baselines.

### Benchmarks and evaluators

GenEval and T2I-CompBench define the core failure modes: object presence, count, attribute binding, and spatial relations. CompAlign extends the evaluation space to more complex multi-subject and 3D relation settings, while CompQuest and T2I-FineEval show that atomic-question and region-aware evaluators are already a live research direction. TIFA and VQAScore further show that decomposition-based and verifier-style evaluation is already well populated. These papers tighten the novelty bar and justify why this proposal should not oversell itself as a general compositional evaluator.

### Generation-time semantic binding

Token Merging, VSC, R-Bind, and InfSplign all intervene during or around generation to improve semantic binding or spatial alignment. Those methods occupy the generation-time novelty space. A&V is orthogonal: it assumes the user already has a small candidate pool and wants a better post-hoc selector without changing the diffusion process.

### DAAM as an uncertain grounding cue

DAAM was introduced for interpretability, not verification. The proposal therefore treats it as a noisy auxiliary signal. The paper’s mechanism claim succeeds only if DAAM materially improves assignment accuracy and the improvement tracks downstream reranking gains. Otherwise the correct conclusion is that DAAM is interesting for visualization but too unstable to anchor a strong verifier.

## Experiments

### Main question

Under a fixed shared SDXL best-of-4 cache, does DAAM-assisted assignment improve explicit assignment metrics and then improve downstream compositional reranking over detector-only and crop-only structured baselines?

### Candidate cache and data splits

Generation setup:

- model: `stabilityai/stable-diffusion-xl-base-1.0`
- resolution: `512 x 512`
- sampler: Euler ancestral
- steps: `30`
- CFG: `7.5`
- seeds: `[11, 22, 33]`
- candidates per prompt-seed: `4`

Data:

- dev set: `40` prompts
- main test set: `200` prompts
- transfer set: `30` hand-written prompts outside benchmark templates

Main test composition:

- `120` GenEval prompts covering `count`, `color`, and `position`
- `80` T2I-CompBench prompts restricted to `attribute binding` and `left/right/above/below` relations

Total main cache:

`200 prompts x 3 seeds x 4 candidates = 2400 images`

All rerankers operate on the exact same cached candidates.

### Baselines

1. `SDXL single-sample` using `cand_0`
2. `best-of-4 + global SigLIP`
3. `best-of-4 + detector-only structured reranking`
4. `best-of-4 + crop-structured SigLIP reranking`
5. `best-of-4 + DAAM assignment without counterfactual negatives`
6. `Assign-and-Verify` full method

The paper lives or dies on comparisons `3` through `6`, because those are the closest alternatives.

### Primary metrics

The exact primary metrics are:

1. **slot-assignment accuracy** on a manually annotated `100` prompt-image pair set
2. **all-atoms-pass** on the selected image for each prompt-seed pair
3. **GenEval overall accuracy** on the selected image under the shared cache

Mechanism-supporting secondary metrics:

- `group_count_accuracy` for repeated-object prompts
- `null_decision_accuracy` for missing-object and under-count cases
- GenEval per-category accuracy on `count`, `color`, and `position`
- T2I-CompBench attribute-binding and relation scores
- pairwise human preference win rate on disagreement cases between the full method and the strongest non-DAAM structured baseline
- verifier latency per image and end-to-end latency per prompt-seed

This metric list intentionally matches the experimental plan framing: assignment accuracy and all-atoms-pass are the paper-centric results, while benchmark metrics provide external validity.

### Manual grounding protocol

The annotation set is sampled after generation so it reuses the shared cache. The set contains `100` prompt-image pairs, stratified over:

- unique-object cases,
- repeated-object count cases,
- attribute binding cases,
- relation cases,
- detector failure cases where `null` is plausible.

Annotators label:

- the correct box for each unique slot, or `null`,
- the correct unordered set of boxes for each repeated-object group,
- whether each parsed atom is satisfied.

To keep the human burden realistic, the study annotates the main `100`-pair set once, then double-labels a `25`-pair overlap for adjudication and agreement checks. With about `2-3` groups per prompt-image pair, this is roughly `250-300` grounding decisions. At about `45-60` seconds per pair for the first pass plus targeted adjudication, the total human time is roughly `2.0-2.5` hours. That is realistic and separate from the GPU budget.

As a secondary sanity check rather than a primary claim, the study also samples up to `60` prompt-seed disagreements between the full method and the strongest non-DAAM structured baseline. Annotators are shown only the prompt and the two selected images and asked which image better satisfies the prompt, with `tie` allowed. This provides a small human reranking check without turning the project into a large annotation effort.

### Ablations

The ablations are aligned with the planned artifact and limited to what the compute budget can support:

1. detector-only assignment vs detector + DAAM assignment
2. positive-only attribute scoring vs in-prompt counterfactual scoring
3. remove explicit `null` handling and exact count penalty
4. geometric-mean soft conjunction vs arithmetic-mean aggregation
5. best-of-4 vs best-of-8 on a `40`-prompt subset

The first and third ablations are most important. If DAAM does not improve assignment, or if null/count handling contributes nothing on repeated-object prompts, the claimed mechanism weakens substantially. The fourth ablation is included only to show that any gain is not an artifact of the score combiner.

### Statistical testing

Use paired bootstrap resampling with `1000` resamples:

- over prompt-seed pairs for reranking metrics,
- over annotated prompt-image pairs for grounding metrics.

Also run McNemar tests on the binary exact-assignment outcome for the manual set.

## Success Criteria

The hypothesis is supported only if all three conditions hold:

1. A&V improves slot-assignment accuracy by at least `+8` absolute points over detector-only structured assignment.
2. A&V improves `all_atoms_pass` by at least `+3` points over the strongest non-DAAM structured baseline.
3. A&V improves GenEval overall by at least `+4` points over best-of-4 global SigLIP reranking on the same candidate cache.

The paper should be written as a negative or cautionary empirical study if:

- DAAM does not improve assignment materially,
- gains vanish against the crop-structured baseline,
- or improvements appear only on benchmark templates and not on the transfer prompts.

## Resource Budget

The work is scoped to one RTX A6000 with an eight-hour total experimental budget.

### Shared generation cache

- main cache: `200 x 3 x 4 = 2400` images
- dev cache: `40 x 3 x 4 = 480` images
- best-of-8 subset: `40 x 3 x 8 = 960` images

At a conservative `5.0 s` per `512 x 512` SDXL image:

- main cache: about `3.33` hours
- dev cache: about `0.67` hours
- best-of-8 subset: about `1.33` hours

### Shared verifier features

Per image:

- GroundingDINO-Tiny: `0.35 s`
- SigLIP crop scoring and cached assignment features: `0.30 s`

The DAAM overhead is budgeted inside generation rather than as a near-zero post-hoc cost, because the traces are captured online during SDXL sampling. At a conservative `5.5 s` per `512 x 512` SDXL image with DAAM hooks enabled, the total generation budget already includes that tracing cost. Shared post-generation verifier extraction is therefore roughly `0.65 s` per image, or about `0.43` hours for the `2400`-image main cache. Baseline rescoring and ablation rescoring over cached features add about `0.55` hours.

### Total

Conservative total wall-clock budget:

- generation with online DAAM tracing: about `5.87` hours
- shared verifier extraction on the main cache: `0.43` hours
- baseline and ablation rescoring: `0.55` hours
- benchmark scripts, bootstrap statistics, plots, and pairwise disagreement sampling: `0.55` hours

Total GPU-critical wall-clock time is about `7.40` hours, leaving roughly `0.60` hours of slack. Human annotation is separate from the GPU budget.

This budget is feasible because:

- generation is shared across all methods,
- DAAM traces are captured once and cached compactly,
- all rerankers reuse the same box proposals and crop features,
- only the small best-of-8 subset requires extra image generation.

## References

1. Deiseroth, B., Schramowski, P., Shindo, H., Dhami, D. S., and Kersting, K. 2022. LogicRank: Logic Induced Reranking for Generative Text-to-Image Systems. arXiv:2208.13518.
2. Tang, R., Liu, L., Pandey, A., Jiang, Z., Yang, G., Kumar, K., Stenetorp, P., Lin, J., and Ture, F. 2023. What the DAAM: Interpreting Stable Diffusion Using Cross Attention. ACL.
3. Ghosh, D., Hajishirzi, H., and Schmidt, L. 2023. GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. NeurIPS Datasets and Benchmarks.
4. Huang, K., Sun, K., Xie, E., Li, Z., and Liu, X. 2023. T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation. NeurIPS Datasets and Benchmarks.
5. Hu, T., Li, L., van de Weijer, J., Gao, H., Khan, F., Yang, J., Cheng, M.-M., Wang, K., and Wang, Y. 2024. Token Merging for Training-Free Semantic Binding in Text-to-Image Synthesis. NeurIPS.
6. Hosseini, S. M. H., Izadi, A. M., Abdollahi, A., Saghafian, A., and Soleymani Baghshah, M. 2025. T2I-FineEval: Fine-Grained Compositional Metric for Text-to-Image Evaluation. arXiv:2503.11481.
7. Wan, Y., and Chang, K.-W. 2025. CompAlign: Improving Compositional Text-to-Image Generation with a Complex Benchmark and Fine-Grained Feedback. arXiv:2505.11178.
8. Hu, X., Li, X., Zhou, Y., and Loy, C. C. 2024. TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering. ICCV.
9. Lin, Z., Pathak, D., Zhang, R., and Agrawala, M. 2024. VQAScore: Evaluating and Selecting Text-to-Image Generation with Visual Question Answering. ECCV.
10. Miranda, I., Salaberria, A., Agirre, E., and Azkune, G. 2025. Adding simple structure at inference improves Vision-Language Compositionality. arXiv:2506.09691.
11. Wen, S., Fang, G., Zhang, R., Gao, P., Dong, H., and Metaxas, D. 2023. Improving Compositional Text-to-image Generation with Large Vision-Language Models. arXiv:2310.06311.
12. Dat, D. H., Nam, H.-W., Mao, P.-Y., and Oh, T.-H. 2025. VSC: Visual Search Compositional Text-to-Image Diffusion Model. ICCV.
13. Zhang, H., and Wan, X. 2025. R-Bind: Unified Enhancement of Attribute and Relation Binding in Text-to-Image Diffusion Models. EMNLP.
14. Rastegar, S., Chatalbasheva, V., Falkena, S., Singh, A., Wang, Y., Gokhale, T., Palangi, H., and Jamali-Rad, H. 2025. InfSplign: Inference-Time Spatial Alignment of Text-to-Image Diffusion Models. arXiv:2512.17851.
