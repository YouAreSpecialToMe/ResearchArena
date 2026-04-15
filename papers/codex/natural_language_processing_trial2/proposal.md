# LIMS-RAG: Localized Minimal-Support Perturbations as a Practical Feature Variant for Sentence-Level RAG Verification

## Introduction

Retrieval-augmented generation (RAG) reduces hallucination by exposing models to external evidence, but recent work makes a sharper point: a response can be correct or even supported after the fact without being faithful to the evidence actually used during generation. This matters for deployment because many lightweight verification pipelines still rely mainly on support detection, while newer faithfulness methods often require richer model access, larger systems, or broader evaluation frameworks than a small lab can reproduce under modest compute.

This proposal deliberately targets a narrower question than a new verification framework:

Can localized perturbations around a greedily predicted support subset provide a useful feature increment over a strong support-only sentence verifier, over support-plus-compactness features, and over coarse full-context removal?

The intended contribution is modest but testable. The method does not claim optimal evidence attribution, principled causal identification, or a new benchmark. It evaluates whether one practical feature variant, localized support-set perturbation, adds measurable signal for sentence-level hallucination detection in RAG. If the gains vanish after stronger controls for label noise, threshold sensitivity, or baseline choice, that negative result is still informative.

The work is scoped to the available budget: one RTX A6000, 60 GB RAM, 4 CPU cores, and roughly 8 hours total experiment time. No generator training is required. All models are small retrieval, reranking, or NLI components, and all evidence comes from benchmark-provided context.

## Proposed Approach

### Task framing

Input:

- a query,
- benchmark-provided retrieved context,
- a generated answer,
- sentence-level labels derived from benchmark annotations.

Output:

- a probability that each answer sentence is unsupported or unfaithful with respect to the provided context.

The paper is a post-hoc detector study. It asks whether a localized necessity signal helps under the exact retrieval context already supplied by the benchmark.

### Conservative supervision setup

`RAGTruth` provides span annotations, not clean sentence labels. The proposal therefore treats sentence labeling as an explicit source of uncertainty rather than hidden ground truth.

Three evaluation slices will be reported:

- `projected-all`: permissive projection for training efficiency and recall-oriented analysis,
- `strict-label`: ambiguity-controlled sentence labels used for the main claim,
- `audited-subset`: manually re-labeled examples for uncertainty-aware analysis.

The headline result will be on `strict-label`, not on permissive projection alone. A sentence enters the strict positive slice only if hallucinated spans dominate the sentence and no adaptive or mixed annotation makes the label ambiguous. Strict negatives must have zero overlap with hallucinated spans. Ambiguous cases are dropped.

### Base support verifier

The anchor system is a strong support-only verifier:

- sentence retrieval over the provided context with `intfloat/e5-base-v2`,
- reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`,
- support scoring with `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.

For each answer sentence, the baseline predicts hallucination from the best support score available in the benchmark context. This is the main baseline. Any added feature must beat this system first.

### Greedy minimal support set `S_min`

For each answer sentence, candidate evidence sentences are ranked and a small support subset `S_min` is built greedily:

1. start from the highest-ranked evidence sentence,
2. score support on the current subset,
3. add the next sentence while support is below a threshold `tau`,
4. stop when support crosses `tau` or a cap of 3 to 4 sentences is reached.

This is a heuristic approximation, not a claim of minimality. The proposal explicitly tests whether the resulting feature remains useful despite heuristic search noise.

Core non-perturbation features:

- support score on full context,
- support score on `S_min`,
- `|S_min|`,
- support margin between `S_min` and the best single-sentence support,
- number of alternative support candidates above threshold,
- dispersion of `S_min` across source documents,
- redundancy proxy from leave-one-out support within `S_min`.

### Localized perturbation features

After constructing `S_min`, compute necessity-style interventions centered only on that subset:

1. `remove-local`: remove all sentences in `S_min` from the context,
2. `drop-one`: remove each support sentence in turn,
3. `swap-local`: replace each support sentence with a semantically nearby but low-support distractor from the same example.

Derived features:

- support drop from full context to `remove-local`,
- support drop from `S_min` to each leave-one-out variant,
- worst and mean leave-one-out drop,
- support drop under `swap-local`,
- best residual support after local removal,
- change in top-ranked evidence bundle after local removal,
- normalized drop relative to original support confidence.

The novelty claim is only this: localized support-set perturbation as a practical feature variant for sentence-level RAG verification.

### Final detector

The final detector remains intentionally simple:

- logistic regression with class balancing,
- z-scored support, compactness, redundancy, and localized perturbation features,
- thresholds selected on validation only.

A training-free additive score will also be reported to test whether any gain comes from the signal itself rather than from flexible feature recombination.

## Related Work

### Lightweight support verification

`MiniCheck` (Tang et al., 2024) and `Auto-GDA` (Leemann et al., 2025) are the direct support-verification lineage. They show that efficient grounding verification is possible with compact models and synthetic adaptation. The proposed paper does not compete with them as a replacement verifier; it uses this family as the base detector and asks whether a localized necessity feature adds value.

### Claim-level and sentence-level RAG faithfulness monitoring

`RAGChecker` (Ru et al., 2024) is the closest broad diagnostic framework. It already combines claim decomposition, faithfulness analysis, and noise sensitivity. `SynCheck` (Wu et al., 2024) is especially important for positioning because it also combines lightweight signals for sentence-level faithfulness monitoring in RAG, albeit during decoding and with access to token-level generation dynamics. These papers narrow the novelty boundary substantially: the present work is not the first sentence-level faithfulness detector and not the first lightweight signal combination for RAG faithfulness.

### Context sensitivity and self-consistency signals

`REFIND` (Lee and Yu, 2025) detects hallucination through coarse context sensitivity, while `SelfCheckGPT` (Manakul et al., 2023) uses self-consistency variation for sentence-level hallucination detection outside the RAG setting. They motivate the idea that support alone is incomplete, but their perturbations are global or sampling-based rather than localized around an inferred support subset.

### Attribution and correctness-versus-faithfulness

`Evidence Contextualization and Counterfactual Attribution for Conversational QA over Heterogeneous Data with RAG Systems` (Roy et al., 2025) studies counterfactual evidence removal for attribution, and `Correctness is not Faithfulness in RAG Attributions` (Wallat et al., 2024) argues that correct support can still reflect post-rationalization. These works motivate the local-removal signal while also limiting the scope of the claim: the proposal borrows attribution intuition for a small detector feature, not a new attribution theory.

### Context sufficiency and conflict-faithful generation

`Sufficient Context` (Joren et al., 2024) and `FaithfulRAG` (Zhang et al., 2025) emphasize that RAG errors arise from insufficient or conflicting evidence, not just unsupported text. The current proposal is narrower: it asks whether localized perturbation around a compact support subset helps detect such cases with a cheap post-hoc classifier.

### Positioning and novelty boundary

The paper should explicitly claim only the following:

- it is not the first claim-level RAG diagnostic system,
- it is not the first sentence-level faithfulness monitor,
- it is not the first context-sensitivity detector,
- it is not the first counterfactual evidence-removal method,
- it is a careful empirical test of whether localized support-set perturbation is a better practical detector feature than support alone, support-plus-compactness, or full-context removal under a shared low-cost setup.

## Experiments

### Primary research questions

1. Does localized support-set perturbation improve over a strong support-only verifier on `RAGTruth`?
2. Does it still help against the strongest in-setting baselines: support-plus-compactness and full-context removal?
3. Are any gains stable across reasonable `S_min` thresholds rather than artifacts of greedy search or threshold tuning?
4. Do localized features remain informative on ambiguity-controlled labels and on a manually audited subset?

### Dataset and splits

- Primary benchmark: `RAGTruth`
- Main slices: `projected-all`, `strict-label`, `audited-subset`
- Robustness setting 1: held-out generator family within `RAGTruth`
- Robustness setting 2: held-out task type within `RAGTruth`

No headline claim will rely on external transfer. Optional out-of-domain testing is dropped from the main proposal to preserve compute and focus.

### Comparator hierarchy

Primary comparators under the same scaffold:

- support-only,
- support-plus-compactness,
- support-plus-compactness-plus-redundancy,
- full-context removal,
- localized perturbation only,
- full feature model.

Secondary, clearly qualified approximations of nearby prior ideas:

- `REFIND-style` coarse context sensitivity,
- `RAGChecker-style` sentence-level diagnostic proxy.

The proposal will not claim superiority to original published systems whose full pipelines are not faithfully reproduced.

### `S_min` calibration and stability analysis

Because `S_min` is heuristic, threshold sensitivity is a first-class experimental question.

Calibration protocol:

- choose `tau` on validation by Brier score and AUPRC for the support-only verifier,
- evaluate a small grid of thresholds around the chosen `tau`,
- lock one threshold before test evaluation.

Stability analysis:

- report how often `S_min` changes across adjacent thresholds,
- report mean Jaccard similarity of `S_min` across threshold settings,
- report the fraction of examples whose localized feature sign flips,
- stratify detector performance by stable versus unstable `S_min` examples.

If the gains arise only where `S_min` is highly unstable, the central claim weakens substantially.

### Experimental protocol

1. Run a pilot on 100 validation sentences to set retrieval depth, max `S_min` size, and threshold grid.
2. Build permissive and strict sentence-label projections from `RAGTruth`.
3. Train and validate the support-only verifier.
4. Compute support-plus-compactness and redundancy baselines.
5. Compute full-context removal features.
6. Compute localized perturbation features.
7. Train logistic regression and training-free additive variants.
8. Evaluate once on the locked test protocol for all three slices.
9. Run held-out-generator and held-out-task robustness tests with the same frozen pipeline.
10. Run threshold-stability and calibration analyses.

### Audited subset and uncertainty reporting

The manual analysis will be expanded to 300 sentences if annotation time permits; otherwise it will remain at 200 with stronger uncertainty reporting. The sample is stratified by:

- strict versus ambiguous projection,
- support-only confidence,
- localized perturbation score,
- disagreement between support-only and localized detectors.

Each example receives two independent annotations plus adjudication.

Labels:

- evidence-dependent,
- post-rationalized,
- redundantly-supported,
- insufficient-context,
- unclear.

Analysis protocol:

- report raw counts and class proportions with bootstrap 95% confidence intervals,
- report Cohen's kappa and binary agreement after collapsing to `evidence-dependent` versus `post-rationalized`,
- report detector AUC with bootstrap intervals,
- avoid strong conclusions if intervals overlap materially or agreement is low.

This manual study is for interpretation and stress-testing, not for the main benchmark headline.

### Metrics

Primary metrics:

- macro F1,
- AUPRC,
- AUROC.

Secondary metrics:

- Brier score,
- expected calibration error,
- response-level hallucination F1,
- performance on stable-`S_min` and unstable-`S_min` subsets,
- audited-subset AUC with bootstrap intervals.

### Ablations

- remove localized perturbation features,
- remove compactness and redundancy features,
- top-1 evidence instead of greedy `S_min`,
- full-context removal versus local removal,
- removal-only versus swap-only versus leave-one-out-only,
- learned logistic regression versus training-free additive scoring,
- strict versus permissive labels,
- stable-`S_min` examples only.

### Feasibility under the compute budget

The proposal fits the stated resources because:

- all models are small encoders or NLI classifiers,
- no generator retraining is needed,
- evidence search stays within benchmark-provided context,
- each perturbation touches only a few sentences,
- the final classifier is linear,
- robustness tests reuse the same cached feature pipeline.

If runtime becomes tight, the fallback order is:

1. keep `strict-label`, support-only, support-plus-compactness, full-context removal, and localized-feature comparisons,
2. keep threshold-stability analysis,
3. keep the audited subset,
4. drop held-out-task evaluation before dropping held-out-generator evaluation.

## Success Criteria

The proposal is supported only if most of the following hold on `strict-label`:

1. The full detector improves over support-only by at least 1.0 macro-F1 point or 0.01 AUPRC.
2. It improves over support-plus-compactness and over full-context removal by at least 0.5 macro-F1 points or 0.005 AUPRC.
3. The gain remains directionally positive across the locked threshold neighborhood for `tau`.
4. On stable-`S_min` examples, gains do not collapse relative to the full strict slice.
5. On the audited subset, localized perturbation improves discrimination between `evidence-dependent` and `post-rationalized` cases, with uncertainty intervals that do not make the effect indistinguishable from zero.

The hypothesis is weakened or refuted if:

- gains appear only on permissive projections,
- support-plus-compactness explains nearly all of the improvement,
- full-context removal matches localized perturbation,
- the feature is beneficial only under unstable `S_min` settings,
- or the audited analysis is too uncertain to distinguish evidence dependence from post-rationalization.

## References

1. Tobias Leemann, Periklis Petridis, Giuseppe Vietri, Dionysis Manousakas, Aaron Roth, and Sergul Aydore. 2025. *Auto-GDA: Automatic Domain Adaptation for Efficient Grounding Verification in Retrieval-Augmented Generation*. In International Conference on Learning Representations.
2. Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, and Cyrus Rashtchian. 2024. *Sufficient Context: A New Lens on Retrieval-Augmented Generation Systems*. arXiv:2411.06037.
3. Jonas Wallat, Maria Heuss, Maarten de Rijke, and Avishek Anand. 2024. *Correctness is not Faithfulness in RAG Attributions*. arXiv:2412.18004.
4. Liyan Tang, Philippe Laban, and Greg Durrett. 2024. *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents*. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing.
5. Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong, Juntong Song, and Tong Zhang. 2024. *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models*. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics.
6. Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang, Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang, and Zheng Zhang. 2024. *RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation*. arXiv:2408.08067.
7. Rishiraj Saha Roy, Joel Schlotthauer, Chris Hinze, Andreas Foltyn, Luzian Hahn, and Fabian Kuech. 2025. *Evidence Contextualization and Counterfactual Attribution for Conversational QA over Heterogeneous Data with RAG Systems*. In Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining.
8. Di Wu, Jia-Chen Gu, Fan Yin, Nanyun Peng, and Kai-Wei Chang. 2024. *Synchronous Faithfulness Monitoring for Trustworthy Retrieval-Augmented Generation*. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing.
9. DongGeon Lee and Hwanjo Yu. 2025. *REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models*. In Proceedings of the 19th International Workshop on Semantic Evaluation.
10. Potsawee Manakul, Adian Liusie, and Mark Gales. 2023. *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models*. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing.
11. Qinggang Zhang, Zhishang Xiang, Yilin Xiao, Le Wang, Junhui Li, Xinrun Wang, and Jinsong Su. 2025. *FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation*. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics.
