# StressAudit-SM: A Compact Robustness-Audit Protocol for Schema Matching Under Metadata Stress

## Introduction

This proposal is an evaluative paper, not a new schema-matching method paper. The problem is practical and narrow: schema matchers are usually reported on clean benchmarks, but data integration failures often happen after metadata quality drops, such as when headers are abbreviated, descriptions disappear, or table-level context is incomplete. Current literature already covers pieces of this story, so the contribution must be framed conservatively.

The closest combined prior art is:

- *Making holistic schema matching robust: an ensemble approach* (He and Chang, 2005), which studies robustness to noisy schema inputs.
- *Prompt-Matcher* (Feng et al., 2024), which studies uncertainty reduction in schema matching.
- *ConStruM* (Chen et al., 2026), which introduces an explicit context-stress benchmark for LLM-based schema matching.

Because these ingredients already exist in adjacent forms, the defensible gap is not "robustness has never been studied" and not "uncertainty in schema matching is new." The gap is that there is still no small, reproducible, CPU-feasible audit recipe that measures, on the same fixed public schema pairs:

1. accuracy degradation under deterministic metadata stress,
2. ranking stability across matcher families,
3. calibration transfer from clean to stressed inputs, and
4. abstention utility at fixed coverage.

The paper's contribution is therefore a compact protocol, called **StressAudit-SM**, plus the empirical findings produced by that protocol. The protocol is intentionally designed to fit `2` CPU cores, `128 GB` RAM, and roughly `8` hours total runtime without GPU training or multi-seed repetition.

**Hypothesis.** On a fixed set of public schema pairs, deterministic header and context stress will change method rankings, degrade clean-fit calibration, and increase the value of abstention, even when the compared methods look similar under clean evaluation.

## Proposed Approach

### 1. Contribution type

StressAudit-SM is a robustness-audit layer over existing public schema-matching pairs. It contributes:

- deterministic stress transforms,
- one fixed split reused everywhere,
- one fixed evaluation matrix,
- one fixed calibration protocol,
- one fixed runtime budget and stop rule.

It does **not** claim:

- a new benchmark family,
- a new state-of-the-art matcher,
- a new training method,
- a large stress-severity sweep.

### 2. Final protocol

The study uses exactly `52` public schema pairs drawn from three public dataset families available in the local Valentine archive: one Valentine-compatible slice, one header-poor slice, and one structure-rich slice with table/type context but no usable descriptions. The split is fixed once and reused for every method and condition:

- `12` clean development pairs,
- `8` stressed development pairs,
- `32` held-out test pairs.

This split is the only split used in the paper. There are no alternative counts, no seed-averaging, and no later resampling stage.

### 3. Stress conditions

The main protocol evaluates four conditions:

- `clean`: original metadata.
- `header-stress`: deterministic abbreviation, token deletion, identifier normalization, and typo injection on column names.
- `context-stress`: removal of shared non-value context, namely table names, neighboring-schema context, and type-context fields.
- `composite-stress`: header-stress plus context-stress.

The stress operators are deterministic and parameter-free once implemented. There is no severity grid in the main paper. Row-budget restriction is excluded from the main experiment and, if run at all, appears only in an appendix.

### 4. Compared methods

To keep the study feasible and aligned with its evaluative contribution, the main matrix compares exactly three methods:

- a lexical matcher built from standard string-similarity signals,
- Similarity Flooding as the classical structural baseline,
- the PTLM baseline from *Schema Matching using Pre-Trained Language Models*.

This reduction is deliberate. The paper does not need a new fourth method to make its claim. A lightweight structure-aware reference model, `StressMatch-Lite`, may be included only as appendix material if time remains after the full main protocol completes, but it is not part of the paper's core claim.

### 5. Calibration and abstention

Each method must emit a scalar score per candidate correspondence. The audit uses one simple protocol:

- fit one logistic calibrator per method on the `12` clean development pairs,
- choose one abstention threshold per method on the same clean development pairs,
- use the `8` stressed development pairs only for one predeclared transfer check,
- report all final claims on the `32` held-out test pairs.

There is no isotonic regression, no multi-threshold sweep for headline claims, and no seed-based confidence averaging.

### 6. Exact execution matrix and stop conditions

The core study executes exactly:

- `32 x 3 x 4 = 384` held-out method-condition evaluations on test pairs,
- plus development-time runs on the fixed `20` development pairs for calibration and threshold setting.

The paper precommits to these execution-time rules:

- the main paper is complete when the three-method, four-condition, `52`-pair protocol has finished;
- appendix analyses run only if the main protocol finishes within `6.5` hours wall-clock time;
- if the projected runtime of the already-fixed main matrix exceeds `7.5` hours after the first `8` test pairs, appendix work is cancelled automatically and only the main matrix is allowed to continue;
- if the main matrix itself cannot finish by `8` hours on `2` CPU cores, the study is counted as infeasible and the feasibility claim is refuted.

These stop conditions are tied to the final claimed study, not to an alternate plan.

## Related Work

### He and Chang 2005

He and Chang are the closest older precedent on robustness framing. Their KDD 2005 paper studies robustness to noisy schema inputs through an ensemble strategy for holistic schema matching. StressAudit-SM does not claim to be the first robustness study. Its difference is evaluative: it standardizes a compact modern audit over public pairwise tasks and adds calibration-transfer and abstention analysis, neither of which is central in the 2005 formulation.

### Prompt-Matcher 2024

Prompt-Matcher is the closest prior work on uncertainty-aware schema matching. That paper already argues that uncertainty handling matters, so StressAudit-SM does not claim novelty on uncertainty itself. The difference is that Prompt-Matcher is a large-model method for reducing uncertainty, whereas StressAudit-SM is a model-agnostic audit protocol asking whether uncertainty estimates learned on clean metadata remain valid after deterministic metadata stress.

### ConStruM 2026

ConStruM is the closest recent prior work on context stress. It introduces HRS-B, a narrow but important benchmark for context-sensitive LLM reranking. StressAudit-SM differs in scope and goal: it is not an LLM context-packing method and not a narrow forced-choice benchmark. It is a compact audit recipe for comparing classical and PTLM-style schema matchers across public pairwise tasks under shared stress conditions.

### Valentine and other public evaluation resources

Valentine provides the most relevant public evaluation harness and dataset interface. StressAudit-SM builds on this style of infrastructure rather than replacing it. More recent systems such as SMUTF, PRISMA, Matchmaker, ReMatch, Magneto, KcMF, and LLMATCH help define the modern matcher landscape, but they are not all required in the core matrix because the contribution here is evaluation design under a strict CPU-only budget.

## Experiments

### Main study

For each of the `32` held-out test pairs, run the three core methods on:

- `clean`,
- `header-stress`,
- `context-stress`,
- `composite-stress`.

This yields `384` held-out evaluations in the main paper. No seeds are used. No row-budget restriction is used in the main matrix.

### Metrics

Primary metrics:

- correspondence-level F1,
- mean rank shift from `clean` to each stress condition,
- pairwise method-order inversions,
- expected calibration error,
- Brier score,
- precision at a fixed preregistered coverage,
- area under the risk-coverage curve,
- runtime and peak memory.

### Statistical analysis

The paper precommits to:

- paired bootstrap confidence intervals over the `32` held-out test pairs,
- one fixed calibration model per method,
- one fixed abstention threshold per method,
- no post-hoc seed averaging,
- no protocol changes after development.

### Optional appendix only

If and only if the main protocol completes within `6.5` hours, the appendix may include:

- `StressMatch-Lite` as a transparent structure-aware reference model,
- a row-budget stress check on the value-aware subset only.

Neither appendix item is needed for the paper's central claim.

## Success Criteria

The proposal succeeds if the fixed audit protocol produces a clear empirical robustness story and fits the declared resources. If ranking changes do not appear, the paper is recast as a negative result showing stability under the available public-metadata stresses rather than as confirmation of the original hypothesis.

Evidence for the hypothesis:

- at least one method ordering changes between `clean` and a stressed condition on held-out test pairs,
- at least one method shows materially worse calibration under stress after clean-only calibration,
- abstention improves precision at the preregistered coverage target under at least one stress condition,
- the main three-method protocol completes within the `8` hour CPU budget.

Evidence against the hypothesis:

- held-out rankings are effectively unchanged under all stresses,
- clean-fit calibration transfers with little degradation,
- abstention adds little deployment value,
- the three-method main matrix cannot complete within the stated resource limits.

## References

1. Bin He and Kevin Chen-Chuan Chang. Making holistic schema matching robust: an ensemble approach. Proceedings of the 11th ACM SIGKDD International Conference on Knowledge Discovery in Data Mining, 2005.
2. Christos Koutras, George Siachamis, Andra Ionescu, Kyriakos Psarakis, Jerry Brons, Marios Fragkoulis, Christoph Lofi, Angela Bonifati, and Asterios Katsifodimos. Valentine: Evaluating Matching Techniques for Dataset Discovery. ICDE, 2021.
3. Yunjia Zhang, Avrilia Floratou, Joyce Cahoon, Subru Krishnan, Andreas C. Muller, Dalitso Banda, Fotis Psallidas, and Jignesh M. Patel. Schema Matching using Pre-Trained Language Models. ICDE, 2023.
4. Longyu Feng, Huahang Li, and Chen Jason Zhang. Cost-Aware Uncertainty Reduction in Schema Matching with GPT-4: The Prompt-Matcher Framework. arXiv:2408.14507, 2024.
5. Marcel Parciak, Brecht Vandevoort, Frank Neven, Liesbet M. Peeters, and Stijn Vansummeren. Schema Matching with Large Language Models: an Experimental Study. TaDA at VLDB Workshops, 2024.
6. Eitam Sheetrit, Menachem Brief, Moshik Mishaeli, and Oren Elisha. ReMatch: Retrieval Enhanced Schema Matching with LLMs. arXiv:2403.01567, 2024.
7. Nabeel Seedat and Mihaela van der Schaar. Matchmaker: Self-Improving Large Language Model Programs for Schema Matching. arXiv:2410.24105, 2024.
8. Yongqin Xu, Huan Li, Ke Chen, and Lidan Shou. KcMF: A Knowledge-compliant Framework for Schema and Entity Matching with Fine-tuning-free LLMs. arXiv:2410.12480, 2024.
9. Jan-Eric Hellenberg, Fabian Mahling, Lukas Laskowski, Felix Naumann, Matteo Paganelli, and Fabian Panse. PRISMA: A Privacy-Preserving Schema Matcher using Functional Dependencies. EDBT, 2025.
10. Yurong Liu, Eduardo Pena, Aecio Santos, Eden Wu, and Juliana Freire. Magneto: Combining Small and Large Language Models for Schema Matching. arXiv:2412.08194, 2024.
11. Sha Wang, Yuchen Li, Hanhua Xiao, Bing Tian Dai, Roy Ka-Wei Lee, Yanfei Dong, and Lambert Deng. LLMATCH: A Unified Schema Matching Framework with Large Language Models. CoRR abs/2507.10897, 2025.
12. Houming Chen, Zhe Zhang, and H. V. Jagadish. ConStruM: A Structure-Guided LLM Framework for Context-Aware Schema Matching. arXiv:2601.20482, 2026.
13. Jaewoo Kang and Jeffrey F. Naughton. Schema Matching Using Interattribute Dependencies. IEEE Transactions on Knowledge and Data Engineering, 2008.
14. Yu Zhang, Mei Di, Haozheng Luo, Chenwei Xu, and Richard Tzong-Han Tsai. SMUTF: Schema Matching Using Generative Tags and Hybrid Features. Information Systems, 2025.
