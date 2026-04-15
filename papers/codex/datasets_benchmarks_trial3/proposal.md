# TwinBench: An Auditable Cluster Benchmark for Coupled Invariance and Boundary Sensitivity in QA

## Introduction

Recent benchmark work has made two points clear. First, model accuracy on a canonical prompt often overstates robustness under meaning-preserving reformulations. Second, models also fail when one local fact changes and the answer should update accordingly. Existing papers already probe these behaviors in isolation or for adjacent goals: RoParQ studies paraphrase robustness, CounterBench studies counterfactual QA, MQuAKE measures answer changes after fact edits, LastingBench uses counterfactual rewriting to preserve benchmark validity, and LGIP combines invariance and sensitivity in an adjacent vision-language setting. This means the right contribution here is not to claim a new semantic phenomenon.

The proposed contribution is instead a **benchmark-engineering paper**: a small, auditable benchmark whose release unit is a **verified cluster** that couples paraphrase invariance and answer-changing sensitivity on the same seed item, with deterministic gold recomputation, explicit acceptance gates, and a release-time audit trail.

TwinBench’s atomic release object contains:

1. `q0`: a seed question,
2. `q1`, `q2`: answer-preserving paraphrases,
3. `q3`: a minimal semantic flip whose gold answer is deterministically recomputed,
4. `q4`: an auxiliary format-control variant reported separately.

The paper’s central claim is therefore narrow and defensible: **cluster-based, auditably verified evaluation can reveal semantic failure modes that single-prompt accuracy and standalone paraphrase or counterfactual tests do not localize as cleanly**. The empirical goal is diagnostic failure analysis, not headline leaderboard reshuffling.

## Proposed Approach

### Released unit and scoring

The semantic release unit is the 4-item cluster `q0`-`q3`. `q4` is never mixed into the primary score.

- `q0`: canonical seed item
- `q1`, `q2`: verified paraphrases with the same normalized gold answer
- `q3`: verified minimal semantic flip with a different normalized gold answer
- `q4`: surface-format control, reported separately

Primary metrics:

1. **Paraphrase Invariance Rate (PIR)**: all of `q0`-`q2` are correct and normalize to the same answer.
2. **Boundary Flip Accuracy (BFA)**: `q3` is answered correctly.
3. **Cluster Semantic Score (CSS)**: the model passes the full coupled condition on `q0`-`q3`.

Auxiliary metrics:

4. **Format Control Accuracy (FCA)** on `q4`.
5. **Update-Only Accuracy (UOA)** on the flip items alone, to test whether cluster coupling adds signal beyond answer updating by itself.

### What is new relative to prior work

TwinBench is explicitly a synthesis benchmark, but not a trivial packaging exercise. The novelty is the **release protocol and evaluation object**, not the individual ingredients.

What is new:

- the **verified cluster** as the atomic released unit rather than independent paraphrase or counterfactual items,
- a **coupled score** that requires both invariance and answer updating on the same seed item,
- **deterministic gold recomputation** for every released flip,
- a **benchmark audit protocol** with rejection reasons, annotation records, and split-wise construction analysis.

What is not claimed:

- not the first paraphrase benchmark,
- not the first counterfactual QA benchmark,
- not the first edited-evidence benchmark,
- not the first benchmark to jointly discuss invariance and sensitivity in the broad sense.

### Scope and benchmark size

The benchmark is sized for one-GPU, one-day execution and for credible human verification.

Target release:

- 96 clusters total if all slices pass
- 72 procedural-core clusters guaranteed
- up to 24 evidence-grounded clusters, included only if they pass stricter gates
- 384 semantically scored prompts in the core `q0`-`q3` set for a 96-cluster release
- 96 auxiliary `q4` prompts

Family allocation:

- 28 arithmetic/symbolic
- 24 temporal
- 20 table/list
- up to 24 evidence-grounded

Fallback:

- if evidence-grounded items fail gates, release a **72-cluster procedural core**
- report the evidence-grounded slice as a diagnostic pilot rather than silently merging weak items into the main benchmark

This reframes the paper around evaluation quality and failure analysis rather than around a large leaderboard benchmark.

### Task families

#### 1. Arithmetic and symbolic

- Latent record: variables, quantities, operators, target slot.
- Gold computation: exact symbolic or numeric executor.
- Valid flips: one quantity change, one operator change, one comparator reversal, or one rate change.
- Rejection rules: unchanged answer after flip, ambiguous parse, or multiple valid normalizations.

#### 2. Temporal

- Latent record: normalized dates, durations, ordering constraints.
- Gold computation: deterministic calendar solver.
- Valid flips: one anchor-date edit, one interval edit, one inclusivity edit, or one before/after reversal.
- Rejection rules: calendar ambiguity, hidden assumptions, or dependence on unstated conventions.

#### 3. Table/list

- Latent record: synthetic typed table or ordered list plus executable query.
- Gold computation: exact Python or SQLite execution.
- Valid flips: one cell edit, one filter edit, one sort-key edit, or one aggregation-target edit.
- Rejection rules: ties after editing, normalization collisions, or underspecified indexing.

#### 4. Evidence-grounded

This is the highest-risk slice and is treated as optional core content.

Allowed sources:

- arXiv abstracts from 2025-2026 with explicit factual metadata,
- official versioned documentation or release notes,
- dated institutional announcements with exact factual statements.

Restrictions:

- evidence snippet length 40-120 words,
- answer must be extractable from the snippet alone,
- exact-answer uniqueness must hold after normalization,
- no external retrieval,
- no pronoun-heavy or multi-hop discourse dependencies.

Valid flips:

- numeric slot replacement,
- same-type named-entity replacement,
- comparator swap,
- attribute-target swap.

For `q1` and `q2`, only the question wording changes. For `q3`, exactly one evidence slot or one tightly localized question slot changes, and the gold answer must be recomputable by exact extraction or deterministic normalization from the edited snippet.

### Annotation and audit protocol

Human verification is part of the contribution, so it is specified concretely.

Annotator pool:

- 3 trained annotators with strong English proficiency and prior QA annotation experience,
- at least 1 adjudicating annotator with graduate-level ML/NLP familiarity,
- compensation budgeted at **$25/hour** including training time.

Training and calibration:

- 2-hour rubric training session,
- 20 practice clusters not used in the benchmark,
- calibration continues until pairwise agreement reaches at least 0.80 on paraphrase validity and flip validity in the practice set.

Per-cluster rubric:

1. **Paraphrase validity**: does `q1`/`q2` preserve the meaning of `q0`?
2. **Flip locality**: does `q3` differ by one targeted semantic change only?
3. **Answer uniqueness**: after normalization, is there exactly one defensible gold answer?
4. **Answer-change validity**: should `q3`’s answer differ from `q0`’s answer?
5. **Outside-knowledge leakage**: can the item be solved from the provided content alone?
6. **Fluency / well-formedness**: is the prompt readable and non-broken?

Production workflow:

1. automatic generation and solver pass,
2. independent annotation by two annotators,
3. disagreement routing to the adjudicator,
4. unresolved or rubric-violating items are dropped,
5. release metadata stores rubric decisions, disagreement type, and final disposition.

Agreement reporting:

- Cohen’s kappa or Krippendorff’s alpha per rubric dimension,
- candidate-to-kept rates by family,
- rejection reasons,
- adjudication frequency.

Estimated annotation load:

- roughly 130-145 candidate clusters to yield a 72-96 cluster release,
- about 6-8 minutes first-pass work per candidate per annotator,
- about 15-20% adjudication rate,
- total human effort about 28-36 hours including training and adjudication.

### Pre-registered gates for evidence-grounded inclusion

The evidence-grounded slice enters the core release only if it clears all gates:

1. from 30 pilot candidates, at least 20 pass automatic uniqueness and answer-change checks;
2. at least 18 survive dual human verification;
3. agreement on answer uniqueness and answer-change validity is at least 0.85;
4. at least half of retained items require editing the evidence snippet itself, not only the question wording;
5. fewer than 10% of retained items are flagged during adjudication as borderline due to normalization brittleness.

If any gate fails, the slice is released only as a pilot appendix.

### Construction splits

TwinBench is built in two construction splits to test benchmark reliability:

- **Split A**: first construction batch finalized before pilot evaluation
- **Split B**: second batch created after freezing the rubric and generator templates

The paper reports:

- candidate-to-kept rates by split,
- family composition by split,
- failure-mode overlap across splits,
- bootstrap uncertainty for split-specific CSS.

Rank comparisons across splits are exploratory only.

## Related Work

### Paraphrase robustness

**RoParQ** is the nearest QA benchmark on paraphrase robustness. It establishes that paraphrased questions expose failures invisible to canonical QA evaluation. TwinBench differs in objective and unit of analysis: it is not an alignment method paper, and it does not release paraphrases as standalone items. Its released object is a cluster that couples paraphrase invariance with answer-changing sensitivity and attaches deterministic recomputation and audit metadata.

Rabinovich et al., **Predicting Question-Answering Performance of Large Language Models through Semantic Consistency**, uses paraphrase consistency to estimate answer reliability without references. TwinBench instead uses verified paraphrases for benchmark scoring, not confidence estimation.

### Counterfactual QA and answer updating

**CounterBench** shows that LLMs struggle with controlled counterfactual reasoning. TwinBench borrows the value of answer-changing interventions, but differs by requiring those flips to be tethered to the exact same seed item that also supports answer-preserving paraphrases.

**MQuAKE** is the most relevant answer-updating neighbor. It evaluates whether answers change after fact edits in a knowledge-editing setting. TwinBench does not compete with MQuAKE on model-editing evaluation. Its distinct contribution is a release-time verified cluster for raw-model evaluation, with tightly scoped flips and deterministic recomputation rather than post-edit knowledge consistency as the primary target.

### Benchmark defense and edited evidence

**LastingBench** already demonstrates that counterfactual rewriting can be used to defend benchmarks from leakage. This substantially weakens any novelty claim around edited evidence alone. TwinBench differs by focusing on small, auditable atomic units and coupled scoring rather than on preserving long-context benchmark freshness.

### Combined invariance and sensitivity in adjacent domains

**LGIP (Language-Guided Invariance Probing of Vision-Language Models)** is important because it explicitly combines invariance to meaning-preserving paraphrases and sensitivity to meaning-changing flips in image-text matching. This further clarifies TwinBench’s position: the broad evaluation idea is not unique. What TwinBench adds is a QA-specific, fully auditable release protocol with deterministic answer recomputation, cluster-level acceptance criteria, and human-verified gold labels.

### Benchmark reliability and evaluation quality

**Do Large Language Model Benchmarks Test Reliability?** argues for platinum-style curation and shows that benchmark noise can mask true model behavior. **On Robustness and Reliability of Benchmark-Based Evaluation of LLMs** shows that paraphrases materially alter benchmark scores. **BetterBench**, **When Benchmarks are Targets**, and **Benchmark^2** argue that benchmark design and ranking stability are often brittle. TwinBench positions itself as a concrete response in this design space: smaller, more verified, more auditable, and explicit about what conclusions a modest benchmark can and cannot support.

## Experiments

### Models

Use four local instruction-tuned models that fit comfortably on one RTX A6000:

- Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct
- Gemma-2-9B-it
- Mistral-7B-Instruct-v0.3

Optional fifth model if time remains:

- Qwen2.5-14B-Instruct in 4-bit quantized form

### Experimental questions

1. Does CSS reveal coupled semantic failures not visible in `q0` accuracy?
2. Is CSS meaningfully lower than UOA, showing that coupling invariance with updating adds diagnostic value?
3. Which families and flip types dominate failures?
4. Are the same qualitative failure modes present in both construction splits?
5. Can a small verified benchmark support reliable failure analysis even if it is underpowered for strong leaderboard claims?

### Main analyses

For each model report:

- `q0` accuracy,
- PIR, BFA, CSS, UOA, FCA,
- family-level breakdowns,
- flip-template breakdowns,
- split A / split B breakdowns,
- rejection statistics from benchmark construction.

### Statistical treatment and power

The benchmark is intentionally modest, so the proposal avoids overclaiming on rankings.

- Use paired bootstrap confidence intervals for `q0`, PIR, BFA, CSS, and UOA.
- Treat model ranking changes between `q0` accuracy and CSS as **exploratory diagnostics**, not a headline claim.
- Report Kendall rank correlation between `q0`-based ranking and CSS-based ranking, but do not claim stable leaderboard reordering unless uncertainty is clearly separated.
- Include a simple detectable-effect calculation: with 72-96 clusters, paired differences smaller than roughly 8-10 percentage points will often be hard to separate cleanly, so the benchmark is powered for moderate semantic gaps and failure-pattern analysis, not fine-grained ranking claims.

### Feasibility

Runtime fits the stated budget:

- 480 prompts per model for a full 96-cluster release including `q4`
- 4 main models = 1,920 generations total
- short answers and exact-match style scoring
- no finetuning and no retrieval training

Expected compute:

- 2-4 GPU hours for inference,
- under 1 CPU hour for scoring and bootstrap analysis,
- the dominant cost is annotation, not model execution.

## Success Criteria

The proposal is supported if:

1. average CSS is at least 8 points below `q0` accuracy across the main four models;
2. CSS is at least 5 points below UOA on at least two families, indicating that cluster coupling adds diagnostic signal beyond answer updating alone;
3. annotator agreement is at least 0.80 on paraphrase validity and flip locality, and at least 0.85 on answer uniqueness and answer-change validity for any evidence-grounded items admitted to the core release;
4. the same qualitative coupled failure modes recur across both construction splits;
5. the paper can produce a credible audit table covering kept rates, rejection reasons, adjudication rates, and per-family brittleness.

The proposal is not supported if:

- CSS is nearly identical to `q0` accuracy,
- CSS adds little beyond UOA,
- agreement or uniqueness is weak,
- evidence-grounded items collapse due to normalization brittleness without yielding useful diagnostic insight,
- or conclusions depend entirely on one construction split.

## References

1. Anka Reuel, Amelia Hardy, Chandler Smith, Max Lamparth, Malcolm Hardy, and Mykel J. Kochenderfer. 2024. *BetterBench: Assessing AI Benchmarks, Uncovering Issues, and Establishing Best Practices*. arXiv:2411.12990. https://arxiv.org/abs/2411.12990
2. Qi Qian, Chengsong Huang, Jingwen Xu, Changze Lv, Muling Wu, Wenhao Liu, Xiaohua Wang, Zhenghua Wang, Zisu Huang, Muzhao Tian, Jianhan Xu, Kun Hu, He-Da Wang, Yao Hu, Xuanjing Huang, and Xiaoqing Zheng. 2026. *Benchmark^2: Systematic Evaluation of LLM Benchmarks*. arXiv:2601.03986. https://arxiv.org/abs/2601.03986
3. Yuefei Chen, Vivek K. Singh, Jing Ma, and Ruixiang Tang. 2025. *CounterBench: A Benchmark for Counterfactuals Reasoning in Large Language Models*. arXiv:2502.11008. https://arxiv.org/abs/2502.11008
4. Joshua Vendrow, Edward Vendrow, Sara Beery, and Aleksander Madry. 2025. *Do Large Language Model Benchmarks Test Reliability?* arXiv:2502.03461. https://arxiv.org/abs/2502.03461
5. Yixiong Fang, Tianran Sun, Yuling Shi, Min Wang, and Xiaodong Gu. 2025. *LastingBench: Defend Benchmarks Against Knowledge Leakage*. Findings of the Association for Computational Linguistics: EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.993/
6. Jae Joong Lee. 2026. *Language-Guided Invariance Probing of Vision-Language Models*. Pattern Recognition Letters 2026 / arXiv:2511.13494. https://arxiv.org/abs/2511.13494
7. Riccardo Lunardi, Vincenzo Della Mea, Stefano Mizzaro, and Kevin Roitero. 2025. *On Robustness and Reliability of Benchmark-Based Evaluation of LLMs*. arXiv:2509.04013. https://arxiv.org/abs/2509.04013
8. Ella Rabinovich, Samuel Ackerman, Orna Raz, Eitan Farchi, and Ateret Anaby Tavor. 2023. *Predicting Question-Answering Performance of Large Language Models through Semantic Consistency*. Proceedings of the Third Workshop on Natural Language Generation, Evaluation, and Metrics. https://aclanthology.org/2023.gem-1.12/
9. Minjoon Choi. 2025. *RoParQ: Paraphrase-Aware Alignment of Large Language Models Towards Robustness to Paraphrased Questions*. arXiv:2511.21568. https://arxiv.org/abs/2511.21568
10. Norah Alzahrani, Hisham Abdullah Alyahya, Yazeed Alnumay, Sultan Alrashed, Shaykhah Alsubaie, Yusef Almushaykeh, Faisal Mirza, Nouf Alotaibi, Nora Altwairesh, Areeb Alowisheq, M. Saiful Bari, and Haidar Khan. 2024. *When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards*. arXiv:2402.01781. https://arxiv.org/abs/2402.01781
11. Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning, Christopher Potts, and Danqi Chen. 2023. *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions*. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. https://aclanthology.org/2023.emnlp-main.971/
