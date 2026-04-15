# TRIAD-Audit: An Audited Evaluation of Route Labels on Context-Memory Conflict Benchmarks

## Introduction

Recent benchmark work has already established that large language models behave unpredictably when retrieved context conflicts with parametric memory. CUB evaluates context-utilization methods across gold, conflicting, and irrelevant contexts; Who’s Who studies same-name entity conflicts in realistic retrieval settings; ConflictBank broadens conflict types across retrieved and embedded knowledge; and ClashEval quantifies the tug-of-war between model priors and retrieved evidence. The crowded prior-art landscape makes a broad “new conflict benchmark” claim untenable.

This proposal therefore makes a narrower claim: **TRIAD-Audit is an audited evaluation study for existing conflict benchmarks, not a new benchmark family or routing architecture.** Its target question is:

> Do benchmark conclusions change once a human-audited oracle explicitly labels the correct action as `context`, `memory`, or `abstain`, instead of relying on answer accuracy or lightly automated route proxies?

The motivating gap is evaluative rather than architectural. Existing conflict benchmarks are strong at measuring whether a model answers correctly under conflicting evidence, but they usually do not provide an audited item-level oracle for the deployment decision itself: use the supplied context, rely on memory, or abstain. That omission matters most on ambiguity-heavy cases, especially alias-rich entity questions where “memory” and “abstain” can easily be conflated.

The paper’s main hypothesis is deliberately modest and testable:

> On existing conflict benchmarks, audited route labels and alias-aware normalization will materially change policy rankings or deployment recommendations relative to the original benchmark scoring or automatic route proxies.

If rankings do not move, that is still a meaningful negative result: it would imply that current benchmark conclusions are more stable than suspected.

## Proposed Approach

### Overview

TRIAD-Audit adds a small, carefully audited evaluation layer to existing benchmarks. The contribution is:

1. a published route-label protocol with adjudication rules;
2. a fixed audited subset from existing conflict benchmarks;
3. an alias-aware evaluation pipeline;
4. a ranking-sensitivity study across evaluation schemes.

The primary result is not a new learned router. The primary result is whether **policy rankings are stable after audit**.

### Data Scope and Audited-Set Design

The study uses two source benchmarks:

- **CUB** as the primary context-conflict benchmark.
- **WhoQA / Who’s Who** as the ambiguity-heavy companion benchmark where aliasing and entity confusion are common.

The audited set is fixed in advance:

- **320 total audited items**
- **160 from CUB**
- **160 from WhoQA**

The split is also fixed in advance:

- **Dev / calibration split: 80 items total**
- 40 CUB + 40 WhoQA
- every dev item is independently labeled by two annotators, then adjudicated
- dev labels are used only to finalize guidelines, alias rules, and one abstention threshold for a simple confidence baseline

- **Locked test split: 240 items total**
- 120 CUB + 120 WhoQA
- every test item receives a final human label
- 80 test items are double-annotated for holdout agreement estimation
- the remaining 160 test items are single-annotated, with mandatory adjudication when the annotator flags low confidence, alias ambiguity, or insufficient evidence

This yields **320 final human-labeled items** and **480 total annotation passes**:

- 160 passes for doubly labeled dev
- 160 passes for the doubly labeled portion of test
- 160 passes for the singly labeled portion of test

The main paper evaluates policies on the **240-item locked test split only**. No ranking claim is made from the dev split.

### Route Label Schema

Each audited item receives one final route label:

- **`context`**: the supplied context contains enough evidence, and using that context is the correct action.
- **`memory`**: the supplied context is stale, misleading, or irrelevant, but the question still has a sufficiently supported answer that can reasonably be produced from model memory.
- **`abstain`**: the item remains too ambiguous or underdetermined for a safe committed answer.

The protocol makes the `memory` versus `abstain` boundary explicit. An item is labeled `memory` only when the answer is sufficiently identifiable despite the bad context. It is labeled `abstain` when entity resolution, temporal grounding, or evidence support remains too weak for safe commitment.

### Annotation Protocol

The annotation instructions will be short and operational. Each item will be labeled using:

- the question
- the provided context
- benchmark metadata
- a small evidence lookup note when required for adjudication

Each audited record stores:

- benchmark and source ID
- split assignment
- final route label
- annotator confidence
- whether alias normalization was needed
- brief supporting evidence note
- adjudication note when applicable

The main appendix will include disagreement examples, especially on `memory` versus `abstain`.

### Alias-Aware Normalization

WhoQA-style cases make naive exact match too brittle. TRIAD-Audit therefore uses a narrow alias-aware normalization layer:

- lowercase and punctuation normalization
- date-format normalization
- benchmark-provided aliases when available
- Wikipedia or Wikidata redirects for entity titles
- manually added aliases discovered during auditing

This is a cleaning step for fair evaluation, not a claim of a new answer metric.

### Compared Policies

To reduce small-sample overfitting risk, the primary comparison is among **fixed policies and simple thresholds**, not a heavily engineered learned router.

Main policies:

1. **Always context**
2. **Always memory**
3. **Always abstain**
4. **Abstain-on-low-confidence**
5. **Context-if-supported-else-abstain**
6. **Memory-if-context-contradiction-and-high-prior-confidence-else-abstain**

These policies use only signals available from frozen model inference:

- answer without context
- answer with context
- token-level confidence or mean log-probability
- agreement or disagreement between the two answers
- a lightweight support or contradiction check

An optional appendix may include a **3-feature multinomial logistic regression** trained **only on the 80-item audited dev split**. It is explicitly exploratory and not needed for the main claim.

### Evaluation Regimes

Each policy is ranked under three regimes:

1. **Original benchmark scoring**
- standard answer correctness as defined by the source benchmark

2. **Proxy route scoring**
- automatically derived route labels from benchmark metadata and simple heuristics

3. **Audited TRIAD scoring**
- final human route labels with alias-aware normalization

The central experiment asks whether rankings change across these regimes.

### Utility-Aware Scoring

Because unsupported wrong answers are often worse than abstention, the paper adds a simple deployment utility matrix.

Default matrix:

- `+1` for a correct committed answer
- `0` for abstain
- `-2` for an unsupported wrong answer

Sensitivity matrices:

- mild penalty: `-1`
- severe penalty: `-3`

The paper reports utility, harmful-answer rate, and coverage alongside route accuracy.

## Related Work

### CUB

**CUB: Benchmarking Context Utilisation Techniques for Language Models** (Hagström et al., 2025) is the closest prior work. CUB already studies model behavior under gold, conflicting, and irrelevant contexts and compares context-utilization methods across multiple datasets and models.

TRIAD-Audit differs from CUB in scope and claim:

- CUB asks which context-utilization methods perform best.
- TRIAD-Audit asks whether the answer to that question changes once the oracle is audited at the item level.

TRIAD-Audit therefore contributes **audited route labels, alias-aware normalization, abstention-aware utility scoring, and sensitivity analysis**, not a broader benchmark replacement.

### Who’s Who / WhoQA

**Who’s Who: Large Language Models Meet Knowledge Conflicts in Practice** (Pham et al., 2024) is important because same-name conflicts make action labeling substantially harder than on synthetic contradiction settings. TRIAD-Audit uses WhoQA precisely because it stresses the `memory` versus `abstain` boundary.

WhoQA contributes a realistic conflict benchmark; TRIAD-Audit contributes an audited action oracle layered on top of such data.

### ConflictBank

**ConflictBank: A Benchmark for Evaluating the Influence of Knowledge Conflicts in LLM** (Su et al., 2024) broadens the space considerably by covering retrieved conflicts, embedded conflicts, and their interaction across misinformation, temporal, and semantic conflict types. This weakens any claim that current benchmarks “miss” conflict evaluation altogether.

TRIAD-Audit differs in two ways:

- ConflictBank is a **large benchmark construction effort**.
- TRIAD-Audit is a **small audited evaluation study** centered on whether benchmark conclusions are label-sensitive.

ConflictBank measures conflict behavior at scale; TRIAD-Audit measures whether **audited action labels alter policy rankings**.

### ClashEval

**ClashEval: Quantifying the tug-of-war between an LLM’s internal prior and external evidence** (Wu, Wu, and Zou, 2024/2025) studies when models follow retrieved evidence versus their prior under controlled perturbations. It is directly relevant because it frames the core descriptive tension between memory and context.

TRIAD-Audit differs because ClashEval is a **descriptive behavior benchmark** about prior-versus-context adoption, whereas TRIAD-Audit is a **normative audited evaluation** of what the model should have done: use context, rely on memory, or abstain.

### Real-document conflict studies and task dependence

**Studying Large Language Model Behaviors Under Context-Memory Conflicts With Real Documents** (Kortukov et al., 2024), **Adaptive Chameleon or Stubborn Sloth** (Xie et al., 2024), **When Context Leads but Parametric Memory Follows** (Tao et al., 2024), **KScope** (Xiao et al., 2025), and **Task Matters** (Sun et al., 2025) all show that conflict behavior depends on realism, evidence structure, and task semantics.

These studies motivate why an audited oracle could matter. None of them center the contribution on a small adjudicated route-label layer used to test ranking stability on existing benchmarks.

### Abstention and utility-aware prediction

Selective prediction and conformal risk work motivates the abstention axis, but TRIAD-Audit is not a new abstention algorithm paper. It only borrows a utility-aware lens to evaluate whether benchmark recommendations change once abstention is treated as a first-class action.

## Experiments

### Models

The paper only needs small frozen models that fit comfortably on one RTX A6000:

- Qwen2.5-1.5B-Instruct
- Llama-3.2-3B-Instruct or comparable 3B model
- one optional extra small model if runtime allows

No fine-tuning is required.

### Runtime Feasibility

The compute budget is modest:

- 320 audited items total
- 2 answer views per item per model
- short-form generation only
- optional lightweight contradiction check
- no large-scale training

This is well within one A6000 and an 8-hour overall budget. Human auditing, not GPU time, is the main bottleneck.

### Metrics

Primary metrics on the locked test set:

- route accuracy against audited labels
- utility under the default and sensitivity matrices
- harmful-answer rate
- abstention rate / coverage
- Kendall tau and Spearman correlation between ranking tables
- number of pairwise ranking reversals

Audit-quality metrics:

- agreement on the 160 double-annotated items
- proxy-to-human agreement
- percentage of items requiring adjudication
- percentage of items requiring alias expansion
- per-error breakdown for alias mismatch, stale context, unsupported memory answer, and unresolved ambiguity

### Pre-Registered Decision Thresholds

The paper will pre-register the following thresholds before running the ranking analysis.

**Evidence that the main claim is supported**

- ranking instability is considered substantial if the audited test ranking shows either:
- a top-1 policy change, or
- at least **2 pairwise ranking reversals** and **Kendall tau <= 0.70** versus original scoring

- audit reliability is considered credible if:
- overall agreement on double-labeled items reaches **Cohen’s kappa >= 0.70**
- the harder `memory` versus `abstain` slice reaches **kappa >= 0.60**

- audit coverage is considered adequate if:
- at least **95% of sampled items** receive a final non-discarded audited label

- proxy weakness is considered practically important if:
- proxy labels disagree with final audited labels on at least **12% of locked-test items**

**Evidence that would weaken the main claim**

- no top-rank change, fewer than 2 pairwise reversals, and Kendall tau above 0.90
- overall agreement below 0.60
- fewer than 90% of sampled items survive audit without being discarded as too underspecified

### Expected Outcomes

There are two publishable outcomes:

1. **Positive audit effect**: audited labels alter rankings or deployment recommendations, showing that current benchmark conclusions are more fragile than standard scoring suggests.
2. **Negative stability result**: rankings remain largely unchanged, showing that current conflict benchmarks are more robust than expected even under careful audit.

Either result is cleaner and more defensible than claiming a new benchmark paradigm.

## Success Criteria

The project succeeds if it demonstrates all of the following:

1. A transparent and reproducible audited route-label protocol can be executed on a fixed subset of existing conflict benchmarks.
2. The resulting audited labels achieve credible agreement and coverage.
3. Policy rankings under audited scoring differ meaningfully from at least one baseline evaluation regime, or the paper can convincingly show that they do not.
4. The paper clearly localizes where disagreement comes from, especially alias handling and the `memory` versus `abstain` boundary.

## References

- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., and Schuster, T. 2024. Conformal Risk Control. ICLR 2024.
- Hagström, L., Kim, Y., Yu, H., Lee, S., Johansson, R., Cho, H., and Augenstein, I. 2025. CUB: Benchmarking Context Utilisation Techniques for Language Models. Findings of ACL 2025.
- Kortukov, E., Rubinstein, A., Nguyen, E., and Oh, S. J. 2024. Studying Large Language Model Behaviors Under Context-Memory Conflicts With Real Documents. arXiv preprint arXiv:2404.16032.
- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., and Hajishirzi, H. 2023. When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories. ACL 2023.
- Moskvoretskii, V., Lysyuk, M., Salnikov, M., Ivanov, N., Pletenev, S., Galimzianova, D., Krayko, N., Konovalov, V., Nikishina, I., and Panchenko, A. 2025. Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty Back Home. arXiv preprint arXiv:2501.12835.
- Pham, Q. H., Ngo, H., Luu, A. T., and Nguyen, D. Q. 2024. Who’s Who: Large Language Models Meet Knowledge Conflicts in Practice. Findings of EMNLP 2024.
- Su, Z., Zhang, J., Qu, X., Zhu, T., Li, Y., Sun, J., Li, J., Zhang, M., and Cheng, Y. 2024. ConflictBank: A Benchmark for Evaluating the Influence of Knowledge Conflicts in LLM. arXiv preprint arXiv:2408.12076.
- Sun, K., Bai, F., and Dredze, M. 2025. Task Matters: Knowledge Requirements Shape LLM Responses to Context-Memory Conflict. arXiv preprint arXiv:2506.06485.
- Tao, Y., Hiatt, A., Haake, E., Jetter, A. J., and Agrawal, A. 2024. When Context Leads but Parametric Memory Follows in Large Language Models. EMNLP 2024.
- Wu, K., Wu, E., and Zou, J. 2025. ClashEval: Quantifying the Tug-of-War Between an LLM’s Internal Prior and External Evidence. arXiv preprint arXiv:2404.10198.
- Xiao, Y., Chen, S., Gallifant, J., Bitterman, D., Hartvigsen, T., and Ghassemi, M. 2025. KScope: A Framework for Characterizing the Knowledge Status of Language Models. arXiv preprint arXiv:2506.07458.
- Xie, J., Zhang, K., Chen, J., Lou, R., and Su, Y. 2024. Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts. ICLR 2024.
