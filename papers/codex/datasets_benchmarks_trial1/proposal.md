# DriftAnswer-Py: A 12-Item Executable Pilot of Accepted Python Stack Overflow Answers Under Documented Library Drift

## Introduction

This proposal is intentionally narrow. It does not claim a new general task for Stack Overflow answer updating, answer enhancement, or comment-guided maintenance. Recent work already covers much of that space, especially:

- Sheikhaei et al., *A Study of Update Request Comments in Stack Overflow Answer Posts* (JSS 2023), which studies maintenance signals in comments;
- Mai et al., *Towards Better Answers: Automated Stack Overflow Post Updating* (ICSE 2025), which introduces Soup for comment-guided post updating;
- Bappon et al., *Human-Aligned Enhancement of Programming Answers with LLMs Guided by User Feedback* (arXiv 2026), which introduces ReSOlve and AUTOCOMBAT for feedback-guided answer enhancement.

The only defensible novelty claim is a combination claim: to the best of a March 21, 2026 search over recent arXiv and related literature, no prior paper appears to release a benchmark that jointly targets:

1. accepted Python Stack Overflow answers;
2. documented library drift as the reason an answer aged;
3. executable validation in both an old and a current library version;
4. official documentation packets justifying the verdict.

The contribution should therefore be framed as a small benchmark artifact paper, not a new-task paper. The benchmark unit is an accepted answer whose current status under library evolution can be reproduced and adjudicated, not an arbitrary answer that could be improved.

The practical motivation is strong despite the narrow scope. Accepted answers remain highly trusted by developers, but API removals, renames, default changes, and migration-induced semantic shifts can silently age them. A reviewer-verifiable pilot benchmark centered on reproducible old/current execution is useful even if the release is small.

### Core hypothesis

For accepted Python Stack Overflow answers with reproducible documented library drift, supplying official library evidence at inference time improves current-version executable repair over a true closed-book baseline and a thread-only baseline. Because the benchmark is small, the main supported claim should be directional and artifact-focused rather than a strong statistical claim.

## Proposed Approach

### Contribution claim

The paper should state its claim conservatively:

> We release a 12-item executable pilot benchmark of accepted Python Stack Overflow answers affected by documented library drift, with official evidence packets and old/current version adjudication. The novelty lies only in the combination of accepted-answer focus, documented drift, and dual-version executable validation.

This is an artifact contribution with a lightweight evaluation component.

### Scope and feasibility

The benchmark must fit a single-machine, eight-hour total project budget. That makes earlier ambitions such as 15-18 retained items, broad three-way balancing, and two-annotator full adjudication unrealistic.

#### Fixed scope

- 12 core executable items
- 2 libraries: `pandas` and `scikit-learn`
- primary label space: `valid` vs `needs_update`
- secondary exploratory label space: `valid`, `partially_stale`, `obsolete`
- one local evaluation model that fits comfortably on 1x RTX A6000

#### Time budget split

- benchmark construction: approximately 5-6 hours
- model evaluation and analysis: approximately 1.5-2 hours
- packaging and sanity checks: approximately 0.5-1 hour

The paper should explicitly separate benchmark construction time from model evaluation time. It should not imply that robust historical reconstruction plus dual-human annotation for a larger set is realistic within one eight-hour run.

### Benchmark definition

Each retained item contains:

- the Stack Overflow question and accepted answer;
- optional selected comments that mention breakage or changed usage;
- a historically plausible old library version;
- a current library version;
- an official evidence packet from library documentation, release notes, or migration guides;
- a primary binary label in `{valid, needs_update}`;
- an optional secondary label in `{valid, partially_stale, obsolete}` when confidence is high;
- a deterministic local harness;
- a minimal reference repair for `needs_update` items.

### Candidate inclusion rules

Each candidate must satisfy all of the following:

1. The thread is on Stack Overflow and tagged `python` plus exactly one target library tag from `{pandas, scikit-learn}`.
2. The answer under study is the accepted answer.
3. The accepted answer contains a primary Python code block of roughly 5-35 lines.
4. The answer is old enough that library drift is plausible, such as at least 18 months old.
5. The answer's intent can be checked with a deterministic local harness.
6. The mismatch between old and current behavior is attributable to documented library evolution, not external services or ambiguous environment setup.
7. The old/current comparison can be justified from official project sources.

### Retrieval and evidence sources

Candidate mining uses:

- Stack Exchange Data Explorer or related Stack Exchange metadata sources for coarse retrieval;
- Stack Exchange API for accepted-answer linkage and timestamps;
- StackPrinter for stable question, answer, and comment text used in annotation.

Drift evidence uses only primary sources:

- official `pandas` documentation and release notes;
- official `scikit-learn` documentation and release notes;
- official migration guides or API reference pages;
- project release pages only for version dating.

Third-party blogs and forum posts are excluded as label evidence.

### Collection pipeline

#### Phase 1: light candidate mining

Mine a manageable pool of roughly 20-30 candidates rather than 50-70. Screening more does not fit the budget and is unnecessary for a 12-item pilot.

Rank candidates using lightweight stale signals:

- comments mentioning `deprecated`, `removed`, `renamed`, `no longer works`, `changed`, or version numbers;
- APIs known from official docs to have changed;
- older answers attached to high-churn APIs.

#### Phase 2: strict feasibility screening

Retain only candidates where:

- the intent is clear;
- a harness can be built quickly;
- an old/current version pair is easy to justify;
- dependencies remain local and simple.

The goal is to finish with 12 retained items, not to maintain a large reserve set.

#### Phase 3: environment linking and harnessing

For each retained item:

1. identify the key library calls;
2. select an old version consistent with the answer date and documented API availability;
3. select one current version;
4. build a local deterministic harness;
5. record official evidence for the drift rationale.

### Labeling protocol

#### Primary label

- `valid`: the accepted answer still works as intended in the current environment without material changes.
- `needs_update`: the accepted answer no longer works as intended in the current environment, but a minimal reference repair or replacement can be demonstrated.

This binary task is the primary analysis because it is realistic for a 12-item executable pilot.

#### Secondary label

When confidence is high, `needs_update` items receive an additional exploratory subtype:

- `partially_stale`: the core solution remains correct but needs a local patch;
- `obsolete`: the current fix requires a materially different solution path.

If subtype agreement is weak, the release should preserve only the binary label and treat the three-way label as optional metadata.

#### Annotation realism

Within the eight-hour budget, the proposal should not promise full two-annotator adjudication for all retained items. A realistic protocol is:

- one primary annotator for all 12 items;
- one secondary reviewer for a small calibration subset of 3-4 items;
- explicit drop rules for ambiguous cases;
- written rationale per item.

This is weaker than full dual annotation, but it is feasible and honest.

## Environment and Artifact Protocol

Each item should include:

- `metadata.json` with post IDs, timestamps, tags, and version pair;
- `thread.md` with question, accepted answer, and selected comments;
- `evidence.md` with official documentation links and a concise rationale;
- `label.json` with binary label and optional secondary subtype;
- `old_requirements.txt`;
- `current_requirements.txt`;
- `harness.py` or equivalent;
- `reference_repair.py` or `repair.patch` for `needs_update` items;
- structured execution outputs for old, current, and repaired runs.

### Executable validity requirements

An item enters the core benchmark only if:

1. it runs locally without network access;
2. inputs are fixed and bundled;
3. the verdict is stable across repeated runs;
4. the harness checks task-relevant behavior;
5. old/current outcomes are interpretable.

For `needs_update`, the benchmark prefers items where the original answer is supported in the old environment and fails in the current environment. If old-version reconstruction is plausible but imperfect, the evidence note must say so explicitly.

## Related Work

### Positioning against closest papers

Mai et al. (Soup) and Bappon et al. (ReSOlve/AUTOCOMBAT) already make it impossible to claim novelty on Stack Overflow answer updating broadly. Zhang et al. establish the obsolete-answer motivation, while BUMP and Byam show the value of executable update artifacts in software evolution. Wang et al. and Zhuo et al. study LLM behavior under API evolution but focus on generated code rather than historical accepted answers.

The proposal differs only by combining:

- accepted-answer trust as the unit of study;
- documented library drift as the inclusion criterion;
- old/current executable adjudication as the artifact core.

That combination is narrow, but it is still publishable as a pilot benchmark if the paper avoids exaggerated task novelty claims.

## Experiments

### 1. Benchmark construction report

The first result is the benchmark itself. Report:

- number of mined candidates;
- number dropped for ambiguity;
- number dropped for unreproducible environments;
- final retained item count;
- per-library counts;
- binary label counts;
- optional three-way subtype counts only if confidence is sufficient.

### 2. Repair evaluation

Use one local model that fits comfortably on the available GPU, such as `Qwen2.5-Coder-7B-Instruct`.

#### Baselines

- true closed-book: question plus accepted answer only;
- thread-aware: question, accepted answer, and selected comments;
- authority-aware: question, accepted answer, and official evidence packet.

If time allows, a hybrid comments-plus-docs setting can be added, but it should not be part of the core claim.

#### Model output

For each item, the model predicts:

- binary status (`valid` or `needs_update`);
- repaired code when it predicts `needs_update`.

The three-way subtype prediction is optional and secondary.

#### Metrics

Primary metrics:

- binary accuracy on `valid` vs `needs_update`;
- current-version repair pass rate on `needs_update` items;
- exact item-level delta between prompting settings.

Secondary metrics:

- exploratory three-way macro-F1 if subtype labels are stable enough;
- evidence citation correctness if citations are requested.

Because the dataset is tiny, report counts and Wilson intervals for pass rates. Do not make strong NHST claims.

### Expected outcomes

The expected pattern is:

- closed-book prompting over-trusts accepted answers;
- comments help in some cases but are noisy;
- official evidence helps most on executable repair for drifted items;
- three-way subtype distinctions are noisier than the binary valid-vs-update split.

## Success Criteria

The proposal is supported if:

1. a 12-item executable benchmark is released with complete artifact files;
2. every item has a defensible binary label and official evidence packet;
3. the authority-aware setting repairs more `needs_update` items than the true closed-book baseline;
4. the benchmark remains reproducible within the stated compute and time budget.

The proposal is weakened if:

- fewer than 10 items survive strict executable filtering;
- old/current reconstruction is too fragile to justify the artifact claim;
- binary labels remain ambiguous;
- authority-aware prompting shows no practical advantage over closed-book.

If subtype consistency is weak, the paper should keep only the binary benchmark claim and present `partially_stale` vs `obsolete` as exploratory notes rather than headline results.

## References

1. Suborno Deb Bappon, Saikat Mondal, Chanchal K. Roy, and Kevin Schneider. *Human-Aligned Enhancement of Programming Answers with LLMs Guided by User Feedback*. arXiv, 2026. https://arxiv.org/abs/2601.17604
2. Yubo Mai, Zhipeng Gao, Haoye Wang, Tingting Bi, Xing Hu, Xin Xia, and Jianling Sun. *Towards Better Answers: Automated Stack Overflow Post Updating*. ICSE 2025 Research Track. https://arxiv.org/abs/2408.09095
3. Mohammad Sadegh Sheikhaei, Yuan Tian, and Shaowei Wang. *A Study of Update Request Comments in Stack Overflow Answer Posts*. Journal of Systems and Software, 2023. https://arxiv.org/abs/2304.07848
4. Haoxiang Zhang, Shaowei Wang, Tse-Hsun Chen, Ying Zou, and Ahmed E. Hassan. *An Empirical Study of Obsolete Answers on Stack Overflow*. IEEE Transactions on Software Engineering, 2019. https://arxiv.org/abs/1903.12282
5. Frank Reyes, Yogya Gamage, Gabriel Skoglund, Benoit Baudry, and Martin Monperrus. *BUMP: A Benchmark of Reproducible Breaking Dependency Updates*. SANER 2024. https://arxiv.org/abs/2401.09906
6. Frank Reyes, May Mahmoud, Federico Bono, Sarah Nadi, Benoit Baudry, and Martin Monperrus. *Byam: Fixing Breaking Dependency Updates with Large Language Models*. arXiv, 2025. https://arxiv.org/abs/2505.07522
7. Chong Wang, Kaifeng Huang, Jian Zhang, Yebo Feng, Lyuye Zhang, Yang Liu, and Xin Peng. *LLMs Meet Library Evolution: Evaluating Deprecated API Usage in LLM-based Code Completion*. ICSE 2025. https://arxiv.org/abs/2406.09834
8. Terry Yue Zhuo, Junda He, Jiamou Sun, Zhenchang Xing, David Lo, John Grundy, and Xiaoning Du. *Identifying and Mitigating API Misuse in Large Language Models*. IEEE Transactions on Software Engineering, 2025. https://arxiv.org/abs/2503.22821
