# RevisionBench: A Compact Benchmark for Abstract-Local Scientific Claim Updates

## Introduction

Benchmarks for evolving knowledge have advanced quickly in 2024-2025, but they mostly evaluate world knowledge, temporal QA, or stale retrieval rather than revision-aware scientific claims. EvoWiki and EvolveBench study evolving knowledge and temporal awareness for LLMs. HoH evaluates how outdated information harms retrieval-augmented generation. ChronoFact studies temporal fact verification with timeline reasoning. These are strong adjacent precedents, and they materially limit any broad novelty claim about "changing facts" or "revision-aware verification."

Scientific revision data also already exists. arXivEdits provides aligned scientific revisions with human edit annotations, and CASIMIR scales scientific version alignment to a much larger OpenReview corpus. VitaminC further shows that document revisions can be turned into fact-verification examples. The defensible gap is therefore narrower than the previous draft claimed: there is still no carefully validated benchmark focused on **abstract-local obsolete-vs-current scientific claim updates**, where inclusion requires adjudicated confirmation that an earlier abstract claim is no longer valid as stated relative to the latest abstract.

This proposal positions RevisionBench as a compact benchmark paper rather than a broad new benchmark family.

**Core claim.** RevisionBench is a compact, human-validated benchmark for abstract-local scientific claim updates mined from real paper revisions. Its contribution is not revision histories in general, and not temporal verification in general, but adjudicated obsolete-versus-current scientific claims under strictly local abstract evidence.

This matters because literature assistants, survey agents, and scientific claim-verification systems routinely encounter multiple paper versions. If a system repeats an older metric, outdated comparison, or softened-away conclusion, it fails in a way that static scientific QA and static claim-verification benchmarks do not expose.

**Hypothesis.** Even when restricted to abstract-local, adjudicated claim updates, strong open models will still over-select stale earlier claims, with the largest failures on comparative updates and semantic softening rather than pure lexical matching cases.

## Proposed Approach

### Benchmark Scope

RevisionBench is intentionally small and narrow.

Primary task:

1. **Current-Claim Selection**
   Input: earlier claim, later claim, latest abstract.
   Output: which claim is current in the latest version.

Secondary task, included only if the pilot clears agreement gates:

2. **Version-Conditioned Verification**
   Input: one claim and the latest abstract.
   Output: `obsolete`, `supported_current`, or `insufficient_evidence`.

The paper is benchmark-first. Task 1 carries the main claim. Task 2 is a controlled extension rather than a headline contribution.

### Data Sources

- arXiv abstracts from papers with at least two versions are the primary source.
- If the pilot shows low yield in CS-only arXiv, add a tightly matched OpenReview slice using CASIMIR-style metadata and keep evaluation split by paper source.

The benchmark remains abstract-local throughout. Full-paper evidence is excluded from the main task to preserve annotation consistency, reproducibility, and compute feasibility.

### Mining Pipeline

1. Extract abstracts for all versions of each paper.
2. Split abstracts into sentences and align adjacent versions with a lexical aligner modeled after arXivEdits-style sentence matching.
3. Add a paraphrase-aware recovery pass using a lightweight sentence encoder such as `all-mpnet-base-v2` to flag likely semantic matches missed by lexical overlap.
4. Retain edited sentence pairs with one dominant claim and discard obvious stylistic edits with rules.
5. Prioritize candidates matching one or more triggers:
   - numeric updates
   - comparative claim changes
   - certainty or hedging changes
   - changed dataset, task, or evaluation setting
   - explicit weakening, narrowing, or removal of a result claim
6. Sample annotation batches with quotas rather than pure frequency sampling so the benchmark is not dominated by numeric edits.

The semantic recovery pass is framed conservatively. It is not a claim of solving deep alignment; it is an audit mechanism to quantify how much benchmark yield would otherwise collapse to local lexical changes.

### Annotation Protocol

Each candidate receives two independent annotations plus adjudication when needed. Annotators see:

- earlier sentence
- later sentence
- earlier abstract
- latest abstract
- version IDs only as identifiers

Step A: substantiveness screen

- `substantive_change`
- `non_substantive_edit`
- `ambiguous`

Step B: earlier-claim status against the latest abstract

- `obsolete`
- `supported_current`
- `insufficient_evidence`

Primary benchmark inclusion rule:

- keep only candidates where adjudication confirms `substantive_change`
- keep Task 1 items only when the earlier claim is adjudicated `obsolete` and the later claim is supported as current by the latest abstract

This avoids the earlier overclaim that any revision pair is useful benchmark evidence.

### Annotation Operations Plan

The real bottleneck is annotation, so the proposal makes it explicit.

- Pilot pool: 120 candidates
- Main pool after go decision: 580 to 780 additional candidates
- Total annotation pool: 700 to 900 candidates

Operational estimate:

- average first-pass annotation time: 2.0 minutes per candidate per annotator
- double annotation over 700 to 900 candidates: 46.7 to 60.0 annotator-hours
- adjudication for an expected 20% to 30% of cases at 2.5 minutes each: 5.8 to 11.3 hours
- handbook calibration, pilot discussion, and spot-audits: 6 to 8 hours

Total planned labor:

- **58 to 79 annotator-hours** including adjudication and calibration

Adjudication policy:

- all label disagreements are adjudicated by a third reviewer
- borderline cases involving abstract insufficiency are resolved in favor of exclusion from Task 1
- handbook updates are allowed only after the pilot; after that, guidelines freeze

Contingency if deeper semantic revisions are rare:

- maintain the paper claim as a compact benchmark for abstract-local scientific claim updates
- merge the fine-grained taxonomy into two preregistered groups for inferential analysis:
  - `local_updates`: numeric and comparative
  - `semantic_updates`: softening/retraction, scope qualification, dataset/setting change
- if `semantic_updates` remain below 30 retained items, report them descriptively and do not make inferential claims by subtype

### Revision Taxonomy and Preregistered Analysis Counts

Fine-grained labels:

- `numeric_update`
- `comparative_update`
- `softening_or_retraction`
- `dataset_or_setting_change`
- `scope_qualification_change`

Preregistered reporting policy:

- target at least **25 retained examples** each for `numeric_update` and `comparative_update`
- target at least **20 retained examples in aggregate** across the three semantic categories during collection, with a hard minimum of **30 total semantic-update items** required for inferential coarse-grained analysis after adjudication
- if any fine-grained category has fewer than 15 examples, report it descriptively only
- inferential hypothesis tests are performed only on the two coarse groups and any fine-grained category with at least 25 examples

This directly addresses the statistical-power concern in the previous draft.

### Pilot and Go/No-Go Gates

Annotate the first 120 candidates before scaling up.

Record:

- substantive-change rate
- obsolete rate among substantive changes
- agreement on substantiveness
- agreement on `obsolete` vs `insufficient_evidence`
- retained-item mix by revision category
- extra yield from the paraphrase-aware recovery pass

Proceed only if the pilot supports a realistic path to:

- at least 150 final Task 1 items from the full annotation pool under Wilson intervals
- at least 30 retained semantic-update items, or else explicitly downgrade the paper framing to a mostly local-update benchmark

Retain Task 2 only if pilot agreement on `obsolete` vs `insufficient_evidence` among substantive items reaches:

- Cohen's kappa >= 0.60
- raw agreement >= 80%

### Baselines

The earlier proposal leaned too heavily on a trivial "later claim wins" heuristic. This revision makes that baseline non-headline and adds a stronger symmetric verifier that does not privilege the later claim by construction.

Non-LLM baselines:

1. `LaterClaimWins`
2. lexical-overlap scorer against the latest abstract
3. embedding-similarity scorer against the latest abstract
4. **symmetric pairwise NLI verifier**: run an off-the-shelf MNLI/DeBERTa-style model on `(latest abstract, claim)` for both claims, convert entailment and contradiction probabilities into a calibrated support score, and select the claim with the higher support margin

LLM baselines:

- `Qwen2.5-7B-Instruct`
- `Llama-3.1-8B-Instruct`

The symmetric NLI verifier is the strongest nontrivial baseline because it evaluates both claims against the same evidence without using temporal position as a shortcut.

### Prompting and Variance Control

To avoid prompt-selection instability:

- primary results are **zero-shot**
- instructions are fixed before test evaluation
- if a few-shot condition is included, it uses one frozen demonstration set chosen from development data once and reused for all models
- no repeated resampling of demonstrations from a tiny dev split
- decoding is greedy or temperature-zero for all main runs

### Experimental Setup

Task 1 settings:

- claim pair only
- claim pair plus latest abstract

Task 2 setting:

- single claim plus latest abstract

Metrics:

- accuracy on Task 1
- macro-F1 on Task 2 if retained
- obsolete-claim attraction rate
- per-group performance for `local_updates` vs `semantic_updates`
- benchmark-construction metrics: raw agreement, Cohen's kappa, adjudication rate, retained-item rate

Statistics:

- Wilson intervals for pilot yield
- paper-level bootstrap 95% confidence intervals for model scores
- paired bootstrap for context-vs-no-context comparisons

## Related Work

### Scientific revisions

**arXivEdits** (Jiang, Xu, and Stevens, EMNLP 2022) is the clearest methodological precursor. It shows that scientific versions can be aligned and annotated at sentence level, but it is a revision-process corpus, not an evaluation benchmark for obsolete scientific claims.

**CASIMIR** (Jourdan et al., LREC-COLING 2024) scales scientific revision alignment substantially and makes clear that revision histories themselves are not novel. RevisionBench differs only at the task level: it keeps adjudicated factual-status changes and evaluates held-out obsolete-vs-current claims.

### Revision-derived verification

**VitaminC** (Schuster, Fisch, and Barzilay, NAACL 2021) is the closest non-scientific benchmark precedent. It already demonstrates that revisions can produce contrastive verification examples. RevisionBench therefore cannot claim novelty for "using revisions as supervision." Its narrower distinction is scientific-domain abstract revisions with adjudicated obsolete-current status under local evidence.

### Evolving knowledge and temporal verification

**EvoWiki** (Tang et al., ACL 2025) evaluates LLMs on evolving Wikipedia knowledge and auto-updatable benchmark construction. It is highly relevant, but its items are encyclopedia-style world knowledge rather than authored scientific claims within a single paper revision chain.

**HoH** (Ouyang et al., ACL 2025) studies the effect of outdated information on RAG. It focuses on stale retrieval sources and answer generation, not revision-aware claim discrimination under fixed local evidence.

**EvolveBench** (Zhu et al., ACL 2025) broadens temporal-awareness evaluation across cognition, awareness, trustworthiness, understanding, and reasoning. It reinforces that temporal robustness is already an active benchmark area, which is why RevisionBench deliberately narrows itself to scientific abstract updates rather than general temporal awareness.

**ChronoFact** (Barik, Hsu, and Lee, IJCAI 2025) studies temporal fact verification with timeline reasoning over complex temporal claims. It is adjacent in verification form but differs sharply in evidence structure: RevisionBench uses paired authored revisions plus a latest abstract, not external event timelines.

**evolveQA** (Nakshatri et al., arXiv 2025) probes evolving external knowledge with timestamped corpora and conflicting facts. It is another strong reminder that the contribution here is not "facts can change," but that scientific abstracts create a distinct, locally evidenced stale-claim setting.

### Scientific claim verification

**SciVer** (Wang et al., ACL 2025) is the most relevant scientific-verification benchmark comparator. It evaluates multimodal claim verification over scientific papers, but it is static. RevisionBench complements it by isolating version change rather than multimodal evidence integration.

**ELAIPBench** (Dai et al., arXiv 2025) evaluates expert-level AI paper understanding rather than revision-aware claim updates.

### Positioning

The proposal is intentionally not "the first revision-aware scientific verification benchmark" in a broad sense. A more defensible positioning is:

- compact benchmark, not broad benchmark family
- scientific abstracts, not general dynamic knowledge
- adjudicated obsolete/current updates, not generic revision alignment
- abstract-local evidence, not full-document or retrieval-heavy verification

## Experiments

Expected benchmark size after the full pipeline:

- Task 1: 150 to 220 items
- Task 2: only if pilot agreement gate is met

Compute fit:

- no model training from scratch
- no fine-tuning required
- small open-model inference only
- sentence embedding and NLI baselines are lightweight
- all evaluation fits comfortably on one RTX A6000 within the stated eight-hour total experiment budget

Main experimental questions:

1. Can strong open models reject obsolete earlier claims when both claims remain superficially plausible?
2. Does the latest abstract materially improve selection accuracy over claim-pair-only input?
3. Are semantic updates harder than local numeric/comparative updates?
4. How competitive is the symmetric pairwise NLI baseline relative to LLMs?
5. How much of benchmark difficulty remains after excluding purely stylistic edits and unstable labels?

## Success Criteria

The proposal is supported if:

1. the pilot supports a Wilson-confidence path to at least 150 retained Task 1 items from 700 to 900 annotated candidates
2. annotation quality is credible, with substantial agreement on substantiveness and acceptable agreement on `obsolete` vs `insufficient_evidence`
3. the strongest model remains below annotator agreement on Task 1
4. the symmetric pairwise NLI baseline is competitive but does not saturate the benchmark
5. access to the latest abstract improves performance over claim-pair-only input
6. semantic-update items are measurably harder than the easiest local-update category, or, if too rare, the paper explicitly narrows its scope claim

The proposal is weakened if:

- abstract-local obsolete cases are too rare to build a 150-item benchmark within the annotation budget
- `obsolete` and `insufficient_evidence` remain too unstable even after handbook calibration
- the retained set collapses almost entirely to numeric edits
- simple symmetric NLI or lexical baselines match annotator-level performance

## References

1. Jiang, C., Xu, W., and Stevens, S. 2022. arXivEdits: Understanding the Human Revision Process in Scientific Writing. Proceedings of EMNLP 2022.
2. Jourdan, L., Boudin, F., Hernandez, N., and Dufour, R. 2024. CASIMIR: A Corpus of Scientific Articles Enhanced with Multiple Author-Integrated Revisions. Proceedings of LREC-COLING 2024.
3. Schuster, T., Fisch, A., and Barzilay, R. 2021. Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence. Proceedings of NAACL 2021.
4. Tang, W., Cao, Y., Deng, Y., Ying, J., Wang, B., Yang, Y., Zhao, Y., Zhang, Q., Huang, X., Jiang, Y.-G., and Liao, Y. 2025. EvoWiki: Evaluating LLMs on Evolving Knowledge. Proceedings of ACL 2025.
5. Ouyang, J., Pan, T., Cheng, M., Yan, R., Luo, Y., Lin, J., and Liu, Q. 2025. HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation. Proceedings of ACL 2025.
6. Zhu, Z., Liao, Y., Chen, Z., Wang, Y., Guan, Y., Wang, Y., and Wang, Y. 2025. EvolveBench: A Comprehensive Benchmark for Assessing Temporal Awareness in LLMs on Evolving Knowledge. Proceedings of ACL 2025.
7. Barik, A. M., Hsu, W., and Lee, M. L. 2025. ChronoFact: Timeline-based Temporal Fact Verification. Proceedings of IJCAI 2025.
8. Nakshatri, N. S., Roy, S., Arivazhagan, M. G., Zhou, H., Kumar, V. B., and Gangadharaiah, R. 2025. When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA. arXiv:2510.19172.
9. Wang, C., Shen, Y., Kuang, Z., Cohan, A., and Zhao, Y. 2025. SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification. Proceedings of ACL 2025.
10. Dai, X., Hu, H., Chen, Y., Li, J., Jin, R., Zhang, Y., Li, X., Shang, L., and Qi, G. 2025. ELAIPBench: A Benchmark for Expert-Level Artificial Intelligence Paper Understanding. arXiv:2510.10549.
11. Das, T., Beigi, M., Aptekar, J., and Sun, J. 2026. AMEND++: Benchmarking Eligibility Criteria Amendments in Clinical Trials. arXiv:2601.06300.
