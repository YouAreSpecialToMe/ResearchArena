# Benchmark-Conditional Robustness Audits for Schema and Entity Matching

## Introduction
Robustness results in data integration are easy to over-interpret because many perturbations that look plausible also change the benchmark label. This matters acutely for schema matching and entity matching, where a corruption may delete the exact evidence that justified a gold correspondence or inject new ambiguity that the released annotation never resolved. Prior work already established the nearby point that entity-matching models can be brittle under label-preserving perturbations, most directly in **Probing the Robustness of Pre-trained Language Models for Entity Matching**. The paper proposed here therefore makes a narrower claim.

The contribution is **not** discovering new perturbation families and **not** proposing a new robust matcher. The contribution is a benchmark-evaluation methodology: robustness estimates should be conditioned on explicit benchmark-specific admissibility rules, deterministic checks, and a dual-sided audit that validates both accepted and rejected perturbations. The paper asks a targeted empirical question:

1. Can a conservative, benchmark-conditional admissibility protocol achieve high precision on accepted perturbations while also rejecting many genuinely label-threatening edits?
2. Does imposing that admissibility layer materially change the robustness conclusions one would draw from naive perturbation protocols?

If the answer to the second question is yes, the paper supports a new caution for robustness evaluation. If the answer is no, but the audit still shows that the admissibility layer is sound, the paper becomes a negative-result cautionary study showing that at least for these benchmarks, conservative label-preserving restrictions do not substantially alter conclusions. Either outcome is publishable if precommitted clearly.

## Proposed Approach

### Overview
The project introduces **ABCA**: an **Audited Benchmark-Conditional Admissibility** protocol for perturbation-based robustness evaluation. ABCA is instantiated on exactly two public benchmarks:

- **T2D-SM-WH** from the WDC Schema Matching Benchmark for schema matching.
- **One fixed WDC Products pairwise slice** for entity matching.

ABCA has four components:

1. benchmark-specific protected-evidence definitions,
2. deterministic admissibility checks with reason codes,
3. a dual-sided human audit over accepted and rejected perturbations,
4. robustness comparisons against naive perturbation protocols under the same CPU budget.

The scope is intentionally narrow. The goal is a defensible methodology paper, not a broad benchmark sweep.

### 1. Benchmark-conditional admissibility
For a benchmark `B`, define:

- released data `D`,
- released labels `G`,
- a benchmark-specific normalizer `N`,
- protected evidence `E`,
- admissibility rules `R`.

A perturbation `c` is admissible for `B` only if:

1. `c(D)` remains format-valid,
2. all rule-specific preconditions in `R` hold,
3. all protected evidence in `E` is unchanged after normalization by `N`,
4. `c` does not create new benchmark-relevant ambiguity according to benchmark-specific competitor checks,
5. audit evidence shows high accepted precision and high rejected violation rate.

ABCA does not claim semantic invariance in the abstract. It claims to preserve the **released benchmark label under explicit benchmark-relative sufficient conditions**.

### 2. Protected evidence and sufficiency arguments

#### Schema matching benchmark: T2D-SM-WH
The label in T2D-SM-WH is a set of gold correspondences between source and target columns. For this benchmark, protected evidence must cover not only the positive matched columns but also the local ambiguity structure around them.

ABCA therefore protects:

- every source or target column participating in a positive gold correspondence,
- every non-matched competitor column whose normalized header or value profile is within a fixed similarity band of a protected matched column,
- one-to-one structure around each protected column, so perturbations cannot create duplicate near-matches.

The normalizer `N_sm` lowercases text, canonicalizes whitespace and punctuation, canonicalizes numbers and parseable dates, and converts each protected column to a normalized value multiset plus simple type profile.

The admissibility rule is conservative:

- for every protected matched column, normalized header semantics and normalized value multiset must be preserved,
- for every protected competitor column, the same invariants must be preserved,
- no perturbation may turn a previously non-competitive column into a near-duplicate of a protected column,
- non-protected columns may be perturbed only if they stay outside the benchmark's ambiguity band.

Allowed perturbation families are therefore restricted to:

- header abbreviation from a verified alias table,
- case and delimiter changes,
- numeric and date formatting changes,
- row reordering,
- non-protected column dropout when the competitor check still passes.

Why this is sufficient: the released T2D-SM-WH labels are column-level correspondences, and the benchmark itself evaluates methods from header and value evidence. Preserving normalized evidence for gold columns and their plausible competitors is a conservative sufficient condition for preserving the released correspondence decision. This does not prove completeness, but it directly addresses the main latent-cue objection better than protecting only positive columns.

#### Entity matching benchmark: fixed WDC Products slice
For WDC Products, the study fixes one official pairwise slice and evaluates binary match labels on record pairs. A key weakness of the earlier draft was that identifiers plus price were not enough to justify label preservation on their own.

ABCA therefore protects:

- identifier-like attributes when present (`gtin`, `gtin13`, `gtin14`, `mpn`, `sku`, `productid`),
- parsed quantities and units on numeric product fields,
- brand and model tokens recoverable either from structured attributes or from title spans aligned to those attributes,
- title spans containing any protected token.

The normalizer `N_em` lowercases, collapses whitespace, removes non-semantic punctuation, canonicalizes decimals and units, and canonicalizes alphanumeric product codes.

The admissibility rule is:

- all protected identifiers and protected numeric quantities must remain identical after normalization,
- protected brand and model tokens must remain identical,
- no edit may touch title spans containing protected tokens,
- title edits are limited to formatting-only rewrites, whitespace or punctuation perturbations, token-order permutations outside protected spans, and seller-boilerplate injection from a fixed whitelist,
- if a record lacks identifier-like and brand-model evidence, only formatting-only perturbations are admissible.

Why this is sufficient: WDC product pairs are often matched by stable product identifiers or by brand-model evidence. When such evidence is present, preserving it and forbidding edits inside the aligned title spans gives a much stronger label-preservation argument than preserving identifiers alone. When such evidence is absent, the admissible space intentionally collapses to superficial formatting edits. That narrowness is acceptable because the paper's claim is about evaluation soundness, not perturbation breadth.

### 3. Deterministic checker and reason codes
Each perturbation passes through a benchmark-specific checker with four stages:

1. family-level preconditions,
2. normalized protected-evidence equality tests,
3. competitor-ambiguity checks,
4. file and schema validation.

The checker outputs accept/reject plus a structured reason code such as:

- `protected_value_changed`,
- `protected_header_semantics_changed`,
- `new_competitor_created`,
- `edited_protected_title_span`,
- `format_invalid`.

These reason codes support the analysis of whether the protocol is merely filtering trivial edits or blocking genuinely label-threatening perturbations.

### 4. Audit design with stronger credibility
The core paper claim depends on audit credibility, so the audit is upgraded beyond the two-author design.

The full audit contains **240 perturbation instances**:

- 120 accepted perturbations,
- 120 rejected perturbations.

Sampling is balanced across:

- benchmark,
- perturbation family,
- severity,
- clean example difficulty bucket.

Annotation protocol:

1. two project authors annotate all 240 examples independently using a written rubric and are blinded to model predictions,
2. a **non-author annotator** with data-management or entity-resolution experience independently annotates a prespecified **72-example subset** stratified across the same factors,
3. disagreements are adjudicated after first-pass annotation,
4. the paper reports both two-author kappa on the full set and external-agreement statistics on the non-author subset.

If a non-author annotator is unavailable, the fallback is a stronger external validation procedure: release the rubric, anonymized examples, and checker decisions for a reproducibility package and obtain an independent annotation from a labmate not involved in rule design before finalizing conclusions. The primary plan, however, assumes one non-author annotator.

Primary audit metrics:

- accepted precision,
- rejected violation rate,
- false-rejection rate,
- two-author Cohen's kappa,
- non-author agreement with adjudicated labels on the audited subset.

### 5. Comparative robustness evaluation
The study compares three perturbation regimes:

1. **ABCA admissible perturbations**,
2. **format-validity-only perturbations**,
3. **naive unrestricted perturbations** matched by perturbation budget and operator families where possible.

Within each regime, two search modes are run:

- random perturbation sampling,
- a small targeted search with beam width 8, program length at most 3, and at most 40 model evaluations per benchmark-method pair.

The search is deliberately modest so the full study fits within the CPU-only budget.

### 6. Models under test
The evaluation uses lightweight baselines because the paper is about robustness conclusions, not state-of-the-art accuracy.

Schema matching:

- Jaccard-Levenshtein baseline in the Valentine style,
- a COMA-style hybrid matcher using header similarity, value overlap, and type compatibility.

Entity matching:

- TF-IDF cosine thresholding over concatenated fields,
- logistic regression over handcrafted similarity features.

These are CPU-feasible, transparent, and adequate for measuring rank and F1 shifts across perturbation regimes.

## Related Work

### Closest prior work: perturbation robustness in entity matching
**Rastaghi, Kamalloo, and Rafiei (2022)** is the key closest paper. It already shows that entity-matching models can fail under label-preserving perturbations. This proposal does not claim novelty over perturbation-based robustness testing. Its novelty is the benchmark-conditional layer: explicit admissibility rules tied to released benchmark evidence, deterministic rejection reasons, and a dual-sided audit showing whether naive protocols overstate brittleness.

### Benchmark realism in entity matching
**Wang et al. (2022)** show that standard EM benchmarks often encode unrealistic assumptions and propose benchmark reconstruction. The proposed paper is complementary. Rather than rebuilding the benchmark, it asks how to evaluate robustness responsibly **given a fixed released benchmark**.

### Schema matching evaluation and benchmarking
Direct literature on perturbation robustness for schema matching appears thin. That gap is part of the motivation, but it raises novelty risk unless the paper is grounded in the broader schema-matching evaluation literature.

The relevant grounding comes from:

- **Rahm and Bernstein (2001)**, which frames schema matching as a signal-combination problem and explains why small schema edits can change match evidence nontrivially.
- **Madhavan, Bernstein, and Rahm (2001)** on Cupid, which demonstrates hybrid dependence on names, types, and structure.
- **Papenbrock et al. (2021)** on Valentine, which emphasizes controlled comparison of matching methods and motivates benchmark-specific evaluation tooling.
- **Crescenzi et al. (2021)** on Alaska, which argues for flexible, task-aware data-integration benchmarks.
- **Hertling and Paulheim (2020)**, which is not schema matching per se but is directly relevant on evaluation methodology: gold-standard creation, benchmark design, and hidden benchmark bias can materially change system conclusions.

These papers support the positioning that the proposal is a robustness-**evaluation** paper informed by schema-matching benchmarking methodology, not a new matching algorithm.

### Benchmark grounding
**Peeters, Der, and Bizer (2024)** define WDC Products. **Peeters, Brinkmann, and Bizer (2024)** describe the broader WDC schema.org table corpora from which WDC SMB is constructed. Together with the benchmark release documentation for WDC SMB, they justify the choice of public, fixed benchmarks with heterogeneous schema and product evidence.

### Corruption search
**Zhu et al. (2025)** show that targeted corruption search can stress-test general ML pipelines. ABCA borrows the search perspective, but only after conditioning on benchmark-preserving admissibility; otherwise the search can optimize toward invalid relabelings.

## Experiments

### Benchmarks
The paper is precommitted to:

- **T2D-SM-WH** from WDC SMB,
- **WDC Products pairwise** using the **80% corner-case, 0% unseen-products, medium development** slice.

Exactly two benchmarks are used so the study can go deep on audit design and failure analysis.

### Metrics
The main reported metrics are:

- clean precision, recall, and F1,
- worst-case F1 drop under each perturbation regime,
- mean F1 drop under random perturbations,
- method-ranking changes across regimes,
- targeted-search minus random-search damage within regime,
- accepted precision,
- rejected violation rate,
- false-rejection rate,
- acceptance rate by perturbation family,
- runtime.

### Core analyses
The paper precommits to five main analyses:

1. ABCA vs format-validity-only perturbations,
2. ABCA vs unrestricted perturbations,
3. accepted vs rejected audit outcomes,
4. which reason codes dominate rejections,
5. whether targeted search inside the admissible space is stronger than random admissible sampling.

### Informative negative-result pathway
The previous draft left failure mode interpretation underspecified. This version predefines what counts as an informative negative result.

The paper will be written as a **careful empirical cautionary study rather than a methodology breakthrough** if all of the following hold:

- accepted precision is high, showing the admissibility rules are sound,
- rejected violation rate is moderate or high, showing the checker is not arbitrary,
- but ABCA and naive protocols yield no ranking reversals and less than **3 absolute F1 points** difference in worst-case drop on both benchmarks.

This outcome would support a narrower conclusion: for these two released benchmarks, conservative benchmark-conditional admissibility does **not** materially change robustness conclusions, implying either that naive protocols already align with benchmark labels or that the defensible admissible space is too narrow to matter. That is still a meaningful result for benchmark methodology.

A second informative negative result is:

- the admissible space is extremely narrow, e.g. overall acceptance rate below **15%** after checker filtering,
- but audit precision remains high.

That would support the claim that some released benchmarks permit only superficial robustness tests if label preservation is taken seriously. Again, this is a useful cautionary finding even without ranking changes.

### Success criteria
The paper's stronger methodology claim is supported if most of the following hold:

- accepted precision is at least **0.95**,
- rejected violation rate is at least **0.70**,
- two-author kappa is at least **0.70**,
- non-author agreement with adjudicated labels is at least **0.75**,
- at least one benchmark shows either a ranking reversal or at least **8 absolute F1 points** difference in worst-case drop between ABCA and a naive protocol,
- targeted search inside the admissible space finds larger damage than random admissible sampling under the same budget,
- all experiments finish within the CPU budget.

The stronger claim is weakened or refuted if:

- accepted precision is low,
- false rejections are common,
- non-author validation disagrees materially with the adjudicated labels,
- or the checker requires repeated benchmark-specific exceptions after inspection of results.

### Feasibility under the CPU-only budget
The study is designed for 2 CPU cores, 128 GB RAM, and about 8 total hours:

- data loading and benchmark-specific normalization: 0.8 h,
- perturbation generators and checker with reason codes: 1.5 h,
- audit packaging and annotation support files: 0.7 h,
- clean baseline runs: 0.3 h,
- perturbation experiments across all regimes and 3 seeds: 2.4 h,
- analysis and plots: 0.9 h,
- contingency: 1.0 h.

Total: **about 7.6 hours** worst-case, with no GPU dependence.

## References
1. Erhard Rahm and Philip A. Bernstein. 2001. *A Survey of Approaches to Automatic Schema Matching*. The VLDB Journal 10(4):334-350.
2. Jayant Madhavan, Philip A. Bernstein, and Erhard Rahm. 2001. *Generic Schema Matching with Cupid*. Microsoft Research Technical Report MSR-TR-2001-58.
3. George Papenbrock, Alexander Fink, Boris G. Trujillo-Rasua, Marcus Samuel, Ziawasch Abedjan, Valter Crescenzi, Michele Dallachiesa, Katja Hose, Christoph Lange, Bruno Marnette, Giansalvatore Mecca, Tova Milo, Jannik Pellenz, Markus Strohmaier, and George Tsaras. 2021. *Valentine: Evaluating Matching Techniques for Dataset Discovery*. IEEE 37th International Conference on Data Engineering, 2745-2748.
4. Sven Hertling and Heiko Paulheim. 2020. *The Knowledge Graph Track at OAEI -- Gold Standards, Baselines, and the Golden Hammer Bias*. Proceedings of the 11th on Knowledge Capture Conference, 343-346.
5. Valter Crescenzi, Andrea De Angelis, Donatella Firmani, Maurizio Mazzei, Paolo Merialdo, Federico Piai, and Divesh Srivastava. 2021. *Alaska: A Flexible Benchmark for Data Integration Tasks*. arXiv:2101.11259.
6. Ralph Peeters, Reng Chiz Der, and Christian Bizer. 2024. *WDC Products: A Multi-Dimensional Entity Matching Benchmark*. Proceedings of the 27th International Conference on Extending Database Technology, 22-33.
7. Ralph Peeters, Alexander Brinkmann, and Christian Bizer. 2024. *The Web Data Commons Schema.org Table Corpora*. Companion Proceedings of the ACM Web Conference 2024, 1079-1082.
8. Tianshu Wang, Hongyu Lin, Cheng Fu, Xianpei Han, Le Sun, Feiyu Xiong, Hui Chen, Minlong Lu, and Xiuwen Zhu. 2022. *Bridging the Gap between Reality and Ideality of Entity Matching: A Revisiting and Benchmark Re-Construction*. Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, 3978-3984.
9. Mehdi Akbarian Rastaghi, Ehsan Kamalloo, and Davood Rafiei. 2022. *Probing the Robustness of Pre-trained Language Models for Entity Matching*. Proceedings of the 31st ACM International Conference on Information and Knowledge Management, 3786-3790.
10. Jiongli Zhu, Geyang Xu, Felipe Lorenzi, Boris Glavic, and Babak Salimi. 2025. *Stress-Testing ML Pipelines with Adversarial Data Corruption*. Proceedings of the VLDB Endowment 18(11):4668-4681.
