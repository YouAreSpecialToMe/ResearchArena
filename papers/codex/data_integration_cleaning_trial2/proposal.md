# CanopyER: Budgeted Rewrite-vs-Match Scheduling with Local Canonicalization and Incremental Blocker Maintenance

## Introduction

Progressive entity matching (EM) optimizes the order of work under a fixed budget so that useful matches appear early. Recent systems already provide strong schedulers over a fixed candidate space, including ProgressER, Progressive Entity Matching via Cost Benefit Analysis, and the 2025 progressive EM design-space study. Cleaning-aware iterative ER systems such as Detective Gadget show that repairing dirty representations can improve downstream matching. What remains missing is a narrower systems question:

**Under a strict runtime budget, when is it better to spend the next unit of work on matching the current candidate graph versus locally rewriting the representation so the candidate graph itself changes?**

CanopyER targets that question directly. It treats progressive EM as competition between two action types:

1. `match`: evaluate the next batch of currently visible candidate pairs.
2. `clean`: apply a blocker-local canonicalization and incrementally update only the touched portion of the candidate graph.

The novelty claim is intentionally narrow: **CanopyER is not a new cleaning operator family, not a new blocker, and not a general iterative dirty-data ER framework. It is a budgeted integration of progressive scheduling, blocker-local canonicalization, and incremental blocker maintenance over `match` and `clean` actions on a mutable candidate graph.**

The hypothesis is that this formulation improves duplicate-yield-versus-time on dirty integration workloads because it can unlock high-value candidate regions without paying the cost of global cleaning or full blocker rebuilds.

## Proposed Approach

### Core Formulation

Input:

- one table for deduplication or two tables for clean-clean EM;
- a deterministic lexical blocker;
- a deterministic pair scorer and match threshold;
- a wall-clock budget `B`.

State:

- current canonicalized record values;
- postings lists and block summaries for the blocker;
- current candidate graph `G_t`;
- frontier of unevaluated candidate pairs;
- action statistics from prior `match` and `clean` executions.

Action space:

- `match(b)`: evaluate the next batch `b` of candidate pairs from the current frontier;
- `clean(c, o)`: apply operator `o` to canopy `c`, then incrementally mutate only the affected blocker state and refresh newly exposed local candidates.

Objective:

- maximize normalized AUC of duplicate yield versus wall-clock time;
- use final F1 at budget `B` as a secondary guardrail.

This action-space definition is the paper's differentiator. Prior progressive work schedules over a fixed representation or over changes caused by incoming records; CanopyER lets the scheduler spend budget on locally changing the existing representation itself.

### Canopies and Deterministic Operators

A **canopy** is a small blocker-centered locality defined by an `(attribute, block-neighborhood)` pair. It contains records participating in a suspiciously fragmented region, such as multiple lexical variants that nearly collide but currently hash into disjoint or weakly connected candidate neighborhoods.

The operator library is lightweight and deterministic:

- case folding and Unicode normalization;
- punctuation and whitespace normalization;
- token sorting for title-like fields;
- numeric, date, and unit normalization;
- abbreviation expansion;
- edit-distance-1 typo consolidation within a canopy.

The contribution is not inventing these operators. The contribution is deciding **whether a particular local rewrite is worth more than another batch of pair evaluations right now**.

### Reproducible Abbreviation and Typo Rules

The operator library is fully materialized and released as plain-text rule files.

Abbreviation dictionaries are built in two stages:

1. a small domain-agnostic seed list for months, units, corporate suffixes, venue terms, and common bibliographic abbreviations;
2. deterministic corpus mining on unlabeled raw records using only string patterns such as `long form (abbr)`, `abbr = long form`, token-initial alignment, and subsequence matching.

An induced pair is kept only if it:

- appears at least twice in the corpus being processed;
- has a unique majority expansion with confidence at least `0.9`;
- does not map two distinct long forms to the same short form inside one attribute domain.

Typo consolidation is similarly deterministic:

- candidate typo groups are generated only within a canopy;
- variants must be edit-distance `1`, share the same first character, and differ by at most one token;
- the canonical form is the most frequent variant in that canopy.

No labels are used in either dictionary-building step. The released artifact will include the final dictionaries per dataset-setting together with the induction script and thresholds.

### Scoring Match and Clean Actions

For each clean action `a`, CanopyER predicts:

- `gain(a)`: expected new true matches unlocked by the rewrite;
- `risk(a)`: expected precision loss from over-canonicalization;
- `cost(a)`: measured rewrite plus incremental blocker-maintenance cost.

It scores clean actions as:

`S_clean(a) = (gain(a) - lambda * risk(a)) / cost(a)`

For each match batch `m`, it predicts:

- `yield(m)`: expected true matches in the next batch;
- `cost(m)`: measured batch-evaluation time.

It scores match actions as:

`S_match(m) = yield(m) / cost(m)`

At each decision point, the scheduler executes the action with the highest score until the budget is exhausted.

### Features for Gain and Risk

`gain(a)` uses a small feature vector:

- near-miss density inside the canopy;
- token-entropy reduction after the hypothetical rewrite;
- number of records likely to move into already promising neighboring blocks;
- fraction of records currently in candidate-starved blocks;
- micro-simulation estimate from applying the rewrite to a small sample and refreshing only sampled postings.

`risk(a)` uses conservative features:

- increase in value collisions among currently dissimilar records;
- shrinkage in matcher score margins for top candidate pairs;
- fraction of newly created candidate pairs whose attribute agreement profile is weak;
- historical precision drop for the same operator family on cross-fitted development settings.

### Estimator Fitting Without Leakage

The action scores use simple linear models with non-negative coefficients plus isotonic calibration on the predicted gain. The fitting protocol is cross-fitted at the dataset-setting level:

1. For each target setting, collect pilot action logs only from the other settings.
2. Generate those logs using a fixed heuristic scheduler, not CanopyER itself, to avoid self-confirming training data.
3. Compute realized action outcomes offline from ground truth only for the training settings.
4. Fit coefficients and calibration maps on the training settings.
5. Freeze them before running the target setting.

Thus, when evaluating one dataset-setting, no labels from that setting influence its estimator weights, calibration, or operator priors.

This estimator is intentionally modest. The paper does not claim a universally portable action-value model. Instead it tests whether a small, conservative ranker is sufficient to beat simple heuristics and sanity baselines in this specific mutable-action regime. The strongest interpretation will therefore be comparative: if the estimator fails to beat these lightweight baselines, the paper's main systems claim weakens accordingly.

### Incremental Candidate-Graph Mutation

After `clean(c, o)`, CanopyER:

- removes only the rewritten records from their current postings lists;
- reinserts only their updated tokens or signatures;
- recomputes candidate pairs only for touched blocks and neighboring blocks;
- keeps untouched postings, blocks, and already evaluated pairs fixed.

This makes local cleaning budget-competitive. The key systems claim is that blocker-local mutation can be much cheaper than global reblocking while still exposing useful new candidate pairs.

## Related Work

### Precise Positioning

The proposal is closest to four lines of prior work, but differs on one exact axis: **budgeted competition between `match` and blocker-local `clean` actions under a mutable candidate graph.**

| System | Primary objective | Action granularity | Candidate graph mutable during run? | Stopping regime |
| --- | --- | --- | --- | --- |
| Detective Gadget (Buoncristiano et al., 2024) | Iterative ER quality improvement on dirty data | Broad iterative repair, alias update, check functions, optional human feedback | Yes, through iterative repair and hashing updates | Iterate to fixpoint or user stop |
| Iterative Blocking (Whang et al., 2009) | Recover more matches by propagating block evidence | Block-level iterative propagation | Yes, via iterative block interactions | Iterate until convergence / no new evidence |
| Progressive ER over Incremental Data (Gazzarri & Herschel, 2023) | Maximize early utility as new data increments arrive | Prioritized pending comparisons after arrivals | Yes, due to arriving records | Budgeted progressive execution between increments |
| Incremental Entity Blocking over Heterogeneous Streaming Data (Araujo et al., 2022) | Maintain blocks efficiently as stream increments arrive | Incremental block updates for new arrivals | Yes, due to arriving records | Continuous streaming |
| ProgressER (Altowim et al., 2018) | Maximize early ER utility under budget | Block, pair, and workflow scheduling | No; representation fixed | Budgeted progressive execution |
| Progressive EM Design Space (Maciejewski et al., 2025) | Characterize progressive scheduling/filtering pipelines | Edge/node/hybrid scheduling over weighted pairs | No; representation fixed | Budgeted progressive execution |
| CanopyER | Maximize early duplicate yield by choosing between comparison and local rewrite | Batch-level `match` versus canopy-level `clean` | **Yes; blocker-local and explicit** | Budgeted progressive execution |

This is the core novelty defense. CanopyER is narrower than Detective Gadget and more mutable than fixed-representation progressive schedulers.

### Closest Papers

**Detective Gadget: Generic Iterative Entity Resolution over Dirty Data** studies iterative dirty-data ER where repairs and matching reinforce one another. It is the nearest conceptual neighbor, but the operating regime is different: Detective Gadget seeks iterative quality improvement, while CanopyER optimizes early utility under a hard budget by explicitly pricing rewrite actions against comparison work.

**Entity Resolution with Iterative Blocking** updates blocking structure as evidence accumulates, but it does not formulate online competition between local canonicalization and matching under a time budget.

**Progressive Entity Resolution over Incremental Data** is a critical near-neighbor because it already studies progressive behavior on a changing comparison space. The distinction is the source of change. Its graph evolves because new records arrive and old comparisons can be deferred between increments; CanopyER instead assumes a fixed input at run start and makes the graph mutable only through chosen local canonicalization actions. That difference matters because the scheduler must decide whether graph mutation itself is worth paying for.

**Incremental Entity Blocking over Heterogeneous Streaming Data** is the key systems-side neighbor for incremental maintenance. It shows that updating blocks without full reconstruction is feasible under noisy streaming arrivals. But it optimizes blocking maintenance for insertions, not progressive duplicate yield under a fixed end-to-end budget, and it does not expose a scheduler that trades blocker mutation against match evaluation.

**ProgressER: Adaptive Progressive Approach to Relational Entity Resolution** is a direct antecedent for progressive utility maximization, but it assumes the record representation is fixed throughout the run.

**Progressive Entity Matching via Cost Benefit Analysis** is close in spirit on benefit-cost scheduling, but schedules static partitions rather than local rewrites that mutate the blocker.

**Progressive Entity Matching: A Design Space Exploration** systematizes edge-centric, node-centric, and hybrid schedulers over weighted candidate graphs. CanopyER reuses that fixed-representation setting as a strong baseline family, then asks what changes once local rewrites become schedulable actions.

**Effective and Efficient Data Cleaning for Entity Matching** shows that targeted cleaning can matter for EM quality. CanopyER moves this idea inside the progressive loop and treats cleaning as a competing budgeted action.

**Sparkly** provides a strong lexical blocking substrate and motivates using a competitive blocker rather than claiming gains from a weak baseline blocker.

## Experiments

### Evaluation Scope

The experiments are scoped for `2` CPU cores, `128 GB` RAM, and about `8` total hours. The proposal therefore uses:

- `5` dataset-settings in the main comparison;
- a single `6`-minute budget per run, with intermediate checkpoints at 1 and 3 minutes from the same run;
- deterministic methods wherever possible, with corruption generation as the only planned source of small repeated variation.

### Dataset-Settings

Natural settings:

- Abt-Buy;
- Amazon-Google Products;
- DBLP-ACM.

Controlled dirty settings:

- Amazon-Google with abbreviation, punctuation, unit, and token-order corruption;
- DBLP-ACM with venue abbreviation, author shortening, token-order, and year-format corruption.

This keeps the suite broad enough for a systems paper while remaining feasible.

### Shared Stack

All methods use the same fixed core:

- blocker: Sparkly-style lexical retrieval with token and character-n-gram TF-IDF features and fixed top-`k` candidate generation;
- pair scorer: deterministic weighted similarity stack over normalized attributes;
- batch size: fixed across all methods;
- all scheduler overhead counted against the same wall-clock budget.

This isolates the contribution to scheduling and candidate-graph maintenance rather than to the matcher.

### Baselines

1. `RawPEM`
   Progressive execution on the raw fixed representation with no cleaning and no blocker mutation.

2. `FullClean+PEM`
   Apply the full operator library globally before blocking, then run the same progressive scheduler. Global cleaning and full rebuild time count against the same `6`-minute budget.

3. `LocalHeuristic`
   Use the same canopy and incremental mutation mechanism as CanopyER, but rank `clean` actions by a simple handcrafted heuristic such as near-miss density divided by touched-postings cost.

4. `MutableGreedy`
   A trivial sanity baseline for the mutable-action setting: if the best available clean action exceeds a fixed hand-tuned heuristic threshold on near-miss density and estimated touched-postings cost, execute it; otherwise execute the next match batch. It uses no learned gain/risk model.

5. `RandomMutable`
   Another sanity baseline for the mutable-action regime: sample between legal `match` and `clean` actions with probability proportional to inverse measured cost, while respecting the same operator library and incremental maintenance stack.

6. `HybridStatic`
   A direct fixed-representation baseline from the 2025 design-space paper: hybrid progressive scheduling over the weighted candidate graph with no cleaning actions and no graph mutation.

The executable study uses the six baselines above plus `CanopyER`, which matches the pre-registered `plan.json` matrix. An earlier draft mentioned `PartitionCBA-Adapted`; that baseline is not part of the final runnable package and should not be interpreted as missing evidence.

### Exact `HybridStatic` Recipe

To reduce ambiguity, the paper pre-registers one concrete `HybridStatic` implementation rather than a flexible family:

- candidate graph: union of top-`k=40` candidates per record from the shared Sparkly-style blocker;
- edge weight: deterministic pair score from the shared similarity stack;
- node score: sum of incident edge weights over still-unseen edges;
- scheduler: at each step, choose the next batch from the higher-scoring option between:
  - `edge` mode: top unseen edges globally by edge weight;
  - `node` mode: highest-scoring node, then emit its unseen incident edges by descending edge weight;
- mixing rule: recompute both scores every batch and take whichever mode has the larger average predicted pair score in its next batch;
- no cleaning, no candidate-graph mutation, no reweighting from observed labels.

This is deliberately simple, faithful to the hybrid static spirit of the design-space paper, and fully reproducible within the CPU budget.

### Fairness Validation for `PartitionCBA-Adapted`

Because `PartitionCBA-Adapted` is not the published system, the paper will validate it as a fair static scheduler rather than present it as an exact reproduction.

Validation steps:

- use the same static candidate graph and pair scores as `RawPEM` and `HybridStatic`;
- count partition construction and scheduling overhead against budget;
- verify that it consistently beats random partition order on the natural settings;
- confirm that its partition ordering tracks observed partition duplicate density better than random;
- report it as an adapted baseline everywhere, including tables and captions.

This avoids overstating what is being reproduced while still testing whether a benefit-cost static scheduler is competitive.

### Ablations

Ablations are deliberately reduced to keep slack in the runtime budget. They run on two representative settings only:

- Amazon-Google corrupted;
- DBLP-ACM natural.

Ablation family:

1. `NoMicroSim`: remove micro-simulation from `gain(a)`.
2. `NoRisk`: remove the precision-risk term.
3. `FullReblock`: replace incremental mutation with full blocker rebuild after each chosen clean action.
4. `FormatOnly`: keep only format and whitespace operators.

This covers the estimator, the risk term, the systems claim, and the operator-library scope without exploding the matrix.

### Failure Analysis

One explicit analysis slice is reserved for the cases where cleaning hurts:

- identify clean actions with negative realized net utility: few unlocked true matches, high false-positive introduction, or high maintenance cost;
- bucket these failures by operator family, canopy size, and pre-action collision rate;
- report concrete examples where canonicalization over-merges semantically different values or spends budget in already saturated neighborhoods;
- compare how often CanopyER, `MutableGreedy`, and `RandomMutable` trigger such harmful actions.

This prevents over-interpreting positive average estimator correlations and clarifies when local cleaning should be avoided.

### Metrics

Primary metric:

- normalized AUC of duplicate yield versus wall-clock time.

Secondary effectiveness metrics:

- recall at 1, 3, and 6 minutes;
- precision, recall, and F1 at 6 minutes.

Systems metrics:

- time spent in matching, scoring, cleaning, and blocker maintenance;
- number of records touched and postings updated per clean action;
- number of new candidate pairs and new true matches unlocked by each clean action.

Estimator-validation metrics:

- Spearman correlation between predicted and realized clean-action gain;
- calibration error of predicted gain after isotonic calibration;
- observed precision drop by operator family.

### Runtime Accounting

The runtime story is designed to be comfortably feasible.

Main comparison:

- `5` settings x `7` methods x `6` minutes = `210` method-minutes, or `3.5` wall-clock hours if run serially.

Ablations:

- `2` settings x `4` variants x `6` minutes = `48` method-minutes, or `0.8` hours.

Additional overhead:

- corruption generation, pilot log collection for cross-fitted estimator weights, exact-baseline validation, and aggregation are budgeted to remain under `1.8` hours total.

That leaves about `1.9` hours of slack for implementation friction, reruns, or slower-than-expected blocker maintenance while staying within the `8`-hour limit.

## Success Criteria

The hypothesis is supported if:

- CanopyER beats `RawPEM` on progressive AUC in at least `4/5` settings;
- CanopyER beats `LocalHeuristic` and `MutableGreedy` on progressive AUC in at least `4/5` settings, showing the estimator matters beyond localized cleaning alone;
- CanopyER matches or exceeds `HybridStatic` on mean progressive AUC across the two corrupted settings;
- incremental mutation is at least `2x` cheaper than `FullReblock` for the same selected clean actions;
- predicted clean-action gain has positive rank correlation with realized unlocked duplicates and limited observed precision loss;
- failure analysis shows that harmful cleaning actions concentrate in identifiable high-collision or already-saturated canopies rather than occurring uniformly at random.

The hypothesis is weakened if:

- gains vanish once `HybridStatic` is included;
- the estimator does not rank useful clean actions above weak ones or fails to beat `MutableGreedy`;
- precision degrades sharply whenever local rewrites are selected;
- incremental maintenance is not materially cheaper than full rebuilds.

## References

Altowim, Y., Kalashnikov, D. V., & Mehrotra, S. (2018). ProgressER: Adaptive progressive approach to relational entity resolution. *ACM Transactions on Knowledge Discovery from Data, 12*(3), Article 33. https://doi.org/10.1145/3154410

Ao, J., & Chirkova, R. (2019). Effective and efficient data cleaning for entity matching. In *Proceedings of the Workshop on Human-In-the-Loop Data Analytics* (Article 2). https://doi.org/10.1145/3328519.3329127

Araujo, T. B., Stefanidis, K., Pires, C. E. S., Nummenmaa, J., & da Nobrega, T. P. (2022). Incremental entity blocking over heterogeneous streaming data. *Information, 13*(12), 568. https://doi.org/10.3390/info13120568

Buoncristiano, M., Mecca, G., Santoro, D., & Veltri, E. (2024). Detective Gadget: Generic iterative entity resolution over dirty data. *Data, 9*(12), 139. https://doi.org/10.3390/data9120139

Gazzarri, L., & Herschel, M. (2023). Progressive entity resolution over incremental data. In *Proceedings of the 26th International Conference on Extending Database Technology* (pp. 80-91). https://doi.org/10.48786/EDBT.2023.07

Maciejewski, J., Nikoletos, K., Papadakis, G., & Velegrakis, Y. (2025). Progressive entity matching: A design space exploration. *Proceedings of the ACM on Management of Data, 3*(1), Article 65. https://doi.org/10.1145/3709715

Paulsen, D., Govind, Y., & Doan, A. (2023). Sparkly: A simple yet surprisingly strong TF/IDF blocker for entity matching. *Proceedings of the VLDB Endowment, 16*(6), 1507-1519. https://doi.org/10.14778/3583140.3583163

Sun, C., Hou, Z., Shen, D., & Nie, T. (2022). Progressive entity matching via cost benefit analysis. *IEEE Access, 10*, 3979-3989. https://doi.org/10.1109/ACCESS.2021.3139987

Whang, S. E., Menestrina, D., Koutrika, G., Theobald, M., & Garcia-Molina, H. (2009). Entity resolution with iterative blocking. In *Proceedings of the ACM SIGMOD International Conference on Management of Data* (pp. 219-232). https://doi.org/10.1145/1559845.1559870
