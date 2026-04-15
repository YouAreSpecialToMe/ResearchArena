# Do LLVM Optimization Remarks Add Signal Beyond Compilation Statistics for Low-Budget LLVM Micro-Search?

## Stage 2 Execution Revision

This rerun is predeclared as a **feasibility and negative-result pilot** on the provided CPU-only machine. The original `StatsGP` comparison cannot be executed here because the installed LLVM 18.1.3 optimized build exposes `-stats -stats-json` flags but emits no usable counters, so `StatsGP` is treated as infeasible rather than replaced with an unplanned surrogate.

The original primary proxy, final LLVM IR instruction count, is also flat across the full six-pass search space on this benchmark suite. In addition, the final text-size outcome often collapses across methods even when text size varies across the full space. Stage 2 therefore keeps final text size as a descriptive executable endpoint but evaluates search behavior mainly through **oracle-gap sample efficiency**: how quickly each 18-evaluation method reaches the exhaustively measured best text size on the non-flat benchmarks.

The remark-normalization interpretability claim remains gated off because real double annotation and Cohen's kappa are still unavailable in this single-operator setting. Any conclusion from this rerun is therefore limited to a restricted six-pass feasibility pilot and may legitimately end as a negative result.

## Introduction

Recent phase-ordering work has already moved beyond pass-identity search. DCO models pass dependence, the CGO 2025 synergistic-search paper reduces exploration through pass synergies, CITROEN (IPDPS 2025) uses pass-related compilation statistics inside online Bayesian optimization, and ECCO uses richer online evidence with causal reasoning. Against that 2025-2026 landscape, this proposal should not be framed as a new evidence-aware optimizer.

The paper is instead a **controlled baseline study**:

**In a tightly restricted LLVM subspace, do normalized optimization remarks provide useful online signal beyond simple probe deltas and beyond prior compiler-emitted compilation statistics?**

That question is still open in a practically relevant sense. LLVM already emits two different online evidence streams:

- `-pass-remarks*`: sparse, event-like records with explicit successes, misses, and blocker explanations
- `-stats -stats-json`: dense numeric pass counters such as numbers of vectorized loops or generated instructions

Recent work, especially CITROEN, already shows that compiler-emitted online statistics can guide search. What has not been cleanly isolated is whether the **more semantic but sparser remark stream** adds anything in a small, CPU-feasible online search regime.

The contribution is therefore intentionally narrow:

- not LLVM phase ordering broadly
- not a replacement for CITROEN or ECCO
- not a new learned optimizer
- a matched-budget, interpretability-oriented comparison of three online signals in a six-pass, sequence-length-at-most-four LLVM micro-search space

If normalized remarks win, the paper contributes a cheap symbolic baseline worth adding to future autotuning studies. If they fail, the paper still contributes a useful negative result: recent online-evidence methods may already get most available value from compilation statistics and simple probe measurements in this low-budget regime.

## Proposed Approach

### Scope and claim boundaries

The search space is deliberately restricted to six LLVM transformation passes that are both common in size/performance tuning and likely to emit remarks or statistics:

- `inline`
- `loop-rotate`
- `licm`
- `loop-unroll`
- `loop-vectorize`
- `slp-vectorizer`

All searched pipelines have maximum length **4**.

All claims are restricted to **low-budget micro-search in this six-pass LLVM subspace**. The paper will not claim general results for full LLVM phase ordering, long sequences, or module-level autotuning.

### Common compilation setup

All methods start from the same canonical bitcode:

`clang -O0 -Xclang -disable-llvm-passes -Xclang -disable-O0-optnone -emit-llvm`

Every evaluated candidate uses the same fixed warm-up prelude outside the search space:

`mem2reg,sroa,instcombine,simplifycfg,loop-simplify,lcssa`

The searched pass sequence is appended after this prelude. This keeps the six-pass space meaningful without hiding additional search freedom inside the prelude.

The paper reports two `-Oz` baselines:

- **standard `default<Oz>`** on canonical IR
- **matched-start `default<Oz>`** on canonical IR after the fixed warm-up prelude

The standard `default<Oz>` baseline is the main externally interpretable compiler baseline. The matched-start variant is a diagnostic control for the micro-search setup.

### Three evidence families to compare

The study compares three increasingly informed online signals under matched budgets.

#### 1. Probe-delta evidence

This baseline sees only measured objective changes from already evaluated candidates:

- instruction-count delta
- basic-block delta
- loop-count delta
- compile-time cost

It does not inspect compiler-emitted semantic evidence. This is the minimal nontrivial online-search baseline.

#### 2. Compilation-statistics evidence

This is the paper's restricted reproduction of the **core evidence representation used by CITROEN**. For each evaluated candidate, collect LLVM `-stats -stats-json` counters restricted to counters touched by the six searched passes and the fixed prelude. Candidate ranking then uses an online surrogate model over these statistics.

This comparison is important because CITROEN is the nearest published neighbor. The proposed study does **not** claim to reproduce full CITROEN, which also includes:

- a much larger search space
- runtime-based tuning
- multi-module dynamic budget allocation
- hundreds of search iterations

Instead, the paper reproduces CITROEN's central idea at the scale feasible here: **compiler-emitted statistics as the online evidence stream**.

#### 3. Normalized-remark evidence

This is the proposed evidence representation. For each evaluated candidate, collect `-pass-remarks`, `-pass-remarks-missed`, and `-pass-remarks-analysis`, then normalize raw remarks into symbolic records with:

- `consumer_pass`
- `outcome` in `{applied, missed, analysis}`
- `family`
- `blocker`
- `scope`
- optional stable `anchor` such as loop or callsite identifiers when present

The online state aggregates:

- active missed opportunities by `(family, blocker, scope)`
- applied transformations by family
- blockers resolved since the previous candidate
- cumulative compile-time cost
- coarse IR deltas only as fallback features

### Why remarks are meaningfully different from compilation statistics

This distinction must be explicit because otherwise the proposal looks like a minor variant of CITROEN.

#### Representation

- **Compilation statistics** are numeric counters keyed by pass or subsystem, such as numbers of vectorized loops or generated instructions. They are usually aggregate, dense, and counter-like.
- **Optimization remarks** are sparse event records attached to specific transformation attempts. They directly encode outcomes such as applied transformations, missed opportunities, and stated blockers like profitability checks, unsupported memory access, or noncanonical loops.

#### Collection and cost

- Both signals are compiler-emitted and cheaper than runtime measurement.
- Statistics are simpler to aggregate because they are already numeric counters.
- Remarks require parsing and normalization, but that parsing cost is small relative to even a single additional compile-and-measure iteration in the planned setup.
- Remarks may be sparser and less stable across passes, which is why the proposal includes an explicit precondition study and accepts a publishable negative result.

#### Decision logic

- CITROEN-style statistics are best suited to **learned continuous ranking** via an online surrogate model.
- Normalized remarks support a **symbolic decision rule** that reasons about blockers and their resolution, without fitting a high-dimensional model from very few samples.

So the actual research question is not "remarks vs other compiler outputs" in general. It is:

**When the budget is tiny, is sparse symbolic blocker information more useful than dense numeric counters or simple probe deltas?**

### Precondition study

Before running the main search comparison, the project measures whether the six-pass space emits enough evidence for the study to be meaningful.

For each benchmark and each of the six passes, run the warm-up prelude followed by that single pass with both remark collection and statistics collection enabled. Record:

- whether any remark is emitted
- raw remark count
- normalized-record count
- remark mix across `applied`, `missed`, and `analysis`
- whether any six-pass-related `-stats-json` counters are nonzero
- number of distinct statistics counters touched

Define reliability metrics per pass:

- **remark coverage**: fraction of benchmarks with at least one remark
- **remark density**: median normalized-record count on emitting benchmarks
- **statistics coverage**: fraction of benchmarks with at least one nonzero relevant counter

Passes are considered remark-reliable if remark coverage is at least **0.75** and remark density is at least **2**.

Fallback policy:

- if at least 4 passes are remark-reliable, run the full study
- if fewer than 4 are remark-reliable but statistics coverage remains strong, reframe the result as a negative evidence study showing that remarks are too sparse for this subspace while statistics remain usable
- if both signals are too sparse, report the micro-search as evidence-poor and stop after the precondition analysis plus lightweight IR-delta comparisons

This makes the project falsifiable and prevents overclaiming.

### Search methods

All search methods share the same pass vocabulary, maximum length, canonical start state, and total budget of **18 `opt` evaluations per benchmark**.

#### Random

- 18 candidate pipelines sampled uniformly from the six-pass, length-at-most-four space

#### Probe-Delta Search

- 6 single-pass probes
- 4 adaptive pair probes from the best first-pass candidates
- 8 remaining evaluations allocated by ranking next candidates only from measured probe deltas

#### Stats-GP Search (CITROEN-core reproduction)

- same 6 single-pass probes
- same 4 adaptive pair probes
- fit an online Gaussian-process regressor on statistics vectors gathered from the already evaluated candidates
- use expected improvement to select the final 8 candidates

This is not a full reproduction of CITROEN's large-budget system. It is an **analytical, budget-matched reproduction of its statistics-driven decision logic** in the restricted single-module setting.

#### Remark-State Search

- same 6 single-pass probes
- same 4 adaptive pair probes
- final 8 candidates selected by beam search over a fixed lexicographic ranking rule:

1. maximize newly resolved blocker count
2. break ties by newly applied transformations in families with prior misses
3. break ties by favorable IR delta
4. break ties by lower compile-time cost

Beam width is **3** by default.

No free score weights are tuned. This keeps the method interpretable and prevents the baseline study from turning into a hidden search-algorithm tuning exercise.

## Related Work

Ashouri, Killian, Cavazos, Palermo, and Silvano's survey frames compiler autotuning and phase ordering as a search-cost problem. It motivates lightweight tuning but predates the current wave of online-evidence methods.

Liu, Fang, Wang, Xie, Huang, and Wang's **Efficient compiler optimization by modeling passes dependence** (DCO) reduces search with dependence structure over passes. The present proposal differs by asking a smaller evidence question inside a much narrower LLVM subspace.

Pan, Wei, Xing, Wu, and Zhao's **Towards Efficient Compiler Auto-tuning: Leveraging Synergistic Search Spaces** (CGO 2025) is relevant because it uses recent structure-aware search-space reduction. Its message for this proposal is that careful search-space restriction is already competitive, so any new signal must beat simple structured baselines under matched budgets.

Zhao, Xia, and Wang's **Leveraging Compilation Statistics for Compiler Phase Ordering** (CITROEN, IPDPS 2025) is the nearest published neighbor and must be treated as such. CITROEN uses pass-related compilation statistics from LLVM `-stats -stats-json` as features for an online Gaussian-process model, then uses Bayesian optimization and dynamic budget allocation across modules. This proposal differs in three precise ways:

- it uses **optimization remarks** rather than statistics as the primary evidence stream
- it uses a **symbolic blocker-resolution rule** rather than a learned probabilistic surrogate
- it evaluates only **single-module, low-budget micro-search** rather than large-budget phase ordering

That difference is substantive only if remark-state search beats a matched statistics-based baseline in the restricted setting. Otherwise the paper becomes a negative result about the limited added value of remarks.

Ashouri et al.'s **Protean Compiler** is a broad infrastructure contribution for fine-grain LLVM phase ordering with large search spaces and feature-rich control. The proposed study is much smaller and baseline-oriented.

Pan, Huang, Dong, Xing, and Wu's **ECCO: Evidence-Driven Causal Reasoning for Compiler Optimization** is the strongest recent evidence-aware comparator conceptually. ECCO reasons over richer evidence with causal modeling and LLM-guided search. This proposal is explicitly not competing at that scope. Instead it tests whether a tiny, interpretable evidence representation is useful before such richer systems are invoked.

Positioning statement: the paper's publishable contribution is **a controlled positive-or-negative study of whether remark-derived blocker states add signal beyond probe deltas and compilation statistics in low-budget LLVM micro-search**.

## Experiments

### Benchmarks

Use **8 short-running CPU benchmarks**:

- 6 PolyBench/C kernels with loop structure
- 2 cBench kernels with nontrivial call structure when available

Inclusion rule:

- warm-up prelude plus one searched pass must finish in at most 15 seconds

To reduce cherry-picking:

- benchmark inclusion is fixed before any search results are inspected
- choose the first eligible kernels in alphabetical order within each source suite
- if fewer than 2 eligible cBench kernels remain, backfill from the next alphabetical eligible benchmark regardless of outcome

### Metrics

Primary search objective:

- post-pipeline LLVM IR instruction count

Mandatory reported endpoints:

- final binary size on all 8 benchmarks
- runtime on a fixed 4-program subset
- tuning wall-clock time
- final one-shot compile time of the selected pipeline

Instruction count is used only because it is cheap enough for the search budget. The paper will explicitly discuss the proxy gap to binary size and runtime.

### Main comparison

Methods:

- standard `default<Oz>`
- matched-start `default<Oz>`
- random search
- probe-delta search
- Stats-GP Search
- Remark-State Search

Common settings:

- six-pass vocabulary
- maximum sequence length of four
- 18 `opt` evaluations per benchmark
- same starting IR and warm-up prelude

### Analytical reproduction against a recent online-evidence method

Because full CITROEN is not feasible under the current budget, the paper includes a pre-registered **CITROEN-core reproduction** rather than claiming a full reimplementation. The reproduction preserves the components that matter for the present question:

- compiler-emitted statistics as evidence
- online surrogate modeling
- acquisition-driven candidate selection

It intentionally omits:

- multi-module budget allocation
- 76-pass search spaces
- 100 to 1000 runtime measurements
- runtime as the search-time objective

This makes the comparison honest and feasible. The outcome directly answers whether remarks help beyond the evidence stream used by the closest 2025 method in the same tiny-search regime.

### Normalization soundness check

Sample about **120 raw remarks** from the precondition study and early search runs. Freeze the normalization rules before inspecting method outcomes. Then:

- manually label a 40-remark subset with two annotators
- report Cohen's kappa for `family`
- report exact-match agreement for `blocker`
- report parser coverage and `other`-bucket rate on the full sample

Proceeding thresholds:

- `family` kappa at least **0.75**
- parser coverage at least **0.80**
- no single pass contributes more than half of all unmapped remarks

If these fail, the paper reports the failure directly and weakens the claim to a negative feasibility result.

### Sensitivity and ablation studies

Run the following on the fixed 4-program runtime subset:

- beam width `2`, `3`, and `4` for Remark-State Search
- Remark-State Search with and without IR-delta fallback features
- Remark-State Search using only applied remarks vs applied plus missed-analysis remarks
- Stats-GP using all observed counters vs only counters touched by the six searched passes

These studies test whether any advantage comes from the evidence representation itself rather than a brittle hyperparameter choice.

### Feasibility accounting

Main comparison cost:

- 8 benchmarks
- 4 search methods
- 18 evaluations per benchmark

That is **576 total `opt` evaluations** for the search methods, plus:

- 16 `default<Oz>` baseline runs
- 48 single-pass precondition runs
- binary-size measurement on all 8 benchmarks
- runtime measurement on 4 benchmarks
- small sensitivity and ablation runs on 4 benchmarks

With a 15-second screen for included benchmarks and only 2 CPU cores, this remains compatible with the overall 8-hour budget.

## Success Criteria

The paper is successful if it produces a clear answer, positive or negative, to the central question under the stated restricted scope.

Evidence confirming the hypothesis:

- at least 4 of the 6 passes are remark-reliable
- Remark-State Search beats Probe-Delta Search on mean IR instruction-count reduction
- Remark-State Search also beats Stats-GP Search by a practically meaningful margin, or ties it while being simpler and more interpretable
- the advantage is reflected in at least one real endpoint, binary size or runtime, on the fixed 4-program subset

Evidence refuting the hypothesis:

- remarks are too sparse or unstable to support a reliable state
- Stats-GP consistently matches or outperforms Remark-State Search
- any small remark advantage disappears on binary size and runtime

Either outcome is publishable only if the paper keeps the claim narrow: this is evidence about **restricted low-budget LLVM micro-search**, not broad phase-ordering generality.

## References

1. Amir H. Ashouri, William Killian, John Cavazos, Gianluca Palermo, and Cristina Silvano. "A Survey on Compiler Autotuning using Machine Learning." *ACM Computing Surveys*, 2018.
2. Jianfeng Liu, Jianbin Fang, Ting Wang, Jing Xie, Chun Huang, and Zheng Wang. "Efficient compiler optimization by modeling passes dependence." *CCF Transactions on High Performance Computing*, 2024.
3. Tamim Burgstaller, Damian Garber, Viet-Man Le, and Alexander Felfernig. "Optimization Space Learning: A Lightweight, Noniterative Technique for Compiler Autotuning." *SPLC 2024*, 2024.
4. Haolin Pan, Yuanyu Wei, Mingjie Xing, Yanjun Wu, and Chen Zhao. "Towards Efficient Compiler Auto-tuning: Leveraging Synergistic Search Spaces." *CGO 2025*, 2025.
5. Jiayu Zhao, Chunwei Xia, and Zheng Wang. "Leveraging Compilation Statistics for Compiler Phase Ordering." *IPDPS 2025*, 2025.
6. Amir H. Ashouri, Shayan Shirahmad Gale Bagi, Kavin Satheeskumar, Tejas Srikanth, Jonathan Zhao, Ibrahim Saidoun, Ziwen Wang, Bryan Chan, and Tomasz S. Czajkowski. "Protean Compiler: An Agile Framework to Drive Fine-grain Phase Ordering." *arXiv preprint arXiv:2602.06142*, 2026.
7. Haolin Pan, Lianghong Huang, Jinyuan Dong, Mingjie Xing, and Yanjun Wu. "ECCO: Evidence-Driven Causal Reasoning for Compiler Optimization." *arXiv preprint arXiv:2602.00087*, 2026.
