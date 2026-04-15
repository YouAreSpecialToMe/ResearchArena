# DebtAware: A Pilot-First Empirical Test of Mutation-Typed Rerun Suppression Beyond LLVM LastRunTracking

## Introduction

LLVM already has strong nearby solutions, so this project should not be presented as a new optimization framework. Jayatilaka et al. (ICPP Workshops 2021) studied ML-guided LLVM optimization skipping by predicting whether a scheduled optimization would modify IR. LLVM's October 10, 2024 RFC on avoiding reruns of transform passes that have just been run, together with the landed `LastRunTracking` direction, already gives a practical no-intervening-change heuristic for suppressing redundant reruns.

The remaining research question is narrower:

**For a very small set of repeatedly invoked LLVM cleanup passes, do mutation-typed intervening summaries add measurable value beyond `LastRunTracking` when deciding whether an already-scheduled rerun can be deferred to the next stock cleanup sweep point?**

This proposal is therefore a bounded empirical test of incremental value over an existing LLVM heuristic. The fixed `-O3` pipeline stays intact. The study does not search pass orders, synthesize pipelines, or claim a new scheduling substrate. Its main contribution is a careful negative-or-positive boundary test: once LLVM already knows whether any compatible intervening change happened, is there any practical benefit to also knowing what kind of change happened?

The key reframing is to target **deferred-rerun quality debt** rather than immediate IR modification. A rerun is only worth forcing now if deferring it to the next stock occurrence of the same pass causes unrecovered downstream divergence, quality loss, or repeated recovery work. That is a much smaller and more defensible claim than proposing a novel pass-skipping method.

## Proposed Approach

### Scope and contribution

DebtAware evaluates only the incremental decision value of mutation-typed change summaries after `LastRunTracking` has already answered the binary question "did any compatible intervening change occur?"

The scope is intentionally narrow to fit the 2-core, CPU-only, roughly 8-hour budget:

- fixed stock `-O3` pipeline
- primary target passes: `InstCombinePass` and `SimplifyCFGPass`
- optional `LCSSAPass` only if the pilot leaves more than 90 minutes of slack
- no neural models
- no pipeline search
- no full-program iterative compilation
- bounded single-rerun counterfactual replay only for sampled opportunities

### Baselines

The comparison is explicitly incremental:

1. **Stock `-O3`**
   Execute every scheduled rerun.
2. **`LastRunTracking` baseline**
   Skip a repeated target pass only when no compatible intervening IR-changing pass has executed since the last run of that same pass on that IR unit.
3. **Learned-change baseline**
   Use the same tiny interpretable model family and the same features as DebtAware, but predict only whether the rerun would change IR now.
4. **DebtAware**
   Predict whether suppressing the current rerun until the next stock sweep point creates downstream debt.
5. **Rule-guardrail fallback**
   Start from `LastRunTracking` and apply fixed mutation-based "never skip after X" rules if labeled data are too sparse for reliable fitting.

The main scientific question is the gap between (2) and (4). If no clear gap appears, the result is still useful because it bounds the practical value of going beyond `LastRunTracking`.

### Mandatory sweep point

For a target pass `P`, a **mandatory sweep point** for rerun opportunity `i` is the earliest later dynamic invocation of the same pass kind on the same IR unit kind that:

- appears in the unmodified stock `-O3` pipeline trace,
- is separated from `i` by at least one non-analysis transform pass on that IR unit kind, and
- would still be scheduled even if `i` were suppressed.

Operationally:

1. Dump the stock `-O3` textual pipeline and record static occurrence ordinals of target passes within each module, CGSCC, function, and loop pass-manager nest.
2. During traced compilation, map each dynamic target-pass invocation to its static occurrence ordinal.
3. Assign the sweep point as the next later matching occurrence on the same IR unit kind.
4. Mark the opportunity `unsweepable` if no such later occurrence exists.

Only sweepable repeated opportunities are eligible for debt labeling or learned-policy evaluation.

### Mutation-typed intervening summaries

Between two runs of the same target pass on the same IR unit, DebtAware accumulates cheap mutation summaries:

- `inst_count_bucket_delta`
- `bb_count_delta`
- `cfg_edge_delta`
- `branch_simplify_or_edge_split_seen`
- `phi_delta_sign`
- `loop_structure_changed`
- `constant_exposure_delta_sign`
- `memop_delta_sign`
- `callsite_delta_sign`
- `simplifycfg_option_compatibility_bit`

These are low-cost counters or booleans. They reset when the target pass executes. The design goal is not expressiveness for its own sake, but a minimal taxonomy that could plausibly separate harmless intervening edits from edits that make immediate rerun execution worthwhile.

### Debt label

For target pass `P` at opportunity `i`, define `debt(P, i) = 1` if suppressing only that rerun until its mandatory sweep point causes either:

- post-sweep IR hash divergence from stock `-O3`,
- final correctness failure,
- text size or total binary size drift greater than 0.1% relative to stock for that benchmark, or
- later runtime validation to show that this benchmark exceeds the preregistered runtime tolerance.

Otherwise assign `debt(P, i) = 0`.

This label is intentionally narrower than "would rerunning now change IR?" A rerun can be locally effectful yet still be safely deferrable if the next stock sweep fully recovers the effect.

### Labeling protocol

The study uses bounded single-rerun counterfactual replay:

1. Run traced stock `-O3` on training benchmarks only.
2. Log every repeated target-pass opportunity, whether `LastRunTracking` would allow skipping, its mutation summary, and its sweep-point mapping.
3. Sample at most **12 repeated opportunities per training benchmark per target pass**, stratified by whether `LastRunTracking` says skip or run.
4. For each sampled opportunity, rerun the compilation once with only that immediate rerun suppressed, force execution again at the mandatory sweep point, and continue to pipeline end.
5. Assign `debt` from the counterfactual outcome.
6. Also save `changed_ir_now` from the stock run for the learned-change baseline.

The cap is conservative because stable labels matter more than broad but noisy coverage under this budget.

### Policy class and safeguards

Use only tiny interpretable per-pass policies:

- depth-2 decision tree with `min_samples_leaf=6`
- sparse decision list as the secondary model family

At inference time, the pass order remains identical to stock `-O3`; the policy only decides whether a repeated rerun executes immediately or is deferred to its precomputed sweep point.

Two safeguards are always active:

- force execution at the next mandatory sweep point already present in stock `-O3`
- force immediate execution after 2 consecutive suppressions of the same pass on the same IR unit

### Low-label fallback

Sparse labels are a realistic outcome and are pre-registered rather than treated as an afterthought.

For every fold and target pass, report:

- `num_candidates`
- `num_sampled`
- `num_unsweepable`
- `num_labeled`
- `num_debt_1`
- `num_debt_0`

If a fold/pass has fewer than 24 labeled opportunities total or fewer than 6 examples in either class:

- do not train `learned_change` or `mutation_debt` for that fold/pass,
- mark the learned result `insufficient_labels`,
- evaluate only `LastRunTracking` and the fixed rule-guardrail fallback,
- report the outcome as a descriptive boundary study, not as evidence of learned-policy success.

## Related Work

### What was checked online and why

On **March 22, 2026**, the novelty check explicitly covered:

- recent LLVM/phase-ordering work on arXiv and public publication pages for 2025-2026, to see whether a newer paper already studied typed intervening mutations or rerun deferral inside stock LLVM pipelines;
- DBLP, Semantic Scholar, and paper landing pages for the closest older work on ML pass skipping and pass-dependence modeling;
- LLVM Discourse and public LLVM sources for the status and intent of the 2024 rerun-suppression RFC and `LastRunTracking`.

This search found active nearby work on pass skipping, pass-order prediction, compilation-statistics-guided phase ordering, and LLVM-integrated autotuning, but did **not** reveal an exact prior study asking whether typed intervening mutations add value **beyond** `LastRunTracking` for deferring already-scheduled cleanup reruns to the next stock sweep point. That does **not** establish strong novelty; it only supports a narrow empirical gap worth testing.

### Jayatilaka et al. 2021

Jayatilaka et al., "Towards Compile-Time-Reducing Compiler Optimization Selection via Machine Learning" (ICPP Workshops 2021), is the closest published pass-skipping paper. It already studies LLVM optimization skipping by predicting whether scheduled optimizations will modify IR. That substantially lowers DebtAware's novelty ceiling.

DebtAware is not subsumed completely, because it asks a different question:

- Jayatilaka et al. target immediate IR modification.
- DebtAware targets deferred-rerun debt under a fixed later recovery point.
- Jayatilaka et al. predate LLVM's later `LastRunTracking` heuristic, so they do not answer whether richer mutation typing adds value **after** the binary no-intervening-change rule is already available.

### LLVM RFC and LastRunTracking

The closest systems baseline is the October 10, 2024 LLVM RFC "[RFC][Pipeline] Avoid running transform passes that have just been run" and the landed `LastRunTracking` implementation direction. That work already suppresses reruns when no compatible intervening change occurred.

This means DebtAware cannot claim novelty for rerun suppression itself. The only defensible question left is whether the **type** of compatible intervening mutation changes the decision in a useful way. If DebtAware does not beat `LastRunTracking`, the right conclusion is negative: the richer bookkeeping was not justified in this bounded setting.

### Pass-ordering and dependence work

Broader pass-ordering work also narrows the contribution. Liu et al. (DCO, 2024), Liang et al. (ICML 2023), Zhao et al. (IPDPS 2025), and recent LLVM-centered systems such as Protean (2026 preprint) and Synergy-Guided nested pipeline tuning (2025 preprint) all study ways to model pass interactions, search the optimization space, or guide phase-ordering decisions.

Those papers do **not** subsume DebtAware's exact question, because they operate at a different level:

- they search or predict pass sequences or subsequences,
- they target broader performance optimization rather than a single rerun-deferral decision inside stock `-O3`,
- they generally optimize over runtime-measured or search-based objectives, not over whether an already-scheduled cleanup rerun can wait until its next stock occurrence.

Their main value here is boundary-setting: they show the surrounding space is crowded, so the paper should be judged as a modest empirical extension study, not as a new autotuning method.

## Experiments

### Research questions

1. Does mutation-typed bookkeeping reduce compile time relative to stock `-O3`?
2. Does it improve on `LastRunTracking`, or does `LastRunTracking` already capture nearly all practical benefit?
3. If gains exist, do they come from reducing debt events rather than from aggressive skipping?
4. Are the labels dense enough to support even a tiny interpretable policy?

### Benchmark set

Use a small cross-suite benchmark pool:

- `CTMark`: 4 programs
- `cBench`: 4 programs
- `PolyBench/C`: 4 programs

Total planned pool: 12 programs.

If the pilot projects the full run beyond budget, reduce immediately to 3 programs per suite, 9 total, while keeping all 3 suites.

### Pilot-first protocol

Feasibility is the main risk, so the study is staged.

**Stage 0: go/no-go pilot**

- select **1 representative benchmark from each suite**, 3 pilot programs total;
- compile each pilot benchmark 3 times with stock `-O3` and 3 times with tracing enabled;
- measure repeated-opportunity counts, traced compile-time overhead, and single-replay median time;
- project total wall time for the full study on 2 CPU lanes only.

Proceed to the full 12-program plan only if all of the following hold:

- traced compile-time overhead is at most 15%,
- projected total wall time is at most 8 hours,
- at least one primary pass has at least 24 candidate repeated opportunities in the pilot.

Otherwise reduce immediately to the 9-program plan and pre-register the possibility that the final paper becomes a rule-based boundary study.

**Stage 1: smallest full evaluation first**

After the pilot passes, execute the first held-out evaluation with:

- `InstCombinePass` first,
- seed `11` first,
- the reduced benchmark count if the pilot demanded it.

Expand to `SimplifyCFGPass` and seeds `{17, 29}` only if the pilot-based time projection still leaves adequate slack after the first stage. If not, report the smaller evaluation as the primary result rather than overcommitting.

This staged execution makes the paper feasible even if the optimistic upper scope proves too expensive on March 22, 2026 hardware assumptions.

### Training and evaluation protocol

Use leave-one-suite-out evaluation:

- train on two suites and test on the held-out suite,
- repeat for all three held-out choices.

Planned seeds are `{11, 17, 29}` for model randomness, labeled-opportunity sampling order, and threshold tie-breaking. Deterministic baselines use the same benchmark order.

Per fold:

1. trace stock `-O3` on training benchmarks only;
2. sample bounded replay opportunities;
3. report label cardinalities before fitting;
4. fit the interpretable model only if minimum label thresholds are met;
5. evaluate on the held-out suite against all baselines.

### Metrics

Primary systems metrics:

- end-to-end compile wall time
- number of target-pass opportunities, immediate executions, and suppressions
- bookkeeping overhead in milliseconds and as a fraction of compile time

Debt and recovery metrics:

- debt-event rate over deferred opportunities
- recovery-event rate at mandatory sweep points
- false-safe rate
- conservative-miss rate
- post-sweep and final IR-hash agreement with stock `-O3`

Artifact guardrails:

- correctness pass rate
- text-section size delta
- total binary size delta
- runtime delta on a small runnable subset

Prediction metrics for learned policies are secondary:

- precision
- recall
- `F0.5`
- balanced accuracy

### Runtime validation

Runtime checks are guardrails, not the main evidence source.

Choose up to 6 runnable held-out benchmarks, at least 2 from different suites if available. Pin execution to one CPU core, warm up once, then measure **10 runs per binary** for stock `-O3`, `LastRunTracking`, and DebtAware. Report the median runtime per benchmark and paired ratios to stock.

The study will not claim broad runtime wins. The runtime checks exist only to rule out harmful regressions while the main evidence remains compile time, suppression behavior, and debt statistics.

## Success Criteria

### Positive main result

DebtAware supports the hypothesis if, on held-out suites:

- compile time improves by at least **2.0%** versus stock `-O3`,
- there are no correctness failures,
- geometric-mean runtime and code-size change stay within ±1.0% of stock,
- no individual runtime benchmark is worse than 3%.

### Positive incremental result beyond LLVM's existing heuristic

DebtAware shows meaningful value beyond `LastRunTracking` if either:

- it beats `LastRunTracking` by at least **0.5 percentage points** of compile-time reduction, or
- it matches compile-time reduction while lowering debt-event rate or recovery-event rate by at least **20%**.

### Boundary-study result

If DebtAware and `LastRunTracking` overlap within 95% paired bootstrap confidence intervals, conclude that typed intervening mutations did not show clear added value in this bounded setting.

### Negative result

If DebtAware beats the learned-change baseline but not `LastRunTracking`, conclude that the debt label did not deliver meaningful incremental value over the existing no-intervening-change rule.

### Feasibility failure

Report the study as infeasible under the stated constraints if any of the following occurs:

- bookkeeping overhead exceeds 10% of compile time,
- debt-event rate exceeds 2% of deferred opportunities,
- the LLVM build plus experiments exceed the 8-hour budget.

## References

1. Amir H. Ashouri, William Killian, John Cavazos, Gianluca Palermo, and Cristina Silvano. "A Survey on Compiler Autotuning using Machine Learning." *CoRR*, abs/1801.04405, 2018.
2. Erik Hellsten, Artur Souza, Johannes Lenfers, Rubens Lacouture, Olivia Hsu, Adel Ejjeh, Fredrik Kjolstad, Michel Steuwer, Kunle Olukotun, and Luigi Nardi. "BaCO: A Fast and Portable Bayesian Compiler Optimization Framework." In *Proceedings of ASPLOS 2023*, 2023.
3. Tarindu Jayatilaka, Hideto Ueno, Giorgis Georgakoudis, Eunjung Park, and Johannes Doerfert. "Towards Compile-Time-Reducing Compiler Optimization Selection via Machine Learning." In *Proceedings of the 2021 International Conference on Parallel Processing Workshops*, 2021.
4. Youwei Liang, Kevin Stone, Ali Shameli, Chris Cummins, Mostafa Elhoushi, Jiadong Guo, Benoit Steiner, Xiaomeng Yang, Pengtao Xie, Hugh Leather, and Yuandong Tian. "Learning Compiler Pass Orders using Coreset and Normalized Value Prediction." In *Proceedings of ICML 2023*, 2023.
5. Jianfeng Liu, Jianbin Fang, Ting Wang, Jing Xie, Chun Huang, and Zheng Wang. "Efficient compiler optimization by modeling passes dependence." *CCF Transactions on High Performance Computing* 6 (2024): 588-607.
6. Jiayu Zhao, Chunwei Xia, and Zheng Wang. "Leveraging Compilation Statistics for Compiler Phase Ordering." In *Proceedings of IPDPS 2025*, 2025.
7. Amir H. Ashouri, Shayan Shirahmad Gale Bagi, Kavin Satheeskumar, Tejas Srikanth, Jonathan Zhao, Ibrahim Saidoun, Ziwen Wang, Bryan Chan, and Tomasz S. Czajkowski. "Protean Compiler: An Agile Framework to Drive Fine-grain Phase Ordering." arXiv:2602.06142, 2026.
8. Haolin Pan, Jinyuan Dong, Mingjie Xing, and Yanjun Wu. "Synergy-Guided Compiler Auto-Tuning of Nested LLVM Pass Pipelines." arXiv:2510.13184, 2025.
9. dtcxzyw. "[RFC][Pipeline] Avoid running transform passes that have just been run." LLVM Discourse RFC, October 10, 2024.
