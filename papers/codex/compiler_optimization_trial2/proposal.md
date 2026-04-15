# Typed Invariant Evidence for Incremental LLVM Cleanup Scheduling

> Execution note for this Stage 2 attempt: the planned pinned LLVM source revision and in-tree `LastRunTrackingAnalysis` extension are not available in the workspace. The experiments in this directory therefore evaluate a narrower proxy scheduler on the installed LLVM 18.1.3 command-line toolchain and locally generated workloads. Any conclusions from `results.json` should be read as preliminary proxy evidence, not validation of the original in-LLVM design.

## Introduction

Repeated cleanup and canonicalization are still a meaningful source of LLVM middle-end compile-time cost. Passes such as `InstCombine`, `SimplifyCFG`, `LoopSimplify`, and `LCSSA` appear multiple times because upstream transforms can invalidate structural assumptions that later optimizations require. Many of those reruns are useful, but some are clearly redundant: an intervening transform may have changed IR without creating relevant cleanup work, or any damage may be confined to one function.

LLVM already established the main baseline. Yingwei Zheng's October 10, 2024 RFC and the merged November 6, 2024 `LastRunTrackingAnalysis` implementation showed that redundant reruns of idempotent cleanup-style passes are common and that coarse last-run evidence can produce compile-time wins in the new pass manager. That prior art substantially narrows the novelty space. This proposal is therefore framed explicitly as an incremental LLVM scheduling extension, not as a new optimization framework.

The original paper question was:

**Can typed, invariant-aware evidence layered on top of `LastRunTrackingAnalysis` improve cleanup scheduling beyond coarse temporal tracking and coarse locality heuristics on a small LLVM pipeline slice?**

For this execution attempt, the narrower empirical question is:

**Can a proxy typed scheduler over a fixed LLVM 18 pass slice improve compile time relative to coarse proxy baselines on generated workloads, and if not, where does the evidence fail?**

The hypothesis is modest but testable. A producer pass can often say more than "some IR changed": it can conservatively say which cleanup-facing invariant may have become dirty, and whether the possible damage is confined to a small region. If that evidence is accurate often enough, the pass manager can make better decisions among skip, function-local rerun, and global fallback. If the evidence is too weak or too expensive, the study will quantify that negative result directly.

## Proposed Approach

### Overview

The mechanism extends `LastRunTrackingAnalysis`; it does not replace it. `LastRunTrackingAnalysis` provides temporal evidence about whether a compatible pass has run since the last successful cleanup execution and whether relevant IR changed afterward. The proposed extension adds two missing dimensions:

- invariant type: what cleanup-relevant property may have been invalidated
- locality: where that possible invalidation is confined

The prototype is intentionally narrow.

Producer passes:

- `SROA`
- `GVN`
- `LICM`
- `LoopRotate`

Primary cleanup consumers:

- `InstCombine`
- `SimplifyCFG`

Optional gated extension:

- `LoopSimplify`
- `LCSSA`

The central claim is the function-local result for `InstCombine` and `SimplifyCFG`. Loop-local repair is explicitly secondary evidence and may be dropped without weakening the main paper.

### Method Details

#### 1. Typed certificate lattice

Each producer emits a conservative summary over a fixed cleanup-facing invariant vocabulary:

- `CFGCanonical`
- `LoopSimplifyForm`
- `LCSSAForm`
- `InstCombineOpportunity`
- `DomTreeFresh`
- `LoopInfoFresh`

Each invariant takes one of five values:

- `preserved`
- `dirty(function-set)`
- `dirty(loop-set)`
- `global_dirty`
- `unknown`

`unknown` is a first-class safe outcome. Any `unknown` on a required invariant forces baseline behavior for that scheduling decision. This design keeps the implementation sound under incomplete producer coverage and makes low-coverage failure scientifically interpretable rather than hidden.

#### 2. Scheduler policy

Each cleanup pass declares the invariants it consumes:

- `InstCombine`: `InstCombineOpportunity`
- `SimplifyCFG`: `CFGCanonical`
- `LoopSimplify`: `LoopSimplifyForm`, `LoopInfoFresh`, `DomTreeFresh`
- `LCSSA`: `LCSSAForm`, `LoopSimplifyForm`, `LoopInfoFresh`, `DomTreeFresh`

Given `LastRunTrackingAnalysis` state plus the current certificates, the scheduler chooses exactly one action:

- `skip_global` when required invariants are `preserved`
- `run_function_local` when all possible dirtiness is confined to a known function set
- `run_loop_local` only when the optional loop-local wrapper is enabled and the certificate is loop-scoped
- `run_global` on any `global_dirty` or `unknown`

This keeps the contribution squarely in LLVM scheduling: fixed pass sequence, conservative evidence, and explicit fallback to stock behavior.

#### 3. Producer transfer functions

The initial transfer-function table is small and hand-audited.

- `SROA`: may create instruction simplification opportunities in touched functions while usually preserving CFG and loop form
- `GVN`: may create `InstCombine` opportunities and some local CFG cleanup opportunities in touched functions while usually preserving loop structure
- `LICM`: may disturb loop-form invariants or LCSSA in the transformed loop, but often only locally
- `LoopRotate`: may dirty loop canonicality in the rotated loop while leaving unrelated functions untouched

Any ambiguous case emits `unknown`. The prototype optimizes for defensible precision over coverage.

#### 4. Why this is better than the obvious baselines

Compared with `LastRunTrackingAnalysis` alone:

- last-run evidence is temporal but not semantic
- an intervening pass can destroy a profitable skip even when it preserved the cleanup-facing invariant that matters
- last-run evidence cannot express function-local rerun instead of global rerun

Compared with a dirty-function heuristic:

- dirty-function is spatial but not semantic
- it reruns cleanup whenever a function was touched, even if the change preserved CFG canonicality or created no combine opportunities
- it cannot distinguish instruction cleanup from loop-form breakage or stale analyses

The proposed contribution is therefore not "metadata helps" in the abstract. It is the narrower claim that typed invariant evidence plus locality can outperform both coarse temporal tracking and coarse spatial tracking on this bounded slice.

#### 5. Publishable fallback hierarchy

The design includes explicit fallback paths so the project remains feasible under the 8-hour budget.

Fallback A: high `unknown` rate

- If the four-pass producer set yields poor coverage, the study narrows to the two producers with the best pilot coverage, expected to be `SROA` and `GVN`.
- The paper then reports a producer-coverage ablation rather than claiming broad applicability.

Fallback B: local wrappers are too costly

- If function-local wrappers around `InstCombine` or `SimplifyCFG` add too much analysis-update or orchestration overhead, the study drops localized reruns and evaluates typed skip-vs-global scheduling only.
- This still tests the core incremental question: whether typed evidence beats `LastRunTrackingAnalysis` alone for deciding when a cleanup pass can be skipped.

Fallback C: loop-local path does not clear the gate

- `LoopSimplify` and `LCSSA` remain an optional extension and are excluded from the main claim if escalation or wrapper cost is too high.

These fallback paths are part of the experimental design, not post-hoc rescue.

## Related Work

### Archival research most relevant to novelty

**Eric Fritz, _Waddle - Always-Canonical Intermediate Representation_ (2018 dissertation).**  
This is the closest conceptual precursor. Waddle advocates maintaining canonical properties continuously through local repair so later transforms can assume them without reconstruction passes. The proposed work is weaker and more incremental: it does not redesign transformation contracts or insist on always-canonical IR. It asks whether LLVM can extract some of that value through scheduler-facing evidence layered on top of its current pass manager.

**Eric Fritz, "Maintaining Canonical Form After Edge Deletion" (ICOOOLPS 2018).**  
This paper makes the comparison sharper by giving a concrete local canonical-form maintenance algorithm for a specific CFG update. It shows that local repair can be real rather than aspirational, but it also highlights the novelty boundary: the present proposal is not inventing local canonical maintenance. Its main question is whether a lighter scheduler extension can exploit typed evidence to avoid or localize later cleanup in stock LLVM.

**Jianfeng Liu et al., "Efficient compiler optimization by modeling passes dependence" (2024).**  
This line of work models pass dependence to improve optimization sequences. The present proposal keeps the sequence fixed and optimizes only cleanup scheduling inside that sequence. The scope is narrower, more systems-oriented, and realistic under a CPU-only budget.

**Florian Huemer et al., "Taking a Closer Look: An Outlier-Driven Approach to Compilation-Time Optimization" (ECOOP 2024).**  
This is less about the mechanism itself and more about evaluation methodology. It argues that compilation-time work should be attributed at the level of specific optimizations and outliers rather than only whole-program totals. That directly motivates the proposed per-pass, per-certificate, and per-fallback accounting.

### Engineering context and direct LLVM baseline

**Yingwei Zheng's 2024 LLVM RFC and PR #112092.**  
This is the direct engineering baseline. It introduced `LastRunTrackingAnalysis` to avoid rerunning idempotent transform passes that already converged and saw no relevant intervening change. The proposed work is explicitly a small extension study on top of that mechanism.

**LLVM new pass manager and local maintenance infrastructure.**  
`PreservedAnalyses`, `LoopStandardAnalysisResults`, `DomTreeUpdater`, `SSAUpdater`, and loop/LCSSA utilities explain why locality-aware scheduling might be implementable. They are engineering enablers, not the paper's scholarly novelty claim.

Separating archival research from LLVM engineering context is important because the paper's novelty rests on how it sits between those two bodies of prior art: more incremental than Waddle, but more typed and locality-aware than `LastRunTrackingAnalysis`.

## Experiments

### Experimental Question

On a bounded LLVM middle-end slice, does typed invariant-aware scheduling reduce compile time more than:

1. stock scheduling
2. `LastRunTrackingAnalysis` alone
3. a dirty-function rerun heuristic
4. a simple throttling baseline

### Prototype Scope

The evaluated slice is deliberately small:

- producer subset: `SROA`, `GVN`, `LICM`, `LoopRotate`
- required cleanup subset: `InstCombine`, `SimplifyCFG`
- optional extension: `LoopSimplify`, `LCSSA`
- pass manager: LLVM new pass manager only

Representative bounded slice:

`sroa -> instcombine -> simplifycfg -> gvn -> instcombine -> simplifycfg -> licm -> loop-rotate -> instcombine -> simplifycfg`

Optional extension:

`-> loop-simplify -> lcssa`

This limited slice is a feature, not a weakness. It isolates a concrete scheduling question and keeps implementation, verification, and benchmarking within a 2-core, 128 GB RAM, roughly 8-hour total budget.

### Benchmarks

To stay within budget while still producing meaningful systems evidence:

- 10-12 `cBench` programs
- 8-10 small `llvm-test-suite` programs
- 20-30 reduced `.ll` regression tests focused on CFG cleanup, loop canonicality, and LCSSA behavior

The reduced tests are essential because they let the paper measure certificate behavior and failure modes on targeted cases without paying full benchmark cost.

### Baselines

1. Stock bounded pipeline slice.
2. `LastRunTrackingAnalysis` only.
3. Dirty-function heuristic:
   rerun cleanup on any function touched since the last cleanup point.
4. Cleanup throttling:
   rerun selected cleanup passes only at every second eligible insertion point.

The dirty-function baseline is mandatory because it tests whether invariant typing adds value beyond cheap locality alone.

### Metrics

Primary metrics:

- optimizer wall time on the bounded slice
- cumulative cleanup-pass wall time
- number of global cleanup reruns avoided

Evidence-quality metrics:

- certificate coverage
- `unknown` rate
- fallback-to-global rate
- function-local rerun rate
- loop-local rerun rate
- loop-local escalation rate

Overhead metrics:

- certificate bookkeeping cost
- wrapper setup and analysis-update cost
- false-positive local rerun count, where a localized rerun executes but makes no IR change

Correctness and output metrics:

- LLVM verifier failures
- differential mismatches versus stock scheduling on identical input bitcode
- benchmark correctness failures
- executable runtime and binary size deltas

### Staged execution plan under the time budget

The budget is tight, so the study is staged:

Stage 1: micro and reduced-test pilots

- measure raw wrapper overhead
- measure certificate coverage and `unknown` rate
- decide whether all four producers remain in scope

Stage 2: benchmark comparison for stock, last-run, dirty-function, and typed scheduling

- use a fixed small repetition count
- focus on pass-time attribution, not broad benchmark breadth

Stage 3: optional loop-local evaluation only if Stage 1 and Stage 2 indicate enough margin

If Stage 1 shows poor coverage or high wrapper cost, the study narrows immediately using the fallback hierarchy above rather than exhausting the full budget on a weak configuration.

### Correctness Methodology

The mechanism is conservative by construction and publishes negative evidence if conservatism forces frequent fallback.

Checks:

1. LLVM verifier after each certificate-controlled localized action and at each fallback boundary
2. differential testing against the same bounded slice in stock mode
3. benchmark execution checks plus reduced regression tests
4. optional translation-validation-style checks on a small reduced subset when practical

Any mismatch forces immediate global fallback and is logged as a certificate-path failure.

### Loop-Local Gate

The loop-local path is included in the paper only if it clears all of the following:

- loop-local coverage above 40%
- escalation-to-global below 50%
- positive net cleanup-time savings after wrapper overhead
- no unresolved verifier or differential failures

If the gate fails, the paper reports loop-local scheduling as negative or partial evidence and centers the contribution on function-local cleanup scheduling.

### Expected Results

The expected outcome is intentionally modest:

- 4-8% optimizer-time reduction on the bounded slice relative to stock
- clear improvement over `LastRunTrackingAnalysis` on cleanup-pass time, even if whole-slice speedup is smaller
- measurable improvement over dirty-function on either optimizer time or unnecessary local reruns
- runtime and binary size changes within 1% on average

The most important negative result would also be publishable: if typed evidence rarely improves on last-run or dirty-function once real `unknown` and wrapper costs are counted, that would quantify the limit of incremental scheduling extensions in LLVM.

## Success Criteria

### Evidence Supporting the Hypothesis

- Function-local typed evidence improves optimizer wall time by at least 4% over stock on the bounded slice.
- Typed evidence beats `LastRunTrackingAnalysis` alone by a statistically clear margin on cleanup-pass time or whole-slice optimizer time.
- Typed evidence beats dirty-function by reducing unnecessary localized reruns, fallback waste, or optimizer time.
- Certificate coverage is high enough to matter, for example above 60% on the retained producer subset, with `unknown` reported explicitly.
- No unresolved verifier failures, differential mismatches, or regression-test failures remain.

### Evidence Refuting the Hypothesis

- Certificate bookkeeping or wrapper overhead erases the savings.
- `unknown` dominates, collapsing decisions back to stock behavior most of the time.
- Dirty-function matches the gains closely, showing typed invariants are not paying for their added complexity.
- `LastRunTrackingAnalysis` already captures nearly all profitable opportunities on this slice.
- Loop-local scheduling escalates too often or costs too much, in which case it is removed from the main claim rather than used to rescue it.

## References

### Archival References

1. Eric Fritz. *Waddle - Always-Canonical Intermediate Representation*. Doctoral Dissertation, University of Wisconsin-Milwaukee, 2018. URL: https://www.eric-fritz.com/assets/papers/Fritz%20-%20Dissertation.pdf
2. Eric Fritz. "Maintaining Canonical Form After Edge Deletion." ICOOOLPS 2018. URL: https://www.eric-fritz.com/assets/papers/Fritz%20-%20Maintaining%20Canonical%20Form%20After%20Edge%20Deletion.pdf
3. Florian Huemer, David Leopoldseder, Aleksandar Prokopec, Raphael Mosaner, and Hanspeter Mössenböck. "Taking a Closer Look: An Outlier-Driven Approach to Compilation-Time Optimization." ECOOP 2024, LIPIcs 313, Article 20, 2024. URL: https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECOOP.2024.20
4. Jianfeng Liu, Jianbin Fang, Ting Wang, Jing Xie, Chun Huang, and Zheng Wang. "Efficient compiler optimization by modeling passes dependence." *CCF Transactions on High Performance Computing*, 6(6):588-607, 2024. URL: https://doi.org/10.1007/s42514-024-00197-9

### Engineering Context

5. Yingwei Zheng. "[RFC][Pipeline] Avoid running transform passes that have just been run." LLVM Discourse, October 10, 2024. URL: https://discourse.llvm.org/t/rfc-pipeline-avoid-running-transform-passes-that-have-just-been-run/82467
6. Yingwei Zheng. "[Analysis] Avoid running transform passes that have just been run." LLVM Pull Request #112092, opened October 12, 2024, merged November 6, 2024. URL: https://github.com/llvm/llvm-project/pull/112092
7. LLVM Project. "Using the New Pass Manager." LLVM documentation. URL: https://llvm.org/docs/NewPassManager.html
8. LLVM Project. `LoopSimplify.h` source documentation. URL: https://llvm.org/doxygen/LoopSimplify_8h_source.html
9. LLVM Project. `LCSSA.cpp` source documentation. URL: https://llvm.org/doxygen/LCSSA_8cpp_source.html
10. LLVM Project. `DomTreeUpdater.h` source documentation. URL: https://llvm.org/doxygen/DomTreeUpdater_8h_source.html
11. LLVM Project. `SSAUpdater.h` source documentation. URL: https://llvm.org/doxygen/SSAUpdater_8h_source.html
