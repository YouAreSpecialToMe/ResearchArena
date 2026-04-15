# How Far Does Probe-Only Recalibration Transfer in Bayesian Quadrature?

## Introduction

Bayesian quadrature (BQ) is attractive because it returns both a point estimate of an integral and a posterior variance. In practice, the point estimate is often usable even when the variance is not: kernel misspecification, empirical-Bayes hyperparameter fitting, and localized roughness can make nominal 90% and 95% BQ intervals substantially overconfident. This is especially problematic in the small-data regime where the interval, not just the mean, is the main output.

This proposal studies a deliberately narrow question: **can external recalibration of BQ uncertainty transfer from a pre-registered bank of analytic probe integrands to unseen target integrands, without using target labels for tuning?** The paper is not positioned as a new quadrature rule. It is a careful empirical and algorithmic study of **post-hoc uncertainty correction** attached to an existing BQ backend.

The key revision relative to the earlier draft is statistical. The paper no longer claims exact split-conformal validity from a tiny calibration set. Instead, it uses a substantially larger pooled calibration sample built from many more probes and repeated probe instances across budgets and design seeds, and it presents the recalibration rules as **heuristic but pre-registered external inflation rules**. The scientific contribution is therefore a transfer study:

- when simple probe-only recalibration works;
- when it fails because the probe bank is poorly aligned with the target family;
- how it compares with stronger internal alternatives such as Bayes-Sard cubature and kernel marginalization.

The central hypothesis is modest and testable: **simple pooled recalibration rules, chosen using probe-only cross-validation, can recover a meaningful fraction of BQ undercoverage under misspecification when the probe bank spans the target roughness regime, but their gains degrade under explicit probe-bank mismatch.**

## Proposed Approach

### Problem Setup

For a target integrand \(f\), a base GP-BQ solver returns
\[
I(f)\mid f(X_n) \approx \mathcal{N}(\mu_f, s_f^2).
\]
The proposed study leaves \(\mu_f\) unchanged and recalibrates only the variance:
\[
I(f)\mid f(X_n) \leadsto \mathcal{N}(\mu_f, \tau_\alpha(r_f)^2 s_f^2),
\]
where \(r_f\) is a scalar roughness diagnostic and \(\tau_\alpha(\cdot)\ge 1\) is learned entirely from probes. The paper compares several external recalibration rules rather than introducing a single supposedly optimal method.

### Pre-Registered Probe Bank

Each measure family gets a fixed bank of **72 analytic probes** with known exact integrals:

- 24 polynomial and low-order orthogonal-basis probes;
- 24 oscillatory probes spanning a frequency ladder;
- 24 localized probes spanning width and location ladders, including a small nonsmooth subset.

The bank is frozen before any target evaluation. It is split once into:

- 48 calibration probes;
- 24 held-out test probes.

Each probe is evaluated under:

- 2 node budgets: \(n\in\{16,32\}\);
- 3 shared design seeds;
- the same base solver family used for targets.

This yields
\[
48 \times 2 \times 3 = 288
\]
calibration probe instances per measure and
\[
24 \times 2 \times 3 = 144
\]
held-out probe instances per measure. The study treats these as a pooled calibration sample with grouped dependence by probe identity. That scale is large enough for stable empirical calibration curves and leave-one-probe-out selection, while still fitting the CPU budget.

### Diagnostics and Probe Scores

For each probe or target, define the standardized BQ error
\[
z = \frac{|I-\mu|}{\max(s,\varepsilon)}.
\]

The primary diagnostic is a scalar roughness score
\[
r(f)=\text{standardized finite-difference variation on the design}.
\]
Two auxiliary diagnostics are logged for analysis but not required by the simplest recalibrators:

- posterior condition number proxy;
- nearest-neighbor fill distance proxy.

The roughness score is used because the filtering-side calibration papers and the earlier draft both point to local irregularity as a main source of undercoverage. The paper still keeps the diagnostic class intentionally simple so the study is reproducible and statistically stable.

### Recalibration Rules

The main experiment compares three external recalibration rules, all trained on the pooled calibration probes and all selected without target labels.

1. **Global variance scaling**
   Estimate a single inflation factor \(\tau_\alpha\) per nominal level by matching the empirical upper quantile of \(z\) on calibration probes. This is the simplest statistically stable baseline and the main fallback if all local rules fail.

2. **Monotone roughness inflation**
   Fit a nondecreasing isotonic map
   \[
   r \mapsto \tau_\alpha(r)
   \]
   on pooled calibration probe instances, using grouped leave-one-probe-out cross-validation to tune clipping and regularization. This rule tests whether roughness carries transferable calibration information without relying on tiny bins.

3. **Piecewise-constant monotone histogram**
   Fit a small monotone step function in roughness with at most 4 bins, selected by grouped cross-validation. This is a more interpretable local rule and a robustness check against isotonic overfitting.

All three rules are forced to satisfy:

- \(\tau_\alpha(r)\ge 1\);
- clipping to a pre-registered upper bound to avoid pathological widening;
- fallback to the global rule when a target lies outside the roughness support of calibration probes.

The paper makes no exact coverage theorem claim for these rules. They are pre-registered heuristics evaluated honestly under held-out probes and held-out targets.

### Probe-Only Model Selection

Rule selection is done in two stages:

1. grouped leave-one-probe-out cross-validation on the 48 calibration probes chooses among global, isotonic, and stepwise rules;
2. the selected rule is refit on all 48 calibration probes and then evaluated on the 24 held-out probes and all targets.

Selection metrics are:

- absolute coverage gap at 90% and 95%;
- weighted interval score;
- width inflation relative to uncalibrated BQ.

No target-family labels or target integrals are used in model selection.

### Probe-Bank Sensitivity as a Main Analysis

Probe-bank sensitivity is not an ablation at the end. It is one of the paper’s main scientific questions. The paper evaluates:

- **random sub-bank sensitivity:** 20 random half-bank recalibrations from the 72-probe bank;
- **leave-one-family-out transfer:** calibrate without one probe family and test on targets aligned with that family;
- **severity-range holdout:** remove the roughest oscillatory or narrowest bump probes during calibration, then measure degradation on matching targets.

These analyses determine whether positive results reflect genuine transfer or accidental alignment to one hand-designed bank.

### Scope of the Contribution

The paper is positioned as a transfer study of external recalibration, not as a claim that post-hoc inflation dominates internal model repair. The intended contribution is:

- a strong baseline study for BQ interval correction under misspecification;
- a practical recommendation about when global scaling is enough;
- evidence about whether simple roughness-aware recalibration transfers at all;
- negative results about bank mismatch if that is what the data show.

## Related Work

### Foundational Probabilistic Integration

O'Hagan's Bayes-Hermite quadrature and the later probabilistic integration perspective of Briol et al. establish the basic GP view of numerical integration and uncertainty quantification. The proposed paper leaves that estimator family intact and studies only the trustworthiness of the reported spread.

### Misspecification and Internal Robustification

Kanagawa, Sriperumbudur, and Fukumizu analyze kernel quadrature under smoothness misspecification. Bayes-Sard cubature improves robustness by altering the internal prior mean space, while Hamid et al. reduce kernel-selection brittleness by marginalizing over stationary kernels. These papers are conceptually close because they all address mismatch, but they do so **inside** the probabilistic integration model. The proposed paper instead studies an **external** recalibration layer applied after fitting.

### Filtering-Side Quadrature Calibration Precursors

The earlier filtering literature is the closest precedent and must be discussed explicitly.

- **Bayesian Quadrature in Nonlinear Filtering** (Prüher and Šimandl, 2015) uses BQ inside sigma-point filtering and shows that accounting for integral variance can improve covariance credibility.
- **Bayesian Quadrature Variance in Sigma-Point Filtering** (Prüher and Šimandl, 2015/2016 chapter version) further studies variance estimation in that setting.
- **Improved Calibration of Numerical Integration Error in Sigma-Point Filters** (Prüher et al., 2021) improves calibration with Bayes-Sard structure inside filtering pipelines.

These papers already establish the idea of fixing quadrature uncertainty, not only the point estimate. That is why the present proposal no longer claims methodological novelty on that axis. Its differentiation is narrower:

- stand-alone Bayesian quadrature rather than nonlinear filtering;
- external post-hoc recalibration rather than internal quadrature redesign;
- explicit held-out transfer from analytic probes to unseen targets;
- probe-bank sensitivity and failure-mode analysis as first-class outcomes.

### Calibration Literature Outside Probabilistic Numerics

Syring and Martin study general posterior calibration, while Pion and Vazquez propose post-hoc calibration for GP predictive distributions. These papers support the broader view that uncertainty can be recalibrated after fitting, but they do not answer whether such recalibration transfers in BQ without target labels.

### Positioning

The novelty claim is intentionally moderate. If successful, the paper’s main value is not a new algorithmic principle. It is a careful answer to a practical question that current BQ papers do not settle: **how much interval reliability can be recovered by external recalibration alone, and how sensitive is that recovery to the probe bank?**

## Experiments

### Solvers and Resource-Constrained Design

To stay within 2 CPU cores and an 8-hour total budget, the study uses low-dimensional synthetic tasks with exact or cheap high-accuracy reference integrals.

Base GP-BQ:

- kernels in \{RBF, Matérn-3/2, Matérn-5/2\};
- empirical-Bayes selection by log marginal likelihood;
- direct linear algebra only;
- dimensions \(d\in\{1,2,3\}\);
- budgets \(n\in\{16,32\}\);
- measures: Uniform\([0,1]^d\) and \(\mathcal{N}(0,I_d)\).

Internal baselines:

- Bayes-Sard cubature on the full target set;
- reduced kernel-marginalization subset benchmark on the hardest targets only.

### Target Families

Per measure-dimension pair, evaluate 4 target families with 2 targets each:

- matched smooth controls;
- oscillatory mismatch;
- localized bump mismatch;
- kinked or cusp-like mismatch.

This gives
\[
2 \text{ measures}\times 3 \text{ dims}\times 4 \text{ families}\times 2 \text{ targets}=48 \text{ targets}.
\]

Targets are chosen so their integrals are either analytic or cheaply computable to near-machine precision. This removes the earlier feasibility concern around expensive reference integration.

### Methods Compared

The main comparison is:

- uncalibrated plug-in GP-BQ;
- global variance scaling;
- monotone isotonic roughness inflation;
- monotone stepwise inflation;
- Bayes-Sard cubature.

The reduced hard-case subset additionally compares against kernel marginalization. The study does not need kernel marginalization on every target to answer the main question.

### Transfer and Sensitivity Analyses

The paper includes four tightly scoped analyses.

1. **Held-out probe transfer**
   Measure whether the selected recalibrator improves calibration on the 24 held-out probes per measure.

2. **Held-out target transfer**
   Evaluate the same recalibrator on the 48 target integrands.

3. **Probe-bank sensitivity**
   Quantify variability across random sub-banks, leave-one-family-out calibration, and severity-range holdout.

4. **Matched-generator sanity study**
   In \(d\in\{1,2\}\), create aligned probe and target ladders for frequency, bump width, and kink sharpness to test whether roughness-aware inflation tracks increasing misspecification in the best-case transfer regime.

### Metrics

Primary metrics:

- empirical 90% and 95% coverage;
- absolute coverage gap;
- weighted interval score;
- interval width and width inflation.

Secondary metrics:

- absolute integration error;
- calibration error on held-out probes;
- abstention rate to the global rule;
- runtime overhead;
- variability across probe-bank resamples.

### Runtime Accounting

The run matrix is designed to be realistic.

Base GP-BQ target fits:
\[
48 \times 2 \times 3 = 288.
\]

Bayes-Sard target fits:
\[
288.
\]

Calibration probe fits:
\[
2 \text{ measures}\times 72 \text{ probes}\times 2 \text{ budgets}\times 3 \text{ seeds}=864.
\]

Reduced kernel-marginalization subset:
\[
16 \text{ hard targets}\times 2 \text{ budgets}\times 3 \text{ seeds}=96.
\]

Total planned fits:
\[
1536.
\]

This is feasible because all solves are low-dimensional, use small design sizes, and share the same CPU-light linear algebra backbone. The sensitivity analyses reuse existing probe evaluations rather than generating a new simulation grid.

### Expected Outcomes

The likely result is nuanced rather than sweeping:

- uncalibrated GP-BQ undercovers on misspecified families;
- global scaling closes much of the gap and is hard to beat when probe-target transfer is weak;
- roughness-aware monotone inflation helps when the bank spans the target regime;
- performance degrades under deliberate bank mismatch;
- internal baselines may remain stronger on the hardest misspecification cases.

That outcome would still be useful because it would define the operational limits of external recalibration.

## Success Criteria

The proposal is supported if most of the following hold:

1. On the union of misspecified target families, at least one probe-only recalibration rule improves 95% coverage over uncalibrated GP-BQ by at least 0.05 absolute.
2. The selected recalibration rule improves mean weighted interval score over uncalibrated GP-BQ and does not require more than 1.75x median width inflation on matched smooth controls.
3. Roughness-aware recalibration beats global scaling on at least one misspecified family and does not lose badly on the others.
4. Held-out probe performance predicts target-side gains in the matched-generator study, supporting the transfer interpretation.
5. Probe-bank sensitivity is measurable but not catastrophic: across random half-bank recalibrations, the standard deviation of target 95% coverage stays below 0.04.
6. Leave-one-family-out and severity-range-holdout experiments show clear degradation when probe support is intentionally misaligned, making the paper’s failure modes explicit.
7. Bayes-Sard remains a meaningful comparator rather than a trivial loser; if it dominates external recalibration everywhere, the paper still yields a publishable negative conclusion only if the sensitivity analyses explain why.
8. The full study completes within the stated CPU budget.

The proposal is not supported if all gains come from indiscriminate interval widening, if held-out probe behavior has no relation to target behavior even in matched-generator settings, or if the results are too bank-specific to survive simple sensitivity checks.

## References

1. O'Hagan, A. Bayes-Hermite Quadrature. *Journal of Statistical Planning and Inference*, 29(3):245-260, 1991.
2. Briol, F.-X., Oates, C. J., Girolami, M., Osborne, M. A., and Sejdinovic, D. Probabilistic Integration: A Role in Statistical Computation? *Statistical Science*, 34(1):1-22, 2019.
3. Kanagawa, M., Sriperumbudur, B. K., and Fukumizu, K. Convergence Analysis of Deterministic Kernel-Based Quadrature Rules in Misspecified Settings. arXiv:1709.00147, 2018.
4. Karvonen, T., Oates, C. J., and Särkkä, S. A Bayes-Sard Cubature Method. In *NeurIPS 2018*, 2018.
5. Hamid, S., Schulze, S., Osborne, M. A., and Roberts, S. J. Marginalising over Stationary Kernels with Bayesian Quadrature. In *AISTATS 2022*, PMLR 151:523-531, 2022.
6. Prüher, J. and Šimandl, M. Bayesian Quadrature in Nonlinear Filtering. In *Proceedings of the 12th International Conference on Informatics in Control, Automation and Robotics (ICINCO)*, pages 380-387, 2015.
7. Prüher, J. and Šimandl, M. Bayesian Quadrature Variance in Sigma-Point Filtering. In *Informatics in Control, Automation and Robotics: 12th International Conference, ICINCO 2015 Revised Selected Papers*, Lecture Notes in Electrical Engineering 383, pages 355-370, 2016.
8. Prüher, J., Karvonen, T., Oates, C. J., Straka, O., and Särkkä, S. Improved Calibration of Numerical Integration Error in Sigma-Point Filters. *IEEE Transactions on Automatic Control*, 66(3):1286-1292, 2021.
9. Syring, N. and Martin, R. Calibrating General Posterior Credible Regions. *Biometrika*, 106(2):479-486, 2019.
10. Pion, A. and Vazquez, E. A Bayesian Framework for Calibrating Gaussian Process Predictive Distributions. In *COPA 2025*, PMLR 266:748-750, 2025.
