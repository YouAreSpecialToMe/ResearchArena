# Confidence Sequences for Markov Chain Monte Carlo: Anytime-Valid Estimation with Sequential Guarantees

## Introduction

### Context

Markov chain Monte Carlo (MCMC) is the workhorse of Bayesian computation, enabling approximate inference in complex probabilistic models by constructing Markov chains whose stationary distributions match the target posterior. A central practical challenge in MCMC is deciding *when to stop*: practitioners must determine when the chain has run long enough that the resulting estimates are sufficiently accurate.

The standard approach constructs confidence intervals using the Markov chain central limit theorem (CLT), estimating the asymptotic variance via batch means or spectral methods (Flegal & Jones, 2010; Vats, Flegal & Jones, 2019). Fixed-width stopping rules (FWSR) terminate sampling when the confidence interval half-width falls below a threshold (Vats et al., 2019; Flegal et al., 2024). However, these intervals are only valid at a *pre-specified* or *fixed* sample size. When practitioners monitor confidence intervals sequentially and stop as soon as they are narrow enough, the resulting coverage guarantees are violated due to the well-known sequential testing problem (optional stopping).

### The Gap

In the sequential analysis literature, *confidence sequences* (CS) provide time-uniform confidence intervals that are valid at *all* stopping times simultaneously (Howard et al., 2021; Ramdas et al., 2023). A (1-alpha) confidence sequence satisfies:

    P(mu in CS_t for all t >= 1) >= 1 - alpha

This is strictly stronger than a fixed-time confidence interval, which only guarantees P(mu in CI_n) >= 1 - alpha for a single pre-specified n. Confidence sequences enable valid sequential monitoring, early stopping, and continuous data analysis without inflating Type I error.

Despite the maturity of both the MCMC output analysis literature and the confidence sequence literature, *no existing work develops confidence sequences specifically for MCMC estimators*. This gap is surprising given that MCMC is inherently a sequential procedure where practitioners routinely monitor estimates and make stopping decisions online.

### Key Insight

MCMC samples are dependent, which precludes direct application of i.i.d. confidence sequence constructions. Our key insight is that two established techniques can bridge this gap:

1. **Batch means**: By averaging over sufficiently large blocks, MCMC samples can be converted into approximately independent batch means, to which confidence sequence techniques can be applied with controlled approximation error.

2. **Coupling-based debiasing**: The unbiased MCMC framework of Jacob et al. (2020) produces exactly independent and unbiased estimators from coupled chains, enabling direct application of standard confidence sequences.

### Hypothesis

We hypothesize that confidence sequences can be constructed for MCMC estimators that (a) maintain valid time-uniform coverage at all stopping times, (b) converge at nearly the optimal rate of O(sqrt(sigma^2_infty * log log n / n)) where sigma^2_infty is the asymptotic variance, and (c) enable substantially more efficient computation by allowing valid early stopping compared to conservative fixed-sample approaches.

## Proposed Approach

### Overview

We propose three methods for constructing confidence sequences for MCMC, each offering different tradeoffs between theoretical guarantees, computational overhead, and practical tightness.

### Method 1: Batch Means Confidence Sequences (BMCS)

**Core idea**: Divide the MCMC output into non-overlapping batches and apply confidence sequence constructions to the resulting batch means.

Let {X_1, X_2, ...} be an MCMC chain targeting distribution pi, and let f be a function whose expectation mu = E_pi[f(X)] we wish to estimate. Partition the chain into batches of size b:

    Y_j = (1/b) * sum_{i=(j-1)b+1}^{jb} f(X_i),  j = 1, 2, ...

Under geometric ergodicity, for sufficiently large b, the batch means Y_1, Y_2, ... are approximately independent with:
- E[Y_j] ≈ mu (with bias decaying exponentially in b)
- Var(Y_j) ≈ sigma^2_infty / b

**Construction**: Apply the empirical Bernstein confidence sequence (Howard et al., 2021) to the batch means. After k batches (n = kb total samples), the confidence sequence takes the form:

    CS_k = [Y_bar_k - w_k, Y_bar_k + w_k]

where Y_bar_k is the running average of batch means and w_k accounts for both the intrinsic variance and the batch approximation error.

**Key theoretical challenge**: The batch means are only *approximately* independent. We must quantify and control the residual dependence and the bias from non-stationarity (early batches may include burn-in). Our approach:
- Use geometric ergodicity to bound the total variation distance between the batch means distribution and a product of independent distributions
- Add a correction term to the confidence sequence width that accounts for this approximation, decaying as O(rho^b) where rho < 1 is the geometric mixing rate
- Use an adaptive batch size b_k that grows logarithmically with k to ensure the approximation error remains controlled

**Adaptive batch size selection**: Set b_k = ceil(c * log(k+1)) for a constant c depending on the mixing time. This ensures:
- The number of batches grows as O(n / log n), maintaining enough batches for tight confidence sequences
- The approximation error from residual dependence decays polynomially in n
- The overall width achieves the near-optimal rate O(sqrt(log log n / n))

### Method 2: Coupling-Based Confidence Sequences (CBCS)

**Core idea**: Use the coupling framework of Jacob et al. (2020) to generate exactly unbiased and independent estimators, then apply standard i.i.d. confidence sequences.

The unbiased MCMC method runs coupled pairs of chains (X_t, Y_t) with a maximal coupling kernel. When the chains meet at time tau, unbiased estimators are constructed as:

    H_k = h(X_k) + sum_{t=k+1}^{tau-1} [h(X_t) - h(Y_{t-1})]

These estimators are exactly unbiased (E[H_k] = mu) and can be generated independently by running multiple coupled chain pairs.

**Construction**: Generate m independent unbiased estimators H_1, ..., H_m and apply the sub-Gaussian or Catoni-style confidence sequence (Wang & Ramdas, 2023) directly.

**Advantages**:
- No approximation: estimators are exactly unbiased and independent
- Full theoretical guarantees from existing confidence sequence theory
- Handles heavy-tailed posteriors via Catoni-style sequences

**Disadvantages**:
- Computational overhead from coupling (typically 2-3x the cost of single-chain MCMC)
- Higher variance per estimator due to the telescopic correction
- Meeting time tau may be large for poorly mixing chains

### Method 3: Martingale Decomposition Confidence Sequences (MDCS)

**Core idea**: Exploit the martingale structure inherent in MCMC to construct confidence sequences directly, without batching or coupling.

Under suitable regularity conditions, the MCMC partial sum admits a Poisson equation decomposition:

    S_n - n*mu = M_n + R_n

where M_n is a martingale and R_n = g(X_n) - g(X_0) is a bounded remainder (with g being the solution to the Poisson equation). This decomposition allows us to:

1. Apply martingale confidence sequences (e.g., the method of mixtures) to M_n
2. Bound R_n using the Poisson equation solution norm
3. Combine to get a confidence sequence for mu = S_n/n

**Advantages**:
- Most direct approach: no batching artifacts or coupling overhead
- Tightest possible width for well-mixing chains
- Naturally adapts to the chain's mixing properties

**Disadvantages**:
- Requires estimating the Poisson equation solution norm (or a bound thereof)
- Theoretical analysis requires stronger assumptions (e.g., drift conditions)

### Comparison and Practical Recommendations

| Method | Validity | Width | Overhead | Assumptions |
|--------|----------|-------|----------|-------------|
| BMCS | Approximate (controlled) | Moderate | Low | Geometric ergodicity |
| CBCS | Exact | Wider | High (coupling) | Faithful coupling exists |
| MDCS | Exact (under assumptions) | Tightest | Moderate | Drift + minorization |

We will provide clear practical recommendations: CBCS for settings where exact validity is paramount, BMCS as the default practical choice, and MDCS for advanced users with well-understood chains.

## Related Work

### MCMC Output Analysis

The foundational work on MCMC confidence intervals uses the Markov chain CLT and asymptotic variance estimation via batch means (Flegal & Jones, 2010), spectral methods (Geyer, 1992), or the initial sequence estimator (Geyer, 2011). Vats, Flegal & Jones (2019) extend this to multivariate settings and propose effective sample size and fixed-width stopping rules. Vats & Flegal (2022) introduce lugsail lag windows to reduce the negative bias of variance estimators. Flegal et al. (2024) provide a comprehensive practical workflow.

**How we differ**: All existing MCMC confidence intervals are valid only at a pre-specified sample size. Our confidence sequences are valid at *all* stopping times, enabling principled sequential monitoring.

### Confidence Sequences and Anytime-Valid Inference

Howard et al. (2021) develop the modern theory of confidence sequences for i.i.d. data, achieving nonasymptotic, nonparametric, time-uniform coverage. Ramdas et al. (2023) survey the game-theoretic foundations, connecting confidence sequences to e-values and test martingales. Wang & Ramdas (2023) extend to heavy-tailed distributions using Catoni's M-estimator.

**How we differ**: Existing confidence sequences assume i.i.d. or martingale data. Our work extends these constructions to the dependent, non-stationary setting of MCMC, which requires fundamentally new technical tools (batch means approximation, coupling debiasing, or Poisson equation decomposition).

### Coupling-Based MCMC Methods

Jacob et al. (2020) introduce unbiased MCMC with couplings, using meeting times of coupled chains to construct exactly unbiased estimators. Corenflos & Dau (2025) use couplings for f-divergence-based convergence diagnostics. Martinez-Taboada & Ramdas (2025) develop sequential goodness-of-fit tests using kernel Stein discrepancy.

**How we differ**: Jacob et al. provide unbiased point estimators but not confidence sequences. Martinez-Taboada & Ramdas address testing (is the sample from the target?) but not estimation (what is E_pi[f]?). Corenflos & Dau provide diagnostic bounds but not estimation guarantees. Our work provides the first *estimation* confidence sequences for MCMC.

### MCMC Stopping Rules

Robertson et al. (2021) and Vats et al. (2019) propose fixed-width stopping rules that terminate when the CLT-based confidence interval is narrow enough. These rules have desirable asymptotic properties but lack finite-sample sequential validity.

**How we differ**: Our confidence sequences maintain valid coverage at *every* intermediate time point, not just at the terminal stopping time. This provides strictly stronger guarantees and enables continuous monitoring without coverage degradation.

## Experiments

### Experimental Setup

All experiments run on CPU (2 cores, 128GB RAM, ~8 hours total budget). We implement all methods in Python using NumPy/SciPy, with MCMC samplers written from scratch for full control over the chain state.

### Experiment 1: Coverage Verification (Gaussian Targets)

**Goal**: Verify that confidence sequences maintain nominal coverage at all stopping times.

**Setup**: Target distribution is a d-dimensional Gaussian N(0, Sigma) with known mean mu = 0. Run Metropolis-Hastings with different proposal scales to control mixing. For each configuration:
- Run 1000 independent replicates
- At each time point t, record whether mu is in CS_t
- Compute empirical coverage at each t and report minimum coverage over all t

**Dimensions**: d in {1, 5, 20, 50}
**Chain lengths**: Up to n = 100,000
**Metrics**: Time-uniform coverage (should be >= 1-alpha), width trajectory, comparison with CLT intervals

### Experiment 2: Practical Bayesian Inference (Logistic Regression)

**Goal**: Demonstrate practical utility on a real Bayesian inference problem.

**Setup**: Bayesian logistic regression on standard UCI datasets (German Credit, Ionosphere, Pima Indians). Use random-walk Metropolis-Hastings and compare:
- BMCS (our method)
- CBCS (our method, coupling-based)
- CLT-based confidence intervals (Flegal & Jones)
- Fixed-width stopping rule (Vats et al.)

**Metrics**: Time to reach target width, actual coverage when stopped early, total computation cost

### Experiment 3: Challenging Mixing (Multimodal Targets)

**Goal**: Test robustness when chains mix poorly.

**Setup**: Mixture of well-separated Gaussians in d dimensions. Vary the separation to control mixing time. Evaluate:
- Coverage degradation under poor mixing
- How BMCS adapts batch size to mixing
- CBCS meeting time distribution
- When diagnostics correctly flag non-convergence

### Experiment 4: Sequential Stopping Efficiency

**Goal**: Quantify computational savings from valid early stopping.

**Setup**: For each target distribution, compare:
- **Oracle**: Runs until CLT interval reaches target width (knows optimal n in advance)
- **FWSR**: Fixed-width stopping rule with CLT intervals (current practice)
- **CS-Stop**: Our confidence sequence with stopping rule CS_width < epsilon

**Metrics**: Ratio of actual to oracle sample size, coverage at stopping time, excess computation

### Experiment 5: Scalability

**Goal**: Assess computational overhead and scaling behavior.

**Setup**: Vary dimension d from 2 to 200. Measure:
- Per-iteration cost of each confidence sequence method
- How width converges as a function of n and d
- Memory usage

### Expected Results

1. **BMCS and CBCS maintain valid coverage at all stopping times**, while CLT intervals show undercoverage when monitored sequentially (expected coverage violation of 5-15% at typical monitoring frequencies).
2. **Sequential stopping with confidence sequences saves 20-40% of computation** compared to conservative fixed-n approaches, with no coverage loss.
3. **CBCS provides tighter guarantees but at 2-3x computational cost** compared to BMCS, making BMCS the recommended default.
4. **Under poor mixing, BMCS with adaptive batch sizing remains valid** while CLT-based methods exhibit severe undercoverage.

## Success Criteria

### Primary criteria (must be met for the paper to succeed):
1. Confidence sequences achieve >= (1-alpha) coverage uniformly over time in controlled experiments with known ground truth
2. At least one method (BMCS or CBCS) is practical: overhead < 2x compared to standard MCMC output analysis
3. Clear computational savings (>= 15% reduction in samples) from valid sequential stopping compared to fixed-width stopping rules

### Secondary criteria (strengthen the paper if met):
4. Theoretical convergence rate within a log log n factor of the optimal i.i.d. rate
5. Robust performance across different MCMC kernels (MH, Gibbs, MALA)
6. Competitive width with CLT intervals at fixed sample sizes (showing minimal cost of anytime validity)

### Criteria that would refute the hypothesis:
- If the approximation error in BMCS is too large to control in practice (confidence sequences are excessively wide)
- If coupling overhead makes CBCS impractical for typical problems
- If sequential monitoring of CLT intervals turns out to have negligible coverage distortion in practice (reducing the motivation)

## References

1. Corenflos, A. and Dau, H.-D. (2025). A coupling-based approach to f-divergences diagnostics for Markov chain Monte Carlo. arXiv:2510.07559.

2. Flegal, J. M. and Jones, G. L. (2010). Batch means and spectral variance estimators in Markov chain Monte Carlo. The Annals of Statistics, 38(2):1034-1070.

3. Flegal, J. M., Hughes, J., Vats, D., and Dai, N. (2024). Implementing MCMC: Multivariate estimation with confidence. arXiv:2408.15396.

4. Geyer, C. J. (1992). Practical Markov Chain Monte Carlo. Statistical Science, 7(4):473-483.

5. Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. The Annals of Statistics, 49(2):1-36.

6. Jacob, P. E., O'Leary, J., and Atchade, Y. F. (2020). Unbiased Markov chain Monte Carlo methods with couplings. Journal of the Royal Statistical Society: Series B, 82(3):543-600.

7. Martinez-Taboada, D. and Ramdas, A. (2025). Sequential Kernelized Stein Discrepancy. Proceedings of AISTATS 2025.

8. Ramdas, A., Grunwald, P., Vovk, V., and Shafer, G. (2023). Game-theoretic statistics and safe anytime-valid inference. Statistical Science, 38(4):576-601.

9. Vats, D. and Flegal, J. M. (2022). Lugsail lag windows for estimating time-average covariance matrices. Biometrika, 109(3):735-750.

10. Vats, D., Flegal, J. M., and Jones, G. L. (2019). Multivariate output analysis for Markov chain Monte Carlo. Biometrika, 106(2):321-337.

11. Wang, H. and Ramdas, A. (2023). Catoni-style confidence sequences for heavy-tailed mean estimation. Stochastic Processes and their Applications, 163:51-79.
