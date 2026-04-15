# Sublinear-Memory Confidence Sequences for Streaming Quantiles

## Introduction

### Context

Quantile estimation from streaming data is a fundamental problem in data analysis, with applications ranging from network latency monitoring (tracking p99 response times) to A/B testing (comparing median user engagement) to financial risk assessment (monitoring Value-at-Risk). Two mature but largely disconnected research threads address this problem:

1. **Streaming quantile sketches** (Greenwald & Khanna, 2001; Karnin et al., 2016; Masson et al., 2019) provide point estimates of quantiles using sublinear memory O(1/epsilon * polylog(n)), but offer only worst-case error bounds, not probabilistic confidence intervals. These sketches are widely deployed in production systems (e.g., DDSketch at Datadog, KLL in Apache DataSketches).

2. **Confidence sequences for quantiles** (Howard & Ramdas, 2022) provide anytime-valid confidence intervals for quantiles that hold uniformly over all sample sizes, with widths shrinking at the optimal sqrt(t^{-1} log log t) rate. However, these methods require maintaining the full empirical CDF, which costs O(n) memory — impractical for long-running streams.

### Problem Statement

No existing method provides **anytime-valid confidence intervals for streaming quantiles with sublinear memory**. Quantile sketches sacrifice statistical guarantees for memory efficiency; confidence sequences sacrifice memory efficiency for statistical guarantees. This gap is practically important: monitoring systems that track tail latencies need both bounded memory (streams run indefinitely) and valid uncertainty quantification (alerting decisions require calibrated confidence).

### Key Insight

The confidence interval for a quantile Q_p = F^{-1}(p) can be obtained by **inverting CDF confidence sequences**: if we have a valid confidence sequence for F(q) at threshold q, we can invert to get a confidence interval for Q_p. Crucially, tracking F(q) at a fixed threshold q requires only O(1) memory — it is a running count of observations below q. By maintaining confidence sequences at a small **adaptive grid** of thresholds concentrated around Q_p, we can construct valid quantile confidence intervals using only O(k log t) total memory, where k is the grid size.

### Hypothesis

An adaptive CDF-tracking algorithm with O(k) active thresholds per epoch (organized in a doubling schedule of O(log t) epochs) can produce anytime-valid confidence intervals for streaming quantiles whose widths converge at the optimal sqrt(t^{-1} log log t) rate, while using only O(k log t) total memory — exponentially less than the O(t) memory required by existing confidence sequence methods.

## Proposed Approach

### Overview

We propose **AdaQuantCS** (Adaptive Quantile Confidence Sequences), an algorithm that maintains anytime-valid confidence intervals for one or more quantile levels using sublinear memory. The algorithm operates in epochs organized on a doubling schedule, with each epoch maintaining a grid of CDF-tracking thresholds that progressively refine around the target quantile.

### Method Details

#### Core Mechanism: CDF Inversion

For a target quantile level p in (0,1), the p-th quantile is Q_p = inf{q : F(q) >= p}. Given a confidence sequence for F(q) — i.e., intervals [L_t(q), U_t(q)] such that P(forall t: L_t(q) <= F(q) <= U_t(q)) >= 1-alpha — we obtain a confidence interval for Q_p by inversion:

  CI_t(Q_p) = [inf{q in G : U_t(q) >= p}, sup{q in G : L_t(q) <= p}]

where G is the set of tracked thresholds. Each F(q) is simply P(X <= q), so its empirical estimate is a running average of Bernoulli indicators I(X_i <= q), trackable in O(1) memory per threshold.

#### Adaptive Grid via Doubling Epochs

The algorithm operates in epochs j = 0, 1, 2, ..., where epoch j spans observations t in [2^j, 2^{j+1}):

1. **Epoch initialization**: At the start of epoch j, choose k threshold values centered around the current best estimate of Q_p. The grid spacing decreases with j, providing finer resolution as more data arrives.

2. **Within-epoch tracking**: For each threshold q in the grid, maintain:
   - A counter c_t(q) = number of observations in this epoch with X_i <= q
   - The epoch length n_j = number of observations in this epoch so far
   - A Bernoulli confidence sequence for F(q) based on (c_t(q), n_j)

3. **CDF inversion**: At each time step, compute the quantile CI by inverting across all thresholds in the current epoch's grid.

4. **Multi-epoch intersection**: Since each epoch provides an independent valid CI (thresholds are chosen deterministically based on epoch boundaries, not data within the epoch), intersecting CIs across epochs can only tighten the result while preserving validity.

#### Grid Placement Strategy

At the start of epoch j, we place k thresholds as follows:
- Use the CI from previous epochs to identify a range [a_j, b_j] containing Q_p with high confidence
- Place thresholds uniformly (or with a density proportional to the estimated f(Q_p)) within [a_j, b_j]
- Include sentinel thresholds outside [a_j, b_j] to detect if Q_p has moved

The grid spacing in epoch j is approximately (b_j - a_j) / k, which shrinks as the CI from previous epochs narrows.

#### Confidence Sequence for F(q)

For a fixed threshold q, the indicators Z_i = I(X_i <= q) are i.i.d. Bernoulli(F(q)). We use the empirical Bernstein confidence sequence from Howard et al. (2021):

  |F_hat_n(q) - F(q)| <= sqrt(2 * V_n * log(log(2n)/alpha) / n) + c * log(log(2n)/alpha) / n

where V_n is the empirical variance and c is a constant. This provides an anytime-valid, nonasymptotic confidence sequence that shrinks at the LIL rate.

### Key Innovations

1. **Epoch-based adaptive refinement**: By using a deterministic doubling schedule, threshold placement decisions at epoch boundaries depend only on data from previous epochs, avoiding selection bias and preserving validity.

2. **Multi-resolution CDF inversion**: Intersecting quantile CIs across epochs creates a "multi-resolution" view that combines the broad coverage of early epochs with the precision of later epochs.

3. **Memory management**: Total memory at time t is O(k * log t) — k thresholds per epoch times O(log t) active epochs. For typical applications (k=50, t=10^9), this is ~1500 counters vs. 10^9 stored observations.

4. **Extension to multiple quantiles**: Track multiple quantile levels simultaneously by maintaining separate grids, or a shared grid covering all target quantile levels.

## Related Work

### Streaming Quantile Sketches

**Greenwald & Khanna (2001)** introduced the GK sketch for deterministic epsilon-approximate quantile estimation in O(1/epsilon * log(epsilon*N)) space. **Karnin, Lang, Liberty (2016)** proved the optimal O(1/epsilon * log log(1/delta)) space bound for randomized sketches (KLL sketch). **Masson, Rim, Lee (2019)** proposed DDSketch with relative-error guarantees and full mergeability, now widely deployed in industry. These sketches provide point estimates with worst-case error bounds but not confidence intervals or anytime-valid guarantees.

**Our work differs** by providing confidence intervals rather than point estimates, with anytime-valid coverage guarantees rather than per-query error bounds. While sketches answer "what is the approximate quantile?", our method answers "what range is the quantile in, with guaranteed coverage?"

### Confidence Sequences and Anytime-Valid Inference

**Howard, Ramdas, McAuliffe, Sekhon (2021)** developed the theory of nonparametric, nonasymptotic confidence sequences with optimal LIL-rate width convergence. **Howard & Ramdas (2022)** applied this to quantile estimation, providing the first confidence sequences for streaming quantiles with coverage uniform over all sample sizes. **Ramdas, Grunwald, Vovk, Shafer (2023)** surveyed the broader landscape of game-theoretic statistics and safe anytime-valid inference (SAVI).

**Our work differs** by addressing the memory bottleneck: Howard & Ramdas (2022) requires O(t) memory to maintain the empirical CDF at time t, making it impractical for long-running streams. Our adaptive CDF-tracking approach achieves similar coverage guarantees in O(k log t) memory.

### CDF Confidence Bands under Nonstationarity

**Mineiro & Howard (2023)** extended time-uniform CDF confidence bands to nonstationary settings using importance weighting, with applications to A/B testing and contextual bandits. Their work addresses nonstationarity but not memory constraints.

**Our work is complementary**: their nonstationary CDF bounds could replace our i.i.d. Bernoulli CS within each epoch, extending our method to nonstationary streams.

### Federated Conformal Prediction with Sketches

Recent work on federated conformal prediction (Lu et al., 2023; Humbert et al., 2023) uses T-Digest or DDSketch to compress calibration scores across distributed clients. These address communication efficiency in federated settings, not memory efficiency in streaming settings. The analysis of sketch error propagation to coverage guarantees is related but technically different from our CDF-inversion approach.

## Experiments

### Planned Setup

All experiments run on CPU (2 cores, 128GB RAM). Implementation in Python using NumPy/SciPy, with the `confseq` library for baseline confidence sequences.

### Experiment 1: Coverage Validation (Synthetic Data)

**Goal**: Verify that AdaQuantCS maintains the nominal coverage rate.

**Setup**: Generate i.i.d. streams from Gaussian(0,1), Exponential(1), Pareto(alpha=2), and Student-t(df=3) distributions. For each distribution, run 1000 independent trials of length n=100,000. Track the median (p=0.5) and 95th percentile (p=0.95).

**Metrics**: Empirical coverage rate (fraction of trials where true quantile is always within CI), average CI width over time.

**Baselines**: Full-memory Howard-Ramdas CS, offline bootstrap CI (computed on sliding window).

### Experiment 2: Memory-Accuracy Tradeoff

**Goal**: Characterize how CI width varies with the memory budget k.

**Setup**: Fix distribution to Gaussian(0,1) and stream length n=1,000,000. Vary k in {5, 10, 20, 50, 100, 200}. Track median CI width at selected time points.

**Metrics**: CI width vs. memory (k), gap relative to full-memory baseline.

**Expected result**: CI width decreases as k increases, approaching the full-memory baseline around k=50-100.

### Experiment 3: Comparison with Sketch + Bootstrap

**Goal**: Compare AdaQuantCS to a natural alternative: quantile sketch (KLL/DDSketch) + bootstrap CI.

**Setup**: Same synthetic streams as Experiment 1. For the baseline, maintain a KLL sketch and compute bootstrap CIs from sketch samples at each time step.

**Metrics**: CI width, coverage, memory, computation time.

**Expected result**: Sketch+bootstrap has no anytime-valid guarantee (coverage degrades with peeking), while AdaQuantCS maintains valid coverage.

### Experiment 4: Real Data Applications

**Goal**: Demonstrate practical utility on real-world monitoring scenarios.

**Datasets**:
- Network latency traces (e.g., CAIDA or synthetic network traces)
- Financial returns data (e.g., daily stock returns for VaR monitoring)

**Metrics**: CI width, memory usage, detection of quantile shifts.

### Experiment 5: Multiple Quantile Tracking

**Goal**: Evaluate simultaneous tracking of multiple quantile levels.

**Setup**: Track p in {0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99} simultaneously on Gaussian and heavy-tailed streams.

**Metrics**: Per-quantile CI width, total memory, joint coverage.

### Experiment 6: Ablation Studies

**Goal**: Understand the contribution of each design choice.

**Ablations**:
- Fixed vs. adaptive grid placement
- Different grid sizes (k)
- Different confidence sequence methods (Hoeffding, empirical Bernstein, betting-based)
- Epoch length schedule (doubling vs. fixed vs. geometric with different ratios)

## Success Criteria

### Primary (confirms hypothesis)
1. AdaQuantCS achieves >= (1-alpha-0.01) empirical coverage across all tested distributions and quantile levels, matching the theoretical guarantee.
2. CI widths converge at rate O(sqrt(log log t / t)), matching the optimal LIL rate achieved by full-memory methods.
3. Memory usage is O(k log t), confirmed empirically to be at least 100x less than full-memory methods at t=10^6.

### Secondary (strengthens contribution)
4. On real data, AdaQuantCS produces practically useful CI widths (e.g., within 2x of full-memory baseline) with <1% of the memory.
5. The adaptive grid strategy outperforms fixed grid by at least 30% in CI width for the same memory budget.
6. AdaQuantCS detects real quantile shifts earlier or with fewer false alarms than sketch-only monitoring.

### What would refute the hypothesis
- If the adaptive grid introduces coverage degradation >0.02 despite theoretical validity, this would suggest implementation issues or that the theoretical analysis has gaps.
- If the memory savings are less than 10x compared to full-memory CS, the practical contribution would be marginal.
- If a simpler approach (e.g., sketch + offline CI) achieves comparable coverage with less complexity, the algorithmic contribution would be diminished.

## References

1. Greenwald, M. and Khanna, S. (2001). Space-efficient online computation of quantile summaries. In *Proceedings of the 2001 ACM SIGMOD International Conference on Management of Data*, pages 58-66.

2. Howard, S. R. and Ramdas, A. (2022). Sequential estimation of quantiles with applications to A/B-testing and best-arm identification. *Bernoulli*, 28(3):1704-1728.

3. Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. *The Annals of Statistics*, 49(2):1055-1080.

4. Karnin, Z., Lang, K., and Liberty, E. (2016). Optimal quantile approximation in streams. In *Proceedings of the 57th IEEE Annual Symposium on Foundations of Computer Science (FOCS)*, pages 71-78.

5. Masson, C., Rim, J. E., and Lee, H. K. (2019). DDSketch: A fast and fully-mergeable quantile sketch with relative-error guarantees. *Proceedings of the VLDB Endowment*, 12(12):2195-2205.

6. Mineiro, P. and Howard, S. R. (2023). Time-uniform confidence bands for the CDF under nonstationarity. In *Advances in Neural Information Processing Systems (NeurIPS)*.

7. Ramdas, A., Grunwald, P., Vovk, V., and Shafer, G. (2023). Game-theoretic statistics and safe anytime-valid inference. *Statistical Science*, 38(4):576-601.
