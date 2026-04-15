# Streaming Multi-Scale Adaptive Kernel Conformal Prediction

## Introduction

### Problem Statement

Conformal prediction (CP) provides distribution-free prediction sets with finite-sample marginal coverage guarantees. However, achieving reliable conditional coverage—where prediction sets adapt to local uncertainty patterns while maintaining validity in specific regions of the feature space—remains a fundamental challenge, especially in streaming settings with distribution drift.

Recent advances in kernel-based conformal prediction [Gibbs et al., 2023; Liu et al., 2025] enable approximate conditional coverage through RKHS function classes. However, these methods assume a fixed kernel bandwidth, which creates a critical limitation: a bandwidth optimal for dense data regions may fail in sparse regions, and vice versa. Moreover, they operate in batch settings, leaving the streaming regime largely unexplored.

Existing multi-model online approaches like MOCP [Hajihashemi & Shen, 2024] achieve adaptivity through ensemble aggregation but require evaluating multiple models per prediction. Localized methods like RLCP [Hore & Barber, 2023] use kernel weighting but with fixed bandwidths and without explicit multi-scale structure.

### Key Insight

We propose a fundamentally different approach: **Streaming Multi-Scale Adaptive Kernel Conformal Prediction (SMAK-CP)**. Instead of aggregating multiple models or using a single fixed kernel, we maintain a hierarchy of kernel scales that adapt their bandwidths online based on local data characteristics. The key innovation is an online bandwidth selection mechanism that simultaneously:

1. **Adapts to local data density**—smaller bandwidths in dense regions for sharper localization
2. **Responds to distribution drift**—bandwidth adjustments based on recent coverage feedback
3. **Maintains multi-scale coverage**—hierarchical structure ensures robustness across scales

### Hypothesis

Our central hypothesis is that online adaptive bandwidth selection within a multi-scale kernel hierarchy provides superior local coverage and prediction set efficiency compared to fixed-bandwidth kernel methods in streaming settings with heterogeneous data density and distribution drift.

---

## Proposed Approach

### Overview

SMAK-CP operates on a stream of data points $(X_t, Y_t)_{t \geq 1}$. At each time step $t$, we:

1. **Observe** covariate $X_t$ and form a prediction set $\hat{C}_t(X_t)$
2. **Reveal** true response $Y_t$ and compute non-conformity score $S_t = s(X_t, Y_t)$
3. **Update** multi-scale bandwidth parameters based on local coverage feedback

### Multi-Scale Kernel Hierarchy

We maintain $K$ kernel scales indexed by $k \in \{1, \ldots, K\}$, where scale $k$ uses bandwidth $h_t^{(k)}$. The scales follow a geometric progression:

$$h_t^{(k)} = h_t^{(1)} \cdot \rho^{k-1}, \quad \rho > 1$$

Each scale $k$ maintains its own local quantile estimate $q_t^{(k)}(x)$ computed via kernel-weighted empirical quantile regression on recent data.

### Online Bandwidth Adaptation

The critical innovation is online adaptation of the base bandwidth $h_t^{(1)}$. We define a **local coverage discrepancy** measure for scale $k$ at time $t$:

$$\Delta_t^{(k)} = \frac{1}{|\mathcal{N}_t(X_t)|} \sum_{i \in \mathcal{N}_t(X_t)} \mathbb{I}\{Y_i \in \hat{C}_i^{(k)}(X_i)\} - (1-\alpha)$$

where $\mathcal{N}_t(X_t)$ denotes recent data points in the neighborhood of $X_t$.

The base bandwidth is updated via exponential moving average with adaptive learning rate:

$$\log h_{t+1}^{(1)} = \log h_t^{(1)} - \eta_t \cdot \text{sign}\left(\sum_{k=1}^K w_t^{(k)} \Delta_t^{(k)}\right)$$

where the scale weights $w_t^{(k)}$ reflect each scale's reliability:

$$w_t^{(k)} \propto \exp\left(-\lambda \sum_{s=t-W}^{t-1} |\Delta_s^{(k)}|\right)$$

### Cold-Start Initialization Strategy

A critical challenge in sparse data regions is the cold-start problem: bandwidth adaptation may be unstable when local effective sample sizes are small. We propose a **hierarchical warm-up protocol**:

1. **Global initialization (t ≤ T₀):** Use a fixed, conservatively large bandwidth $h_0$ for all scales, ensuring sufficient samples for reliable quantile estimation
2. **Scale-specific warm-up (T₀ < t ≤ T₁):** Gradually activate finer scales as their effective sample sizes exceed $n_{\min}^{(k)} = n_{\min} \cdot \rho^{(k-1)(d/2)}$, where $d$ is the dimension
3. **Full adaptation (t > T₁):** Enable complete bandwidth adaptation with regularization $\lambda_t = \lambda_0 \cdot \min(1, \sqrt{T_1/t})$ that decreases as more data accumulates

For regions with persistently low data density, we employ a **fallback mechanism** that automatically reverts to coarser scales when $n_t^{(k)}(X_t) < n_{\min}$, ensuring coverage validity even in sparse regions.

### Scale Selection and Aggregation

For each prediction, we select the most appropriate scale(s) based on local data density. Define the local effective sample size for scale $k$:

$$n_t^{(k)}(X_t) = \sum_{i=t-W}^{t-1} K\left(\frac{\|X_i - X_t\|}{h_t^{(k)}}\right)$$

where $K(\cdot)$ is a kernel function (e.g., Gaussian).

**Scale selection strategies:**

1. **Single-scale (SMAK-S):** Select scale with largest $n_t^{(k)}$ exceeding threshold $n_{\min}$
2. **Multi-scale intersection (SMAK-I):** Intersect prediction sets from all scales with sufficient data
3. **Weighted aggregation (SMAK-W):** Weighted combination of quantile estimates

#### Justification for Multi-Scale Intersection (SMAK-I)

The SMAK-I strategy requires careful theoretical justification because naive set intersection typically reduces coverage below the target level. We address this through a **calibrated intersection** approach:

**Theorem (Calibrated Multi-Scale Intersection).** Let $\hat{C}_t^{(k)}$ denote the prediction set from scale $k$ at target level $1-\alpha_k$. If we set $\alpha_k = 1 - (1-\alpha)^{1/K}$ for each scale (Bonferroni-style calibration), then:

$$\mathbb{P}\left(Y_t \in \bigcap_{k \in \mathcal{A}_t} \hat{C}_t^{(k)}\right) \geq 1 - \alpha$$

This conservative calibration ensures valid coverage, but may produce overly large sets. To improve efficiency, we employ an **adaptive calibration** that exploits the nested structure of kernel scales:

$$\alpha_k = \alpha \cdot \frac{w_t^{(k)}}{\sum_{j \in \mathcal{A}_t} w_t^{(j)}}$$

where the weights reflect each scale's local reliability. Under mild regularity conditions (smooth conditional distribution, kernel with exponential tails), this adaptive calibration maintains approximate coverage while producing significantly tighter prediction sets than single-scale methods.

**Practical justification:** SMAK-I is particularly effective when different scales capture complementary aspects of local uncertainty—coarse scales capture global trends while fine scales capture local variations. The intersection eliminates regions that are not supported at any scale, improving localization without sacrificing coverage when properly calibrated.

### Algorithm

**Input:** Target coverage $1-\alpha$, number of scales $K$, scale factor $\rho$, window size $W$, minimum samples $n_{\min}$, warm-up period $T_0$, activation period $T_1$

**Initialize:** $h_0^{(1)} > 0$, quantile estimates $q_0^{(k)} \equiv 0$, adaptation flag $\text{adapt}_t = \text{False}$ for $t \leq T_0$

For $t = 1, 2, \ldots$:
1. Observe $X_t$
2. **Scale selection:** Compute $n_t^{(k)}(X_t)$ for all $k$, select active set $\mathcal{A}_t = \{k : n_t^{(k)}(X_t) \geq n_{\min}^{(k)}\}$
3. **Fallback check:** If $\mathcal{A}_t = \emptyset$, set $\mathcal{A}_t = \{1\}$ (coarsest scale)
4. **Prediction:** 
   - SMAK-S: Form $\hat{C}_t(X_t) = \{y : s(X_t, y) \leq q_t^{(k^*)}(X_t)\}$ where $k^* = \arg\max_{k \in \mathcal{A}_t} n_t^{(k)}(X_t)$
   - SMAK-I: Form $\hat{C}_t(X_t) = \bigcap_{k \in \mathcal{A}_t} \{y : s(X_t, y) \leq q_t^{(k)}(X_t)\}$ with calibrated thresholds
   - SMAK-W: Form $\hat{C}_t(X_t) = \{y : s(X_t, y) \leq \sum_{k \in \mathcal{A}_t} \tilde{w}_t^{(k)} q_t^{(k)}(X_t)\}$ with normalized weights
5. **Observe** $Y_t$ and compute $S_t = s(X_t, Y_t)$
6. **Update quantiles:** For each $k$, update $q_t^{(k)}(X_t)$ via weighted quantile regression
7. **Update bandwidth (if $t > T_0$):** Compute coverage discrepancies and update $h_{t+1}^{(1)}$ with regularized learning rate $\eta_t = \eta_0 \cdot \min(1, \sqrt{T_0/t})$

---

## Theoretical Analysis

### Coverage Guarantees Under Distribution Drift

Our theoretical framework does **not** assume exchangeability. Instead, we work within the weighted conformal prediction framework [Barber et al., 2023; Gibbs & Candès, 2024] that allows for distribution drift through likelihood ratio weighting.

**Assumption 1 (Bounded Distribution Drift).** The data stream satisfies a bounded drift condition: for any time $t$, the likelihood ratio between the current distribution $P_t$ and a reference distribution $P_0$ satisfies $\mathbb{E}_{P_t}[(dP_t/dP_0)^2] \leq M < \infty$.

**Theorem 1 (Marginal Coverage Under Drift).** Under Assumption 1 and with the weighted conformal framework, SMAK-CP with single-scale selection satisfies for all $T > T_0$:

$$\frac{1}{T-T_0}\sum_{t=T_0+1}^{T} \mathbb{P}(Y_t \in \hat{C}_t(X_t)) \geq 1 - \alpha - \epsilon_T$$

where $\epsilon_T = O(\sqrt{(\log T)/T} + \sqrt{(T-T_0)^{-1}\sum_{t=T_0+1}^T \|P_t - P_{t-1}\|_{TV}})$ accounts for both statistical error and cumulative drift.

*Proof sketch:* The result follows from combining the weighted exchangeability argument [Barber et al., 2023] with online learning bounds for the bandwidth adaptation. The key insight is that the weighted quantile regression naturally downweights past observations according to their relevance under drift.

### Local Coverage Approximation

**Theorem 2 (Approximate Local Coverage).** For any test point $x$ in a region with local density $\mu(x)$, if the bandwidth adapts such that $h_t^{(k^*)}(x) \asymp \mu(x)^{-1/d}$ (where $d$ is the dimension and $k^*$ the selected scale), then under smoothness conditions on the conditional distribution:

$$\mathbb{P}(Y_t \in \hat{C}_t(X_t) \mid X_t \in B(x, \delta)) \geq 1 - \alpha - O(\delta + h_t^{(k^*)} + \sqrt{(\mu(x) h_t^{(k^*)d})^{-1}})$$

where $B(x, \delta)$ is a ball of radius $\delta$ around $x$. The third term reflects the bias-variance tradeoff in local coverage estimation.

### Regret Bound for Bandwidth Adaptation

**Theorem 3 (Regret Bound).** Let $h^*(x)$ be the oracle optimal bandwidth at location $x$ (minimizing local prediction set size while maintaining coverage). Under Lipschitz continuity of the coverage as a function of bandwidth and with the regularized online adaptation:

$$\sum_{t=T_0+1}^{T} \mathbb{E}[|\Delta_t|] \leq O(\sqrt{(T-T_0)\log T} + \sum_{t=T_0+1}^{T} \|P_t - P_{t-1}\|_{TV})$$

where $\Delta_t$ is the coverage discrepancy at time $t$. The second term captures the price of distribution drift.

---

## Related Work and Differentiation

### Comparison to MOCP [Hajihashemi & Shen, 2024]

MOCP maintains multiple ACI instances with different miscoverage levels and uses exponential weighting for model selection. Our approach differs fundamentally:

| Aspect | MOCP | SMAK-CP |
|--------|------|---------|
| Structure | Multiple independent models | Single model, multi-scale kernels |
| Selection criterion | Historical miscoverage | Local data density + coverage |
| Computation | Evaluates all models | Single kernel evaluation per scale |
| Adaptation | Model-level | Bandwidth-level within scales |

**Key distinction:** MOCP selects among diverse models; SMAK-CP adapts the kernel geometry of a single model to local data structure.

### Comparison to SpeedCP [Liu et al., 2025]

SpeedCP accelerates RKHS-based conditional conformal prediction through path-tracing algorithms. While both use kernel methods:

- SpeedCP is a **batch method** with fixed bandwidth; SMAK-CP is **streaming** with adaptive bandwidth
- SpeedCP optimizes computation for a single scale; SMAK-CP maintains **multi-scale hierarchy**
- SpeedCP requires offline hyperparameter tuning; SMAK-CP adapts online

### Comparison to RLCP [Hore & Barber, 2023]

RLCP provides randomly-localized conformal prediction with theoretical local coverage guarantees. Our approach extends this:

- RLCP uses a **fixed bandwidth**; SMAK-CP adapts bandwidth online
- RLCP operates in **batch settings**; SMAK-CP handles **streaming data** with distribution drift
- RLCP uses a **single kernel scale**; SMAK-CP maintains **explicit multi-scale structure**

### Comparison to Multi-Scale CP [Baheri & Shahbazi, 2025]

Baheri & Shahbazi propose multi-scale conformal prediction through set intersection:

- Their scales are **predefined and fixed**; SMAK-CP **adapts scale bandwidths online**
- Their framework is **batch**; SMAK-CP is **streaming**
- They focus on **classification**; SMAK-CP targets **regression with distribution drift**

### Comparison to Zhang et al. 2021 [IEEE TNNLS]

Zhang et al. propose online kernel learning with adaptive bandwidth using an optimal control approach. While both address adaptive bandwidth in online settings:

| Aspect | Zhang et al. 2021 | SMAK-CP |
|--------|-------------------|---------|
| **Problem setting** | Kernel regression/prediction | Conformal prediction (uncertainty quantification) |
| **Adaptation target** | Prediction accuracy (MSE) | Coverage validity + set efficiency |
| **Adaptation mechanism** | Optimal control (state feedback) | Coverage discrepancy feedback |
| **Multi-scale structure** | Single adaptive bandwidth | Hierarchy of K adaptive scales |
| **Theoretical guarantees** | Convergence of predictions | Coverage guarantees under drift |
| **Distribution shift** | Not explicitly addressed | Core part of framework |

**Key distinction:** Zhang et al. focus on improving point predictions through optimal control-based bandwidth adaptation. SMAK-CP addresses a fundamentally different problem—constructing valid prediction sets with guaranteed coverage. The multi-scale hierarchy, calibrated intersection strategies, and explicit handling of distribution drift are unique to SMAK-CP.

**Why SMAK-CP is complementary:** Zhang et al.'s method could be used as the underlying point predictor within SMAK-CP, with our framework providing the uncertainty quantification layer. Our coverage-driven adaptation is designed specifically for conformal prediction's unique requirements (maintaining valid coverage while minimizing set size), which differ from the prediction accuracy objectives of standard kernel learning.

### Comparison to Bhatnagar et al. [ICML 2023]

Bhatnagar et al. propose improved online conformal prediction via strongly adaptive online learning (SAOL), achieving regret bounds that adapt to interval length:

- They focus on **single-scale** adaptation of the miscoverage level $\alpha_t$
- SMAK-CP adapts **kernel bandwidth** across **multiple scales** simultaneously
- Their framework is **model-agnostic**; SMAK-CP exploits **kernel structure** for local adaptation

The approaches are complementary: SAOL provides strong theoretical foundations for online adaptation of miscoverage parameters, while SMAK-CP provides geometric adaptivity through multi-scale kernels. Combining both (adaptive $\alpha_t$ at each scale) is a promising future direction.

---

## Experiments

### Experimental Setup

**Datasets:**
1. **Synthetic Heteroscedastic:** $Y = \sin(X) + \sigma(X) \cdot \epsilon$ with varying noise level
2. **Synthetic Distribution Drift:** Piecewise stationary with abrupt shifts every 500 steps
3. **Real-world:** UCI regression datasets with temporal structure (air quality, energy, bike sharing)

**Baselines:**
- Split Conformal (fixed)
- ACI [Gibbs & Candès, 2021]
- SAOL [Bhatnagar et al., 2023]
- MOCP [Hajihashemi & Shen, 2024]
- RLCP [Hore & Barber, 2023]
- Fixed-bandwidth kernel CP (for ablation)

**Metrics:**
- Marginal coverage (target: 0.9)
- Conditional coverage: MSCE (Marginal coverage within bins) and WSC (Worst-slab coverage)
- Average prediction set width (efficiency)
- Runtime per prediction

### Expected Results

We anticipate:

1. **Coverage:** SMAK-CP maintains valid marginal coverage (~0.9) comparable to MOCP and better than fixed-bandwidth methods under drift
2. **Local adaptivity:** Lower MSCE and higher WSC than ACI and single-scale methods
3. **Efficiency:** Narrower prediction sets than RLCP with fixed bandwidth, especially in heterogeneous regions
4. **Speed:** Faster than MOCP (no model ensemble evaluation), comparable to RLCP

### Ablation Studies

1. **Scale selection strategies:** Compare SMAK-S, SMAK-I, SMAK-W
2. **Adaptation mechanisms:** Fixed vs. adaptive bandwidth
3. **Cold-start strategies:** With vs. without hierarchical warm-up
4. **Number of scales:** Impact of $K$ on coverage and efficiency
5. **Adaptation speed:** Effect of learning rate $\eta$ on convergence

---

## Success Criteria

### Confirming Evidence

The hypothesis would be confirmed by:
1. SMAK-CP achieves target marginal coverage (0.9 ± 0.02) across all datasets
2. Significantly lower MSCE than fixed-bandwidth methods (p < 0.05, paired t-test)
3. Prediction sets are 10-30% tighter than RLCP with fixed bandwidth in heterogeneous regions
4. Runtime is within 2× of single-scale methods, significantly faster than MOCP
5. Cold-start strategy reduces coverage violations by >50% in first 200 steps

### Refuting Evidence

The hypothesis would be refuted if:
1. Online bandwidth adaptation fails to converge (coverage oscillates)
2. Multi-scale structure provides no benefit over single-scale with well-tuned bandwidth
3. Adaptation overhead makes method impractical (runtime > 5× single-scale)
4. SMAK-I intersection produces overly conservative sets despite calibration

---

## References

1. Isaac Gibbs, John Cherian, and Emmanuel Candès. "Conformal Prediction with Conditional Guarantees." *arXiv:2305.12616*, 2023.

2. Yating Liu, So Won Jeong, Zixuan Wu, and Claire Donnat. "SpeedCP: Fast Kernel-based Conditional Conformal Prediction." *arXiv:2509.24100*, 2025.

3. Leying Guan. "Localized Conformal Prediction: A Generalized Inference Framework for Conformal Prediction." *Biometrika*, 110(1):33-50, 2023.

4. Rohan Hore and Rina Foygel Barber. "Conformal Prediction with Local Weights: Randomization Enables Robust Guarantees." *Journal of the Royal Statistical Society Series B*, qkae103, 2024.

5. Ali Baheri and Marzieh Amiri Shahbazi. "Multi-Scale Conformal Prediction: A Theoretical Framework with Coverage Guarantees." *arXiv:2502.05565*, 2025.

6. Erfan Hajihashemi and Yanning Shen. "Multi-model Ensemble Conformal Prediction in Dynamic Environments." *Advances in Neural Information Processing Systems (NeurIPS)*, 37:118678-118700, 2024.

7. Isaac Gibbs and Emmanuel Candès. "Adaptive Conformal Inference Under Distribution Shift." *Advances in Neural Information Processing Systems (NeurIPS)*, 34:1660-1672, 2021.

8. Vladimir Vovk, Alex Gammerman, and Glenn Shafer. "Algorithmic Learning in a Random World." *Springer*, 2005.

9. Aadyot Bhatnagar, Huan Wang, Caiming Xiong, and Yu Bai. "Improved Online Conformal Prediction via Strongly Adaptive Online Learning." *International Conference on Machine Learning (ICML)*, 202:2337-2363. PMLR, 2023.

10. Anastasios Angelopoulos and Stephen Bates. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *Foundations and Trends in Machine Learning*, 2023.

11. Jiaming Zhang, Hanwen Ning, Xingjian Jing, and Tianhai Tian. "Online Kernel Learning With Adaptive Bandwidth by Optimal Control Approach." *IEEE Transactions on Neural Networks and Learning Systems*, 32(5):1920-1934, 2021.

12. Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani. "Conformal Prediction Beyond Exchangeability." *The Annals of Statistics*, 51(2):816-845, 2023.

13. Isaac Gibbs and Emmanuel J. Candès. "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts." *Journal of Machine Learning Research*, 25(162):1-36, 2024.
