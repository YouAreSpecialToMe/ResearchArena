# Research Proposal: Comparative Analysis of Adaptation Criteria for Gradient-Based Discrete MCMC: When Acceptance-Rate Trumps Jump-Distance

## 1. Introduction

### 1.1 Context and Problem Statement

Discrete Markov Chain Monte Carlo (MCMC) methods are fundamental to Bayesian inference in discrete spaces, with critical applications in statistical physics (Ising models), energy-based models, and combinatorial optimization. Recent gradient-based samplers—Gibbs-With-Gradients (GWG) [Grathwohl et al., 2021], Norm-Constrained Gradient (NCG) [Rhodes & Gutmann, 2022], and Discrete Langevin Monte Carlo (DLMC) [Zhang et al., 2022]—have improved efficiency but require hyperparameter tuning.

Three distinct adaptive approaches have emerged:
- **Rhodes & Gutmann [2022]**: Adaptive step-size via jump distance maximization (Appendix E.1)
- **Sun et al. [2022]**: Adaptive LBP (ALBP) targeting ρ* = 0.574 acceptance rate via stochastic approximation
- **Sun et al. [2023a]**: Any-Scale Balanced (AB) sampler maximizing empirical jump distance
- **Pynadath et al. [2024]**: Automatic Cyclical Scheduling (ACS) with acceptance-rate-based tuning

**The Gap**: While these methods exist, **no systematic comparison** evaluates which adaptation criterion works best under what conditions. Each paper evaluates on different problems with different metrics, leaving practitioners without guidance. Moreover, **failure modes are poorly understood**: when do these adaptive methods fail, and why?

### 1.2 Key Insight and Hypothesis

**Key Insight**: The relative performance of acceptance-rate vs. jump-distance adaptation depends critically on problem characteristics:
- **High-dimensional product distributions**: Both criteria may work (theory predicts 0.574 optimal)
- **Critical temperature/phase transitions**: Jump-distance adaptation becomes unstable (as noted by Sun et al. [2023a])
- **Multimodal distributions**: Fixed-target adaptation may trap in local modes

**Core Hypothesis**:
> Acceptance-rate-based adaptation (ALBP) provides superior robustness compared to jump-distance maximization (AB-sampler) for gradient-based discrete MCMC, particularly in high-dimensional correlated distributions, but both methods fail near phase transitions where local correlation structure invalidates theoretical assumptions. Cyclical scheduling (ACS) provides advantages for multimodal distributions where fixed-target adaptation may trap in local modes.

### 1.3 Proposed Contribution

We provide a **systematic empirical comparison** addressing:

1. **Head-to-head comparison**: ALBP [Sun et al., 2022] vs. AB-sampler [Sun et al., 2023a] vs. ACS [Pynadath et al., 2024] under unified framework
2. **Additional baselines**: Fixed-step GWG and a simple hand-tuned grid-search baseline to contextualize adaptive methods
3. **Failure mode characterization**: When and why does each adaptation criterion fail? Explicit analysis of phase transition behavior
4. **Pilot validation**: Small-scale experiments validating theoretical predictions before full benchmark
5. **CPU-focused benchmark**: Practical wall-clock comparison for practitioners (using same problem instances as DISCS for direct comparability)

**Explicit Non-Claim**: We do NOT claim to be the first to propose acceptance-rate-based adaptation—Sun et al. [2022] already proposed ALBP with Eq. 22. Our contribution is **comparative analysis and failure characterization**.

---

## 2. Proposed Approach

### 2.1 Background: Existing Adaptive Methods

#### 2.1.1 ALBP: Acceptance-Rate Targeting [Sun et al., 2022]

Sun et al. [2022] established that for locally-balanced proposals with single-coordinate updates, the optimal acceptance rate is ρ* = 0.574. They proposed ALBP with update rule:

$$R_{t+1} \leftarrow R_t + \eta_t(a_t - \delta)$$

where $a_t$ is acceptance probability, $\delta = 0.574$ is target. This is a Robbins-Monro stochastic approximation that drives the empirical acceptance rate toward the theoretically optimal value.

**Strengths**: Theoretically grounded; works well for product distributions  
**Limitations (acknowledged in paper)**: Theory assumes product distributions; behavior near phase transitions is unknown

#### 2.1.2 AB-Sampler: Jump-Distance Maximization [Sun et al., 2023a]

The AB-sampler adaptively tunes step size $\sigma$ and balancing parameter $\alpha$ to maximize empirical jump distance:

$$(\sigma, \alpha) = \arg\max_{\sigma, \alpha} \mathbb{E}[\|x' - x\|]$$

**Strengths**: Joint optimization of multiple parameters; uses second-order information  
**Limitations**: Jump distance is a proxy for mixing; the paper notes adaptation is "not stable until the Markov chain reaches its stationary distribution"

#### 2.1.3 ACS: Cyclical Acceptance-Rate Targeting [Pynadath et al., 2024]

ACS uses cyclical schedules with automatic tuning targeting a specific acceptance rate $\rho^*$:

**Strengths**: Designed for multimodal distributions; non-asymptotic convergence guarantees  
**Limitations**: Adds cyclical complexity; unclear when this is necessary vs. monotonic adaptation

### 2.2 Additional Baselines for Contextualization

#### 2.2.1 Fixed-Step GWG [Grathwohl et al., 2021]

Standard Gibbs-With-Gradients with fixed step size provides a baseline showing the value of adaptation. We use step size σ = 0.1 as reported in the original paper.

#### 2.2.2 Grid-Search Heuristic Baseline

A simpler adaptive approach: run short pilot chains with step sizes σ ∈ {0.01, 0.05, 0.1, 0.2, 0.5}, select best by empirical jump distance, then freeze. This tests whether complex online adaptation is necessary or if simple heuristics suffice.

### 2.3 Critical Analysis: When Does Each Method Fail?

#### Failure Mode 1: Phase Transitions (All Methods)

Roberts & Rosenthal [2001] note that optimal scaling theory "fails when the target distribution has phase-transition behavior." At critical temperature, correlation length diverges, and local approximations break down. We hypothesize:
- **ALBP**: Target rate 0.574 may be suboptimal or unattainable
- **AB-sampler**: Jump distance maximization becomes unstable due to critical slowing down
- **ACS**: Cyclical scheduling may provide some robustness

#### Failure Mode 2: Pre-Convergence Adaptation (AB-Sampler)

Sun et al. [2023a] explicitly note: "the adaptation is not stable until the chain reaches stationary distribution." This creates a chicken-and-egg problem: you need good parameters to reach stationarity, but you need stationarity to adapt parameters.

#### Failure Mode 3: Multimodality (ALBP, AB-Sampler)

Both ALBP and AB-sampler use monotonic adaptation to fixed targets. For multimodal distributions, this may:
- Converge to parameters optimal for one mode
- Fail to explore other modes
- ACS addresses this via cyclical schedules

### 2.4 Interaction Between Adaptation Criteria and Design Choices

The effectiveness of adaptation criteria depends on other design choices:

**Balancing Function Type**: The choice between Barker (\sqrt{t}/(1+\sqrt{t})) and Tanh balancing affects optimal acceptance rates. ALBP assumes Barker balancing; AB-sampler adapts the balancing function itself.

**Local vs. Global Proposals**: ALBP uses single-coordinate updates (local); AB-sampler can use block updates (more global). This interacts with jump-distance measurement.

**Warmup Strategy**: Different adaptation methods require different warmup budgets. We evaluate adaptation stability during warmup, not just final performance.

### 2.5 Our Approach: Systematic Comparison

We do NOT propose a new algorithm. Instead, we:

1. **Implement all methods** under unified framework (same codebase, same evaluation metrics)
2. **Design targeted experiments** to expose failure modes:
   - **Phase 1 (Pilot)**: Validate theoretical predictions on tractable problems
   - **Phase 2 (Comparison)**: Head-to-head on Ising models (varying temperature)
   - **Phase 3 (Failure Analysis)**: Critical temperature, multimodality, high correlation
3. **Measure robustness**: Performance variance across random seeds, initializations
4. **Statistical rigor**: Effect sizes and power analysis for comparing methods

---

## 3. Related Work

### 3.1 Gradient-Based Discrete Sampling

**Grathwohl et al. [2021]** introduced GWG, the first gradient-based discrete sampler using locally-balanced proposals. **Zanella [2020]** provided the theoretical framework for locally-informed proposals.

### 3.2 Adaptive Discrete Sampling (Our Comparison Targets)

**Sun et al. [2022]** (NeurIPS): Established optimal acceptance rate ρ* = 0.574 for locally-balanced proposals and proposed ALBP with stochastic approximation adaptation. This is the most directly relevant prior work—we explicitly acknowledge their priority on acceptance-rate-based adaptation.

**Sun et al. [2023a]** (ICLR): Proposed AB-sampler with jump-distance maximization. They note limitations: adaptation unstable until stationarity, not suitable for contrastive divergence learning.

**Pynadath et al. [2024]** (NeurIPS): Proposed ACS for multimodal distributions with cyclical schedules and acceptance-rate-based tuning.

**Rhodes & Gutmann [2022]** (TMLR): Included adaptive step-size via jump distance maximization in Appendix E.1.

### 3.3 DISCS Benchmark [Goshvadi et al., 2023]

DISCS provides a unified framework for comparing discrete samplers. **Our CPU benchmark uses the same problem instances as DISCS** (Lattice Ising, Random Ising, RBM) for direct comparability, but with CPU-focused wall-clock measurements and evaluation of adaptive variants not included in the original benchmark.

### 3.4 Positioning Summary

| Aspect | Sun et al. [2022] | Sun et al. [2023a] | Pynadath et al. [2024] | **This Work** |
|--------|-------------------|-------------------|----------------------|---------------|
| **Criterion** | Acceptance rate | Jump distance | Cyclical + acceptance rate | **Comparison of all three** |
| **Theory** | Optimal rate ρ* = 0.574 | None explicit | Non-asymptotic convergence | **Failure mode analysis** |
| **Contribution** | New algorithm | New algorithm | New algorithm | **Systematic comparison** |
| **Scope** | Single method | Single method | Single method | **Unified benchmark** |
| **Baselines** | Fixed step | Fixed step | Fixed step | **+ Grid-search heuristic** |
| **Statistics** | Point estimates | Point estimates | Point estimates | **Effect sizes, power analysis** |

**Our Position**: We do not claim algorithmic novelty. Our contribution is empirical: the first systematic comparison of adaptive criteria, with explicit failure mode characterization and statistical rigor.

---

## 4. Experiments

### 4.1 Phase 1: Pilot Validation (1 hour)

**Goal**: Validate theoretical predictions on small tractable problems before large-scale experiments

**Experiments**:
1. **Reproduce Sun et al. [2022] Figure 5**: Verify optimal acceptance rate 0.574 on Bernoulli model
2. **Pilot comparison**: ALBP vs. AB-sampler on 100D Gaussian with ground truth
   - Verify ESS peaks at predicted acceptance rate
   - Measure adaptation convergence time

**Success Criteria**:
- Reproduce key results from Sun et al. [2022] within ±0.05 acceptance rate
- Both adaptive methods converge to near-optimal parameters
- Pilot confirms feasibility of larger experiments

### 4.2 Phase 2: Head-to-Head Comparison (2.5 hours)

**Goal**: Compare adaptation criteria under identical conditions with statistical rigor

**Problems** (same instances as DISCS for comparability):
1. **Lattice Ising (100-900 variables)**: 
   - Below critical (T > Tc): Weak correlations
   - At critical (T = Tc): Strong long-range correlations
   - Above critical (T < Tc): Ordered phase
2. **Random Ising (100-500 variables)**: Frustrated interactions, glassy landscapes

**Methods Compared**:
- ALBP (acceptance-rate targeting, ρ* = 0.574)
- AB-sampler (jump-distance maximization)
- ACS (cyclical scheduling)
- Fixed-step GWG (σ = 0.1 baseline)
- Grid-search heuristic (pilot + freeze)

**Metrics**:
- ESS per second (wall-clock on CPU)
- Adaptation stability (parameter trajectories)
- Robustness (coefficient of variation across 10 seeds)

**Statistical Analysis**:
- **Effect size**: Cohen's d for ESS differences between methods
- **Statistical power**: Post-hoc power analysis for Mann-Whitney U tests
- **Meaningful difference**: >0.5 standard deviations in ESS (medium effect size)
- **Significance threshold**: p < 0.05 with Bonferroni correction for multiple comparisons

**Key Questions**:
- Does ALBP maintain near-0.574 acceptance rate across problem types?
- Does AB-sampler achieve higher jump distance but lower ESS?
- Which method is more robust to initialization?
- Does the simple grid-search heuristic match adaptive methods?

### 4.3 Phase 3: Failure Mode Analysis (2.5 hours)

**Goal**: Characterize when each method fails

**Experiment 1: Phase Transition Stress Test**
- Ising model at critical temperature (J = 0.44 for 2D lattice)
- Measure: acceptance rate trajectory, ESS, gradient variance
- **Expected finding**: Both methods struggle; acceptance rate deviates from 0.574

**Experiment 2: Pre-Convergence Adaptation**
- Start chains from "bad" initializations (far from typical set)
- Compare: ALBP vs. AB-sampler stability during burn-in
- **Expected finding**: AB-sampler more unstable early; ALBP more robust

**Experiment 3: Multimodality**
- Mixture of Gaussians in discrete space (2-10 modes)
- Compare: ALBP/AB-sampler (fixed targets) vs. ACS (cyclical)
- **Expected finding**: ACS discovers more modes; fixed-target methods may trap

### 4.4 Phase 4: CPU Wall-Clock Benchmark (1.5 hours)

**Goal**: Practical guidance for CPU-only practitioners

**Relationship to DISCS**: We use the **same problem instances** as Goshvadi et al. [2023] (Lattice Ising 400-variable, Random Ising 400-variable) for direct comparability. Where DISCS reports GPU wall-clock times, we report CPU wall-clock times. This allows practitioners to understand the performance gap and choose appropriate hardware.

**Implementation**: Pure NumPy/Numba (no JAX/GPU dependency)

**Measurements**:
- Total time = adaptation + sampling to target ESS
- Memory usage
- Scaling with dimension (100 → 2500 variables)

**Deliverables**:
- Performance tables by problem class
- Decision tree: "If your problem is X, use method Y"
- CPU vs GPU comparison showing where CPU is viable

### 4.5 Phase 5: Analysis and Documentation (0.5 hours)

**Deliverables**:
- Final report with all results
- Code release with reproducible scripts

---

## 5. Success Criteria

### 5.1 Confirmation Criteria

The hypothesis is **confirmed** if:

1. **ALBP robustness**: ALBP achieves ≥80% of optimal ESS on ≥70% of problems with <20% variance across seeds
   - *Rationale*: 80% threshold based on efficiency robustness (Sun et al. [2022] notes 0.5-0.7 acceptance rate range retains high efficiency); 70% coverage allows for known failure modes

2. **AB-sampler instability**: AB-sampler shows >2× higher variance in early adaptation phase compared to ALBP
   - *Rationale*: Validates the "unstable until stationarity" claim from Sun et al. [2023a]

3. **Phase transition failure**: Both methods show >50% performance degradation at critical temperature compared to off-critical
   - *Rationale*: Validates Roberts & Rosenthal [2001] warning about phase transitions

4. **Multimodality advantage**: ACS discovers ≥2× more modes than ALBP/AB-sampler on multimodal targets
   - *Rationale*: Demonstrates value of cyclical scheduling for multimodality

5. **Practical guidance**: Clear decision criteria emerge for method selection

### 5.2 Refutation Criteria

The hypothesis is **refuted** if:

1. **No clear winner**: Different methods perform best on different problems with no systematic pattern

2. **Jump-distance wins**: AB-sampler consistently outperforms ALBP across problem types

3. **Theory mismatch**: Empirically optimal acceptance rates differ systematically from 0.574 even on product distributions

### 5.3 Mixed Results Handling

**Expected outcome**: Results may be mixed—e.g., ALBP wins on correlated problems, AB-sampler on product distributions, ACS on multimodal targets.

**Framing**: Mixed results are still **valuable scientific contributions**:
- They provide nuanced guidance for practitioners
- They identify boundary conditions for theoretical predictions
- They motivate future theoretical work

**Success criteria for mixed results**:
- Clear characterization of which method works for which problem class
- Statistical evidence supporting the classification (not just anecdotal)
- Actionable decision criteria for practitioners

### 5.4 Statistical Justification of Thresholds

**Why 80% ESS threshold?** Based on Sun et al. [2022] Figure 4, efficiency remains within 80% of optimal for acceptance rates in [0.45, 0.7], suggesting 80% is a robustness threshold.

**Why 70% coverage?** Allows for known failure modes (phase transitions, multimodality) while still claiming general robustness.

**Effect sizes**: We use Cohen's d conventions: small (0.2), medium (0.5), large (0.8). A "meaningful" difference requires d > 0.5 (medium effect).

**Power analysis**: With n=10 seeds per condition, we have 80% power to detect medium effects (d=0.5) at α=0.05.

### 5.5 Novel Findings

Significant findings would include:
1. **Phase transition characterization**: Quantitative measure of how critical temperature affects adaptive methods
2. **Adaptation budget guidelines**: Minimum adaptation steps needed for stability
3. **Method selection criteria**: Clear problem characteristics predicting which method works best
4. **Grid-search comparison**: Whether simple heuristics match complex adaptive methods

---

## 6. Impact and Significance

### 6.1 Scientific Contribution

1. **First systematic comparison** of adaptive criteria for discrete MCMC
2. **Failure mode characterization** providing understanding of method limitations
3. **Empirical validation** (or refutation) of theoretical predictions from Sun et al. [2022]
4. **Statistical rigor** with effect sizes and power analysis in MCMC comparisons

### 6.2 Practical Impact

1. **For practitioners**: Evidence-based guidance on adaptive method selection
2. **For method developers**: Understanding of when to use which adaptation criterion
3. **For the field**: Resolution of fragmentation in adaptive discrete MCMC literature

### 6.3 Target Venues

- **Primary**: AISTATS, UAI (empirical methodology and comparison)
- **Secondary**: NeurIPS/ICML workshops (benchmarks, sampling)
- **Tertiary**: Journal of Computational and Graphical Statistics

---

## 7. References

1. Grathwohl, W., Swersky, K., Hashemi, M., Duvenaud, D., & Maddison, C. J. (2021). Oops I took a gradient: Scalable sampling for discrete distributions. *ICML 2021*.

2. Rhodes, B., & Gutmann, M. (2022). Enhanced gradient-based MCMC in discrete spaces. *TMLR*.

3. Zhang, R., Liu, X., & Liu, Q. (2022). A Langevin-like sampler for discrete distributions. *ICML 2022*.

4. Sun, H., Dai, H., & Schuurmans, D. (2022). Optimal scaling for locally balanced proposals in discrete spaces. *NeurIPS 2022*.

5. Sun, H., Dai, B., Sutton, C., Schuurmans, D., & Dai, H. (2023a). Any-scale balanced samplers for discrete space. *ICLR 2023*.

6. Pynadath, P., Bhattacharya, R., Hariharan, A., & Zhang, R. (2024). Gradient-based Discrete Sampling with Automatic Cyclical Scheduling. *NeurIPS 2024*.

7. Goshvadi, K., Sun, H., Liu, X., Nova, A., Zhang, R., Grathwohl, W., Schuurmans, D., & Dai, H. (2023). DISCS: A benchmark for discrete sampling. *NeurIPS 2023 Datasets and Benchmarks*.

8. Zanella, G. (2020). Informed proposals for local MCMC in discrete spaces. *JASA*.

9. Roberts, G. O., & Rosenthal, J. S. (2001). Optimal scaling for various Metropolis-Hastings algorithms. *Statistical Science*.

10. Andrieu, C., & Thoms, J. (2008). A tutorial on adaptive MCMC. *Statistics and Computing*.

---

## 8. Timeline and Resource Requirements

| Phase | Duration | Key Deliverable | Priority |
|-------|----------|-----------------|----------|
| Phase 1: Pilot Validation | 1 hour | Validation of theoretical predictions | **Essential** |
| Phase 2: Comparison | 2.5 hours | Head-to-head results across methods | **Essential** |
| Phase 3: Failure Analysis | 2.5 hours | Phase transition and robustness results | **Essential** |
| Phase 4: CPU Benchmark | 1.5 hours | Wall-clock performance tables | **Essential** |
| Phase 5: Documentation | 0.5 hours | Final report and code | **Essential** |

**Total Time**: ~8 hours (fits within budget)

**Computational Resources**: CPU-only (2 cores, 128GB RAM)

**Contingency Plan** (if time runs short):
- **Cut first**: Phase 4 CPU benchmark (reduce problem sizes/dimensions)
- **Keep**: Phase 1-3 (core scientific contribution)
- **Minimal viable**: Phase 1 + Phase 2 on subset of problems

**Risk Mitigation**: 
- Phase 1 pilot validates approach before committing to full experiments
- Grid-search heuristic is simpler to implement than full AB-sampler; provides fallback
- Prioritize Phase 2 (core comparison) over extensive failure analysis

---

## 9. Differentiation from Prior Work (Explicit Statement)

### What Sun et al. [2022] Already Did
- Proved optimal acceptance rate ρ* = 0.574 for locally-balanced proposals
- Proposed ALBP with stochastic approximation adaptation (Eq. 22)
- Validated on Bernoulli, Ising, FHMM, RBM

### What Sun et al. [2023a] Already Did
- Proposed AB-sampler with jump-distance maximization
- Used quadratic approximation and Gaussian integral trick
- Noted adaptation instability before stationarity

### What Pynadath et al. [2024] Already Did
- Proposed ACS with cyclical schedules for multimodality
- Provided non-asymptotic convergence guarantees
- Targeted acceptance rate ρ* = 0.5

### What We Do (Our Contribution)
1. **Systematic comparison**: First head-to-head evaluation of all three approaches
2. **Additional baselines**: Grid-search heuristic to contextualize adaptive methods
3. **Statistical rigor**: Effect sizes and power analysis
4. **Failure mode analysis**: Explicit characterization of when each method fails
5. **Mixed results handling**: Framework for interpreting non-binary outcomes
6. **DISCS comparability**: CPU benchmark using same problem instances

**We do NOT claim**: Novel adaptive algorithm, new theoretical optimal rates, or first adaptive discrete sampler. Our contribution is **understanding, comparison, and statistical rigor**.
