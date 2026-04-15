# CoSBC: Dependence-Specialized Enriched SBC with Symmetric Pooled Ranking

## Introduction

Simulation-based calibration (SBC) is exact under calibrated Bayesian inference because, conditional on each simulated dataset, the planted parameter draw is exchangeable with posterior draws. In practice, however, standard SBC is often run on coordinates or a few scalar summaries, so it can miss the specific failures that matter most for approximate posteriors: distorted covariance, broken pairwise dependence, and joint-tail errors.

This proposal does not claim a new calibration paradigm. The closest conceptual starting point is enriched SBC, where sensitivity is improved by choosing better, possibly data-dependent, test quantities. The narrower gap is this: there is not yet a dependence-specialized instantiation of enriched SBC that (i) uses a symmetric multivariate ranking rule rather than many separate scalar ranks, (ii) calibrates its global test with the same randomization machinery used by SBC, and (iii) is evaluated fairly against a feature-matched enriched-SBC baseline with the same summaries and the same aggregation/calibration pipeline.

The paper's main claim is therefore conditional and pre-registered. If the proposed pooled ranking beats the feature-matched enriched-SBC baseline by a clear margin, the contribution is a stronger dependence diagnostic within enriched SBC. If not, the contribution narrows to a careful negative or neutral result plus an exact-valid per-family construction and a matched empirical study clarifying when richer summaries alone already suffice.

The central hypothesis is that pooled symmetric multivariate ranking can turn dependence-focused feature banks into a more powerful and more localizable SBC diagnostic than ranking those same features one-by-one, while preserving exact finite-sample rank validity for each family and using only conditionally valid Monte Carlo randomization for the practical global test.

## Proposed Approach

### Setup

For replicate \(i \in \{1,\dots,R\}\):

1. Draw \((\theta_i^\star, y_i)\) from the prior predictive distribution, or from the posterior-SBC conditional-on-data construction in the fixed-data study.
2. Draw \(M\) approximate posterior samples \(\theta_{i1},\dots,\theta_{iM}\) from \(q(\theta \mid y_i)\).
3. Form the candidate set
   \[
   X_i = (X_{i0},\dots,X_{iM}) := (\theta_i^\star,\theta_{i1},\dots,\theta_{iM}).
   \]

Under exact inference, conditional on \(y_i\), the \(M+1\) candidates are exchangeable.

### Pooled Rank Transform

For each coordinate \(j\) and candidate \(r\), compute the randomized pooled rank \(\widetilde R_{irj}\) of \(X_{irj}\) among \(\{X_{isj}\}_{s=0}^M\). Define
\[
U_{irj} = \frac{\widetilde R_{irj}-1/2}{M+1},
\qquad
Z_{irj} = \Phi^{-1}(U_{irj}).
\]

This pooled transform is permutation-equivariant in the candidate labels, so exchangeability is preserved.

### Dependence Feature Families

The method uses two small, pre-registered feature families:

1. **Covariance family**
   \[
   \psi_{ir}^{\mathrm{cov}} = \{ Z_{ir,j} Z_{ir,k} : 1 \le j < k \le d \}.
   \]

2. **Tail family**
   \[
   \psi_{ir}^{\mathrm{tail}} =
   \big\{
   \mathbf{1}\{U_{ir,j}>\alpha,\ U_{ir,k}>\alpha\},
   \mathbf{1}\{U_{ir,j}<1-\alpha,\ U_{ir,k}<1-\alpha\}
   : 1 \le j < k \le d
   \big\},
   \]
   with \(\alpha=0.9\).

These families are deliberately narrow. The goal is not omnibus detection; it is a targeted dependence diagnostic for covariance and joint-tail misspecification.

### Symmetric Pooled Ranking

For family \(\ell \in \{\mathrm{cov}, \mathrm{tail}\}\), let \(\psi_{ir}^{(\ell)}\) denote candidate \(r\)'s family feature vector. Compute a leave-one-out energy-style score
\[
A_{ir}^{(\ell)} =
\frac{1}{M}\sum_{s \ne r}
\left\|
\psi_{ir}^{(\ell)}-\psi_{is}^{(\ell)}
\right\|_2.
\]

The planted statistic for replicate \(i\) and family \(\ell\) is the randomized rank of \(A_{i0}^{(\ell)}\) among \(\{A_{ir}^{(\ell)}\}_{r=0}^M\). Because the score is computed from the pooled candidate set and is permutation-equivariant, the planted rank remains exactly uniform under the SBC null.

This is the paper's only genuinely new mechanism. Everything else is intentionally conservative: dependence features come from enriched SBC intuition, and global testing uses standard randomization logic.

### Per-Family Validity Claim

**Claim 1.** For any measurable permutation-equivariant pooled transform, any measurable permutation-equivariant tuning rule, and any measurable permutation-equivariant score with uniform random tie-breaking, the randomized planted rank is exactly uniform on \(\{0,\dots,M\}\) under exchangeability.

This follows from the same symmetry argument as standard SBC. The proposal's theory contribution is modest: it packages that argument for pooled multivariate scores so the method is clearly positioned as a valid enriched-SBC instantiation.

### Practical Global Test

Let \(V_i^{(\ell)} \in (0,1)\) be the PIT-scaled rank for family \(\ell\). Define
\[
S_\ell = \sum_{i=1}^R \left(V_i^{(\ell)}-\frac{1}{2}\right)^2,
\qquad
T_{\max} = \max_{\ell \in \{\mathrm{cov},\mathrm{tail}\}} S_\ell.
\]

The practical p-value is computed by Monte Carlo randomization over independently resampled within-replicate candidate labels, using the standard plus-one correction
\[
\hat p = \frac{1 + \sum_{b=1}^B \mathbf{1}\{T_{\max}^{(b)} \ge T_{\max}^{\mathrm{obs}}\}}{B+1}.
\]

The main study does **not** rely on exhaustive global relabeling. Exhaustive enumeration is used only in one tiny implementation check with \(R=4\) and \(M=7\), where \((M+1)^R = 4096\) is actually tractable. All main experiments use Monte Carlo randomization and claim conditional validity, not exact exhaustive computation.

### Feature-Matched Enriched-SBC Baseline

The decisive comparison is a feature-matched enriched-SBC baseline implemented with the same ingredients except for pooled ranking:

1. Use the same pooled \(U\) and \(Z\) transforms.
2. Use the same covariance and tail feature bank.
3. Rank each scalar feature directly as an ordinary SBC test quantity.
4. Aggregate those scalar ranks within each family using the same quadratic score form and calibrate the family and global statistics with the same Monte Carlo relabeling scheme used by CoSBC.

This matching matters. If CoSBC wins only because its aggregation is better calibrated than the baseline's, the result is not interesting. The proposal therefore fixes aggregation and calibration across methods and isolates only one difference: pooled multivariate ranking versus many scalar ranks.

### Optional Secondary Score

One small ablation replaces the energy score with a Gaussian-kernel similarity score using the pooled median distance as bandwidth. This is not a second method; it is only a robustness check on the score choice.

## Related Work

Talts et al. established SBC as an exchangeability-based validation tool. Modrak et al. are the closest precursor, because they show that richer and data-dependent test quantities can substantially improve SBC sensitivity. CoSBC should therefore be positioned as a dependence-specialized enriched-SBC construction, not as a new framework.

Yao and Domke propose discriminative calibration, a broader multivariate diagnostic that often gains power through flexible classifiers but does not preserve SBC's exact rank-uniform interpretation. Säilynoja et al. extend SBC to the conditional-on-observed-data setting; the proposal reuses that framework for a realistic fixed-data study. Gretton et al. and Székely and Rizzo supply the kernel and energy discrepancy machinery used inside the pooled score. Aich and Aich motivate dependence-specific discrepancy design from a copula viewpoint. Wang et al. and Zhao and Marriott are also relevant because they target the practical evaluation of approximate posteriors with special attention to covariance structure and targeted functionals, reinforcing the motivation for a dependence-focused rather than omnibus diagnostic.

The novelty boundary is therefore clear. The paper is new only if the specific combination of dependence feature families, symmetric pooled ranking, and matched SBC-style calibration materially improves over feature-matched enriched SBC. Otherwise the work should be framed as a careful instantiation and benchmark rather than a methodological leap.

## Experiments

### Compute Budget

The full study is scoped for 2 CPU cores, 128GB RAM, and about 8 total hours:

- 3 seeds.
- Main settings: \(R=100\) replicates and \(M=24\) posterior draws.
- Small exact-check setting: \(R=4\), \(M=7\).
- Main Monte Carlo relabelings: \(B=199\).
- One stability ablation with \(B=999\) in a single small setting.

BLAS threads are pinned to one core per process. No experiment requires neural-network training.

### Study A: Gaussian Dependence Benchmarks

Use conjugate multivariate Gaussian models with exact posteriors so null validity is transparent and approximate posteriors are cheap to generate.

Settings:

1. **Toeplitz shrinkage benchmark** with \(d \in \{8,16\}\) and posterior correlation shrinkage \(\lambda \in \{0.25,0.5,0.75\}\).
2. **Single-pair corruption benchmark** with block-diagonal covariance, where one designated pair has its sign flipped or correlation set to zero.
3. **TailMix benchmark** where the approximation matches posterior mean and covariance but replaces the exact Gaussian with a symmetric scale mixture to induce radial and co-occurrence tail errors.

These studies test power and localization under controlled alternatives.

### Study B: Realistic Fixed-Data Posterior-SBC Case

Add one less-synthetic study using posterior SBC on a real dataset:

1. **Minnesota radon varying-intercept model** or an equivalently small hierarchical regression dataset with strong posterior dependence.
2. Use a well-mixed NUTS run as the reference posterior generator inside posterior SBC.
3. Compare against two practical approximations: mean-field ADVI and Laplace.

This study matters because closed-form Gaussian corruptions alone do not show that the diagnostic is useful in a real Bayesian workflow. A small hierarchical model is CPU-feasible, exhibits nontrivial dependence, and is exactly the kind of setting where coordinate-wise checks can fail while approximate posteriors still look superficially plausible.

### Baselines

All baselines use the same replicate budget and, where applicable, the same relabeling calibration:

1. **Scalar SBC** on coordinates plus Mahalanobis radius.
2. **Feature-matched enriched SBC** with the same covariance and tail features and the same randomization-calibrated family/global aggregation.
3. **Discriminative calibration** using a CPU-feasible logistic classifier on raw coordinates, pairwise products, and tail indicators.
4. **Standalone energy-distance check** on the same family feature vectors.

The main scientific comparison is CoSBC versus feature-matched enriched SBC. The other baselines provide context only.

### Metrics

The primary metrics are:

1. **Null calibration**: empirical rejection at levels \(0.05\) and \(0.1\).
2. **Power**: rejection probability on shrinkage, single-pair, TailMix, and realistic posterior-approximation alternatives.
3. **Localization**: top-1 and top-3 recovery of the corrupted pair in the single-pair benchmark.
4. **Runtime**: wall-clock seconds per method.

### Pre-Registered Decision Rule

The stronger novelty claim is made only if CoSBC beats the feature-matched enriched-SBC baseline by:

1. at least **0.08 absolute power** at level \(0.05\) on at least two synthetic dependence-only alternatives, and
2. at least **0.05 absolute power** on at least one realistic posterior-SBC alternative,

while both methods remain near nominal size under the null.

If that margin is not met, the paper explicitly drops the stronger claim and reframes the result as evidence that richer dependence summaries, rather than pooled ranking itself, explain most of the gain.

### Runtime Breakdown

Expected runtime on 2 CPU cores:

1. Null-validity and tiny exact-check study: 1.5 hours.
2. Synthetic power and localization benchmarks: 3.5 hours.
3. Realistic posterior-SBC radon study: 2.0 hours.
4. Discriminative baseline and kernel-score ablation: 0.7 hours.
5. Plotting, tables, and reruns of failed seeds: 0.3 hours.

Total: about 8 hours.

## Success Criteria

The proposal succeeds only if all of the following hold:

1. **Per-family validity**: randomized pooled ranks are uniform under exact inference, and the tiny \(R=4, M=7\) exhaustive check matches the Monte Carlo implementation.
2. **Fair baseline comparison**: CoSBC and feature-matched enriched SBC use the same feature bank and the same family/global relabeling calibration.
3. **Empirical usefulness**: the pre-registered power margin over feature-matched enriched SBC is met; otherwise the novelty claim is explicitly weakened.
4. **Real-world relevance**: at least one realistic fixed-data posterior approximation study shows the dependence diagnostic matters beyond conjugate Gaussian benchmarks.
5. **Budget compliance**: all experiments finish within the stated CPU-only budget.

The proposal is refuted if null calibration fails, if the matched enriched-SBC baseline ties CoSBC across settings, or if the realistic study shows no practical value beyond standard enriched SBC.

## References

1. Talts, S., Betancourt, M., Simpson, D., Vehtari, A., and Gelman, A. (2018). *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*. arXiv:1804.06788.
2. Modrák, M., Moon, A. H., Kim, S., Bürkner, P.-C., Huurre, N., Faltejskova, K., Gelman, A., and Vehtari, A. (2023). *Simulation-Based Calibration Checking for Bayesian Computation: The Choice of Test Quantities Shapes Sensitivity*. Bayesian Analysis.
3. Yao, Y. and Domke, J. (2023). *Discriminative Calibration: Check Bayesian Computation from Simulations and Flexible Classifier*. NeurIPS 2023.
4. Säilynoja, T., Schmitt, M., Bürkner, P.-C., and Vehtari, A. (2025). *Posterior SBC: Simulation-Based Calibration Checking Conditional on Data*. arXiv:2502.03279.
5. Wang, Y., Kasprzak, M., and Huggins, J. H. (2023). *A Targeted Accuracy Diagnostic for Variational Approximations*. AISTATS 2023.
6. Zhao, H. and Marriott, P. (2013). *Diagnostics for Variational Bayes Approximations*. arXiv:1309.5117.
7. Aich, A. and Aich, A. B. (2025). *Copula Discrepancy: Benchmarking Dependence Structure*. arXiv:2507.21434.
8. Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., and Smola, A. J. (2012). *A Kernel Two-Sample Test*. Journal of Machine Learning Research, 13, 723-773.
9. Székely, G. J. and Rizzo, M. L. (2013). *Energy Statistics: A Class of Statistics Based on Distances*. Journal of Statistical Planning and Inference, 143(8), 1249-1272.
