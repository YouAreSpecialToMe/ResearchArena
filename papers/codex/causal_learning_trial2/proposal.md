# PACER-Cert as a Benchmark for Stopping-Certificate Calibration in CPU-Only Active Causal Discovery

## Introduction

Active causal discovery already has adaptive online design, Bayesian finite-sample stopping, and decision-oriented intervention scoring. That prior art makes the original PACER-Cert framing too strong as a method paper. This proposal therefore repositions the work as an **empirical benchmark and analysis paper** about stopping-certificate calibration in a narrow but realistic regime:

- linear-Gaussian SEMs,
- perfect single-node interventions,
- small observational warm start,
- strict CPU-only execution,
- limited intervention budget.

The core question is:

> Can a cheap edge-level stopping certificate reliably predict when the remaining intervention budget is no longer worth spending on the unresolved edges?

PACER-Cert is the vehicle for that study, not the main novelty claim by itself. The paper's contribution is an empirical and diagnostic framework for testing whether a lightweight resolvability surrogate is calibrated, decision-useful, and trustworthy enough to support stopping.

The central hypothesis is deliberately falsifiable: if the surrogate only tracks generic edge ambiguity, or if it fails against direct Monte Carlo lookahead on small instances, then PACER-Cert should be reported as an untrustworthy heuristic rather than a useful certificate.

## Proposed Approach

### Study framing

The paper is positioned as a benchmark/analysis paper with one lightweight heuristic instantiation. The contribution is:

1. a formal decomposition of active-discovery policies into uncertainty representation, action scoring, and stopping rule,
2. a CPU-feasible stopping-certificate heuristic, PACER-Cert,
3. a calibration protocol that compares surrogate predictions against empirical continuation outcomes,
4. a trust diagnostic and a direct-lookahead ablation that can falsify the surrogate.

### PACER-Cert summary

PACER-Cert maintains a compact weighted set of plausible DAG particles, uses those particles to score candidate interventions, and stops when the remaining ambiguity appears non-resolvable under the remaining budget.

The machinery is intentionally lightweight:

- maximum `M = 24` DAG particles,
- observational initialization from bootstrap FGES CPDAGs and DAG extensions,
- linear-Gaussian SEM parameter fitting,
- single-node interventions only,
- batch sizes `m in {25, 50, 100}`.

### 1. Compact uncertainty representation

Given observational warm-start data `D_obs`:

1. Draw `B = 24` bootstrap resamples.
2. Run FGES on each resample to obtain a CPDAG.
3. Sample up to `R = 3` consistent DAG extensions per CPDAG.
4. Fit linear-Gaussian SEM parameters for each DAG.
5. Score each DAG with an interventional BIC-style criterion on all accumulated data.
6. Deduplicate and retain at most `M = 24` particles with normalized weights.

This is not claimed to approximate the full posterior accurately. It is a compressed uncertainty representation designed to expose disputed edge states cheaply.

### 2. Edge ambiguity statistics

For each unordered pair `e = {u, v}`, compute weighted particle mass on the three local states:

- `p_fwd(e)` for `u -> v`,
- `p_rev(e)` for `v -> u`,
- `p_0(e)` for no edge.

Define:

- ambiguity `A(e) = H(p_fwd, p_rev, p_0) / log 3`,
- eliminable mass `R(e) = 1 - (p_fwd^2 + p_rev^2 + p_0^2)`.

`A(e)` captures disagreement. `R(e)` captures how much particle mass would collapse if the edge became known.

### 3. Local resolvability surrogate

For disputed edge `e`, intervention target `I`, batch size `m`, and particle `G_k`:

1. Construct a local rival graph `G_k^{alt(e)}` by reversing `e` if acyclic, otherwise deleting `e`.
2. Restrict attention to a local variable set containing the endpoints of `e` plus their non-intervened Markov blanket neighborhood.
3. Under `do(I)`, compute the induced local Gaussian interventional distributions for `G_k` and `G_k^{alt(e)}`.
4. Form the local noncentral likelihood-ratio proxy

   `lambda_{k,e}(I,m) = 2m * KL(P^{do(I)}_{G_k} || P^{do(I)}_{G_k^{alt(e)}})`.

5. Map this to asymptotic power at significance level `alpha = 0.05`:

   `pow_{k,e}(I,m) = Pr[chi^2_1(lambda_{k,e}(I,m)) > chi^2_{1,0.95}]`.

Average over particles where the edge is unresolved:

`D(e | I,m) = sum_k \bar w_{k,e} pow_{k,e}(I,m)`.

`D(e | I,m)` is not a correctness guarantee. It is a surrogate estimate of how resolvable the edge appears under the candidate action.

### 4. Action scoring

Candidate intervention `(I, m)` receives score

`Score(I,m) = (1 / m) * sum_{e in E_amb} A(e) R(e) D(e | I,m)`.

This scoring rule is intentionally simple:

- uncertainty term `A(e)`,
- consequence term `R(e)`,
- detectability term `D(e | I,m)`.

An ablation with `D(e | I,m) = 1` tests whether detectability adds value beyond ambiguity.

### 5. Stopping certificate

For remaining budget `B_rem`, define edge-level best predicted resolvability

`q_e(B_rem) = max_{I, m <= B_rem} D(e | I,m)`.

Define total remaining budget-resolvable ambiguity mass

`C(B_rem) = sum_{e in E_amb} A(e) R(e) q_e(B_rem)`.

PACER-Cert stops when `C(B_rem) < epsilon_stop`.

For each unresolved edge the method reports:

- `(p_fwd, p_rev, p_0)`,
- `q_e(B_rem)`,
- best candidate action `(I*, m*)`,
- binary flag `unlikely-to-resolve-under-budget` when `q_e(B_rem) < tau`.

The intended interpretation is operational: the current policy predicts that further budget is unlikely to resolve these edges.

### 6. Trust diagnostic

The main soundness gap in the original draft was the lack of justification for when the local KL/power surrogate should be trusted. This proposal addresses that with a falsifiable diagnostic rather than an overstated theorem.

For a sampled decision state, define a small direct Monte Carlo lookahead estimate

`MC(e | I,m)`

by simulating continuation data from the true SCM, updating the particle set, and measuring the frequency with which the correct edge state becomes top-weighted.

The surrogate is considered trustworthy in that state only if:

1. `D(e | I,m)` and `MC(e | I,m)` agree in rank over candidate actions for the same edge,
2. their absolute discrepancy stays below a tolerance band on average,
3. high surrogate confidence is not concentrated in low effective-sample-size or high particle-collapse states.

This yields a concrete failure mode analysis:

- if trust degrades mainly when particle ESS is low, the certificate is a particle-degeneracy artifact;
- if trust degrades mainly on dense local neighborhoods, the locality assumption is suspect;
- if trust degrades everywhere, the certificate is not informative enough for stopping.

### 7. Minimal theoretical justification

The paper will not claim a full stopping guarantee. The defensible theoretical statement is narrower:

Under linear-Gaussian SEMs, fixed intervention target `I`, fixed local rival pair `(G, G^{alt})`, and correct local parameterization, the log-likelihood ratio for testing the two local models grows linearly with batch size `m`, with slope equal to the relevant interventional KL divergence. Therefore:

- ranking candidate actions by `KL / cost` is asymptotically aligned with ranking them by local test power per sample,
- `q_e(B_rem)` should be monotone nondecreasing in `B_rem` if the surrogate is internally coherent.

Both claims are empirically checkable. The proposal uses them as sanity conditions, not as headline theory.

## Related Work

### Positioning by design axis

The paper explicitly contrasts prior work along three axes:

1. stopping rule,
2. action scoring,
3. uncertainty representation.

| Method | Stopping rule | Action scoring | Uncertainty representation | Role in this paper |
|---|---|---|---|---|
| AOED (Elahi et al., 2024) | Track-and-stop style online termination under finite samples | Bayesian online expected information gain / adaptive design objective | Bayesian posterior over graphs/interventional outcomes | Mandatory baseline for stopping behavior |
| Sample Efficient Bayesian Learning (Zhou et al., 2024) | Explicit finite-sample Bayesian stopping | Bayesian sample-efficient acquisition tied to posterior uncertainty | Bayesian graph posterior | Main novelty blocker; used for positioning, not required implementation |
| Bayesian Intervention Optimization (Wang et al., 2024) | No edge-level stopping certificate; decision objective is intervention selection | Bayesian decision-oriented optimization for globally decisive interventions | Bayesian uncertainty with optimization over interventions | Main action-scoring comparator in concept |
| GIT (Olko et al., 2023) | No explicit budget-aware stopping certificate | Gradient-based intervention targeting | Parameter-gradient sensitivity inside a discovery framework | Lightweight empirical baseline |
| PACER-Cert | Edge-level predicted resolvability threshold `C(B_rem) < epsilon_stop` | Ambiguity x eliminable mass x predicted local power per sample | Capped weighted DAG particle set | Heuristic under study |

### What is and is not novel

This proposal does **not** claim:

- the first active causal discovery method under sample budgets,
- the first stopping rule for active discovery,
- the first decision-oriented intervention scoring rule.

The narrower claim is:

- prior methods do not study **edge-level stopping-certificate calibration** as the main object of interest,
- prior methods do not provide a CPU-feasible benchmark that asks whether a cheap resolvability surrogate is actually informative enough to justify stopping.

If experiments show that PACER-Cert does not offer materially better calibration than simple ambiguity thresholds, the resulting paper should still be written as a negative-result benchmark/analysis paper rather than defended as a new method.

## Experiments

### Main goal

The main empirical question is:

> Does detectability-aware stopping provide calibrated and decision-useful stopping signals beyond generic graph ambiguity?

### Feasible scope

All experiments are scoped for:

- 2 CPU cores,
- 128 GB RAM,
- roughly 8 hours total runtime,
- no GPU,
- no neural-network training.

### Data-generating process

All core experiments use synthetic linear-Gaussian SEMs with:

- graph sizes `p in {10, 15}`,
- graph families `{Erdos-Renyi, scale-free}`,
- edge-weight regimes `{weak: [0.10, 0.25], mixed: [0.10, 0.80]}`,
- observational warm start of 200 samples,
- maximum interventional budget of 300 samples,
- allowed intervention batches `m in {25, 50, 100}`,
- 3 SCM seeds per setting.

This yields `2 x 2 x 2 x 3 = 24` core SCM instances.

### Methods

The mandatory empirical set is:

1. `FGES-only`: no interventions.
2. `Random active`: random target, fixed batch size 50, stop only at budget exhaustion.
3. `PACER-no-D`: same particles and stopping scaffold but `D(e | I,m) = 1`.
4. `PACER-Cert`: full heuristic.
5. `GIT`: lightweight practical acquisition baseline.
6. `AOED-lite`: implementation or simplified reproduction of AOED sufficient to preserve its online stopping and acquisition logic on the same small synthetic setting.

AOED is mandatory because the paper is about stopping. Sample Efficient Bayesian Learning and Bayesian Intervention Optimization remain comparison targets in the related-work analysis and discussion, but they are not mandatory implementations under the CPU/time budget.

### Experiment 1: Stopping-certificate calibration

This is the centerpiece experiment.

Protocol:

1. Build an auxiliary set of small `p = 8` SCMs from both graph families.
2. Sample decision states from rollouts of random active, GIT, AOED-lite, and PACER-no-D.
3. For each sampled state and candidate `(e, I, m)`, estimate empirical resolvability using **40 continuation simulations** from the true SCM.
4. Empirical resolvability is the fraction of continuations in which the correct edge state becomes top-weighted after updating the particle set.
5. Compare `D(e | I,m)` against this empirical target.

Metrics:

- Brier score,
- expected calibration error,
- AUROC,
- Spearman rank correlation.

### Experiment 2: Stopping-label prediction

For a state with remaining budget `B_rem`, define oracle label

`y_e^*(B_rem) = 1`

iff there exists an allowed `(I,m)` with `m <= B_rem` such that the empirical resolvability from 40 continuation simulations is at least `0.8`.

Evaluate whether `q_e(B_rem)` predicts this label.

Metrics:

- AUROC of `q_e(B_rem)`,
- Brier score,
- precision/recall of the binary `unlikely-to-resolve-under-budget` flag,
- calibration plots stratified by graph family and weight regime.

### Experiment 3: Direct-lookahead ablation

To test whether the surrogate is informative rather than merely correlated with ambiguity, run a stronger ablation on a tiny subset of the auxiliary `p = 8` instances:

1. replace `D(e | I,m)` in the action score with direct Monte Carlo lookahead `MC(e | I,m)` estimated from a small number of continuation simulations,
2. compare intervention rankings and stopping decisions against PACER-Cert,
3. measure how often surrogate-selected and lookahead-selected actions agree.

The subset is intentionally tiny so runtime remains feasible. The point is diagnostic, not leaderboard performance.

Outcomes:

- if PACER-Cert tracks direct lookahead closely, the surrogate is substantively informative;
- if not, the paper should state that its usefulness comes only from cheap correlation with ambiguity.

### Experiment 4: Discovery efficiency

Measure practical discovery performance on the 24 core instances.

Metrics:

- SHD versus interventional samples,
- directed-edge F1 versus samples,
- area under directed-edge-F1 versus budget,
- fraction of runs that stop before budget exhaustion,
- residual ambiguity mass at stop time,
- wall-clock time per rollout.

Primary comparison questions:

1. Does PACER-Cert outperform `PACER-no-D` in area under directed-edge-F1 versus budget?
2. Does PACER-Cert stop earlier than GIT and AOED-lite on weak-edge regimes without materially harming final F1?
3. Are gains explained by better stopping rather than by trivial under-spending?

### Experiment 5: Trust-diagnostic stratification

Analyze surrogate failures by state descriptors:

- particle effective sample size,
- number of unresolved edges,
- local neighborhood size around the queried edge,
- graph family,
- weak versus mixed edge-weight regime.

This experiment turns the diagnostic into a paper result: reviewers can see when the certificate should and should not be trusted.

### Optional external replay

If time remains after the core benchmark, include a small perturbation replay on Sachs as a sanity check for stopping behavior. This is explicitly optional and not part of the success criteria.

## Success Criteria

Because this is an analysis paper, success is defined first by calibration and diagnostic value, then by discovery performance.

The proposal is supported if:

1. `q_e(B_rem)` achieves AUROC above `0.7` and better Brier score than `PACER-no-D` for oracle stopping-label prediction.
2. `D(e | I,m)` shows positive rank correlation with empirical resolvability and acceptable calibration on the 40-continuation benchmark states.
3. On the direct-lookahead ablation, PACER-Cert agrees with Monte Carlo lookahead on a clear majority of action rankings or stopping choices.
4. PACER-Cert stops earlier than GIT and AOED-lite on weak-edge settings while losing at most `2` directed-edge-F1 points at the full budget, or improving on them.
5. Median rollout time stays below roughly `90` seconds for `p = 15`, keeping the full study within the CPU budget.

The hypothesis is refuted, or at least significantly weakened, if:

1. the certificate is no better calibrated than a plain ambiguity threshold,
2. direct lookahead disagrees strongly with the surrogate even on tiny problems,
3. stopping gains come mostly from premature termination with large final-F1 degradation,
4. calibration collapses in the exact states where PACER-Cert claims strongest confidence,
5. AOED-lite matches or exceeds PACER-Cert on stopping efficiency without needing the certificate machinery.

## References

1. Eberhardt, F. (2005). *On the Number of Experiments Sufficient and in the Worst Case Necessary to Identify All Causal Relations Among N Variables*. UAI 2005.
2. He, Y.-B., and Geng, Z. (2008). *Active learning of causal networks with intervention experiments and optimal designs*. Journal of Machine Learning Research, 9, 2523-2547.
3. Squires, C., Magliacane, S., Greenewald, K., Katz, D., Kocaoglu, M., and Shanmugam, K. (2020). *Active Structure Learning of Causal DAGs via Directed Clique Tree*. NeurIPS 2020.
4. Tigas, P., Annadani, Y., Jesson, A., Schölkopf, B., Gal, Y., and Bauer, S. (2022). *Interventions, Where and How? Experimental Design for Causal Models at Scale*. NeurIPS 2022.
5. Toth, C., Lorch, L., Knoll, C., Krause, A., Pernkopf, F., Peharz, R., and von Kügelgen, J. (2022). *Active Bayesian Causal Inference*. NeurIPS 2022.
6. Olko, M., Zając, M., Nowak, A., Scherrer, N., Annadani, Y., Bauer, S., Kuciński, Ł., and Miłoś, P. (2023). *Trust Your ∇: Gradient-based Intervention Targeting for Causal Discovery*. NeurIPS 2023.
7. Zhang, J., Cammarata, L., Squires, C., Sapsis, T. P., and Uhler, C. (2023). *Active Learning for Optimal Intervention Design in Causal Models*. Nature Machine Intelligence, 5, 1446-1455.
8. Elahi, M. Q., Wei, L., Kocaoglu, M., and Ghasemi, M. (2024). *Adaptive Online Experimental Design for Causal Discovery*. arXiv:2405.11548.
9. Wang, Y., Liu, M., Sun, X., Wang, W., and Wang, Y. (2024). *Bayesian Intervention Optimization for Causal Discovery*. arXiv:2406.10917.
10. Zhou, W., Huang, H., Zhang, G., Shi, R., Yin, K., Lin, Y., and Liu, B. (2024). *OCDB: Revisiting Causal Discovery with a Comprehensive Benchmark and Evaluation Framework*. arXiv:2406.04598.
11. Zhou, Z., Elahi, M. Q., and Kocaoglu, M. (2024). *Sample Efficient Bayesian Learning of Causal Graphs from Interventions*. NeurIPS 2024.
12. Kummerfeld, E., Williams, L., and Ma, S. (2024). *Power analysis for causal discovery*. International Journal of Data Science and Analytics, 17, 255-274.
13. Elrefaey, A., and Pan, R. (2025). *From Observation to Orientation: an Adaptive Integer Programming Approach to Intervention Design*. arXiv:2504.03122.
14. Gregorini, M., Boldrini, C., and Valerio, L. (2025). *DODO: Causal Structure Learning with Budgeted Interventions*. arXiv:2510.08207.
