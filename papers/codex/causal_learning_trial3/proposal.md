# When Do Path-Dependent Setup Costs Matter in Sequential Causal Design?

## Stage 2 Scope Note

This workspace now supports a narrower Stage 2 study than the original Stage 1 proposal. It does not implement a full CPDAG-consistent linear-Gaussian DAG posterior with exact sufficient-statistic tracking. Instead, it studies a CPU-feasible orientation-only surrogate: start from a learned or oracle skeleton, maintain a posterior over acyclic orientations of that skeleton, and compare finite-budget intervention planners on terminal orientation entropy. Claims from this workspace should therefore be read as evidence about that surrogate orientation-planning problem, not about the original full CPDAG/DAG posterior program.

## Introduction

This project is positioned as a **careful empirical characterization paper**, not as a new general causal design framework. The closest literature already covers additive-cost intervention design, budgeted Bayesian experimental design, and adaptive or non-myopic sequential causal design. The narrower question left open is whether **path-dependent setup costs** actually change which intervention sequence is preferred once the objective is stated in the right way for a finite experimental budget.

That question matters because many real intervention platforms have family-level setup overheads. Changing from CRISPR to pharmacological perturbations, or from one assay protocol to another, can consume budget even when each individual experiment is cheap. Existing causal design papers in the provided reference set treat intervention cost as additive across actions, while switching-cost papers outside causal discovery do not study causal graph orientation. This paper asks when those two views diverge in a finite-sample sequential causal learning problem.

The scope is deliberately narrow and CPU-feasible. We start from an estimated skeleton, restrict attention to orienting edges inside that skeleton, and study short-horizon planners for sparse linear-Gaussian SCMs under a fixed total budget. The goal is not to sell another heuristic, but to identify the regimes in which path dependence matters, when it provably cannot matter much, and how robust those conclusions are to posterior approximation and skeleton error in this surrogate setting.

### Problem statement

Given observational data, an estimated CPDAG `C_hat`, a catalog of singleton interventions from a small set of families, and a total budget `B`, choose interventions sequentially to minimize expected terminal orientation uncertainty inside the estimated skeleton, where each action incurs execution cost, sample cost, and a setup cost that depends on the previously chosen family.

### Key insight

The right primary objective is **budget-aligned terminal utility**, not an information-to-cost ratio. Once the objective is written this way, path-dependent setup costs can change the optimal policy only through two mechanisms:

1. they change the feasible policy set under the budget;
2. among jointly feasible policies, they are large enough to outweigh the additive-value advantage of a higher-switch policy.

This yields a stronger contribution than the previous two-step threshold observation: a general structural characterization of switching-cost invariance and crossover for any finite horizon, plus an empirical audit of how often those conditions are active in realistic small-budget causal design settings.

### Hypothesis

For sparse linear-Gaussian SCMs with hard and soft singleton interventions, transition costs affect the optimal sequential design policy only in a restricted regime characterized by small additive-value gaps and/or tight budget margins. A switching-aware planner will outperform additive-cost planners only in those regimes, and the crossover pattern will persist under exact posterior computation on small graphs.

## Proposed Approach

### Overview

Maintain a posterior over acyclic orientations of an estimated skeleton, update that posterior after each intervention, and evaluate short candidate intervention sequences using a **fixed-budget terminal entropy objective**. The paper contributes:

1. a finite-horizon structural condition for when switching costs can and cannot change the preferred policy;
2. a CPU-feasible transition-aware planner used as an analysis instrument rather than the main novelty claim;
3. an empirical characterization across switching regimes, posterior backends, and learned-vs-oracle skeletons.

### Primary objective

Let `S_hat` be the skeleton induced by `C_hat`. For each undirected edge `{u,v}` in `S_hat`, define the posterior orientation probability `p_t(u -> v)` and skeleton-conditional orientation entropy

- `H_orient(t) = sum_{{u,v} in S_hat} h(p_t(u -> v))`
- `h(p) = -p log p - (1-p) log(1-p)`.

For a policy `pi` executed until budget exhaustion or horizon `T`, define the primary value

- `V_B(pi) = E[H_orient(0) - H_orient(T_pi)]`

subject to

- `sum_t C_exec(a_t) + C_samp(a_t) + C_sw(f_{t-1}, f_t) <= B`.

This objective aligns directly with a fixed experimental budget. The previous ratio score `Delta/C_path` is retained only as a **secondary efficiency diagnostic**. One experiment explicitly checks that the qualitative crossover conclusions are unchanged when comparing policies by `V_B`, area-under-entropy-vs-budget, or the older ratio summary.

### Structural characterization

Write the total cost of a policy as

- `C(pi; lambda) = C_add(pi) + lambda N_sw(pi)`

for a scalar switching-penalty sweep, where `N_sw(pi)` is the number of family switches. Let `V_0(pi)` denote the expected terminal entropy reduction of `pi`, which is independent of `lambda` because `lambda` changes only the cost model.

The paper will prove the following finite-horizon characterization.

**Proposition 1 (penalized invariance condition).** Under the penalized objective

- `J_lambda(pi) = V_0(pi) - lambda N_sw(pi)`,

switching cost can change the ranking between two policies `pi_1` and `pi_2` only if

- `0 < V_0(pi_1) - V_0(pi_2) <= lambda (N_sw(pi_1) - N_sw(pi_2))`

for the higher-switch policy `pi_1`. Therefore, if

- `V_0(pi_1) - V_0(pi_2) > lambda_max (N_sw(pi_1) - N_sw(pi_2))`

for every lower-switch competitor `pi_2`, then `pi_1` remains optimal for all `lambda in [0, lambda_max]`.

This gives a general **invariance region**: transition costs are irrelevant unless the additive-value advantage is small relative to the switch-count gap.

**Proposition 2 (budgeted crossover condition).** Under the budgeted objective `V_B`, transition costs can change the optimal policy only if at least one of the following holds:

1. a previously optimal additive-cost policy becomes infeasible because `C_add(pi) + lambda N_sw(pi) > B`;
2. two jointly feasible policies have additive-value gap no larger than `lambda` times their switch-count gap, as in Proposition 1.

Equivalently, if every higher-switch policy is either infeasible or has additive-value advantage larger than its maximum switching surcharge, the budgeted optimum is unchanged.

This is the main theoretical strengthening over the previous draft. The old two-step threshold becomes a corollary of Proposition 2.

### Data model

Use sparse linear-Gaussian SCMs with Normal-Inverse-Gamma priors:

- main settings: `d in {10, 15}`;
- runtime slice: `d = 20`;
- exact-posterior diagnostics: `d = 8`.

Each node satisfies

- `X_j = beta_j^T X_pa_G(j) + eps_j`,
- `eps_j ~ N(0, sigma_j^2)`.

Intervention families:

1. `hard(i)`: replace `X_i` by an exogenous Gaussian draw;
2. `soft(i)`: preserve parents but perturb local mechanism via mean shift and variance inflation.

Both choices preserve conjugate updates and are cheap on CPU.

### Cost model

For action `a_t = (i_t, f_t, n_t)`,

- `C(a_t | a_{t-1}) = c_exec(f_t) + n_t c_sample(f_t) + c_sw(f_{t-1}, f_t)`.

The experiments use both symmetric and asymmetric switching matrices. Budgets are defined in units that permit roughly 6 to 10 batches depending on the switching regime.

### Posterior representation

The implemented posterior state contains:

- acyclic orientation particles `G^(1), ..., G^(M)` over a fixed skeleton;
- weights `w^(m)`;
- accumulated observational and interventional datasets;
- previous family `f_{t-1}`;
- remaining budget.

Initialization:

1. estimate a learned skeleton from observational data with GES+BIC when available and a correlation fallback otherwise;
2. generate acyclic orientations exactly on the `d=8` slice when feasible and approximately otherwise;
3. weight them by a Gaussian regression score over the accumulated records.

### Orientation-posterior generation and audit

This is treated as an approximation with explicit diagnostics, not as a novel inference contribution.

1. enumerate all acyclic orientations of the fixed skeleton when the state space is small enough;
2. otherwise sample legal acyclic orientations from random topological orders consistent with any compelled directions.

Approximation quality is measured on `d=8` cases where exact enumeration is feasible:

- total variation error of edge-orientation marginals;
- KL divergence between exact and particle posteriors over DAGs;
- calibration of posterior direction probabilities;
- whether planner rankings change under exact versus particle backends.

### Posterior update

After each intervention batch, rescore the orientation particles on the expanded dataset, update particle weights, monitor ESS, and optionally rejuvenate the particle pool with newly sampled legal orientations inside the fixed skeleton. This is a practical surrogate update, not a claim of exact CPDAG/DAG posterior tracking.

### Planning algorithms

The planner is intentionally short-horizon and CPU-feasible.

Action space:

- target node `i` over all nodes;
- family `f in {hard, soft}`;
- batch size `n in {25, 50}`.

All methods share the same posterior backend within a table.

**Method A: Myopic budgeted gain**

- choose the feasible action with largest one-step expected entropy reduction.

**Method B: Horizon-2 additive-cost planner**

- ignores switching costs in search, but is evaluated under the true budget/cost model.

**Method C: Horizon-2 switching-aware receding-horizon planner**

- searches feasible depth-2 sequences with state `(posterior, previous family, remaining budget)`;
- objective: expected terminal entropy reduction at horizon end.

**Method D: Horizon-3 switching-aware planner**

- used only on the smallest main setting.

**Method E: Transition-aware budgeted tree search baseline**

- an exhaustive finite-horizon tree search on reduced settings (`d=8`, coarse budget grid, top-`K` pruned first actions);
- optimizes the same fixed-budget terminal objective with transition costs;
- serves as a stronger baseline outside the receding-horizon heuristic, directly addressing the concern that the previous comparison set was too weak.

This baseline is expensive, so it is limited to the exact-posterior slice where it is most informative.

### Rollout evaluation

For each candidate sequence:

1. sample `R = 16` posterior particles by weight;
2. sample local parameters from the conjugate posterior;
3. simulate interventional data for `n in {25, 50}`;
4. update the posterior analytically;
5. average terminal entropy reduction.

The rollout counts are fixed across methods to keep runtime controlled.

## Related Work

### Additive-cost causal intervention design

Kocaoglu, Dimakis, and Vishwanath (2017), Ghassami et al. (2018), and Lindgren et al. (2018) study intervention design with additive intervention costs and graph-identification objectives. They are structurally closest on the cost side, but they do not model transition-dependent setup costs across rounds. This proposal borrows their cost-aware motivation while focusing on the narrower question of when path dependence changes a sequential policy under finite samples and finite budgets.

### Budgeted and sequential causal Bayesian design

ABCD-Strategy (Agrawal et al., 2019) formulates Bayesian budgeted experimental design for causal structure targets using mutual information and explicit design constraints. Adaptive Online Experimental Design for Causal Discovery (Elahi et al., 2024) studies adaptive intervention allocation with sample-complexity analysis. Sample Efficient Bayesian Learning of Causal Graphs from Interventions (Zhou et al., 2024) maintains posterior uncertainty over graphs under interventions. GO-CBED (Zhang et al., 2025) is particularly important because it already provides a non-myopic sequential causal BED formulation. Relative to these papers, the present contribution is not a richer planner or a broader causal objective. It is a focused characterization of **transition-dependent setup costs under a fixed budget**, with theory about invariance/crossover and an exact-posterior audit of when that extra modeling changes decisions.

### Variable-cost sequential design outside causal discovery

Sequential Bayesian Experimental Design with Variable Cost Structure (Zheng et al., 2020) shows that non-uniform cost models alter sequential design decisions, though the variable cost there is computational cost of utility evaluation rather than intervention setup cost. Bayesian Optimization with Setup Switching Cost (Pricopie et al., 2024) is closer in spirit because it models path-dependent setup overheads, but it is not a causal discovery problem. These papers motivate the cost structure; they do not settle whether the same issue matters in causal graph learning.

### Positioning and novelty claim

The novelty claim is intentionally narrow: **I am not aware, within the provided reference set and the targeted novelty check, of a prior causal design paper that directly characterizes when path-dependent family-switching costs change the finite-budget optimal sequential intervention policy, while separating that effect from posterior approximation and skeleton error.** If a very close concurrent paper exists, the proposal still remains viable as a rigorous characterization study only if the structural result and exact-posterior audit are both retained.

## Experiments

### Resource-aware study design

The full study must fit roughly 8 wall-clock hours on 2 CPU cores, so the evaluation is intentionally compact:

- `d in {10, 15}` for the main experiments;
- `d = 8` for exact-posterior and exhaustive-search diagnostics;
- `d = 20` for a small runtime slice only;
- 10 shared seeds in the main study;
- no neural components;
- singleton interventions only;
- horizon 3 only where explicitly stated.

Shared seeds:

- `[11, 23, 37, 41, 53, 67, 79, 83, 97, 101]`.

### Synthetic setup

Graphs:

- Erdős-Rényi DAGs with expected degree 2;
- scale-free DAGs as a secondary topology.

Data:

- 500 observational samples;
- interventional batch sizes 25 and 50.

Switching regimes:

1. zero;
2. low;
3. medium;
4. high;
5. asymmetric high.

Soft-intervention regimes:

1. weak perturbation;
2. strong perturbation.

### Baselines

1. Random feasible schedule.
2. Myopic budgeted gain.
3. Myopic gain divided by immediate additive cost.
4. Horizon-2 additive-cost planner.
5. Horizon-2 switching-aware planner.
6. Horizon-3 switching-aware planner on the smallest main setting.
7. Transition-aware budgeted tree search on the exact-posterior small-graph slice.

The last baseline is crucial because it optimizes the fixed-budget transition-cost objective directly and is not just a matched heuristic variant.

### Metrics

Primary:

- terminal skeleton-conditional orientation entropy at matched budget;
- area under orientation-entropy-versus-spent-budget curve;
- fraction of estimated-skeleton edges correctly oriented.

Secondary:

- restricted SHD on `S_hat`;
- full SHD to the true DAG;
- number of family switches;
- wall-clock planning time;
- posterior approximation error on exact-enumeration slices;
- the old ratio metric `Delta/C_path` as a diagnostic only.

### Core experiment 1: budgeted crossover sweep

Sweep switching penalties and budgets jointly. For each setting, compare the best additive-cost policy and the best switching-aware policy under the fixed-budget terminal objective. Test whether policy crossover occurs only in the regimes predicted by the structural condition: small additive-value gap, high switch-count gap, or tight feasibility margin.

### Core experiment 2: exact-posterior and exact-search audit

On `d=8` graphs:

1. compare exact and particle posterior orientation marginals under the same data sequence;
2. compare horizon-2 switching-aware planning against the stronger transition-aware exhaustive tree-search baseline;
3. check whether the qualitative crossover claims remain true under exact posterior inference and stronger search.

This addresses the two strongest concerns from the previous draft: posterior bias and weak baselines.

### Core experiment 3: learned versus oracle CPDAG

Run the main switching sweep with:

1. learned `C_hat` from observational data;
2. oracle CPDAG from the true skeleton.

If the effect disappears under learned skeletons, the paper cannot claim a meaningful switching-cost phenomenon in practice. This experiment is therefore mandatory.

### Core experiment 4: objective-robustness check

Compare policy rankings under:

1. terminal entropy reduction at fixed budget;
2. area under the entropy-versus-budget curve;
3. the old ratio summary.

The claim survives only if the first two, budget-aligned criteria agree and the ratio metric does not reverse the main conclusion.

### Core experiment 5: family-strength and asymmetry sensitivity

Vary:

- soft-intervention strength;
- family-specific execution cost;
- symmetric versus asymmetric switching matrices.

This tests whether the result is structural rather than an artifact of a single stylized family model.

## Success Criteria

The proposal is supported if:

1. the structural condition correctly predicts broad invariance regions where switching costs do not change the preferred policy;
2. switching-aware planning beats the horizon-matched additive-cost planner on the primary budgeted metrics in medium/high switching regimes;
3. the same qualitative crossover survives under exact posterior inference on `d=8`;
4. the stronger transition-aware tree-search baseline does not eliminate the observed advantage of modeling switching costs;
5. the main conclusion is unchanged when moving from the old ratio diagnostic to budget-aligned objectives;
6. the learned-CPDAG setting still shows a measurable effect, even if smaller than the oracle-CPDAG setting;
7. the full study completes within the CPU budget.

The proposal is refuted or substantially weakened if:

1. exact-posterior or exact-search comparisons remove the switching-cost effect;
2. the effect appears only under the ratio metric and disappears under fixed-budget evaluation;
3. learned-skeleton error dominates the results so strongly that switching costs have no stable effect in the practical setting;
4. benefits appear only in one artificial family-strength configuration;
5. planning overhead makes the study infeasible on CPU.

## References

1. Raj Agrawal, Chandler Squires, Karren Yang, Karthik Shanmugam, and Caroline Uhler. "ABCD-Strategy: Budgeted Experimental Design for Targeted Causal Structure Discovery." AISTATS 2019. https://proceedings.mlr.press/v89/agrawal19b.html
2. Yashas Annadani, Panagiotis Tigas, Desi R. Ivanova, Andrew Jesson, Yarin Gal, Adam Foster, and Stefan Bauer. "Differentiable Multi-Target Causal Bayesian Experimental Design." arXiv:2302.10607, 2023. https://arxiv.org/abs/2302.10607
3. Muhammad Qasim Elahi, Lai Wei, Murat Kocaoglu, and Mahsa Ghasemi. "Adaptive Online Experimental Design for Causal Discovery." ICML 2024 Spotlight. https://arxiv.org/abs/2405.11548
4. AmirEmad Ghassami, Saber Salehkaleybar, Negar Kiyavash, and Elias Bareinboim. "Budgeted Experiment Design for Causal Structure Learning." ICML 2018. https://proceedings.mlr.press/v80/ghassami18a.html
5. Murat Kocaoglu, Alex Dimakis, and Sriram Vishwanath. "Cost-Optimal Learning of Causal Graphs." ICML 2017. https://proceedings.mlr.press/v70/kocaoglu17a.html
6. Erik M. Lindgren, Murat Kocaoglu, Alexandros G. Dimakis, and Sriram Vishwanath. "Experimental Design for Cost-Aware Learning of Causal Graphs." CausalML Workshop at NeurIPS 2018. https://arxiv.org/abs/1810.11867
7. Stefan Pricopie, Richard Allmendinger, Manuel Lopez-Ibanez, Clyde Fare, Matt Benatan, and Joshua Knowles. "Bayesian Optimization with Setup Switching Cost." GECCO Companion 2024. https://doi.org/10.1145/3638530.3664101
8. Sue Zheng, David S. Hayden, Jason Pacheco, and John W. Fisher III. "Sequential Bayesian Experimental Design with Variable Cost Structure." NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/hash/2adee8815dd939548ee6b2772524b6f2-Abstract.html
9. Zheyu Zhang, Jiayuan Dong, Jie Liu, and Xun Huan. "Goal-Oriented Sequential Bayesian Experimental Design for Causal Learning." arXiv:2507.07359, 2025. https://arxiv.org/abs/2507.07359
10. Jiaqi Zhang, Louis Cammarata, Chandler Squires, Themistoklis P. Sapsis, and Caroline Uhler. "Active Learning for Optimal Intervention Design in Causal Models." arXiv:2209.04744, 2022. https://arxiv.org/abs/2209.04744
11. Zihan Zhou, Muhammad Qasim Elahi, and Murat Kocaoglu. "Sample Efficient Bayesian Learning of Causal Graphs from Interventions." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/113e6f1d94af5df4f306fbcb3f82339f-Abstract-Conference.html
