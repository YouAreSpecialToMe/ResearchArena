# Benchmarking Subset Aggregation for Classical Causal Discovery Under Marginalization Error

## Introduction

This project is explicitly positioned as an **empirical benchmark paper**, not as a new causal discovery method. The core ingredients already exist in prior work: subset disagreement as an internal reliability signal in Self-Compatibility, leave-one-variable-out validation in LOVO, subset-estimate-aggregate pipelines in SEA, higher-order claim aggregation beyond naive edge voting, and consistency-guided aggregation in the recent work of Ninad et al. The defensible gap is therefore narrower:

> When local CPDAGs are learned on overlapping variable subsets of one fixed observed-variable system, do compatibility-style weights improve the final global CPDAG relative to simple controls, and under what levels of subset distortion should such gains be interpreted only as empirical heuristics?

The paper makes three contributions.

1. It defines a reproducible CPU-scale benchmark for fixed-variable subset aggregation in classical causal discovery.
2. It isolates weighting and merge choices by holding one shared local-graph bank fixed across all subset methods.
3. It audits the subset-marginalization problem directly before interpreting any aggregation gain.

The scope is intentionally modest. The study is restricted to synthetic regimes with \(p=20\) variables under a 2-core, CPU-only budget. The intended claim is empirical: some aggregation heuristics may be useful despite local subset misspecification. The paper does **not** claim that arbitrary subset CPDAGs provide sound globally aggregatable evidence in general.

## Proposed Approach

### Study Object

The study object is a fixed family of symbolic subset aggregators built on top of classical PC outputs. The paper does not propose a new discovery principle. It studies which weighting and merge choices help once the local graphs are already available.

Every subset method shares the same pipeline:

1. construct one bank of local CPDAGs on overlapping variable subsets;
2. extract only explicit symbolic claims from each local CPDAG;
3. assign either uniform, compatibility-based, or non-falsification control weights to the local graphs;
4. merge weighted claims into one global CPDAG with either a Wilson-style shrinkage rule or a deterministic rule.

This yields a clean factorial benchmark:

- weighting: `Uniform` vs `CompatExp` vs `CompatRank` vs `CompatTopHalf` vs `BootstrapStability`
- merger: `Wilson` vs `DetThreshold` vs `DetRank`

### Shared Local-Graph Bank

For each synthetic dataset with \(p=20\) observed variables:

- subset size: \(k=8\)
- number of subsets: \(M=20\)
- row bootstraps per subset: \(B=2\)
- base learner inside every subset cell: `PC-Stable` with fixed \(\alpha=0.01\)

This produces 40 local CPDAGs per dataset. The subset list is generated once by a deterministic coverage heuristic:

- every variable appears in at least 6 subsets;
- most variable pairs co-occur at least twice;
- the same subset list and bootstrap seeds are reused by all subset methods.

This is the core identification strategy of the benchmark. If two subset methods differ, they differ only in weighting and merging, not in the local evidence they saw.

### Claim Extraction

From each local CPDAG \(g\), extract only explicit claims:

- `adj(i,j)`: \(i\) and \(j\) are adjacent;
- `nonadj(i,j)`: \(i\) and \(j\) are both present and non-adjacent;
- `dir(i->j)`: \(g\) explicitly orients \(i \to j\);
- `coll(i->k<-j)`: \(g\) explicitly contains an unshielded collider.

Undirected edges provide adjacency evidence only. Missing orientation is treated as abstention, not as contradiction.

### Compatibility Weighting

For local graphs \(g_a\) and \(g_b\) with overlap \(O=S_a\cap S_b\), compare only claims whose variables lie entirely in \(O\). Count contradictions only when both graphs make explicit opposing claims:

- `adj(i,j)` versus `nonadj(i,j)`;
- `dir(i->j)` versus `dir(j->i)`, or an orientation versus `nonadj(i,j)`;
- `coll(i->k<-j)` versus an explicit incompatible triple claim on the same nodes.

Abstention is neutral. A graph is not penalized for failing to orient an edge.

Let \(\mathrm{contr}(g_a,g_b)\) be the number of explicit conflicts and \(\mathrm{comp}(g_a,g_b)\) the number of comparable overlap claims. Define

\[
r(g_a,g_b)=
\begin{cases}
\mathrm{contr}(g_a,g_b)/\mathrm{comp}(g_a,g_b), & \mathrm{comp}(g_a,g_b)>0 \\
0, & \mathrm{comp}(g_a,g_b)=0.
\end{cases}
\]

For each local graph \(g\),

\[
C_g=\frac{1}{|\mathcal N(g)|}\sum_{h\in\mathcal N(g)} r(g,h),
\]

where \(\mathcal N(g)\) is the set of overlapping local graphs.

The primary weight is pre-registered as

\[
w_g=\exp(-2C_g).
\]

Two deterministic compatibility variants are included:

- `CompatRank`: convert the ordering induced by \(C_g\) into linearly spaced rank weights;
- `CompatTopHalf`: keep the 50% most compatible local graphs and weight them equally.

### Non-Falsification Control Weight

To separate overlap contradiction from generic local stability, the benchmark includes one non-falsification control weight:

- `BootstrapStability`: for each subset, compute agreement between the two bootstrap CPDAGs learned on the same variable set; assign both graphs the same weight based on that within-subset agreement score.

This control uses local stability without cross-subset contradiction, making it a cleaner benchmark comparator than an informal "SEA-like" label while still addressing the same question: is incompatibility information special, or is any stability proxy enough?

### Merge Rules

All mergers operate on weighted support tables and enforce CPDAG validity at the end via cycle checks and Meek closure.

#### Wilson-Style Screen

For a binary claim family, define weighted support

\[
\hat p=\frac{\text{support}}{\text{support}+\text{opposition}}
\]

and effective sample size

\[
n_{\mathrm{eff}}=\frac{(\sum_g w_g)^2}{\sum_g w_g^2}.
\]

Accept a claim only if the one-sided 90% Wilson lower bound exceeds \(0.5\).

This is treated only as a shrinkage heuristic. Because the local graphs overlap heavily, \(n_{\mathrm{eff}}\) is not a valid independence-adjusted sample size. The paper will not interpret the Wilson quantity as a calibrated confidence interval.

#### Deterministic Controls

To test whether any result depends on the Wilson heuristic, include two purely deterministic merge rules:

- `DetThreshold`: accept a claim when weighted support ratio exceeds \(0.6\);
- `DetRank`: rank candidate claims by weighted support margin and greedily accept compatible claims in descending order.

The central benchmark question is whether compatibility weighting helps under both heuristic and deterministic mergers.

## Subset-Marginalization Problem

### Soundness Caveat

This is the key methodological risk. A PC CPDAG learned on an arbitrary subset \(S\subset[p]\) of variables from a larger system need not equal the CPDAG of the induced subgraph on \(S\). Omitted observed variables can create effective latent confounding, induce extra dependencies, destroy faithfulness on the subset, or alter collider patterns. As a result:

- local `nonadj` claims may be false relative to the induced subgraph;
- local orientations may not transport to the full system;
- disagreement between local graphs may reflect omission artifacts rather than finite-sample error.

The paper therefore avoids any claim that subset CPDAG statements are theoretically aggregatable in general. The intended conclusion is limited to empirical utility under controlled finite-sample regimes.

### Targeted Audit Before Main Interpretation

Before interpreting any global aggregation gain, the benchmark will quantify subset distortion directly.

For every sampled subset \(S\):

1. compute the true induced subgraph \(G^\star[S]\) from the simulation DAG \(G^\star\);
2. compute the true induced CPDAG \(\mathrm{CPDAG}(G^\star[S])\);
3. compare the local learned claims on \(S\) against that induced CPDAG.

Report the following validity rates:

- adjacency claim validity;
- non-adjacency claim validity;
- orientation claim validity;
- collider claim validity.

Also report the fraction of local claims that disagree with the induced CPDAG before any aggregation happens. This is the paper's main sanity check on the subset-marginalization problem.

### Interpretation Policy

The benchmark will use the audit to qualify every main result:

- if local subset claims already agree well with the induced CPDAG, aggregation gains are easier to interpret;
- if local claim distortion is substantial, any gain is reported only as a practical denoising effect, not as evidence of sound structural transport;
- if orientation gains appear exactly where local induced-subgraph validity is very poor, the paper will present that as a warning sign rather than a success story.

This restriction is deliberate and addresses the main soundness objection directly.

## Related Work

### Internal Reliability and Falsification

Faller et al. (2024), *Self-Compatibility: Evaluating Causal Discovery without Ground Truth*, are the closest conceptual precursor. They already use subset disagreement as an internal reliability signal. This proposal does not claim a new compatibility principle; it tests whether that signal is useful as a fixed-bank aggregation weight.

Schkoda et al. (2025), *Cross-validating causal discovery via Leave-One-Variable-Out*, provide another internal validation signal. LOVO is selection-oriented rather than aggregation-oriented and is therefore included only as a global baseline, not as local evidence inside the subset bank.

Viinikka et al. (2018), *Intersection-Validation*, are older but relevant prior work on agreement-based evaluation without ground truth.

### Subset Aggregation and Consistency-Guided Aggregation

Wu et al. (2025), *Sample, estimate, aggregate*, already show that subset-estimate-aggregate pipelines can be effective, though with learned neural aggregation and larger compute. That prior work substantially narrows novelty: this paper isolates a smaller symbolic CPU-only question rather than proposing a new aggregation paradigm.

Ninad et al. (2025), *Causal discovery on vector-valued variables and consistency-guided aggregation*, are the closest novelty threat. They already study consistency-guided aggregation and explicitly analyze when aggregation can fail. The main difference here is not theory but setting: fixed observed variables, overlapping subsets, and a benchmark centered on empirical failure modes under subset distortion.

### Classical DAG Aggregation and Bagging

Wang and Peng (2014), *Learning directed acyclic graphs via bootstrap aggregating (DAGBag)*, are essential prior art. DAGBag shows that graph aggregation can reduce false positives in finite samples via bootstrap ensembles and SHD-based aggregation. This paper is not the first to use graph aggregation in causal structure learning. Its narrower contribution is to compare fixed-variable subset aggregation against these older bagging intuitions and newer compatibility signals.

Debeire et al. (2024), *Bootstrap aggregation and confidence measures to improve time series causal discovery*, provide a recent bagging analogue in time-series causal discovery.

Zanga et al. (2025), *Causal Discovery on Higher-Order Interactions*, further narrow the novelty of any aggregation claim by showing that naive edge-frequency voting can miss higher-order structure.

### Alternative Full-Graph Baselines

Malinsky (2024), *A cautious approach to constraint-based causal model selection*, motivates conservative finite-sample structure estimation.

Chan et al. (2024), *AutoCD*, represent the model-selection alternative family: instead of aggregating many local graphs, select one global learner or one hyperparameter setting using internal criteria.

These papers collectively motivate the benchmark framing. If compatibility weighting helps, it must help relative to both subset controls and strong non-aggregation baselines.

## Experiments

### Positioning

The paper is a **synthetic finite-sample stress test**. Real data are intentionally out of scope for the core claim because they do not provide clean ground truth for this question.

### Benchmark Regimes

Main benchmark:

- graph families: Erdős-Rényi and scale-free
- structural regimes:
  - linear Gaussian
  - nonlinear additive noise
  - near-unfaithful linear
  - mild misspecification
- variables: \(p=20\)
- sample sizes: \(n\in\{200,1000\}\)
- random seeds: 3

Total datasets:

\[
2 \times 4 \times 2 \times 3 = 48.
\]

This is sized to fit the 8-hour CPU budget while preserving both easy and hard regimes.

### Methods Compared

Full-graph baselines:

- `PC-Stable`: full-graph PC-Stable with fixed \(\alpha=0.01\);
- `FGES`: full-graph score-based search with standard Gaussian BIC;
- `NOTEARS-L1`: full-graph continuous optimization baseline, projected to a CPDAG for evaluation;
- `SC-Select`: choose full-graph PC-Stable over a fixed \(\alpha\)-grid using self-compatibility;
- `LOVO-Select`: choose full-graph PC-Stable over the same \(\alpha\)-grid using LOVO, only if the scoring recipe is operationally unambiguous.

Subset-family cells:

- `Uniform+Wilson`
- `Uniform+DetThreshold`
- `Uniform+DetRank`
- `CompatExp+Wilson`
- `CompatExp+DetThreshold`
- `CompatExp+DetRank`
- `CompatRank+DetRank`
- `CompatTopHalf+DetRank`
- `BootstrapStability+Wilson`
- `BootstrapStability+DetRank`

The primary comparison is `CompatExp` versus `Uniform` under the same merger. `BootstrapStability` is the main control for the claim that contradiction information carries unique value.

### Hyperparameter Policy

To avoid the fairness problem in the previous draft, the study will use **pre-registered defaults only**. No method hyperparameter is tuned on SHD using a pilot subset of benchmark datasets.

Fixed subset-family defaults:

- `PC-Stable` significance level inside subsets: \(\alpha=0.01\)
- `CompatExp` temperature: 2
- `CompatTopHalf` keep fraction: 0.5
- `DetThreshold` acceptance threshold: 0.6
- `Wilson` level: one-sided 90%
- subset design: \(k=8, M=20, B=2\)

Internal-selection baselines use the same fixed PC-Stable grid

\[
\alpha \in \{0.001, 0.01, 0.05, 0.1\},
\]

with the score deciding the winner. This preserves symmetry across selection methods.

### Metrics

Primary metrics:

- SHD to the true global CPDAG;
- skeleton precision, recall, and F1;
- orientation precision and recall;
- false-orientation rate among oriented edges.

Subset-audit metrics:

- local adjacency validity against the true induced CPDAG;
- local non-adjacency validity;
- local orientation validity;
- local collider validity;
- local claim disagreement rate with the induced CPDAG.

Secondary diagnostics:

- self-compatibility score;
- LOVO score if computed;
- fraction of undirected edges;
- graph density;
- runtime and peak memory.

### Key Ablations

1. `Subset Distortion Audit`
   Quantify how often local subset claims disagree with the true induced CPDAG before any aggregation result is interpreted.

2. `Wilson vs Deterministic Merge`
   Test whether qualitative conclusions survive removal of the Wilson-style screen.

3. `Compatibility vs Stability`
   Compare `CompatExp` against `BootstrapStability` to isolate the value of contradiction information itself.

4. `Compatibility Parameterization`
   Compare exponential weighting, rank weighting, and top-half truncation.

5. `Subset Budget`
   On a reduced hard-regime slice, compare \(M\in\{16,20,24\}\) to test bank-size sensitivity.

6. `Order Sensitivity`
   On the reduced slice, rerun `DetRank` with reversed and random tie orders.

### Expected Outcomes

The strongest plausible outcome is not universal dominance. A publishable positive result would look like this:

- compatibility weighting improves orientation precision or lowers false orientations in the harder regimes;
- the gain survives at least one deterministic merger;
- the gain is not fully reproduced by within-subset bootstrap stability alone;
- the subset audit shows the local bank is noisy but not completely uninformative.

A publishable negative result is also possible:

- once deterministic merging replaces Wilson, compatibility weighting adds little over uniform weighting; or
- the subset audit shows local orientation claims are too distorted for compatibility weighting to carry meaningful signal.

Because the benchmark isolates these failure modes cleanly, either outcome is scientifically useful.

## Success Criteria

The hypothesis is supported if all of the following hold:

1. `CompatExp` lowers median false-orientation rate by at least 10% relative to `Uniform` on the 48-dataset benchmark under at least one merger family.
2. The direction of the `CompatExp` effect is consistent under both `Wilson` and at least one deterministic merger on the hard regimes.
3. `CompatExp` matches or exceeds `BootstrapStability` under the same merger, so gains are not explained by generic local stability alone.
4. The subset audit shows that reported gains do not arise only by collapsing to extremely sparse or mostly unoriented graphs.
5. The full benchmark and reduced-slice ablations fit within the stated 2-core, CPU-only, roughly 8-hour budget.

The hypothesis is weakened or refuted if the following patterns dominate:

- compatibility weighting helps only under the Wilson-style heuristic and disappears under deterministic merging;
- `Uniform` or `BootstrapStability` matches `CompatExp` across the hard regimes;
- the subset audit reveals severe induced-subgraph disagreement exactly where the method appears to improve, making the gain hard to interpret;
- strong full-graph baselines such as `FGES`, `NOTEARS-L1`, or `SC-Select` match the gains with less complexity.

## Compute Feasibility

The design is constrained to the stated hardware:

- no GPU;
- 2 CPU cores only;
- all subset discovery runs on 8-variable problems rather than 20-variable problems;
- the expensive local bank is cached once per dataset and reused across all subset methods;
- mergers and audits are symbolic and cheap relative to structure learning;
- heavier sensitivity checks are limited to reduced hard-regime slices.

This keeps the full study feasible within roughly 8 hours while still supporting the main benchmark and the critical soundness audit.

## References

- Chan, G., Claassen, T., Hoos, H. H., Heskes, T., and Baratchi, M. (2024). *AutoCD: Automated Machine Learning for Causal Discovery Algorithms*. Probabilistic Graphical Models.
- Debeire, K., Gerhardus, A., Runge, J., and Eyring, V. (2024). *Bootstrap aggregation and confidence measures to improve time series causal discovery*. Proceedings of the Third Conference on Causal Learning and Reasoning.
- Faller, P. M., Vankadara, L. C., Mastakouri, A. A., Locatello, F., and Janzing, D. (2024). *Self-Compatibility: Evaluating Causal Discovery without Ground Truth*. Proceedings of the 27th International Conference on Artificial Intelligence and Statistics.
- Malinsky, D. (2024). *A cautious approach to constraint-based causal model selection*. arXiv:2404.18232.
- Ninad, U., Wahl, J., Gerhardus, A., and Runge, J. (2025). *Causal discovery on vector-valued variables and consistency-guided aggregation*. arXiv:2505.10476.
- Schkoda, D., Faller, P. M., Blobaum, P., and Janzing, D. (2025). *Cross-validating causal discovery via Leave-One-Variable-Out*. Proceedings of the 4th Conference on Causal Learning and Reasoning.
- Viinikka, J., Eggeling, R., and Koivisto, M. (2018). *Intersection-Validation: A Method for Evaluating Structure Learning without Ground Truth*. Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics.
- Wang, R., and Peng, J. (2014). *Learning directed acyclic graphs via bootstrap aggregating*. arXiv:1406.2098.
- Wu, M., Bao, Y., Barzilay, R., and Jaakkola, T. (2025). *Sample, estimate, aggregate: A recipe for causal discovery foundation models*. Transactions on Machine Learning Research.
- Zanga, A., Scutari, M., and Stella, F. (2025). *Causal Discovery on Higher-Order Interactions*. arXiv:2511.14206.
