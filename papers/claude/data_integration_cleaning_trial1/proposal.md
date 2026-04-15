# Cross-Constraint Repair Safety: Decomposing Multi-Constraint Data Cleaning via Constraint Interaction Analysis

## Introduction

### Context

Data cleaning is a critical step in any data integration and analytics pipeline. Real-world datasets contain heterogeneous errors---missing values, duplicate records, constraint violations, inconsistencies---that must be addressed before the data can be reliably used. To specify what "clean" data looks like, practitioners employ multiple types of integrity constraints simultaneously: functional dependencies (FDs), conditional functional dependencies (CFDs), denial constraints (DCs), and metric constraints.

The challenge intensifies in data integration scenarios where data from multiple heterogeneous sources must be merged. Each source may satisfy its local constraints but the integrated dataset often violates constraints that span across sources. The resulting constraint set is typically heterogeneous (mixing FDs, CFDs, and DCs) and large.

### Problem Statement

Current approaches to multi-constraint data repair fall into two extremes:

1. **Independent repair**: Each constraint type is handled by a specialized algorithm in isolation (e.g., an FD repair tool followed by a DC repair tool). This is fast but ignores interactions---repairing violations of one constraint can introduce new violations of another. In practice, this leads to "repair cascades" where multiple rounds of cleaning are needed.

2. **Holistic repair**: All constraints are considered simultaneously in a single optimization or probabilistic inference framework (e.g., HoloClean). This captures interactions but is computationally expensive, often quadratic or worse in the number of constraints, and scales poorly to large constraint sets.

The fundamental issue is that **the interaction structure between constraints is neither systematically analyzed nor exploited**. In practice, many constraint pairs are non-interfering (repairs for one cannot violate the other), yet holistic methods pay the cost of considering all constraints jointly. Conversely, independent methods miss the critical interactions between the few constraint pairs that do interfere.

### Key Insight

We observe that the interaction between data quality constraints exhibits **structural sparsity**: in typical real-world constraint sets, most constraint pairs operate on disjoint or weakly overlapping attribute sets, making their repairs mutually safe. Only a small fraction of constraint pairs have repairs that can potentially interfere. By formally characterizing this interaction structure via a **Constraint Interaction Graph (CIG)**, we can decompose the multi-constraint repair problem into largely independent subproblems, achieving near-holistic quality at near-independent-repair cost.

### Hypothesis

**A Constraint Interaction Graph that captures repair safety relationships between constraint pairs can be efficiently computed and used to decompose multi-constraint data repair into independent subproblems, achieving repair quality comparable to holistic methods while providing speedups proportional to the degree of interaction sparsity in the constraint set.**

## Proposed Approach

### Overview

We propose **CRIS** (Cross-constRaint Interaction-aware repair Scheduling), a framework that:
1. Formally defines when repairs for one constraint are "safe" with respect to another
2. Constructs a Constraint Interaction Graph encoding these safety relationships
3. Decomposes the repair problem into maximally independent subproblems
4. Schedules repair execution to minimize cascading violations
5. Provides theoretical guarantees on repair quality and convergence

### Step 1: Repair Safety Analysis

Given a set of heterogeneous constraints Sigma = {sigma_1, ..., sigma_n} over a relational schema R(A_1, ..., A_m), we analyze pairwise repair safety at three levels:

**Level 1 -- Structural Safety (syntactic, O(n^2)):**
Two constraints sigma_i and sigma_j are *structurally safe* if the set of attributes that sigma_i's repair can modify is disjoint from the set of attributes that sigma_j references. Formally, let Mod(sigma_i) be the attributes that a repair for sigma_i may change, and Ref(sigma_j) be the attributes that sigma_j's predicates reference. If Mod(sigma_i) intersection Ref(sigma_j) = empty set, then repairs for sigma_i cannot introduce violations of sigma_j. This is a purely syntactic check on constraint definitions.

For FDs X -> Y, Mod(FD) = Y (only RHS attributes are modified in repair) and Ref(FD) = X union Y. For denial constraints, Mod(DC) includes all attributes in the DC's predicates (since any could be modified), and Ref(DC) is the same set. This asymmetry means FD repairs are structurally safe w.r.t. many other constraints (they only modify determined attributes), while DC repairs have broader potential for interference.

**Level 2 -- Predicate-Aware Safety (semantic, O(n^2 * |predicates|)):**
Even with overlapping attributes, constraint pairs may be safe if the repair operations and constraint predicates are compatible. For example, if sigma_i repairs attribute A by setting it to a value that satisfies a predicate "A > 0", and sigma_j requires "A > -10", then sigma_i's repairs trivially satisfy sigma_j's requirement on A. We analyze predicate entailment to identify such cases.

**Level 3 -- Data-Dependent Safety (empirical, O(n^2 * sample_size)):**
For constraint pairs that pass neither structural nor predicate-aware checks, we estimate the empirical interference rate by sampling: apply sigma_i's repair to a sample of the data, then check how many new sigma_j violations are introduced. If the interference rate is below a threshold epsilon, we treat the pair as approximately safe.

**Justification of the safety threshold epsilon:** We set epsilon based on the principle that the expected number of new violations introduced should be below a tolerable absolute count. Specifically, for a dataset of N tuples and a sample of size s, we use a one-sided binomial confidence interval: we declare a pair safe if the upper bound of the 95% confidence interval on the interference rate is below epsilon. We propose epsilon = 0.01 as a default, meaning fewer than 1% of tuples are affected by cross-constraint interference. This choice is motivated by the observation that typical error rates in benchmark datasets are 5-10%, so an interference rate of 1% introduces at most a relative degradation of ~10-20% of the error rate---small enough to be absorbed by the verification pass. We include a **sensitivity analysis** over epsilon in {0.001, 0.005, 0.01, 0.02, 0.05} in our experiments to validate this choice empirically and characterize the precision-recall trade-off: smaller epsilon yields more conservative (safer) decompositions with fewer components but higher quality, while larger epsilon yields more aggressive decompositions with more speedup but potential quality loss.

### Step 2: Constraint Interaction Graph Construction

The CIG is a directed graph G = (V, E) where:
- Each vertex v_i in V represents a constraint sigma_i
- A directed edge (v_i, v_j) in E indicates that repairs for sigma_i are **not safe** w.r.t. sigma_j (i.e., repairing sigma_i may introduce violations of sigma_j)
- Edge weight w(v_i, v_j) encodes the estimated interference severity (from Level 3 analysis, or a default weight for Level 1/2 unsafe pairs)

The CIG is sparse when constraint interactions are sparse, which we hypothesize is the common case in practice.

### Step 3: Graph-Based Decomposition

Using the CIG, we apply the following decomposition:

1. **Independent components**: Compute weakly connected components of G. Constraints in different components have no interference and can be repaired fully independently, in parallel.

2. **DAG components**: Within a connected component, if the induced subgraph is acyclic (a DAG), we repair constraints in reverse topological order---repairing "downstream" constraints first (those that no unsafe edge points to), then working upward. This ensures that later repairs (for upstream constraints) don't undo the work of earlier repairs (for downstream constraints).

3. **Cyclic components**: For components with cycles, we compute a minimum-weight feedback arc set (FAS) to identify the weakest interference edges. We break cycles at these edges, yielding a DAG. For the removed edges, we apply one additional round of checking/repair after the DAG-ordered repair completes.

### Formal Guarantee: Decomposition Equivalence (Proof Sketch)

**Theorem.** Let Sigma be a constraint set whose CIG G is acyclic. Under *minimum-change update repair* semantics (i.e., each constraint is repaired by making the minimum number of cell-level changes to satisfy it), the decomposed repair computed by CRIS yields the same result as a monolithic repair that considers all constraints simultaneously, provided that each individual constraint repair produces a unique minimum-change solution.

**Proof Sketch.** We prove this by induction on the topological order of G.

*Base case:* Consider the set of sink nodes S in G (constraints with no outgoing edges). These constraints do not interfere with any other constraint's repair. Since no edge (v_i, v_j) exists for v_j in S from any other node that has not yet been repaired, the repair of each sink constraint is identical whether computed independently or as part of the monolithic optimization.

*Inductive step:* Assume that all constraints at topological depth > k have been correctly repaired (matching the monolithic solution). Consider a constraint sigma_i at depth k. By the CIG construction, sigma_i has outgoing edges only to constraints at depth > k, which are already repaired. The key observation is that the absence of an edge (v_j, v_i) for any not-yet-repaired v_j (at depth <= k) means that repairs for those constraints cannot affect sigma_i's violations. Furthermore, since constraints at depth > k are already in their final repaired state, the violations that sigma_i sees are exactly the same as in the monolithic setting. Under unique minimum-change repair, sigma_i's repair is therefore identical to its monolithic counterpart.

*Uniqueness requirement:* The uniqueness condition is necessary because if multiple minimum-change repairs exist for a constraint, the decomposed and monolithic approaches may select different ones (breaking ties differently), leading to divergent downstream effects. In practice, tie-breaking can be enforced by a deterministic rule (e.g., lexicographic ordering of repair candidates), which restores the equivalence guarantee.

**Remark.** When the CIG contains cycles, the guarantee does not hold in general. CRIS handles this by breaking cycles at minimum-weight edges and applying a verification pass. The verification pass detects any violations introduced by the cycle-breaking approximation and applies targeted repairs. We bound the number of verification rounds needed by the feedback arc number of the CIG (typically 0-2 for real-world constraint sets).

### Step 4: Scheduled Repair Execution

CRIS composes existing single-constraint repair algorithms according to the computed schedule:

- For each independent component: select the appropriate repair algorithm (e.g., a minimum-change FD repairer for FD-only components, a DC repair algorithm for DC-only components)
- Execute components in parallel where possible
- Within each component, follow the DAG order from Step 3
- For cyclic residual edges: run a verification pass and apply targeted repairs if new violations are detected

### Key Innovations

1. **Formal hierarchy of repair safety conditions**: Three progressively finer levels of analysis (structural, predicate-aware, data-dependent) that trade off analysis cost for decomposition quality.

2. **Constraint Interaction Graph**: A novel graph structure that captures cross-constraint interference, enabling principled decomposition.

3. **Provable decomposition guarantees**: When the CIG is acyclic and individual repairs are unique minimum-change solutions, decomposed repair is equivalent to monolithic repair (see proof sketch above). When the CIG has independent components, speedup is proportional to the number of components.

4. **Practical composability**: CRIS works as a meta-algorithm that composes existing repair tools, not replacing them. It can use any single-constraint repair algorithm as a subroutine.

## Related Work

### Holistic Data Cleaning

**Chu, Ilyas, and Papotti (ICDE 2013)** introduced holistic data cleaning using denial constraints and conflict hypergraphs. Their work was the first to recognize the importance of considering constraint interactions, encoding all constraints into a single conflict hypergraph. However, they did not analyze *which* constraints actually interact---all are treated as potentially interacting, leading to a monolithic optimization. CRIS explicitly identifies safe (non-interacting) constraint pairs and exploits this structure for decomposition.

**Rekatsinas et al. (PVLDB 2017)** proposed HoloClean, which unifies integrity constraints with statistical signals via probabilistic inference. HoloClean achieves high repair quality (~90% precision) but its probabilistic program scales poorly with constraint set size. CRIS can use HoloClean as a subroutine for small dense subproblems while handling the overall decomposition efficiently.

### Scalable Constraint-Based Repair

**Rezig et al. (PVLDB 2021)** proposed Horizon, achieving linear-time FD repair by leveraging data patterns induced by FDs. However, Horizon handles only FDs---a single constraint type. CRIS generalizes interaction analysis to heterogeneous constraints (FDs, CFDs, DCs) and their cross-type interactions.

**Zhao, Han, and Wan (arXiv 2026)** proposed topology-aware subset repair via graph decomposition. Their work decomposes conflict graphs for CFD repair into independent subgraphs. CRIS differs by analyzing interactions *between different constraint types* rather than within a single type, and uses update repair rather than subset repair (tuple deletion).

### Pipeline Optimization

**Krishnan and Wu (arXiv 2019)** proposed AlphaClean, which searches for good cleaning pipeline orderings. **Siddiqi, Kern, and Boehm (SIGMOD 2024)** developed SAGA for scalable pipeline optimization. Both treat operator ordering as a black-box search problem. CRIS provides formal analysis of *why* certain orderings are better (based on repair safety), and its CIG can be used to prune these systems' search spaces.

### Constraint Discovery and Repair Complexity

**Chu, Ilyas, and Papotti (PVLDB 2013)** developed algorithms for discovering denial constraints, which subsume FDs and many other dependency types. **Livshits, Kimelfeld, and Roy (ACM TODS 2020)** showed that computing optimal repairs for FDs is NP-hard even for a single FD, but identified tractable special cases. CRIS's decomposition reduces the effective problem size for each subproblem, making approximation algorithms more tractable.

**Khayyat et al. (SIGMOD 2015)** proposed BigDansing for scalable data cleaning with a logical plan of operators. BigDansing optimizes execution of individual rules but does not analyze cross-rule interactions.

### Incremental and Streaming Repair

Recent work on incremental DC violation detection (PVLDB 2024) and non-blocking FD discovery from data streams (Information Sciences 2024) addresses the dynamic setting. CRIS currently targets the batch setting but its decomposition naturally supports incremental updates: when new data arrives, only the affected components of the CIG need re-repair.

### Comparison Table

| System | Constraint Types | Interaction Analysis | Decomposition | Guarantees |
|--------|-----------------|---------------------|---------------|------------|
| Holistic (Chu+ '13) | DCs | Implicit (hypergraph) | No | Minimal repair |
| HoloClean ('17) | DCs + statistical | None (joint inference) | No | Probabilistic |
| Horizon ('21) | FDs only | Intra-FD patterns | No | Linear time |
| Topo-Aware ('26) | CFDs | Single-type conflict | Single type | Density-based |
| AlphaClean ('19) | Any (pipeline) | None (search) | No | No |
| SAGA ('24) | Any (pipeline) | None (search) | No | No |
| **CRIS (ours)** | **Heterogeneous** | **Cross-type CIG** | **Yes, multi-type** | **Safety-based** |

## Experiments

### Setup

**Datasets:**
- **Hospital** (~1,000 tuples, 19 attributes): The standard Hospital dataset from HoloClean's public repository (github.com/HoloClean/holoclean/testdata/hospital.csv), widely used as a benchmark for data repair evaluation. Contains FDs (e.g., HospitalName, Address -> City, State, ZipCode; Provider_Number -> HospitalName) and DCs (e.g., consistency of quality measures). We use the actual dataset and its associated denial constraints (hospital_constraints.txt) from the HoloClean repository, with additional FDs discovered using standard algorithms. Errors are present natively; we additionally inject controlled errors using BART for systematic evaluation.
- **Tax** (~200K tuples, 15 attributes): The Tax dataset originally from Fan et al., used as a benchmark in BART and multiple data cleaning studies. Contains FDs (e.g., zip -> city, state; salary, filing_status -> tax_rate) and DCs (e.g., salary ordering constraints). We obtain this dataset from the BART project repository (github.com/dbunibas/BART) which provides the canonical version with associated constraints. Errors are injected using BART's error generator at controlled rates.
- **Flights** (~2,400 tuples, 7 attributes): The Flights dataset used in HoloClean's original evaluation, containing flight departure and arrival times as reported by different data sources. Contains FDs (e.g., flight_id, source -> departure_time, arrival_time) and DCs on temporal consistency (e.g., arrival after departure). Obtained from the HoloClean project's benchmark data. This dataset exhibits data fusion challenges where multiple sources provide conflicting values for the same flights.
- **Beers** (~2,400 tuples, 11 attributes): The Beers dataset from the REIN benchmark. Contains FDs on brewery-location mappings and type consistency constraints. Included as a small-scale dataset to test CRIS behavior when decomposition opportunities may be limited.
- **Synthetic**: Controlled datasets with varying numbers of constraints (5-50), attribute overlap ratios (0%-100%), and error rates (1%-15%). This allows systematic evaluation of how CIG density affects CRIS's performance.

**Constraint sets**: For each real dataset, we use a mix of:
- 5-10 FDs (discovered using standard FD discovery algorithms, e.g., HyFD)
- 3-5 CFDs (manually specified based on domain knowledge documented in prior work)
- 5-10 DCs (discovered using Hydra/FastDC, or taken from the dataset's accompanying constraint files)

**Error injection**: For Hospital (which contains native errors), we use both the native errors and BART-injected errors at controlled 5% and 10% rates. For Tax, Flights, and Beers, we use the BART error generator to inject 5-10% errors that violate the specified constraints.

**Baselines:**
1. **Sequential-Independent**: Apply FD repair, then CFD repair, then DC repair sequentially with no interaction awareness
2. **Reverse-Independent**: Reverse order (DC repair, then CFD, then FD)
3. **Greedy-Holistic**: A greedy joint optimizer that considers all constraints simultaneously, selecting the repair for each cell that resolves the maximum number of violations. This is our own implementation and is **not** HoloClean. It serves as a proxy for holistic approaches, trading optimality for simplicity. We clearly distinguish it from HoloClean in all results.
4. **HoloClean** (where feasible): We attempt to run the open-source HoloClean system (github.com/HoloClean/holoclean) on the Hospital dataset (its native benchmark) as a reference point. Due to HoloClean's PostgreSQL dependency and scaling characteristics, we may not be able to run it on all datasets within our time budget; in such cases, we report only the Greedy-Holistic baseline and note the limitation.
5. **CRIS-Structural**: Our approach using only Level 1 (structural) safety
6. **CRIS-Full**: Our approach with all three levels of safety analysis

**Metrics:**
- Repair precision: fraction of changed cells that were actually erroneous
- Repair recall: fraction of erroneous cells that were correctly repaired
- Repair F1: harmonic mean of precision and recall
- Runtime: wall-clock time for the full repair process
- CIG density: fraction of edges present in the CIG (measures interaction sparsity)
- Cascade count: number of new violations introduced during repair

### Planned Experiments

**Experiment 1: Interaction Sparsity in Real Datasets.**
Compute the CIG for each real dataset's constraint set. Measure the fraction of constraint pairs that are safe at each level (structural, predicate-aware, data-dependent). We hypothesize that >60% of constraint pairs are structurally safe, and >80% are safe at the data-dependent level.

**Experiment 2: Repair Quality Comparison.**
Compare repair F1 of all methods across the 4 real datasets (Hospital, Tax, Flights, Beers) with 3 random seeds for error injection. Report mean and standard deviation. We expect CRIS-Full to achieve F1 within 3% of Greedy-Holistic while significantly outperforming Sequential-Independent repair. Where HoloClean results are available (Hospital), we compare against those as well.

**Experiment 3: Runtime Scalability.**
Measure runtime as: (a) dataset size increases (subsampling Tax from 50K to 200K tuples), and (b) constraint set size increases from 5 to 50 on Synthetic data. We expect CRIS speedup over Greedy-Holistic to grow with constraint set size due to increased decomposition opportunity.

**Experiment 4: Decomposition Analysis.**
For each dataset, visualize the CIG and report the number of independent components, DAG structure, and cycle presence. Analyze how much of the speedup comes from parallelizing independent components vs. ordered execution of DAG components.

**Experiment 5: Cascade Analysis.**
Count the number of new constraint violations introduced at each step of repair for Sequential-Independent vs. CRIS. We expect CRIS to introduce significantly fewer cascading violations due to its safety-aware scheduling.

**Experiment 6: Safety Level Ablation.**
Compare CRIS-Structural vs. CRIS with predicate-aware safety vs. CRIS-Full. Measure the incremental benefit of each finer safety level in terms of additional safe pairs identified and quality improvement.

**Experiment 7: Epsilon Sensitivity Analysis.**
Evaluate CRIS-Full with epsilon in {0.001, 0.005, 0.01, 0.02, 0.05} on Hospital and Tax. For each value, measure: (a) number of CIG edges (decomposition aggressiveness), (b) number of independent components, (c) repair F1, and (d) cascade count. This characterizes the precision-recall trade-off of the data-dependent safety threshold and validates our default choice of epsilon = 0.01.

### Runtime Budget

| Component | Estimated Time |
|-----------|---------------|
| Dataset acquisition and preprocessing | 30 min |
| Constraint discovery (FDs, DCs) | 30 min |
| CIG construction (all levels) | 20 min |
| CRIS repair on 4 real datasets (3 seeds) | 2 hours |
| Baseline comparisons (Independent, Greedy-Holistic) | 1.5 hours |
| HoloClean on Hospital (if feasible) | 30 min |
| Scalability experiments | 1 hour |
| Sensitivity analysis (epsilon) | 30 min |
| Synthetic experiments | 1 hour |
| **Total** | **~7.5 hours** |

### Expected Results

- Real-world constraint sets produce sparse CIGs: >60% of pairs are structurally safe
- CRIS-Full achieves F1 within 2-5% of Greedy-Holistic on real datasets
- CRIS runs 3-10x faster than Greedy-Holistic, especially with >20 constraints
- Sequential-Independent repair introduces 30-50% more cascading violations than CRIS
- Structural safety alone captures >70% of safe pairs; predicate-aware adds ~15%; data-dependent adds the remaining ~10%
- CIG analysis overhead is <5% of total repair time
- Epsilon sensitivity shows a smooth trade-off: F1 degrades by <1% as epsilon increases from 0.001 to 0.01, then degrades more sharply beyond 0.02

## Success Criteria

### Confirming the hypothesis
- CIG-based decomposition produces at least 2 independent components on all real-world benchmark datasets
- Repair F1 within 5% of the strongest holistic baseline on all benchmarks
- Runtime speedup of at least 2x over Greedy-Holistic on constraint sets with >15 constraints
- Iterative repair converges within the theoretically bounded number of rounds for cyclic CIGs

### Refuting the hypothesis
- Real-world constraint sets produce dense CIGs (>80% of edges present), providing negligible decomposition
- CIG construction overhead exceeds the repair time savings
- Decomposed repair quality degrades by >10% F1 compared to holistic repair
- The three-level safety hierarchy provides insufficient discrimination (most pairs classified as unsafe regardless of level)

## References

1. Xu Chu, Ihab F. Ilyas, Paolo Papotti. "Holistic Data Cleaning: Putting Violations Into Context." ICDE, pp. 458-469, 2013.
2. Theodoros Rekatsinas, Xu Chu, Ihab F. Ilyas, Christopher Re. "HoloClean: Holistic Data Repairs with Probabilistic Inference." PVLDB 10(11): 1190-1201, 2017.
3. Sanjay Krishnan, Eugene Wu. "AlphaClean: Automatic Generation of Data Cleaning Pipelines." arXiv:1904.11827, 2019.
4. Shafaq Siddiqi, Roman Kern, Matthias Boehm. "SAGA: A Scalable Framework for Optimizing Data Cleaning Pipelines for Machine Learning Applications." SIGMOD 2024.
5. El Kindi Rezig, Mourad Ouzzani, Walid G. Aref, Ahmed K. Elmagarmid, Ahmed R. Mahmoud, Michael Stonebraker. "Horizon: Scalable Dependency-driven Data Cleaning." PVLDB 14(11): 2546-2559, 2021.
6. Guoqi Zhao, Xixian Han, Xiaolong Wan. "Topology-Aware Subset Repair via Entropy-Guided Density and Graph Decomposition." arXiv:2601.19671, 2026.
7. Xu Chu, Ihab F. Ilyas, Paolo Papotti. "Discovering Denial Constraints." PVLDB 6(13): 1498-1509, 2013.
8. Ester Livshits, Benny Kimelfeld, Sudeepa Roy. "Computing Optimal Repairs for Functional Dependencies." ACM TODS 45(1): 4:1-4:46, 2020.
9. Zuhair Khayyat, Ihab F. Ilyas, Alekh Jindal, Samuel Madden, Mourad Ouzzani, Paolo Papotti, Jorge-Arnulfo Quiane-Ruiz, Nan Tang, Si Yin. "BigDansing: A System for Big Data Cleansing." SIGMOD, pp. 1215-1230, 2015.
10. Qixu Chen, Yeye He, Raymond Chi-Wing Wong, Weiwei Cui, Song Ge, Haidong Zhang, Dongmei Zhang, Surajit Chaudhuri. "Auto-Test: Learning Semantic-Domain Constraints for Unsupervised Error Detection in Tables." SIGMOD 2025.
11. Nataliya Prokoshyna, Jaroslaw Szlichta, Fei Chiang, Renee J. Miller, Divesh Srivastava. "Combining Quantitative and Logical Data Cleaning." PVLDB 9(4): 300-311, 2015.
12. Ihab F. Ilyas, Xu Chu. "Data Cleaning." ACM Books, Morgan & Claypool, 2019.
13. Patricia C. Arocena, Boris Glavic, Giansalvatore Mecca, Renee J. Miller, Paolo Papotti, Donatello Santoro. "Messing Up with BART: Error Generation for Evaluating Data-Cleaning Algorithms." PVLDB 9(2): 36-47, 2015.
14. Mohamed Abdelaal, Christian Hammacher, Harald Schoening. "REIN: A Comprehensive Benchmark Framework for Data Cleaning Methods in ML Pipelines." EDBT 2023. (Source for Beers benchmark dataset.)
