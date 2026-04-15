# The Bandwidth Knapsack: Optimal Migration Scheduling for Tiered Memory Systems

## Introduction

### Context

Modern server architectures increasingly feature heterogeneous memory tiers: fast local DRAM, slower CXL-attached memory, and optionally persistent memory (NVM). Operating systems manage these tiers by monitoring page access patterns and migrating pages between tiers -- promoting hot pages to fast memory and demoting cold pages to slow memory. This tiered memory management is critical: poor placement can degrade application performance by 2-5x due to the latency gap between tiers (e.g., ~80ns for local DRAM vs. ~170-250ns for CXL memory).

A rich body of recent systems work has addressed the question of **which pages to migrate** -- ranking pages by access frequency (TPP, MEMTIS), access latency impact (Colloid), amortized offcore latency (SOAR/ALTO), or using non-exclusive copies (Nomad). However, all these systems treat the complementary question of **how to schedule migrations under limited bandwidth** as secondary, using simple heuristics: fixed migration-rate caps, FIFO queuing of candidates, or ad-hoc throttling.

### Problem Statement

Migration bandwidth between memory tiers is a scarce, shared resource. On current CXL hardware, the interconnect bandwidth is shared between regular memory accesses and migration traffic. Excessive migration can saturate the interconnect, degrading performance for all tenants. Conversely, under-migration leaves hot data stranded in slow tiers. The migration scheduling problem -- deciding which subset of candidate pages to migrate in each scheduling epoch, given a bandwidth budget -- is a constrained optimization problem that current systems solve with greedy heuristics.

### Key Insight

We observe that migration scheduling in tiered memory is structurally equivalent to the **online knapsack problem with renewable capacity**: each epoch presents a set of migration candidates (items), each with a benefit (expected latency reduction times access frequency) and a cost (migration bandwidth consumed). The bandwidth budget (knapsack capacity) renews each epoch. Pages not migrated in one epoch may or may not be candidates in the next (their hotness changes), and migrations in one epoch alter the system state for subsequent epochs.

This connection to online knapsack theory is powerful because it immediately yields: (1) lower bounds on what any online algorithm can achieve, (2) provably near-optimal algorithms from the online optimization literature, and (3) a framework for analyzing existing heuristics' worst-case behavior.

### Hypothesis

We hypothesize that existing greedy-by-rank migration schedulers (which simply migrate the highest-ranked pages until bandwidth is exhausted) are provably suboptimal in the worst case, and that a knapsack-aware migration scheduler can improve application performance by 3-12% on bandwidth-constrained tiered memory configurations, with larger gains under high contention (multiple tenants competing for migration bandwidth).

## Proposed Approach

### Overview

We propose **BKS (Bandwidth Knapsack Scheduler)**, a migration scheduling framework for tiered memory systems that formulates per-epoch migration decisions as online knapsack instances and solves them using theoretically-grounded algorithms. BKS is designed to be a drop-in replacement for the migration scheduling component of any existing tiered memory system -- it takes as input the ranked list of migration candidates produced by any page ranking policy (TPP's hotness, Colloid's latency, ALTO's AOL) and outputs the subset to actually migrate, respecting bandwidth constraints.

### Formal Problem Definition

**The Bandwidth-Constrained Migration Scheduling (BCMS) Problem:**

- **Input per epoch t**: A set of migration candidates C_t = {c_1, ..., c_n}, where each candidate c_i has:
  - benefit b_i: expected performance improvement from migration (e.g., access_frequency x latency_reduction)
  - cost w_i: bandwidth consumed by migrating the page (depends on page size: 4KB base page or 2MB huge page)
  - direction d_i: promotion (slow to fast) or demotion (fast to slow)
- **Constraint**: Total migration bandwidth per epoch B (e.g., 500MB/s x epoch_length)
- **Objective**: Maximize total benefit sum(b_i * x_i) subject to sum(w_i * x_i) <= B, x_i in {0,1}
- **Online aspect**: The candidate set C_{t+1} depends on decisions made at epoch t (migrating a page changes the system state). Future candidate sets are unknown.

This is a variant of the online knapsack problem where: (a) capacity renews each epoch, (b) items (pages) persist across epochs with changing benefits, and (c) decisions have state-dependent consequences.

### Key Innovations

**1. Competitive Analysis of Existing Policies**

We prove that the greedy-by-rank policy (migrate in order of benefit until bandwidth exhausted, ignoring cost heterogeneity) has a competitive ratio of Omega(R) in the worst case, where R = w_max / w_min is the ratio between the largest and smallest page sizes. For systems with both 4KB and 2MB pages, R = 512, meaning greedy can be 512x worse than optimal in adversarial scenarios. This worst case arises when many small high-benefit pages are passed over in favor of a single large page that happens to rank first.

**2. Benefit-Density Scheduling**

Our primary algorithm ranks candidates by benefit density (b_i / w_i) rather than raw benefit, with a threshold-based acceptance rule derived from the online knapsack literature. Specifically, we use an exponential threshold function: accept candidate c_i if b_i / w_i >= phi(L_t), where L_t is the current epoch's bandwidth utilization fraction and phi is a monotonically increasing function calibrated to achieve O(log R)-competitive ratio. This approach is inspired by the threshold-based algorithms of Sun et al. (SIGMETRICS 2023) for online knapsack with departures.

**3. Multi-Epoch Lookahead via Decay Estimation**

Pure per-epoch optimization ignores temporal dynamics: a page that is hot now but cooling should be deprioritized relative to a page that is warming up. We augment the per-epoch knapsack with a decay-adjusted benefit: b'_i = b_i * decay_factor(c_i), where the decay factor is estimated from the page's recent access trend (increasing, stable, or decreasing). This is computed using exponentially weighted moving averages of per-epoch access counts, requiring only O(1) additional state per tracked page.

**4. Fairness-Aware Multi-Tenant Extension**

In multi-tenant settings, migration bandwidth must be shared across tenants. We extend BCMS to the multi-tenant case by formulating it as a multi-dimensional online knapsack: in addition to the bandwidth constraint, each tenant has a minimum bandwidth guarantee (fairness floor). We adapt the competitive algorithms of Lechowicz et al. (SIGMETRICS 2022) for online multidimensional knapsack to this setting.

### Algorithm Summary

```
BKS Migration Scheduler (per epoch):
1. Receive candidate list C_t from page ranking policy
2. For each candidate c_i:
   a. Compute benefit b_i (from ranking policy's score)
   b. Compute cost w_i (page size)
   c. Compute decay-adjusted benefit b'_i = b_i * decay(c_i)
   d. Compute density d_i = b'_i / w_i
3. Sort candidates by density d_i (descending)
4. Initialize utilized bandwidth L = 0
5. For each candidate in sorted order:
   a. If d_i >= threshold(L / B):
      - Accept: schedule migration, L += w_i
   b. If L >= B: stop
6. Execute accepted migrations
```

### Lightweight Implementation

BKS adds minimal overhead to existing systems:
- Per-page state: one 8-byte counter for decay estimation
- Per-epoch computation: O(n log n) sort of candidates (n is typically 100s-1000s per epoch)
- No new kernel data structures: reuses existing page tracking from the underlying tiering system
- Composable: works with any page ranking policy as a plug-in scheduling layer

## Related Work

### Tiered Memory Management Systems

**TPP** (Al Maruf et al., ASPLOS 2023) introduced transparent page placement for CXL-enabled tiered memory using page reclamation for demotion and NUMA hinting faults for promotion. TPP's migration scheduling is FIFO-based with a configurable rate limit. BKS can replace TPP's scheduler to optimize which pages actually get migrated within the rate budget.

**MEMTIS** (Lee et al., SOSP 2023) uses access distribution histograms for page classification and dynamic page size determination. MEMTIS profiles pages to determine the hotness distribution but uses a simple threshold-based migration policy without optimizing migration bandwidth allocation.

**Colloid** (Vuppalapati et al., SOSP 2024) identifies access latency (not just hotness) as the key metric for tier placement. Colloid includes a migration limit per scheduling quantum as an explicit parameter, acknowledging bandwidth as a constraint. However, it selects pages greedily by latency impact without considering the cost heterogeneity of migrating different page sizes.

**Nomad** (Xiang et al., OSDI 2024) sidesteps migration bandwidth pressure through non-exclusive tiering: promoted pages retain shadow copies in the slow tier, enabling instant demotion without bandwidth cost. This is complementary to BKS -- even with shadow copies, promotion bandwidth remains limited.

**SOAR/ALTO** (Liu et al., OSDI 2025) go beyond hotness by using Amortized Offcore Latency (AOL) as the ranking metric. ALTO dynamically regulates migration rates but does not formalize the scheduling problem or provide theoretical guarantees on bandwidth utilization efficiency.

**Jenga** (Kadekodi et al., 2025) addresses migration thrashing through gradual hotness decay and context-based allocation. Its approach to preventing unnecessary migrations is complementary to BKS's goal of optimizing which necessary migrations to prioritize.

**Nimble** (Yan et al., ASPLOS 2019) focuses on the mechanism of fast page migration (reducing per-page migration latency) rather than the policy of which pages to migrate. BKS addresses the complementary policy question.

### Online Optimization Theory

**Competitive Algorithms for Online Multidimensional Knapsack** (Lechowicz et al., SIGMETRICS 2022) provides threshold-based algorithms with O(log(theta * alpha))-competitive ratios for online knapsack problems. We adapt their framework to the tiered memory setting.

**The Online Knapsack Problem with Departures** (Sun et al., SIGMETRICS 2023) extends online knapsack to settings where items have finite lifetimes, analogous to pages whose hotness changes. Their threshold-based approach and data-driven extensions are directly applicable to our decay-adjusted formulation.

**The Primal-Dual Method for Online Algorithms** (Buchbinder and Naor, 2009) provides the foundational framework we use for competitive analysis.

### How BKS Differs

All prior tiered memory systems treat migration scheduling as a secondary concern -- a simple rate limiter or FIFO queue applied after page ranking. BKS is the first to: (1) formalize migration scheduling as an online optimization problem, (2) prove worst-case bounds on existing heuristics, (3) provide a scheduler with provable competitive guarantees, and (4) cleanly separate the "what to migrate" question (page ranking) from the "how much to migrate" question (bandwidth scheduling) into composable layers.

## Experiments

### Experimental Setup

All experiments run on CPU only (no GPU required). We build a trace-driven tiered memory simulator that models:
- Two or three memory tiers with configurable latency and bandwidth parameters
- Page-granularity access tracking (4KB and 2MB pages)
- Migration bandwidth as a shared, limited resource
- Multiple concurrent workloads (multi-tenant scenarios)

### Memory Access Traces

We use three categories of traces:

1. **Synthetic traces**: Generated with configurable parameters (working set size, access distribution skew, phase change frequency) to systematically explore the parameter space and create adversarial inputs.

2. **SPEC CPU 2017 memory traces**: We extract memory access traces from SPEC CPU 2017 benchmarks using Pin or Valgrind's Lackey tool. Target benchmarks: mcf (pointer-chasing, irregular), lbm (streaming), xalancbmk (mixed), and omnetpp (object-heavy). These provide realistic single-application access patterns with diverse memory behaviors.

3. **Multi-programmed workloads**: Combinations of SPEC traces to simulate multi-tenant scenarios with competing migration demands.

### Baselines

1. **Greedy-by-rank**: Migrate pages in order of benefit score until budget exhausted (current practice in TPP, Colloid, ALTO)
2. **Fixed-rate FIFO**: Migrate in arrival order up to a fixed bandwidth limit (TPP default)
3. **Offline optimal (LP relaxation)**: Solve the per-epoch knapsack optimally with hindsight (upper bound on achievable performance)
4. **Random scheduling**: Randomly select candidates up to budget (lower bound)

### Evaluation Metrics

- **Application performance**: Simulated weighted memory access latency (lower is better)
- **Migration bandwidth utilization**: Fraction of budget used for beneficial migrations
- **Migration efficiency**: Benefit achieved per byte migrated
- **Fairness (multi-tenant)**: Jain's fairness index across tenant performance
- **Competitive ratio**: Empirical ratio of online algorithm's cost to offline optimal

### Planned Experiments

**Experiment 1: Single-Tenant Performance.** Compare BKS against baselines across all traces. Vary bandwidth budget from 10% to 100% of maximum migration rate. Measure application latency reduction.

**Experiment 2: Page Size Heterogeneity Impact.** Evaluate with workloads that mix 4KB and 2MB pages. The benefit-density approach should show the largest advantage here, where greedy-by-rank fails most.

**Experiment 3: Multi-Tenant Contention.** Run 2, 4, 8 concurrent workloads sharing a fixed migration budget. Compare fairness and total throughput of BKS vs. greedy partitioning.

**Experiment 4: Sensitivity Analysis.** Vary key parameters: epoch length (1ms to 100ms), bandwidth budget (100MB/s to 10GB/s), tier latency ratio (1.5x to 5x), working set sizes. Identify operating regimes where BKS provides the largest gains.

**Experiment 5: Composability.** Plug BKS into simulated versions of TPP, Colloid, and ALTO ranking policies. Show that BKS improves all of them, demonstrating composability.

**Experiment 6: Adversarial Analysis.** Construct worst-case inputs that maximize the gap between greedy and BKS. Verify that empirical competitive ratios match theoretical predictions.

### Expected Results

- BKS improves application performance by **3-12%** over greedy-by-rank under bandwidth-constrained scenarios (modest gains when bandwidth is abundant, larger gains when constrained).
- Gains are largest (8-12%) when page size heterogeneity is high (mixed 4KB/2MB workloads) and bandwidth is most constrained (< 30% of max rate).
- Multi-tenant fairness improves by **10-25%** (Jain's index) under contention.
- Empirical competitive ratio of BKS is within 1.1-1.3x of offline optimal, vs. 1.5-3x for greedy-by-rank.
- BKS scheduling overhead is < 0.1% of epoch time (microseconds for sorting ~1000 candidates).

## Success Criteria

1. **Theory**: Prove that greedy-by-rank has Omega(R) competitive ratio and BKS achieves O(log R), where R is the page size ratio.
2. **Practice**: BKS improves simulated application latency by >= 3% over greedy-by-rank on at least 4 of 6 SPEC workloads under bandwidth-constrained settings.
3. **Composability**: BKS improves performance when combined with at least 2 of 3 tested ranking policies (TPP, Colloid, ALTO simulations).
4. **Fairness**: BKS achieves higher Jain's fairness index than equal-share bandwidth partitioning in multi-tenant experiments.

### What Would Refute the Hypothesis

- If greedy-by-rank performs within 2% of BKS across all workloads, the optimization is not practically useful (even if theoretically justified).
- If bandwidth is never the bottleneck in realistic configurations (all beneficial migrations always fit within the budget), the problem doesn't manifest.
- If page size heterogeneity is rare in practice (most systems use only 4KB or only 2MB, not mixed), the key advantage of density-based scheduling disappears.

## References

1. Hasan Al Maruf, Hao Wang, Abhishek Dhanotia, Johannes Weiner, Niket Agarwal, Pallab Bhattacharya, Chris Petersen, Mosharaf Chowdhury, Shobhit Kanaujia, and Prakash Chauhan. "TPP: Transparent Page Placement for CXL-Enabled Tiered-Memory." ASPLOS 2023.

2. Taehyung Lee, Sumit Kumar Monga, Changwoo Min, and Young Ik Eom. "MEMTIS: Efficient Memory Tiering with Dynamic Page Classification and Page Size Determination." SOSP 2023.

3. Midhul Vuppalapati, Rachit Agarwal, Dan Ports, and Abraham Silberschatz. "Tiered Memory Management: Access Latency is the Key!" SOSP 2024.

4. Weiqian Xiang, Ke Liu, Lei Wang, and Ziqi Shuai. "Nomad: Non-Exclusive Memory Tiering via Transactional Page Migration." OSDI 2024.

5. Jinshu Liu, Hamid Hadian, Hanchen Xu, and Huaicheng Li. "Tiered Memory Management Beyond Hotness." OSDI 2025.

6. Rohan Kadekodi et al. "Jenga: Responsive Tiered Memory Management without Thrashing." arXiv:2510.22869, 2025.

7. Musa Unal, Vishal Gupta, Yueyang Pan, Yujie Ren, and Sanidhya Kashyap. "Tolerate It if You Cannot Reduce It: Handling Latency in Tiered Memory." HotOS 2025.

8. Adam Lechowicz, Bo Sun, Mohammad Hajiesmaili, and Adam Wierman. "Competitive Algorithms for Online Multidimensional Knapsack Problems." SIGMETRICS 2022.

9. Bo Sun, Lin Yang, Mohammad Hajiesmaili, Adam Wierman, John C.S. Lui, Don Towsley, and Danny H.K. Tsang. "The Online Knapsack Problem with Departures." SIGMETRICS 2023.

10. Niv Buchbinder and Joseph Naor. "The Design of Competitive Online Algorithms via a Primal-Dual Approach." Foundations and Trends in Theoretical Computer Science, 2009.

11. Zi Yan, Daniel Lustig, David Nellans, and Abhishek Bhattacharjee. "Nimble Page Management for Tiered Memory Systems." ASPLOS 2019.

12. Yuhong Zhong, Drew Roselli, and Mark Silberstein. "Managing Memory Tiers with CXL in Virtualized Environments." OSDI 2024.
