# ShareArb: Evictor-Side Responsibility for Shared Linux Page-Cache Arbitration

## Introduction

Linux page-cache research no longer lacks mechanism. `FetchBPF`, PageFlex, and `cache_ext` show that Linux paging and file-cache policy are now programmable enough to support specialized replacement and control policies. Likewise, broad claims about tenant-aware accounting and page-cache partitioning are no longer available: `pCache` already separates private and shared page-cache accounting for containers, and Per-VM Page Cache Partitioning already studies utility-driven share control plus tenant-specific replacement.

The narrower unresolved problem is this: when multiple tenants overlap on the same file-backed pages, who should be held responsible when one tenant's fault displaces a shared page that another tenant will soon need again? Existing occupancy accounting tells us who currently consumes memory, but not who caused cross-tenant harm at eviction time. That distinction matters because a tenant can evict a page that is only fractionally charged to it under proportional sharing, yet the resulting miss cost may fall mostly on another tenant with stronger near-term reuse.

This proposal therefore makes a deliberately narrow claim.

**Claim.** `ShareArb` is a new eviction-responsibility heuristic for overlap-heavy shared page-cache arbitration.

It is not:

- a new shared-page accounting framework;
- a new programmable page-cache substrate;
- a claim that online joint share-plus-policy control is itself novel.

Instead, the paper asks whether an evictor-side responsibility signal, layered on top of a `pCache`-style shared/private occupancy baseline, materially improves fairness and regret on overlap-heavy workloads.

**Key insight.** The right arbitration signal is not just "who occupies this shared page now?" but also "who caused harm by evicting a shared page that others were likely to reuse soon?"

**Hypothesis.** Relative to occupancy-only arbitration built on `pCache`-style shared/private charging, adding an evictor-side responsibility debt will reduce shared-page regret and improve worst-tenant slowdown on overlap-heavy workloads, while collapsing to baseline behavior when overlap is weak or absent.

## Proposed Approach

### Overview

`ShareArb` has four components:

1. a `pCache`-style private/shared occupancy baseline;
2. a lightweight per-page Shared Responsibility Vector (SRV);
3. an evictor-side debt signal that charges predicted cross-tenant harm to the tenant that triggered the eviction;
4. a small controller that chooses coarse cache shares and one of a few existing replacement profiles.

The contribution is the third component and the way it changes arbitration decisions; the rest is intentionally conservative scaffolding that makes the comparison scientifically defensible.

### 1. Baseline shared/private accounting

For each resident file-backed page `g`, let `C_g` be the set of tenants that referenced it within a bounded recent window. Occupancy is charged as follows:

- if `|C_g| = 1`, the page is private and the sole tenant receives charge `1`;
- if `|C_g| > 1`, the page is shared and its occupancy is divided proportionally across current consumers.

This baseline intentionally mirrors the role played by `pCache`: it answers "who is using the cache?" but not "who caused harmful evictions?"

The main baselines are:

- **PrivateOnly-Utility**: private-only charging plus utility-based share choice;
- **pCache-Account**: proportional shared/private charging only;
- **pCache-Account+Policy**: the same occupancy baseline, plus per-tenant replacement-profile selection.

These baselines ensure the paper is judged against adjacent prior art rather than against a weak strawman.

### 2. Shared Responsibility Vector

For each shared page `g`, `ShareArb` maintains a normalized consumer-weight vector:

`r_g(i) = reuse_g(i) / sum_j reuse_g(j)`

where `reuse_g(i)` is estimated from either:

- a bounded window of recent references to `g`; or
- an exponentially decayed reuse counter.

This yields a compact estimate of which tenants are most likely to benefit if the page remains resident. The vector satisfies:

- `sum_i r_g(i) = 1`;
- identical consumers receive equal weight;
- private pages degenerate to a single tenant with weight `1`.

SRV is not presented as a new accounting theory. It is simply the state needed to estimate downstream harm at eviction time.

### 3. Evictor-side responsibility debt

When tenant `i` faults and the chosen victim is a shared page `g`, `ShareArb` assigns debt to the evictor:

`debt_i += sum_{j != i} r_g(j) * cost_j(g)`

where `cost_j(g)` is the predicted miss penalty if tenant `j` re-references `g` after eviction.

This is the core heuristic. It separates three signals that prior occupancy-only schemes conflate:

- **occupancy**: who currently uses cache space;
- **benefit**: who is likely to reuse the shared page soon;
- **responsibility**: who triggered the eviction that destroyed that shared benefit.

Debt is decayed each control epoch:

`debt_i(t+1) = lambda * debt_i(t) + new_harm_i(t)`

with half-life chosen relative to observed working-set turnover and tested in sensitivity analysis.

### 4. Miss-cost proxy

The work does not require a perfect miss predictor. It needs a stable relative ranking of harm. `cost_j(g)` will therefore use a calibrated family-level proxy:

- run each workload family in isolation;
- measure median cache-hit service time and cold-miss service time;
- use the family-specific miss gap as the default penalty;
- test robustness against a unit-cost alternative.

If the conclusion holds under both, then the gains are attributable to responsibility attribution rather than delicate cost tuning.

### 5. Small online controller

Each tenant is assigned:

- one replacement profile from `{LRU, SCAN, FREQ}`;
- one coarse cache-share allocation.

For tenant `i`, the controller uses:

`pressure_i = occ_i + eta * debt_i`

and scores decisions with a simple utility objective that trades throughput proxy, slowdown, and pressure. The controller is intentionally small because the paper is not about advanced optimization.

The share grid is defined as an integer simplex over 10 equal cache quanta with a minimum of one quantum per tenant. This fixes the inconsistency in the previous draft:

- for 2 tenants, the action space is the 9 valid allocations from `(10, 90)` through `(90, 10)`;
- for 3 tenants, the action space is all 36 valid allocations whose quanta sum to 10 with each tenant receiving at least one quantum.

This same grid supports both the 2-tenant and 3-tenant cases without changing controller logic.

### 6. Reduction behavior

`ShareArb` includes an explicit fallback rule:

- if measured overlap is below a threshold, or SRV confidence is too weak, set `eta = 0`;
- the controller then reduces to occupancy-only arbitration.

This makes the method falsifiable. On disjoint workloads, it should not invent harmful behavior.

## Related Work

### Shared/private page-cache accounting

Wang et al.'s `pCache` is the most important adjacent prior work. It directly addresses inaccurate page-cache accounting for containers by distinguishing private and shared cache occupancy and proportionally charging shared pages. `ShareArb` does not claim that contribution. The question here is whether occupancy accounting remains incomplete because it still lacks evictor-side attribution of cross-tenant harm.

Park et al.'s weight-aware page-cache management for application-level I/O proportionality similarly shows that occupancy matters for fairness objectives. However, it aims to align cache residency with weights rather than estimate which tenant caused downstream loss from evicting shared pages.

### Tenant-aware page-cache control

Per-VM Page Cache Partitioning already studies utility-based cache allocation plus per-tenant eviction control for VMs. That prior art rules out any claim that a combined share-and-policy controller is novel. In this paper, the controller is intentionally just a test harness for the new arbitration signal.

### Programmable Linux paging mechanisms

`FetchBPF`, PageFlex, and `cache_ext` demonstrate that Linux now has practical mechanisms for customized page-cache behavior. These systems are important because they remove "Linux cannot express this policy" as an objection. But they do not define the evictor-responsibility objective studied here.

### Shared-page eviction interference

Kim and Rajkumar's shared-page management work for temporal isolation in resource kernels is a missing adjacent citation from the earlier draft. Their Shared-Page Conservation and Shared-Page Eviction Lock mechanisms show that shared pages can create cross-domain interference when one workload causes another to suffer unexpected timing penalties. That paper is conceptually important because it identifies the interference phenomenon clearly.

It does not, however, solve the problem addressed here:

- it is aimed at real-time memory reservations, not best-effort multi-tenant page-cache arbitration;
- it uses protection/conservation mechanisms rather than an online evictor-side harm attribution signal;
- it does not estimate which tenant should be charged for future miss cost under overlap-heavy file-cache sharing.

### Fair sharing of shareable cached objects

Kunjir et al.'s `ROBUS` is another missing adjacent citation. It studies fairness when cached objects can be simultaneously useful to multiple tenants, which is conceptually closer to `ShareArb` than private-resource fairness models. `ROBUS` shows that fair allocation for shareable cache objects differs from ordinary partitioning because a single cached object may benefit many tenants at once.

That said, `ROBUS` still does not solve evictor-side page-cache responsibility attribution:

- it operates on batch allocations of shared cached objects in data-parallel systems, not online page eviction in Linux;
- it reasons about fair cache placement, not about which tenant should bear debt when a shared object is displaced;
- it does not use near-term reuse evidence to charge evictors for downstream harm.

### Utility and fairness signals from hardware-cache partitioning

Utility-Based Cache Partitioning and Fair Cache Sharing and Partitioning established the broader lesson that allocation quality depends on the signal being optimized, not only on the replacement rule. `ShareArb` applies that lesson to file-backed shared pages, where occupancy and responsibility can diverge sharply.

### Replay and trace-analysis methodology

Miniature Simulations, SHARDS, and CounterStacks show that compact replay and stack-distance methods can estimate cache utility efficiently. This makes a CPU-only experimental design credible: the proposal focuses on relative arbitration quality under trace replay, with live experiments used only as secondary sign checks rather than as proof of deployment readiness.

## Experiments

### Experimental questions

1. Does evictor-side responsibility improve arbitration beyond `pCache`-style shared/private occupancy charging?
2. Are the gains concentrated in overlap-heavy regimes, while vanishing on disjoint workloads?
3. Does the debt signal improve worst-tenant slowdown by reducing realized shared-page regret?
4. Are the conclusions robust to miss-cost and debt-decay choices?
5. Do replay-based method rankings survive a limited live sanity check?

### Implementation scope

The main artifact is a user-space page-cache replay engine and controller simulator, not a new kernel implementation. That scope matches the available resources:

- 2 CPU cores;
- 128 GB RAM;
- roughly 8 hours total runtime.

Live experiments are explicitly secondary and are reported only as limited sanity checks on method ordering.

### Workloads

The final matrix has five workload families:

1. **OverlapShift-2T**: two tenants repeatedly reuse a shared file region whose hot subset migrates over time.
2. **ScanVsLoop-2T**: one tenant streams through a shared region while the other repeatedly reuses a smaller hot subset.
3. **SQLiteTraceMix-2T**: a trace-derived family built from captured `pread64` accesses of two concurrent `sqlite3` query streams against the same immutable database file.
4. **SQLiteTraceMix-3T**: a three-tenant stress version formed by interleaving three captured query streams with unequal overlap.
5. **DisjointPhase-2T**: a negative control with no shared pages.

The trace-derived families address the external-validity weakness in the previous draft, but the artifact scope is narrower than the original draft. Concretely:

- build a moderate-size synthetic SQLite database with TPC-H-like tables generated inside the artifact;
- run representative analytical and point-query templates using stock `sqlite3`;
- capture file-offset traces through `pread64` instrumentation;
- convert file offsets to 4 KiB page IDs;
- replay and interleave the traces to form 2-tenant and 3-tenant overlap scenarios.

This produces a real file-backed trace family without requiring a full kernel implementation, but it should be interpreted as a replay-only external-validity proxy rather than as evidence from a public 2 GB benchmark dataset.

### Methods compared

The primary comparison set is:

- **PrivateOnly-Utility**
- **pCache-Account**
- **pCache-Account+Policy**
- **ShareArb**

Targeted ablations are:

- **ShareArb-NoDebt**: SRV state but no evictor-side debt;
- **ShareArb-UnitCost**: debt uses unit miss cost;
- **OracleOverlap**: an offline upper bound using future re-reference knowledge on a small subset.

The key novelty test is `pCache-Account` versus `ShareArb`; `ShareArb-NoDebt` isolates the added value of the debt term itself.

### Experimental grid and runtime budget

Primary matrix:

- 5 workload families
- 3 cache budgets: tight, medium, loose
- 3 random seeds
- 4 primary methods

Total primary replay runs:

`5 x 3 x 3 x 4 = 180`

Targeted ablations:

- `ShareArb-NoDebt` on the four overlap-heavy families: `4 x 3 x 3 = 36` runs
- `ShareArb-UnitCost` on the same four families: `36` runs
- `OracleOverlap` on one synthetic and one trace-derived family at tight budget: `2 x 3 = 6` runs

Total planned replay runs:

`180 + 36 + 36 + 6 = 258`

At an expected 30 to 45 seconds per replay, total replay time is roughly 2.2 to 3.3 hours. This leaves headroom within the 8-hour budget for trace capture, calibration, and a small number of live sanity checks.

### Metrics

- aggregate throughput proxy;
- worst-tenant slowdown;
- Jain fairness index;
- shared-page regret: the fraction of near-term reuses lost because another tenant evicted a shared page;
- responsibility precision: correlation between charged debt and realized downstream harm;
- controller stability: number of share/profile changes per replay hour;
- replay overhead: runtime and memory footprint.

### Live sanity checks

The live study is intentionally downgraded relative to the earlier draft. It is not used to claim that the full proposed controller has been realized in stock Linux.

Instead, the live component only tests whether replay-based rankings have the right sign:

- replay a small subset of `SQLiteTraceMix-2T` and `ScanVsLoop-2T` live under `cgroup v2 memory.high`;
- approximate the controller's decisions with coarse share changes and `posix_fadvise()`-based profile hints;
- sample residency using `mincore()` and workload-owned offset maps.

The interpretation is narrow: if replay predicts `ShareArb > pCache-Account`, the live check only asks whether that ordering is directionally preserved. If not, the main claim must rest on replay and be stated accordingly.

## Success Criteria

The paper is successful if all of the following hold:

- On the four overlap-heavy settings, `ShareArb` improves worst-tenant slowdown by at least 8 to 12 percent on average over `pCache-Account`.
- Aggregate throughput proxy is within 3 percent of `pCache-Account` in the worst case and typically improves.
- `ShareArb` materially reduces shared-page regret relative to both `pCache-Account` and `ShareArb-NoDebt`.
- Responsibility debt correlates positively with realized downstream harm in both synthetic and trace-derived families.
- On `DisjointPhase-2T`, `ShareArb` stays within 3 percent of `pCache-Account`, confirming correct reduction behavior.
- Live sanity checks preserve replay ordering in at least 3 of 4 tested conditions.

The hypothesis is weakened or refuted if any of the following occur:

- `pCache-Account` matches `ShareArb` on the overlap-heavy families;
- gains vanish when moving from synthetic traces to the SQLite-derived traces;
- the debt term helps only under one fragile decay setting or one miss-cost proxy;
- fairness gains come from severe throughput loss or unstable controller oscillation;
- live checks fail to preserve even the sign of replay-predicted improvements.

## References

Wang, K., Wu, S., Li, S., Huang, Z., Fan, H., Yu, C., and Jin, H. Precise control of page cache for containers. *Frontiers of Computer Science*, 18(2):182102, 2024.

Sharma, P., Kulkarni, P., and Shenoy, P. J. Per-VM Page Cache Partitioning for Cloud Computing Platforms. In *International Conference on Communication Systems and Networks (COMSNETS)*, 2016.

Park, J., Oh, K., and Eom, Y. I. Towards Application-level I/O Proportionality with a Weight-aware Page Cache Management. In *IEEE 36th Symposium on Mass Storage Systems and Technologies (MSST)*, 2020.

Cao, X., Patel, S., Lim, S. Y., Han, X., and Pasquier, T. FetchBPF: Customizable Prefetching Policies in Linux with eBPF. In *USENIX Annual Technical Conference (ATC)*, 2024.

Yelam, A., Wu, K., Guo, Z., Yang, S., Shashidhara, R., Xu, W., Novakovic, S., Snoeren, A. C., and Keeton, K. PageFlex: Flexible and Efficient User-space Delegation of Linux Paging Policies with eBPF. In *USENIX Annual Technical Conference (ATC)*, 2025.

Zussman, T., Zarkadas, I., Carin, J., Cheng, A., Franke, H., Pfefferle, J., and Cidon, A. `cache_ext`: Customizing the Page Cache with eBPF. In *ACM Symposium on Operating Systems Principles (SOSP)*, 2025.

Kim, H. and Rajkumar, R. Shared-Page Management for Improving the Temporal Isolation of Memory Reservations in Resource Kernels. In *IEEE 18th International Conference on Embedded and Real-Time Computing Systems and Applications (RTCSA)*, 2012.

Kunjir, M., Fain, B., Munagala, K., and Babu, S. ROBUS: Fair cache allocation for data-parallel workloads. In *Proceedings of the ACM SIGMOD International Conference on Management of Data*, pages 219-234, 2017.

Qureshi, M. K. and Patt, Y. N. Utility-Based Cache Partitioning: A Low-Overhead, High-Performance, Runtime Mechanism to Partition Shared Caches. In *IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 2006.

Kim, S., Chandra, D., and Solihin, Y. Fair Cache Sharing and Partitioning in a Chip Multiprocessor Architecture. In *International Conference on Parallel Architecture and Compilation Techniques (PACT)*, 2004.

Waldspurger, C., Saemundsson, T., Ahmad, I., and Park, N. Cache Modeling and Optimization using Miniature Simulations. In *USENIX Annual Technical Conference (ATC)*, 2017.

Waldspurger, C. A., Park, N., Garthwaite, A., and Ahmad, I. Efficient MRC Construction with SHARDS. In *13th USENIX Conference on File and Storage Technologies (FAST)*, 2015.

Tian, D., Jiang, H., Agrawal, K., and Li, K. Characterizing Storage Workloads with Counter Stacks. In *USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, 2015.
