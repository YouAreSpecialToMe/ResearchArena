# Anticipatory Page Migration via Markov Phase Models for Tiered Memory Systems

## Introduction

### Context

Modern data centers increasingly deploy tiered memory architectures combining fast local DRAM with slower but higher-capacity CXL-attached memory. These systems require intelligent page migration policies to place frequently accessed ("hot") pages in fast memory while relegating infrequently accessed ("cold") pages to slower tiers. The performance of such systems hinges on how quickly and accurately the OS can identify and migrate the right pages to the right tiers.

### Problem Statement

All existing page migration policies for tiered memory systems are fundamentally **reactive**: they observe page access patterns, detect hot or cold pages after a significant number of accesses have occurred, and then initiate migrations. This reactive approach incurs an inherent **migration lag** -- the time interval between when a workload's memory access pattern changes (e.g., due to a program phase transition) and when the affected pages have been migrated to the appropriate tier. During this lag, the workload suffers degraded performance because its newly-hot pages reside in slow memory while its newly-cold pages wastefully occupy fast memory.

This migration lag is not merely theoretical. MTM (EuroSys'24) demonstrated that DAMON-based profiling takes 3x longer than their approach to achieve 50% profiling accuracy after a phase change. ARMS (2025) introduced change-point detection to accelerate reaction to phase shifts but still fundamentally reacts to changes after they manifest. Even the state-of-the-art ALTO system (OSDI'25) -- which moves beyond simple hotness to the more sophisticated amortized offcore latency (AOL) metric -- remains reactive in its core design.

### Key Insight

Programs exhibit well-documented **phase behavior**: they cycle through distinct execution phases with markedly different memory access patterns, instruction mixes, and working sets. This phase behavior has been extensively studied in the computer architecture community (Sherwood et al., ASPLOS'02; Dhodapkar & Smith, ISCA'02) and exploited for cache management, DVFS, and simulation sampling. Critically, phase transitions in many workloads are **predictable** -- programs often cycle through phases in repeatable sequences, and transitions can be anticipated before they fully manifest.

### Hypothesis

We hypothesize that by modeling workload phase transitions as a Markov chain and using this model to **proactively** migrate pages before phase transitions occur, we can significantly reduce the migration lag penalty and improve application performance in tiered memory systems compared to state-of-the-art reactive approaches. Specifically, we conjecture that anticipatory migration can reduce the number of slow-tier accesses during phase transitions by 30-60% relative to reactive policies, while incurring at most 5-10% overhead from mispredicted migrations.

## Proposed Approach

### Overview

We propose **MarkovTier**, a phase-aware anticipatory page migration framework for tiered memory systems. MarkovTier operates in three stages:

1. **Online Phase Detection**: Continuously monitors workload behavior using lightweight access pattern signatures to identify distinct execution phases and detect phase transitions in real time.

2. **Markov Transition Learning**: Maintains a Markov chain model of phase transitions for each workload, learning transition probabilities online from observed phase sequences.

3. **Anticipatory Migration**: When the model predicts an imminent phase transition with sufficient confidence, proactively begins migrating pages that belong to the predicted next phase's working set to fast memory, while demoting pages that will become cold.

### Method Details

#### Phase Identification

We represent each phase by a **working set signature** -- a compact hash-based fingerprint of the set of pages accessed during a monitoring interval. Specifically, we use a Bloom filter or count-min sketch of accessed page frame numbers, updated at each profiling interval (e.g., every 100ms). Phase boundaries are detected when the Jaccard similarity between consecutive signatures drops below a threshold, indicating a significant change in the active working set.

Each unique signature is mapped to a phase ID via locality-sensitive hashing, allowing the system to recognize when a previously-seen phase recurs. This is inspired by the Basic Block Vector (BBV) approach of Sherwood et al. but operates at the memory access level rather than the instruction level, making it directly relevant to page migration decisions.

#### Markov Transition Model

Phase transitions are modeled as a first-order Markov chain:

- **States**: The set of identified phases {P_1, P_2, ..., P_k}
- **Transition matrix T**: T[i][j] = P(next phase = P_j | current phase = P_i)
- **Working set profiles**: For each phase P_i, the system maintains a compact representation of the hot pages (pages accessed more than a frequency threshold during the phase)

The transition matrix is learned online using exponential moving averages to adapt to changing workload behavior:

    T[i][j] = alpha * I(last_transition was i->j) + (1-alpha) * T[i][j]

where alpha is a learning rate (e.g., 0.1) that controls the balance between recent and historical transitions.

#### Anticipatory Migration Algorithm

At each profiling interval, MarkovTier:

1. Updates the current phase signature and checks for phase transition
2. If in stable phase P_i, computes the predicted next phase: P_j* = argmax_j T[i][j]
3. If T[i][j*] > confidence_threshold (e.g., 0.6), begins **anticipatory migration**:
   - Computes the set difference: pages_to_promote = hot_pages(P_j*) - hot_pages(P_i)
   - Computes pages_to_demote = hot_pages(P_i) - hot_pages(P_j*)
   - Initiates background migration at a rate that is proportional to T[i][j*] (higher confidence = more aggressive migration)
4. If a phase transition occurs and the prediction was correct, the working set is already (partially) in place, reducing the migration lag
5. If the prediction was incorrect, a rollback mechanism demotes speculatively promoted pages

#### Migration Rate Control

To prevent anticipatory migration from consuming too much bandwidth and interfering with application performance, MarkovTier employs a **budget-based rate controller**:

- A fixed fraction of total memory bandwidth (e.g., 10%) is allocated to anticipatory migrations
- Within this budget, migrations are prioritized by expected access frequency in the predicted phase
- If the system detects that anticipatory migration is causing measurable performance degradation (via latency monitoring), the budget is dynamically reduced

### Key Innovations

1. **First use of Markov phase models for page migration**: While phase detection has been used in caches and DVFS, and reactive phase detection has been applied to memory tiering (ARMS), no prior work has used a predictive phase transition model to drive proactive page migration decisions.

2. **Anticipatory vs. reactive migration**: All prior tiered memory systems (TPP, Nomad, ALTO, ARMS, ArtMem, FlexMem) are fundamentally reactive. MarkovTier is the first to migrate pages BEFORE the access pattern changes.

3. **Confidence-weighted migration**: The migration aggressiveness is proportional to prediction confidence, providing a principled way to trade off between proactive benefit and misprediction cost.

4. **Budget-based rate control**: Prevents anticipatory migration from becoming a performance liability under low-confidence predictions or bandwidth-constrained conditions.

## Related Work

### Reactive Tiered Memory Management

**TPP** (Al Maruf et al., ASPLOS'23) introduced transparent page placement for CXL-enabled tiered memory in the Linux kernel, using NUMA hinting faults for promotion and page reclamation for demotion. TPP is the baseline policy in modern Linux kernels but reacts slowly to working set changes.

**Nomad** (Xiang et al., OSDI'24) proposed non-exclusive memory tiering via transactional page migration, maintaining shadow copies to reduce demotion cost. While Nomad reduces the cost of individual migrations, it does not address the timing of migrations relative to phase changes.

**ALTO** (Liu et al., OSDI'25) moved beyond hotness to use amortized offcore latency (AOL) as a migration metric, throttling unnecessary migrations. ALTO represents the state-of-the-art in single-application tiering but remains reactive -- it throttles migrations rather than anticipating when they will be needed.

**MTM** (EuroSys'24) improved profiling speed over DAMON but still detects phases after they occur rather than predicting them. MTM demonstrated that faster profiling reduces but does not eliminate migration lag.

### Adaptive Tiered Memory

**ARMS** (2025) introduced change-point detection on slow-tier bandwidth to detect hot set shifts, using longer access histories during stable phases. ARMS is the closest existing work to ours in spirit, as it adapts behavior based on phase stability. However, ARMS reacts to detected changes rather than anticipating future phases, and does not maintain a model of phase transition probabilities.

**FlexMem** (ATC'24) combines different profiling methods to respond to application phase changes, using adaptive profiling granularity. Like ARMS, FlexMem adapts its profiling but not its migration timing.

**ArtMem** (Yi et al., ISCA'25) applies Q-learning to tiered memory management, learning adaptive migration policies through reinforcement learning. While ArtMem's RL agent implicitly captures some phase behavior through its state representation, it does not explicitly model phase transitions or perform anticipatory migration. The RL approach also requires more state space exploration time to converge.

### Program Phase Detection

**SimPoint** (Sherwood et al., ASPLOS'02) demonstrated that programs exhibit phase behavior identifiable through Basic Block Vectors, enabling representative simulation sampling. This foundational work established that phase structure is pervasive and predictable.

**Dhodapkar & Smith** (ISCA'02) proposed working set signatures for dynamic phase detection, showing that phase changes can be detected online with compact representations. Our working set signature approach builds on this work but adapts it to the memory page level for tiering decisions.

**POP Detector** (2019) provided a lightweight online phase detection framework using hardware performance counters, demonstrating that phase detection can be performed with minimal overhead.

### How MarkovTier Differs

MarkovTier is distinguished from all prior work by its **anticipatory** nature: rather than reacting to phase changes after they occur (TPP, Nomad, ALTO, ARMS, FlexMem) or learning implicit patterns through model-free RL (ArtMem), MarkovTier explicitly models the sequence of phase transitions as a Markov chain and uses this model to proactively migrate pages before they are needed. This transforms the migration timing from a reactive detect-then-migrate paradigm to a predictive anticipate-then-migrate paradigm.

## Experiments

### Experimental Setup

**Simulator**: We will build a trace-driven tiered memory simulator in Python that models:
- A two-tier memory system: fast tier (DRAM, 100ns access latency) and slow tier (CXL memory, 300ns access latency)
- Configurable fast-tier capacity (e.g., 25%, 50%, 75% of total working set)
- Page migration with realistic bandwidth constraints (e.g., 10 GB/s migration bandwidth)
- Page-level access tracking with configurable profiling intervals

**Workloads**: We will use three categories of workloads:

1. **Synthetic phase workloads**: Programs with controlled phase structure (number of phases, phase duration, working set overlap between phases, transition predictability). These allow us to systematically vary phase characteristics and measure their impact.

2. **SPEC-inspired traces**: Synthetic traces modeled after known phase behavior in SPEC CPU benchmarks (e.g., gcc with ~10 distinct phases, mcf with 3-4 phases), using publicly documented phase characterizations from the SimPoint literature.

3. **Server-like workloads**: Traces modeling request-driven workloads with periodic behavior (e.g., web servers with diurnal patterns, batch processing with map/reduce phases).

**Baselines**:
- **LRU-based reactive**: Promotes pages on access, demotes LRU pages when fast tier is full (similar to basic NUMA balancing)
- **TPP-like reactive**: Promotes on NUMA fault, demotes via reclamation scan (models Linux TPP)
- **ARMS-like adaptive-reactive**: Detects phase changes via bandwidth change-point detection, accelerates profiling after changes
- **ALTO-like throttled**: Uses access-cost-weighted migration with throttling
- **Oracle**: Offline optimal with perfect knowledge of future accesses (Belady-like)

**Metrics**:
- **Slow-tier access ratio**: Fraction of accesses served from slow memory (primary metric)
- **Effective memory latency**: Weighted average of fast and slow tier accesses
- **Migration overhead**: Total data moved (GB) and bandwidth consumed
- **Migration lag**: Time from phase transition to steady-state page placement
- **Misprediction rate**: Fraction of anticipatory migrations that prove unnecessary
- **Throughput**: Simulated instructions per unit time

### Planned Experiments

1. **Phase predictability study**: Characterize how phase structure (number of phases, transition regularity, working set overlap) affects the accuracy of Markov prediction and the benefit of anticipatory migration.

2. **Sensitivity to fast-tier capacity**: Vary the ratio of fast to total memory (10%-90%) and measure how anticipatory migration's benefit changes. We expect larger benefits when fast tier is scarce.

3. **Sensitivity to prediction confidence threshold**: Sweep the confidence threshold from 0.3 to 0.9 and measure the tradeoff between proactive benefit and misprediction cost.

4. **Migration bandwidth budget**: Vary the fraction of bandwidth allocated to anticipatory migration (1%-20%) and identify the sweet spot.

5. **Comparison across workload types**: Compare MarkovTier against all baselines across synthetic, SPEC-inspired, and server-like workloads.

6. **Ablation study**: Evaluate contributions of individual components:
   - Phase detection only (without Markov model)
   - Markov model without confidence thresholding
   - Full MarkovTier with all components

7. **Robustness to irregular phases**: Test with workloads that have unpredictable phase transitions to characterize when anticipatory migration helps vs. hurts.

### Expected Results

- For workloads with regular phase structure (e.g., iterative algorithms, server workloads), MarkovTier should reduce slow-tier accesses during transitions by 30-60% compared to reactive baselines.
- For workloads with irregular phases, MarkovTier should perform no worse than reactive approaches (the confidence threshold prevents wasteful migrations).
- Migration overhead should increase by no more than 15% compared to reactive approaches, since anticipatory migrations replace (rather than add to) reactive migrations.
- The benefit should be most pronounced when fast-tier capacity is limited (25-50% of working set) -- when there's enough fast memory for everything, tiering doesn't matter; when there's too little, anticipatory migration can't help because the working set doesn't fit.

## Success Criteria

### Confirmation of Hypothesis
The hypothesis is confirmed if MarkovTier achieves:
1. At least 20% reduction in slow-tier access ratio during phase transitions compared to the best reactive baseline (ALTO-like)
2. At most 10% increase in total migration data volume
3. Prediction accuracy above 70% for workloads with repetitive phase patterns

### Refutation of Hypothesis
The hypothesis is refuted if:
1. The migration lag from reactive approaches is too short to matter (profiling is fast enough that proactive migration adds no value)
2. Misprediction costs consistently outweigh anticipatory benefits even with confidence thresholding
3. The overhead of phase tracking and Markov model maintenance exceeds the saved migration lag

## References

1. Hasan Al Maruf, Hao Wang, Abhishek Dhanotia, Johannes Weiner, Niket Agarwal, Pallab Bhattacharya, Chris Petersen, Mosharaf Chowdhury, Shobhit Kanaujia, Prakash Chauhan. "TPP: Transparent Page Placement for CXL-Enabled Tiered-Memory." ASPLOS, 2023.

2. Jinshu Liu, Hamid Hadian, Hanchen Xu, Huaicheng Li. "Tiered Memory Management Beyond Hotness." OSDI, 2025.

3. Lingfeng Xiang, Zhen Lin, Weishu Deng, Hui Lu, Jia Rao, Yifan Yuan, Ren Wang. "NOMAD: Non-Exclusive Memory Tiering via Transactional Page Migration." OSDI, 2024.

4. Zi Yan, Daniel Lustig, David Nellans, Abhishek Bhattacharjee. "Nimble Page Management for Tiered Memory Systems." ASPLOS, 2019.

5. Jie Ren, Dong Xu, Junhee Ryu, Kwangsik Shin, Daewoo Kim, Dong Li. "MTM: Rethinking Memory Profiling and Migration for Multi-Tiered Large Memory." EuroSys, 2024.

6. Sujay Yadalam, Konstantinos Kanellis, Michael Swift, Shivaram Venkataraman. "ARMS: Adaptive and Robust Memory Tiering System." arXiv:2508.04417, 2025.

7. Xinyue Yi, Hongchao Du, Yu Wang, Jie Zhang, Qiao Li, Chun Jason Xue. "ArtMem: Adaptive Migration in Reinforcement Learning-Enabled Tiered Memory." ISCA, 2025.

8. Dong Xu, Junhee Ryu, Jinho Baek, Kwangsik Shin, Pengfei Su, Dong Li. "FlexMem: Adaptive Page Profiling and Migration for Tiered Memory." USENIX ATC, 2024.

9. Timothy Sherwood, Erez Perelman, Greg Hamerly, Brad Calder. "Automatically Characterizing Large Scale Program Behavior." ASPLOS, 2002.

10. Ashutosh S. Dhodapkar, James E. Smith. "Managing Multi-Configuration Hardware via Dynamic Working Set Analysis." ISCA, 2002.

11. Yuhong Zhong, Drew Zagieboylo, Asaf Cidon, Emin Gun Sirer, G. Edward Suh. "Managing Memory Tiers with CXL in Virtualized Environments." OSDI, 2024.
