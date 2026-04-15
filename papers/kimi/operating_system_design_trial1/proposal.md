# UniSched: Proactive Cross-NUMA Scheduling for CXL Memory via sched_ext

## Research Proposal

**Research Area:** Operating Systems, CPU Scheduling, Memory Tiering  
**Keywords:** sched_ext, CXL, tiered memory, CPU scheduling, eBPF, NUMA, proactive scheduling, discrete-event simulation  
**Target Venues:** OSDI, SOSP, ASPLOS, EuroSys  

---

## 1. Introduction

### 1.1 Context and Problem Statement

Modern datacenter servers are experiencing a fundamental architectural shift driven by heterogeneous memory hierarchies enabled by Compute Express Link (CXL). A typical server now features multiple memory tiers:
- Local DDR5 DRAM (80-100 ns latency, high bandwidth)
- CXL-attached DRAM (150-300 ns latency, asymmetric bandwidth)
- CXL memory pools (shared across sockets, variable latency 200-500 ns)

Linux 6.12 merged `sched_ext`, a revolutionary framework enabling custom CPU schedulers as eBPF programs. This creates an opportunity for deployable, kernel-modification-free scheduling policies.

**The Critical Gap:** Current schedulers treat CPU scheduling decisions in isolation from memory topology. When selecting a CPU for a task, schedulers consider CPU load and cache topology but largely ignore where the task's memory resides, leading to suboptimal decisions.

### 1.2 Key Insight

Memory-topology-aware CPU scheduling should be **proactive, lightweight, and deployable without kernel modifications**. Rather than reacting to page faults (incurring latency penalties), the scheduler should:

1. Use hardware PMU sampling (PEBS/IBS) to proactively track memory access patterns
2. Make CPU placement decisions considering both compute efficiency AND memory access efficiency
3. Be deployable via sched_ext as a userspace-loaded BPF program

### 1.3 Research Hypothesis

> A proactive, PMU-driven cross-NUMA scheduler can achieve comparable or better performance than kernel-modification-based approaches (e.g., Tiresias), while being deployable without kernel patches and introducing less than 3% scheduling overhead.

### 1.4 Why This Matters

CXL is being rapidly adopted by major cloud providers. Tiresias (ACM TURC 2024) recently demonstrated CXL-aware CPU scheduling but requires kernel modifications for PTSR and uses reactive page-fault-based migration. UniSched offers a deployable alternative.

---

## 2. Proposed Approach

### 2.1 System Architecture

We propose **UniSched**, a scheduling framework with three components:

#### Component 1: Topology Discovery and Characterization
- Enumerate memory tiers via `/sys/devices/system/node/` interfaces
- Characterize each NUMA node: local DRAM, CXL-attached, or CPU-less
- Build latency/bandwidth matrix using PMU-based microbenchmarks
- Export topology to scheduler via BPF maps (in real implementation) or simulation parameters

#### Component 2: Proactive Per-Task Memory Profiling
- Leverage Intel PEBS (Precise Event-Based Sampling) or AMD IBS
- Sample memory accesses at ~1% frequency (target overhead <2%)
- Track per-task statistics: access distribution across NUMA nodes, bandwidth consumption
- Classify tasks: "local-dominant", "cxl-bandwidth-bound", "latency-sensitive"

#### Component 3: Unified Scheduling Algorithm
The scheduler scores CPU candidates using:

```
scheduling_score(CPU, task) = 
    w1 * compute_score(CPU_load, task_priority) +
    w2 * memory_score(task_pattern, CPU_NUMA_node, mem_controller_load) +
    w3 * migration_cost(cache_warmth)
```

**Policy 1: Latency-Aware Placement**
- For tasks with >60% local DRAM access: prefer same-node CPUs
- For tasks using CXL memory: balance between CXL bandwidth availability and CPU load

**Policy 2: Bandwidth Balancing**
- Monitor CXL controller saturation via PMU counters
- Migrate bandwidth-heavy tasks to nodes with available CXL bandwidth

**Policy 3: Memory Coordination**
- Export per-task memory access hints
- Userspace daemon issues `move_pages()` for high-value migrations

### 2.2 Implementation Strategy

**Phase 1: Hardware Validation and PMU Overhead Measurement (CRITICAL)**
Before any simulation work, we **must** validate PMU overhead:
1. Measure baseline application performance without profiling
2. Enable PEBS/IBS sampling at target frequencies (0.5%, 1%, 2%)
3. Measure performance degradation
4. **Gate:** If overhead exceeds 3%, the approach is not viable

**Phase 2: Discrete-Event Simulation (Due to Kernel Version Constraint)**
- **Current hardware:** Linux 6.8.0 (sched_ext requires 6.12+)
- **Approach:** Build a discrete-event simulator that models:
  - Task arrivals, completions, and CPU scheduling decisions
  - Memory access patterns with configurable locality
  - NUMA topology with parameterized latency/bandwidth matrices
  - CXL memory controller contention
- **Validation:** Simulator parameters calibrated from real hardware measurements

**Technology Stack:**
- Python-based discrete-event simulation (SimPy or custom)
- eBPF/BPF prototypes for scheduler logic validation (can run without sched_ext)
- PMU profiling validation on actual hardware

---

## 3. Detailed Comparisons with Related Work

### 3.1 UniSched vs. Tiresias

| Aspect | Tiresias | UniSched |
|--------|----------|----------|
| **Deployment** | Kernel modifications required | sched_ext BPF (no kernel patches) |
| **Migration Trigger** | Reactive: page-fault-driven | Proactive: PMU-based profiling |
| **Mechanism** | Page-table self-replication (PTSR) | sched_ext BPF programs |
| **Memory Tracking** | NUMA hint faults (256MB sampling) | PEBS/IBS sampling (~1% frequency) |
| **Scope** | Cross-NUMA thread migration | Cross-NUMA topology-aware placement |

**Key Distinction:** Tiresias achieves performance through deep kernel integration (PTSR). UniSched demonstrates that comparable benefits can be achieved through modern kernel extensibility mechanisms.

### 3.2 UniSched vs. CXLAimPod (Critical Distinction)

**CXLAimPod** (Yang et al., arXiv 2025) also uses sched_ext for CXL-aware scheduling, but addresses a fundamentally different problem:

| Aspect | CXLAimPod | UniSched |
|--------|-----------|----------|
| **Primary Goal** | Exploit CXL full-duplex channels | Cross-NUMA topology-aware placement |
| **Scope** | **Intra-node** (single NUMA node) | **Inter-node** (across NUMA nodes) |
| **Problem** | Read/write bandwidth balancing within one node | Which NUMA node to schedule tasks on |
| **Mechanism** | Co-schedule read-heavy and write-heavy tasks | Score-based CPU selection across nodes |
| **Target** | AI inference with mixed R/W patterns | General-purpose diverse workloads |

**Complementary Relationship:** These are orthogonal problems at different levels:
- **UniSched** decides: "Which NUMA node should this task run on?"
- **CXLAimPod** decides: "Given tasks on this NUMA node, how do we balance read/write traffic?"

A complete solution could combine both: UniSched decides which NUMA node, CXLAimPod optimizes within that node.

### 3.3 UniSched vs. AutoNUMA

**AutoNUMA** (Linux kernel) is the current state-of-the-art for NUMA-aware scheduling in production Linux:

| Aspect | AutoNUMA | UniSched |
|--------|----------|----------|
| **Approach** | Reactive sampling + page migration | Proactive PMU-based scheduling |
| **Information** | Page access sampling (software) | Hardware PMU events (PEBS/IBS) |
| **Mechanism** | Memory follows CPU / CPU follows memory | Unified scoring with topology awareness |
| **CXL Awareness** | None specific | Explicit CXL latency/bandwidth modeling |

**Why UniSched May Outperform AutoNUMA:**
1. **Proactive vs. Reactive:** PMU sampling provides immediate access pattern data vs. AutoNUMA's periodic scanning
2. **Hardware vs. Software:** PEBS/IBS provides precise instruction-to-data correlation
3. **CXL-Specific:** AutoNUMA predates CXL; UniSched explicitly models CXL topology

---

## 4. Experimental Plan

### 4.1 Research Questions

1. **RQ1 (Performance):** Does UniSched improve application performance compared to baseline Linux (EEVDF + AutoNUMA)?
2. **RQ2 (vs. Tiresias approach):** How does proactive PMU-based scheduling compare to reactive page-fault-based approaches?
3. **RQ3 (Overhead):** What is the runtime overhead of PMU profiling and scheduling decisions?
4. **RQ4 (Simulation Fidelity):** How well do simulation results correlate with real hardware behavior?

### 4.2 Hardware and Simulation Setup

**Hardware Platform (for PMU validation and calibration):**
- Standard x86_64 server with 2 sockets, 32+ cores
- 256+ GB DRAM
- **Kernel:** Linux 6.8.0 (sched_ext NOT available - see simulation approach)

**Pre-Experiment Validation (MANDATORY):**
1. **PMU Overhead Test:**
   - Run STREAM benchmark without PMU profiling → baseline throughput
   - Run with PEBS sampling at 0.5%, 1%, 2% frequencies
   - Calculate overhead percentage
   - **If overhead > 3%:** Document and reconsider approach
   - **If overhead ≤ 3%:** Proceed with confidence

2. **Hardware Feature Check:**
   - Verify PEBS/IBS availability: `grep -E 'pebs|ibs' /proc/cpuinfo`
   - Test PMU access: `perf record -e mem-loads -c 10000 ./test` as root
   - If PMUs unavailable, document software fallback approach

**Simulation Approach (Due to Kernel 6.8.0 vs. 6.12+ Requirement):**
- Build discrete-event simulator modeling:
  - Task scheduling events (arrival, dispatch, completion, migration)
  - Memory access patterns with configurable NUMA affinity
  - CXL latency/bandwidth characteristics
  - Memory controller contention
- Calibrate simulator using real hardware measurements

### 4.3 Simulation Limitations (Explicitly Acknowledged)

We explicitly acknowledge the following simulation limitations:

1. **CXL Protocol Overheads:** Simulation uses parameterized latency/bandwidth values. It cannot capture:
   - Actual CXL protocol handshaking delays
   - Real contention behavior at the link level
   - Cache coherency protocol overheads specific to CXL

2. **Real-World Contention:** Simulated memory controller contention is approximated based on queuing models, not actual hardware arbitration

3. **sched_ext Behavior:** The simulator approximates BPF scheduler behavior. Actual sched_ext may have:
   - Different scheduling latency characteristics
   - BPF verifier limitations not captured
   - Kernel integration subtleties

4. **Migration Costs:** Cache warmth and TLB effects are modeled heuristically

**Mitigation:** We will calibrate simulator parameters using real hardware measurements where possible (PMU overheads, memory access latencies) and explicitly note these limitations in all results.

### 4.4 Workloads

**Microbenchmarks (for simulator validation):**
1. **STREAM:** Memory bandwidth with varying read/write ratios
2. **Random Access:** Pointer-chasing for latency sensitivity
3. **Mixed Workload:** Latency-sensitive + bandwidth-heavy co-location

**Application Workloads (simulated based on real-world characteristics):**
1. **Redis/YCSB:** Key-value store with memory-intensive patterns
2. **Graph Analytics (PageRank, BFS):** Graph processing exceeding local DRAM
3. **SPEC CPU2017:** General compute with varying footprints

### 4.5 Baselines

1. **Linux EEVDF:** Default kernel scheduler (simulated)
2. **EEVDF + AutoNUMA:** Default with AutoNUMA enabled (CRITICAL BASELINE)
3. **Tiresias-like:** Reactive page-fault-driven migration (simulated)
4. **UniSched Variants:**
   - Topology-only (no per-task profiling)
   - Profiling-only (standard load balancing)
   - Full system (topology + profiling)

### 4.6 Metrics

**Performance:**
- Application throughput (ops/sec)
- Tail latency (p50, p95, p99)
- Memory bandwidth utilization per tier

**Scheduler:**
- Scheduling overhead (cycles per schedule)
- Task migration frequency and latency
- Fairness: Jain's index

**Deployability:**
- Lines of kernel code modified (target: 0)

### 4.7 Expected Results

- **5-15% improvement** in throughput for memory-bandwidth-bound workloads vs. EEVDF+AutoNUMA
- **10-20% reduction** in tail latency for latency-sensitive workloads
- **<3% overhead** from profiling (validated on real hardware)
- **Zero kernel modifications** required for deployment

---

## 5. Success Criteria

### 5.1 Confirming Evidence (Hypothesis Supported)

- PMU profiling overhead ≤ 3% on real hardware
- Simulated UniSched achieves ≥5% throughput improvement on 3+ benchmarks vs. EEVDF+AutoNUMA
- Memory bandwidth utilization on CXL links improves by ≥10%
- Fairness index remains above 0.85 under competitive workloads
- Deployable without kernel modifications (verified via code analysis)

### 5.2 Refuting Evidence (Hypothesis Rejected)

- PMU profiling overhead exceeds 5%
- Simulated performance improvement <3% across all benchmarks
- Simulated results diverge significantly from expected hardware behavior
- Severe fairness degradation (Jain's index <0.7)

---

## 6. Timeline and Milestones (8-Hour Budget)

| Phase | Duration | Activities |
|-------|----------|------------|
| 1 | 1 hour | **PMU overhead validation** - CRITICAL gate. Measure PEBS/IBS overhead on real hardware. If >3%, document and adjust approach. |
| 2 | 1.5 hours | Topology discovery, memory characterization, simulator framework setup |
| 3 | 2 hours | Baseline schedulers (EEVDF, AutoNUMA, Tiresias-like) implementation in simulator |
| 4 | 2 hours | UniSched implementation, benchmark evaluation |
| 5 | 1.5 hours | Ablation studies, results analysis, limitation documentation |

**Simulation-Only Justification:** Due to kernel 6.8.0 vs. 6.12+ requirement and 8-hour budget constraint, we use discrete-event simulation rather than real sched_ext implementation. The simulator is calibrated using real hardware PMU measurements from Phase 1.

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Simulation vs. Real sched_ext:** Results are from simulation, not real sched_ext deployment
2. **CXL Simulation:** Bandwidth throttling and latency injection approximate but don't fully capture CXL protocol behavior
3. **PMU Dependency:** Requires hardware PMU support (validated in Phase 1)
4. **Single-machine scope:** Distributed scheduling across nodes not addressed

### 7.2 Future Directions

1. **Kernel 6.12+ Upgrade:** Validate simulation findings on real sched_ext with kernel upgrade
2. **Real CXL Hardware:** Validate on production CXL hardware when available
3. **Hybrid PMU/Page-Fault:** Combine proactive PMU with reactive page-fault mechanisms

---

## 8. Conclusion

UniSched proposes a deployable, kernel-modification-free approach to CXL-aware CPU scheduling. By differentiating from Tiresias (kernel-modification-based, reactive) and CXLAimPod (intra-node duplex optimization) through its proactive PMU-based profiling and cross-NUMA focus, UniSched offers a practical alternative. While simulation-based validation has limitations, the approach includes mandatory PMU overhead validation on real hardware and explicit acknowledgment of simulation constraints.

---

## References

1. Wenda Tang, Tianxiang Ai, Jie Wu. "Tiresias: Optimizing NUMA Performance with CXL Memory and Locality-Aware Process Scheduling." ACM TURC 2024.

2. Yiwei Yang, Yusheng Zheng, et al. "CXLAimPod: CXL Memory is all you need in AI era." arXiv:2508.15980, 2025.

3. Jack Tigar Humphries, Neel Natu, et al. "ghOSt: Fast & Flexible User-Space Delegation of Linux Scheduling." SOSP 2021.

4. Tejun Heo, David Vernet, et al. "sched_ext: BPF extensible scheduler class." Linux kernel 6.12, 2024.

5. Huaicheng Li, Daniel S. Berger, et al. "Pond: CXL-Based Memory Pooling Systems for Cloud Platforms." ASPLOS 2023.

6. Hasan Al Maruf, Hao Wang, et al. "TPP: Transparent Page Placement for CXL-Enabled Tiered-Memory." ASPLOS 2023.

7. Lingfeng Xiang, Zhenlin Liu, et al. "NOMAD: Non-Exclusive Memory Tiering via Transactional Page Migration." OSDI 2024.

8. Midhul Vuppalapati, Rachit Agarwal. "Tiered Memory Management: Access Latency is the Key!" SOSP 2024.

9. Andrea Arcangeli. "AutoNUMA: Automatic NUMA balancing." Linux kernel, 2012-2024.

10. Jonathan Corbet. "The EEVDF CPU scheduler for Linux." LWN.net, 2023.

11. David Vernet. "The current status and future potential of sched_ext." Linux Plumbers Conference 2024.

12. Baptiste Lepers et al. "Thread and Memory Placement on NUMA Systems: Asymmetry Matters." USENIX ATC 2015.
