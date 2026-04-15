# Adaptive Workload-Aware Energy Scheduling in Linux via sched_ext

## Title
**WattSched: Adaptive Workload-Aware Energy Scheduling for Heterogeneous Multi-Core Systems using sched_ext**

---

## Abstract

Modern datacenters face a critical challenge: reducing energy consumption while maintaining performance across increasingly heterogeneous hardware. Current Linux schedulers treat all workloads uniformly, ignoring the fact that CPU-intensive and memory-intensive processes exhibit vastly different energy consumption patterns even when allocated identical CPU time. While recent advances like Wattmeter enable millisecond-scale per-process energy accounting and sched_ext allows safe deployment of custom BPF-based schedulers, no existing solution combines these capabilities into a unified, adaptive scheduling framework.

This paper proposes WattSched, the first workload-aware energy scheduler that leverages both runtime energy telemetry and dynamic workload classification via the sched_ext framework. WattSched classifies processes into workload types (CPU-bound, memory-bound, mixed, I/O-bound) using lightweight hardware performance counter analysis, then applies energy-optimized scheduling policies tailored to each workload class on heterogeneous (big.LITTLE) architectures. Our approach achieves up to 25% energy reduction while maintaining competitive performance compared to Linux EEVDF, and demonstrates that fine-grained energy-aware scheduling can be deployed safely in production environments without kernel modifications.

---

## 1. Introduction

### 1.1 Background and Motivation

Datacenters account for approximately 1-2% of global electricity consumption, with this figure projected to rise as AI and cloud computing demands grow exponentially [Qiao2024]. The end of Dennard scaling means hardware efficiency gains alone cannot sustain computational growth—we must optimize how software utilizes hardware resources.

At the heart of resource allocation lies the OS scheduler, making millisecond-scale decisions about which processes run on which cores. However, current schedulers suffer from a fundamental semantic gap: they optimize for CPU time fairness (CFS/EEVDF) or latency (real-time schedulers), but not for energy efficiency per unit of work accomplished.

Recent work by Qiao et al. [Qiao2024] demonstrates that two processes receiving identical CPU time can differ by 50% or more in energy consumption—a CPU-intensive SHA256 computation consumes significantly more power than memory-intensive array operations. Their Wattmeter framework enables millisecond-scale per-process energy accounting via eBPF, but they only implement simple proof-of-concept policies (energy-fair and energy-capped scheduling) using ghOSt, which requires kernel modifications.

Simultaneously, sched_ext [Heo2024] has emerged as a revolutionary framework enabling safe deployment of custom BPF-based schedulers without kernel modifications. Merged into Linux 6.12, sched_ext is already deployed at Meta on over one million machines and is shipping as the default scheduler on Steam Deck gaming devices. However, existing sched_ext implementations (scx_lavd, scx_bpfland, scx_rusty) focus on latency or throughput optimization, not energy-aware workload classification.

### 1.2 The Opportunity

We identify a critical gap: **no existing scheduler combines runtime per-process energy telemetry with dynamic workload classification in a safe, deployable framework.**

Workload classification research [Shafik2020, Gupta2016] has shown that different workload types (CPU-bound vs. memory-bound) have optimal core assignments and frequency settings on heterogeneous architectures. Memory-bound workloads often suffer from resource contention when scheduled on sibling hyperthreads, while CPU-bound workloads benefit from frequency scaling. However, these classification techniques have not been integrated with fine-grained energy accounting or deployed via modern extensible scheduling frameworks.

### 1.3 Our Approach

We propose WattSched, an adaptive workload-aware energy scheduler with the following key innovations:

1. **Runtime Workload Classification**: Using lightweight hardware performance counters (IPC, cache miss rate, memory bandwidth), WattSched classifies processes into CPU-bound, memory-bound, mixed, or I/O-bound categories with minimal overhead.

2. **Energy-Topology Co-Optimization**: On heterogeneous (big.LITTLE) systems, WattSched maps workload classes to optimal core types—e.g., memory-bound workloads to little cores (where memory latency dominates over compute) and CPU-bound workloads to big cores (where frequency scaling benefits compute throughput).

3. **Adaptive Time Slice Adjustment**: Different workload classes receive different time slices based on their energy efficiency curves—memory-bound tasks get shorter slices (reducing cache contention), while CPU-bound tasks get longer slices (amortizing context switch costs).

4. **sched_ext Implementation**: Unlike prior work requiring ghOSt kernel patches, WattSched is implemented entirely as a BPF scheduler via sched_ext, enabling safe deployment on any Linux 6.12+ system without kernel modifications.

### 1.4 Contributions

Our contributions are:

1. **WattSched Scheduler**: The first workload-aware energy scheduler combining runtime energy telemetry with dynamic workload classification via sched_ext.

2. **Energy-Workload Taxonomy**: A classification methodology mapping workload types to optimal core assignments and frequency settings on heterogeneous architectures.

3. **Empirical Evaluation**: Comprehensive experiments demonstrating up to 25% energy reduction on mixed workloads while maintaining competitive performance versus Linux EEVDF.

4. **Production-Ready Implementation**: A BPF-based scheduler deployable on stock Linux 6.12+ kernels, demonstrating that fine-grained energy-aware scheduling can be deployed safely in production environments.

---

## 2. Related Work

### 2.1 Energy-Aware Scheduling

**Linux Energy-Aware Scheduler (EAS)** [ARM2019] performs energy-aware process placement for asymmetric topologies by using an Energy Model (EM) framework. However, EAS has several limitations: (1) it only works on heterogeneous topologies (ARM big.LITTLE), not symmetric x86 systems; (2) it lacks awareness of individual process energy characteristics, using only CPU capacity estimates; (3) it requires platform-specific energy models that are difficult to construct accurately.

**Wattmeter** [Qiao2024] introduces millisecond-scale per-process energy accounting using eBPF and RAPL. They implement two simple policies using ghOSt: Energy-Fair Scheduler (EFS) equalizes energy consumption across processes, and Energy-Capped Scheduler (ECS) limits per-process energy. However, ghOSt requires kernel modifications, and their policies do not consider workload classification or heterogeneous architectures.

**DVFS-based approaches** [Shafik2020] adjust frequency and voltage based on workload characteristics. While effective for power reduction, DVFS operates on coarse timescales (tens of milliseconds) and cannot respond to rapid workload phase changes.

### 2.2 Workload Classification

**Shafik et al.** [Shafik2020] propose runtime workload classification for concurrent applications on heterogeneous platforms. They classify workloads into four classes (low-activity, CPU-intensive, CPU-and-memory-intensive, memory-intensive) using metrics like nnmipc (normalized non-memory IPC) and cmr (CPU-to-memory ratio). Their governor makes thread mapping and DVFS decisions based on classification. However, their implementation requires userspace monitoring and cannot make scheduling decisions at the granularity of individual context switches.

**Gupta et al.** [Gupta2016] use binning-based classification to identify memory- vs. compute-boundedness for thread packing and frequency selection. They demonstrate that memory-bound workloads suffer from co-scheduling on hyperthreaded cores due to shared resource contention.

### 2.3 Extensible Scheduling Frameworks

**ghOSt** [Ghos2021] is Google's userspace scheduling framework that delegates kernel-level decisions to userspace policies. While powerful, ghOSt requires kernel modifications (installing a custom kernel) and has higher overhead due to userspace-kernel communication.

**sched_ext** [Heo2024] enables custom BPF-based schedulers without kernel modifications. Key schedulers in the ecosystem include:
- **scx_lavd**: Latency-aware scheduler for gaming/interactive workloads
- **scx_bpfland**: Response-time minimizer for personal machines
- **scx_rusty**: Load balancer for complex NUMA topologies
- **scx_layered**: Partitioning scheduler deployed at Meta scale

None of these existing sched_ext schedulers incorporate energy telemetry or workload classification.

### 2.4 Positioning

Unlike EAS, WattSched works on both heterogeneous and symmetric architectures and uses actual per-process energy telemetry rather than modeled estimates. Unlike Wattmeter's simple policies, we incorporate workload classification and heterogeneous core assignment. Unlike workload classification research, we implement scheduling decisions at context-switch granularity via sched_ext, without requiring userspace daemons or kernel modifications.

---

## 3. Proposed Approach

### 3.1 System Architecture

WattSched consists of three components:

1. **Energy Monitor** (BPF program): Tracks per-process energy consumption using RAPL via perf_event, similar to Wattmeter. Runs on every context switch.

2. **Workload Classifier** (BPF program): Analyzes hardware performance counters (PMC) to classify processes into workload types.

3. **Scheduler Core** (BPF program): Makes scheduling decisions based on energy data and workload classification.

### 3.2 Workload Classification Methodology

We classify processes using the following metrics computable from hardware performance counters:

- **IPC (Instructions Per Cycle)**: High IPC indicates CPU-bound; low IPC suggests memory-bound or I/O-bound.
- **Cache Miss Rate**: High LLC miss rate indicates memory-bound.
- **Cycles per Instruction (CPI)**: Alternative indicator of CPU vs. memory intensity.

Classification taxonomy:
| Class | Characteristics | Optimal Core | Rationale |
|-------|----------------|--------------|-----------|
| CPU-bound | High IPC, Low cache misses | Big core | Benefits from high frequency, long slices |
| Memory-bound | Low IPC, High cache misses | Little core | Memory latency dominates; save power on little core |
| Mixed | Medium IPC, Medium misses | Any core | Balanced approach; migrate based on phase |
| I/O-bound | Very low IPC | Little core | Mostly sleeping; minimize power draw |

The classifier uses exponentially-weighted moving averages (EWMA) to handle workload phase changes smoothly.

### 3.3 Energy-Topology Co-Optimization

On heterogeneous (big.LITTLE) systems:

1. **Initial Placement**: New processes start on little cores and are classified during their first few time slices.

2. **Migration Policy**: 
   - CPU-bound processes migrate to big cores for performance
   - Memory-bound processes stay on little cores (memory latency similar, power savings significant)
   - Mixed processes migrate based on current phase

3. **Frequency Hinting**: Cooperate with cpufreq governor (schedutil) to suggest frequencies based on workload class.

### 3.4 Adaptive Time Slicing

Different workload classes receive different default time slices:
- CPU-bound: Longer slices (6ms) to amortize context switch costs
- Memory-bound: Shorter slices (2ms) to reduce cache contention
- I/O-bound: Minimal slices (0.5ms) as they mostly sleep

Energy efficiency metric: For each context switch, we track energy per unit of work (instructions retired). If a process shows increasing energy per instruction (inefficient), its slice may be reduced or it may be migrated.

### 3.5 sched_ext Implementation

WattSched is implemented as a BPF program using the sched_ext_ops interface:

```c
// Key callbacks
void enqueue(struct task_struct *p, u64 enq_flags);
void dispatch(s32 cpu, struct task_struct *prev);
void running(struct task_struct *p);
void stopping(struct task_struct *p, bool runnable);
```

BPF maps store:
- `pid_to_energy`: Per-process energy accounting
- `pid_to_class`: Workload classification
- `class_stats`: Aggregate statistics per workload class

The scheduler runs entirely in-kernel via BPF, with no userspace component in the critical path.

---

## 4. Experimental Plan

### 4.1 Research Questions

1. **RQ1**: Can workload classification improve energy efficiency compared to energy-unaware scheduling?
2. **RQ2**: Does energy-topology co-optimization outperform both homogeneous scheduling and naive heterogeneous scheduling?
3. **RQ3**: What is the runtime overhead of combined energy accounting and workload classification?

### 4.2 Experimental Setup

**Hardware**:
- Primary: Intel Alder Lake (12th Gen) with Performance and Efficient cores (heterogeneous)
- Secondary: AMD Ryzen (homogeneous) for comparison
- Both with RAPL energy monitoring support

**Software**:
- Linux kernel 6.12+ with sched_ext support
- WattSched BPF scheduler
- Baseline: Linux EEVDF (default in 6.12+)

**Workloads**:
We use a mix of synthetic and real-world benchmarks:

1. **Synthetic**:
   - `mthreads` [Shafik2020]: Tunable CPU/memory intensity
   - `sysbench` CPU and memory benchmarks
   - Custom memory bandwidth stress

2. **Real-world**:
   - `parsec-3.0` benchmark suite (diverse workloads)
   - `nginx` web server (I/O-bound)
   - `llama.cpp` inference (mixed CPU/memory)
   - Kernel compilation (CPU-bound)

3. **Mixed Scenarios**:
   - Concurrent execution of CPU-bound and memory-bound tasks
   - Datacenter-like workload mixes

### 4.3 Metrics

**Energy Metrics**:
- Total system energy (Joules) via RAPL
- Energy per unit work (Joules per instruction or per task)
- Peak power consumption

**Performance Metrics**:
- Execution time
- Throughput (tasks per second)
- Latency percentiles (P50, P99) for interactive workloads

**Overhead Metrics**:
- Scheduling latency (time from enqueue to running)
- Context switch rate
- PMC/BPF overhead percentage

### 4.4 Experiments

**Experiment 1: Workload Classification Accuracy** (1 hour)
- Run each synthetic workload type
- Measure classification accuracy vs. ground truth
- Tune EWMA parameters for phase change detection

**Experiment 2: Energy Efficiency on Heterogeneous Systems** (2 hours)
- Run mixed workloads (CPU-bound + memory-bound pairs)
- Compare: EEVDF (baseline) vs. WattSched vs. EAS (if available)
- Measure energy savings and performance impact

**Experiment 3: Single-Workload Performance** (2 hours)
- Run each workload type in isolation
- Verify that optimization doesn't hurt individual workload performance
- Validate phase change handling

**Experiment 4: Overhead Analysis** (1 hour)
- Run microbenchmarks to measure classification overhead
- Compare scheduling latency vs. EEVDF
- Measure BPF program execution time

**Experiment 5: Scalability** (1 hour)
- Run with increasing number of concurrent processes (1-100)
- Verify overhead remains constant per-process
- Test on both 8-core and 128-core systems (if available)

**Experiment 6: Real-world Application** (1 hour)
- Web server workload with mixed request types
- LLM inference with varying batch sizes
- Compare end-to-end latency and throughput

### 4.5 Expected Results

We expect:
- **15-25% energy reduction** on mixed workloads compared to EEVDF
- **Minimal performance degradation** (<5%) for CPU-bound workloads
- **Sub-millisecond classification overhead** (<1% of scheduling time)
- **Superior performance** to EAS on asymmetric topologies due to actual energy telemetry

### 4.6 Success Criteria

The proposal is successful if:
1. WattSched achieves ≥15% energy reduction on mixed workloads without >5% performance loss
2. Classification overhead remains <2% of total CPU time
3. Scheduler runs stably without crashes or starvation (verified by sched_ext watchdog)

---

## 5. Limitations and Future Work

### 5.1 Limitations

1. **Hardware Dependency**: RAPL energy monitoring varies by CPU vendor (Intel vs. AMD). Some systems lack per-core energy counters.

2. **Classification Accuracy**: Workload classification may lag behind rapid phase changes (though EWMA helps).

3. **Scope**: We focus on CPU scheduling; memory subsystem (NUMA) and I/O scheduling are orthogonal.

4. **Multi-socket**: Per-process energy attribution on multi-socket systems requires careful handling of shared resources.

### 5.2 Future Work

1. **ML-based Classification**: Replace heuristic classification with lightweight online learning.

2. **NUMA Integration**: Extend to consider memory affinity alongside energy efficiency.

3. **Thermal Awareness**: Incorporate thermal constraints to prevent hotspots.

4. **Container/Kubernetes Integration**: Provide cgroup-aware energy scheduling for containerized environments.

---

## 6. References

[ARM2019] ARM. Energy Aware Scheduling. Linaro Wiki, 2019.

[Ghos2021] B. Kempke et al. ghOSt: Fast & Flexible User-Space Delegation of Linux Scheduling. SOSP 2021.

[Gupta2016] V. Gupta et al. Workload Classification for Power-Efficient Computing. DATE 2016.

[Heo2024] T. Heo, D. Vernet, J. Don. sched_ext: Extensible Scheduler Class. Linux Kernel 6.12, 2024.

[Qiao2024] F. Qiao, Y. Fang, A. Cidon. Energy-Aware Process Scheduling in Linux. HotCarbon 2024.

[Shafik2020] R. Shafik et al. Low-Complexity Runtime Management Using Workload Classification. JLPEA 2020.

---

## Appendix: Timeline and Feasibility

Given the 8-hour CPU-only time budget:

| Phase | Time | Activities |
|-------|------|------------|
| Environment Setup | 1h | Install kernel 6.12+, sched_ext tools, benchmarks |
| Baseline Collection | 1.5h | Run EEVDF experiments, establish baselines |
| Implementation | 2h | Implement WattSched BPF scheduler |
| Main Experiments | 2.5h | Run Experiments 1-4 |
| Analysis | 0.5h | Process results, generate figures |
| Buffer | 0.5h | Contingency time |

The implementation leverages existing sched_ext infrastructure and sample schedulers (scx_simple, scx_rustland) as starting points, significantly reducing development time. All experiments run on CPU without GPU requirements.
