# Invisible Cycles: Characterizing and Quantifying CPU Time Displacement from Asynchronous Kernel Execution in Modern Linux

## Introduction

### Context and Motivation

Modern Linux kernels increasingly rely on asynchronous execution mechanisms to achieve high performance: io_uring offloads I/O operations to kernel worker threads, eBPF programs execute in softirq and interrupt contexts, workqueues defer filesystem and block-layer operations to kernel threads, and network stack processing occurs in NAPI softirq contexts. These mechanisms are foundational to contemporary high-performance systems — io_uring is now the preferred I/O interface for databases (ScyllaDB, PostgreSQL), web servers, and storage engines, while eBPF is ubiquitous in networking, observability, and security infrastructure.

However, these asynchronous mechanisms share a fundamental property that has not been systematically studied: **they displace CPU work from the requesting process's scheduling context to a different execution context** (kernel worker threads, softirq handlers, or kworker threads). The Linux CPU scheduler (EEVDF, as of kernel 6.6+) tracks CPU time consumption per-task via virtual runtime (vruntime). When a process's work is executed by a kernel worker thread or in softirq context, that CPU time is either charged to the worker thread (which has its own scheduling entity) or not charged to any user-visible process at all. We term this phenomenon **CPU time displacement**.

### Problem Statement

CPU time displacement creates a systematic gap between the CPU resources a process *causes* to be consumed and the CPU resources the scheduler *attributes* to that process. This gap undermines three core OS abstractions:

1. **Scheduler Fairness**: The EEVDF scheduler guarantees bounded lag (the difference between ideal and actual CPU allocation) for each task. But if a task's actual CPU consumption is partially hidden in worker threads, the scheduler's fairness invariant is maintained *de jure* but violated *de facto* — the task receives more effective CPU service than its fair share.

2. **Cgroup Resource Control**: Container orchestrators (Kubernetes, Docker) rely on cgroup CPU controllers to enforce resource limits and ensure isolation between tenants. If a container's io_uring worker threads execute outside the container's cgroup, or if their CPU time is not properly attributed, the container effectively escapes its resource constraints.

3. **Resource Accounting and Billing**: Cloud providers bill tenants based on CPU usage metrics derived from scheduler accounting. Displaced CPU time is invisible to these metrics, creating systematic under-billing for I/O-intensive workloads and cross-subsidization from CPU-bound tenants.

This problem is not merely theoretical. A Kubernetes issue (#125409) documents how kernel threads starve when pod cgroups consume most CPU weight, and io_uring developers have acknowledged that "it is important to ensure that the work a process induces is billed back to that process from a scheduling perspective."

### Key Insight

We observe that asynchronous kernel execution mechanisms can be modeled as a **work-conserving relay**: a requesting task submits work that is then executed by a relay entity (worker thread, softirq handler, etc.) in a different scheduling context. The CPU time consumed by the relay entity constitutes the *displaced time*. By formally modeling this relay abstraction, we can analytically characterize the magnitude of displacement for each mechanism, derive bounds on the resulting fairness violation, and design attribution strategies that restore scheduler invariants.

### Hypothesis

We hypothesize that (1) CPU time displacement from asynchronous kernel mechanisms is quantitatively significant — accounting for 10-40% of total system CPU time under I/O-intensive workloads — and that (2) this displacement creates measurable fairness violations that grow superlinearly with the number of competing processes, and that (3) a cost-attribution mechanism that propagates CPU charges across async boundaries can restore scheduler fairness to within 5% of the ideal while incurring less than 3% overhead.

## Proposed Approach

### Overview

Our approach has four components:

1. **Taxonomy and Characterization**: We systematically catalog asynchronous execution mechanisms in the Linux kernel, characterize their displacement properties, and measure their prevalence in representative workloads.

2. **Formal Model**: We develop a queueing-theoretic model of CPU time displacement that captures the interaction between the scheduler, requesting processes, and relay entities. The model yields closed-form bounds on fairness violation as a function of displacement intensity and system load.

3. **Discrete-Event Simulation**: We build a high-fidelity discrete-event simulator of the Linux EEVDF scheduler augmented with async execution mechanisms. The simulator enables evaluation under diverse workload scenarios at scale, including mixed workloads with varying displacement intensities.

4. **Mitigation Design and Evaluation**: We design a CPU cost attribution mechanism based on *causal charge propagation* — tracking the causal chain from request submission to async execution and propagating the CPU charge back to the originating task. We evaluate this mechanism's effectiveness and overhead in simulation.

### Method Details

#### Component 1: Taxonomy of Displacement Mechanisms

We catalog the following asynchronous execution mechanisms in the Linux kernel (v6.8+):

| Mechanism | Relay Entity | Displacement Type | Cgroup Attribution |
|-----------|-------------|-------------------|-------------------|
| io_uring (io-wq) | Per-task worker threads | Bounded: workers inherit task's cgroup | Partial (since 5.12) |
| io_uring (SQPOLL) | Kernel polling thread | Unbounded: shared across rings | Weak |
| Workqueues (cmwq) | Per-CPU kworker threads | Unbounded: shared across all tasks | None (root cgroup) |
| Softirq/ksoftirqd | Per-CPU ksoftirqd threads | Unbounded: interrupt context | None |
| Network NAPI | Softirq context | Unbounded: per-device | None |
| eBPF programs | Various (softirq, kprobe, etc.) | Context-dependent | None |
| Block layer completion | Softirq or irq thread | Brief but frequent | None |

For each mechanism, we characterize:
- **Trigger condition**: What causes work to be displaced?
- **Duration distribution**: How much CPU time does the displaced work consume?
- **Attribution status**: Is the CPU time attributed to any user-visible entity?
- **Priority inheritance**: Does the relay entity inherit the requesting task's priority?

#### Component 2: Formal Model

We model the system as a multi-class queueing network with N processes sharing M CPU cores. Each process i generates two types of CPU demand:

- **Direct demand** d_i(t): CPU time consumed in the process's own context (counted by the scheduler)
- **Displaced demand** r_i(t): CPU time consumed by relay entities on behalf of process i (not counted by the scheduler)

The total CPU demand of process i is D_i(t) = d_i(t) + r_i(t), but the scheduler only observes d_i(t).

We define the **displacement ratio** as α_i = E[r_i] / E[D_i], representing the fraction of process i's total CPU demand that is displaced.

Under EEVDF, the scheduler maintains the fairness invariant: lag_i = ideal_time_i - actual_time_i, bounded by |lag_i| ≤ Q (where Q is the time quantum). However, when displacement is present, the *effective lag* becomes:

effective_lag_i = ideal_total_time_i - (actual_time_i + displaced_time_i)

We derive:
- **Theorem 1**: For a system with N processes and displacement ratios {α_1, ..., α_N}, the maximum fairness violation (difference between the highest and lowest effective CPU share) is bounded by max_i(α_i) - min_i(α_i) in steady state.
- **Theorem 2**: Under uniform displacement (all α_i equal), fairness is preserved. Unfairness arises from *heterogeneous* displacement — when processes displace different fractions of their work.
- **Theorem 3**: The rate of fairness violation growth with N is Θ(N · Var(α)) where Var(α) is the variance of displacement ratios across processes.

#### Component 3: Discrete-Event Simulator

We implement a discrete-event simulator (in Python) that models:
- **EEVDF scheduler**: Full implementation of the EEVDF algorithm with virtual runtime tracking, lag computation, and deadline-based selection
- **Async relay mechanisms**: Configurable relay entities with parameterizable displacement ratios, service time distributions, and cgroup attribution policies
- **Workload generator**: Generates mixed workloads with configurable numbers of processes, arrival rates, service time distributions, and displacement intensities
- **Metrics collection**: Tracks per-process effective CPU share, fairness index (Jain's fairness index applied to effective shares), tail latency distribution, and cgroup-level accounting accuracy

The simulator is validated against analytical bounds from the formal model.

**Workload scenarios:**
1. **Homogeneous I/O-intensive**: All processes use io_uring heavily (α ≈ 0.3). Tests whether uniform displacement preserves fairness (Theorem 2).
2. **Heterogeneous mixed**: Mix of io_uring-heavy and CPU-bound processes (α ranges from 0 to 0.4). Tests fairness violation bounds (Theorem 1).
3. **Container co-location**: Multiple cgroups with different workload profiles sharing a host. Tests cgroup accounting accuracy.
4. **Scaling**: Vary N from 4 to 256 processes. Tests scaling behavior (Theorem 3).
5. **Bursty displacement**: Time-varying displacement ratios simulating workload phase changes.

#### Component 4: Cost Attribution Mechanism

We propose **Causal Charge Propagation (CCP)**, a mechanism that attributes displaced CPU time back to the originating process. CCP works by:

1. **Tagging**: When a process submits work to an async mechanism, the submission is tagged with the originating process's task ID and cgroup.
2. **Tracking**: The relay entity tracks CPU time consumed on behalf of each tagged submission.
3. **Propagation**: Periodically (at scheduler tick granularity), accumulated displaced time is propagated back to the originating process's vruntime and cgroup accounting.

We evaluate three propagation strategies:
- **Immediate**: Charge displaced time at each scheduler tick (most accurate, highest overhead)
- **Batched**: Accumulate and charge in batches at configurable intervals (trade-off)
- **Statistical**: Estimate displacement from historical ratios and charge proactively (lowest overhead, least accurate)

### Key Innovations

1. **First formal characterization** of CPU time displacement as a unified phenomenon across Linux async execution mechanisms
2. **Analytical fairness bounds** for EEVDF under heterogeneous displacement — no prior work provides these bounds
3. **Causal Charge Propagation** mechanism with formal correctness guarantees
4. **Comprehensive simulation framework** for evaluating scheduler fairness under async execution

## Related Work

### CPU Scheduling Fairness

The Completely Fair Scheduler (CFS) by Ingo Molnar (2007) introduced virtual runtime-based fairness to Linux. It was replaced by EEVDF (Earliest Eligible Virtual Deadline First) in Linux 6.6, based on the theoretical work by Stoica et al. (1996). Both schedulers guarantee per-task fairness based on observed CPU time, but neither accounts for displaced CPU time from async execution. Isstaif et al. (2025) study context switching overhead in densely packed clusters under EEVDF but do not address the displacement problem. Our work extends fairness analysis to include displaced CPU time.

### Asynchronous I/O and io_uring

io_uring, introduced by Jens Axboe in Linux 5.1 (2019), provides high-performance asynchronous I/O through shared ring buffers between user and kernel space. Recent work has evaluated io_uring for database systems (Du et al., 2024), characterized its performance on NVMe SSDs, and explored its integration with polling and completion-based models. However, no prior work systematically studies the scheduling implications of io_uring's kernel worker threads. The Cloudflare engineering blog (2022) documented io_uring worker pool behavior, and a liburing issue (#916) raised the question of worker thread priority, but these are practitioner observations, not formal analyses.

### Tail Latency and Scheduling

Li et al. (2014) identified hardware, OS, and application-level sources of tail latency. Concord (Iyer et al., SOSP 2023) achieves microsecond-scale tail latency with approximate optimal scheduling. Shinjuku (Kaffes et al., NSDI 2019) provides preemptive scheduling for microsecond-scale workloads. Shenango (Ousterhout et al., NSDI 2019) achieves high CPU efficiency for latency-sensitive datacenter workloads. ghOSt (Humphries et al., SOSP 2021) delegates scheduling to userspace for policy flexibility. These works optimize scheduling for specific workload types but do not address the fundamental accounting gap from async execution.

### Resource Isolation and Cgroups

Linux cgroups (v2) provide CPU, memory, and I/O isolation for containers. Shue et al. (OSDI 2012) studied fairness and isolation for multi-tenant cloud storage. The Kubernetes community has documented issues with kernel thread starvation under aggressive cgroup CPU limits (Issue #125409). Our work provides the first formal analysis of how async kernel execution undermines cgroup CPU accounting.

### sched_ext and Extensible Scheduling

sched_ext (merged in Linux 6.12) enables BPF-based custom scheduling policies. While sched_ext provides unprecedented flexibility in scheduling policy design, it inherits the displacement problem — custom schedulers still rely on the kernel's CPU time accounting, which does not capture displaced time. Our formal model and attribution mechanism are applicable to sched_ext schedulers.

### Priority Inversion in Asynchronous Contexts

Priority inversion is well-studied for lock-based synchronization (Sha et al., 1990). PREEMPT_RT provides priority inheritance for kernel mutexes. However, priority inversion across asynchronous execution boundaries (e.g., io_uring worker threads, softirq handlers) has not been formally characterized. Our displacement model subsumes async priority inversion as a special case.

## Experiments

### Experimental Setup

All experiments run on a commodity Linux server with:
- 2 CPU cores (to amplify contention effects)
- 128GB RAM (sufficient for large-scale simulation)
- Python discrete-event simulator (no GPU required)
- Workloads derived from published traces and synthetic distributions

### Experiment 1: Displacement Characterization (Taxonomy Validation)

**Goal**: Quantify the displacement ratio for each async mechanism under representative workloads.

**Method**: For each mechanism in our taxonomy, we construct a workload that exercises it (e.g., io_uring random read/write, network packet processing, filesystem journal commits) and measure the fraction of CPU time consumed by relay entities vs. the requesting process.

**Data source**: We use published system call traces (from the SNIA IOTTA repository) and network traffic traces (from CAIDA) to parameterize realistic workloads.

**Expected results**: io_uring worker threads account for 15-35% of total CPU time in I/O-intensive workloads. Network softirq processing accounts for 10-25% under high packet rates. Workqueue processing accounts for 5-15% during filesystem-heavy operations.

### Experiment 2: Fairness Violation Quantification

**Goal**: Measure the fairness violation under heterogeneous displacement.

**Method**: Simulate N processes with varying displacement ratios sharing M cores under EEVDF. Measure Jain's fairness index on effective CPU shares (including displaced time). Compare with the scheduler's reported fairness (based only on direct CPU time).

**Configurations**:
- N ∈ {4, 8, 16, 32, 64, 128, 256}
- M ∈ {2, 4, 8} (simulated cores)
- α_i drawn from Beta(2, 5) distribution (skewed toward low displacement)
- Baseline: all processes CPU-bound (α = 0)

**Expected results**: Jain's fairness index on effective shares drops from ~1.0 (under the scheduler's view) to 0.7-0.85 (under the true view including displacement). The gap widens with N, confirming Theorem 3.

### Experiment 3: Cgroup Accounting Accuracy

**Goal**: Quantify how much CPU time "leaks" out of cgroup accounting due to displacement.

**Method**: Simulate a multi-tenant scenario with K cgroups, each containing processes with different displacement profiles. Compare the cgroup controller's reported CPU usage with the true usage (including displaced time).

**Configurations**:
- K ∈ {2, 4, 8, 16} cgroups
- Each cgroup: 4-16 processes with displacement ratios drawn from different distributions
- CPU limit: each cgroup limited to 1/K of total CPU

**Expected results**: I/O-intensive cgroups exceed their CPU limits by 15-30% when displacement is counted. CPU-bound cgroups are correspondingly under-served.

### Experiment 4: CCP Mechanism Evaluation

**Goal**: Evaluate the effectiveness and overhead of Causal Charge Propagation.

**Method**: Implement three CCP strategies (immediate, batched, statistical) in the simulator. Measure:
- Fairness restoration: How close to ideal fairness after CCP?
- Overhead: Additional scheduling decisions and context switches
- Convergence time: How quickly does CCP restore fairness after a workload change?

**Expected results**: Immediate CCP restores fairness to within 2-3% of ideal with ~5% overhead. Batched CCP (10ms intervals) restores fairness to within 5% with ~1% overhead. Statistical CCP achieves ~8% fairness gap with <0.5% overhead.

### Experiment 5: Sensitivity Analysis

**Goal**: Understand which parameters most affect displacement-induced unfairness.

**Method**: Vary one parameter at a time:
- Displacement ratio variance (Var(α))
- System load (utilization from 50% to 95%)
- Number of cores (2 to 32)
- Displacement burstiness (coefficient of variation of inter-displacement times)
- Relay entity scheduling priority

**Expected results**: Fairness violation is most sensitive to displacement ratio variance and system load, confirming analytical predictions.

### Experiment 6: Trace-Driven Validation

**Goal**: Validate the model with realistic workload parameters.

**Method**: Parameterize the simulator using published workload traces:
- YCSB-like database workloads (high io_uring usage)
- Web server workloads (mixed I/O and CPU)
- Machine learning inference serving (CPU-intensive with periodic I/O)

**Expected results**: Fairness violations in trace-driven scenarios match within 10% of model predictions, validating the analytical framework.

## Success Criteria

The hypothesis is **confirmed** if:
1. Displacement accounts for ≥10% of total CPU time in at least one representative workload class
2. Jain's fairness index on effective CPU shares drops below 0.9 under heterogeneous displacement with ≥8 competing processes
3. CCP restores fairness index to ≥0.95 with ≤5% overhead for at least one propagation strategy
4. Analytical bounds from the formal model match simulation results within 15%

The hypothesis is **refuted** if:
1. Displacement never exceeds 5% of total CPU time across all workload classes
2. Fairness violations remain within the scheduler's inherent tolerance (|lag| ≤ Q) even with heterogeneous displacement
3. No CCP strategy can improve fairness without >10% overhead

## References

1. Stoica, I., Abdel-Wahab, H., Jeffay, K., Baruah, S.K., Gehrke, J.E., and Plaxton, C.G. "A Proportional Share Resource Allocation Algorithm for Real-Time, Time-Shared Systems." In Proceedings of the IEEE Real-Time Systems Symposium (RTSS), 1996.

2. Molnar, I. "CFS Scheduler." Linux Kernel Documentation, 2007. https://docs.kernel.org/scheduler/sched-design-CFS.html

3. Zijlstra, P. "EEVDF Scheduler." Linux Kernel Documentation, 2023. https://docs.kernel.org/scheduler/sched-eevdf.html

4. Axboe, J. "Efficient IO with io_uring." Linux Kernel Documentation, 2019. https://kernel.dk/io_uring.pdf

5. Jasny, M., El-Hindi, M., Ziegler, T., Leis, V., and Binnig, C. "High-Performance DBMSs with io_uring: When and How to Use It." arXiv:2512.04859, 2025.

6. Isstaif, A.A.T., Kalyvianaki, E., and Mortier, R. "Mitigating Context Switching in Densely Packed Linux Clusters with Latency-Aware Group Scheduling." arXiv:2508.15703, 2025.

7. Li, J., Sharma, N.K., Ports, D.R.K., and Gribble, S.D. "Tales of the Tail: Hardware, OS, and Application-level Sources of Tail Latency." In Proceedings of the ACM Symposium on Cloud Computing (SoCC), 2014.

8. Humphries, J., Nandivada, N., Zhuo, D., Zhou, E., Weiss, G., Stutsman, R., Caulfield, A., Mogul, J.C., and Ghobadi, M. "ghOSt: Fast & Flexible User-Space Delegation of Linux Scheduling." In Proceedings of the 28th ACM Symposium on Operating Systems Principles (SOSP), 2021.

9. Shue, D., Freedman, M.J., and Shaikh, A. "Performance Isolation and Fairness for Multi-Tenant Cloud Storage." In Proceedings of the 10th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2012.

10. Ousterhout, A., Fried, J., Behrens, J., Belay, A., and Balakrishnan, H. "Shenango: Achieving High CPU Efficiency for Latency-sensitive Datacenter Workloads." In Proceedings of the 16th USENIX Symposium on Networked Systems Design and Implementation (NSDI), 2019.

11. Kaffes, K., Chong, T., Humphries, J.T., Belay, A., Mazières, D., and Kozyrakis, C. "Shinjuku: Preemptive Scheduling for μsecond-scale Tail Latency." In Proceedings of the 16th USENIX Symposium on Networked Systems Design and Implementation (NSDI), 2019.

12. Sha, L., Rajkumar, R., and Lehoczky, J.P. "Priority Inheritance Protocols: An Approach to Real-Time Synchronization." IEEE Transactions on Computers, 39(9):1175-1185, 1990.

13. Waldspurger, C.A. and Weihl, W.E. "Lottery Scheduling: Flexible Proportional-Share Resource Management." In Proceedings of the 1st USENIX Symposium on Operating Systems Design and Implementation (OSDI), 1994.

14. Jia, J. "eBPF and Kernel Extensions." PhD Dissertation, University of Illinois Urbana-Champaign, 2025.

15. Iyer, R., Unal, M., Kogias, M., and Candea, G. "Achieving Microsecond-Scale Tail Latency Efficiently with Approximate Optimal Scheduling." In Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP), 2023.

16. Waldspurger, C.A. and Weihl, W.E. "Lottery Scheduling: Flexible Proportional-Share Resource Management." In Proceedings of the 1st USENIX Symposium on Operating Systems Design and Implementation (OSDI), 1994.

17. Valente, P. and Checconi, F. "High Throughput Disk Scheduling with Fair Bandwidth Distribution." IEEE Transactions on Computers, 59(9):1172-1186, 2010.

18. Lozi, J.-P., Lepers, B., Funston, J., Gaud, F., Quema, V., and Fedorova, A. "The Linux Scheduler: A Decade of Wasted Cores." In Proceedings of the 11th European Conference on Computer Systems (EuroSys), 2016.

19. Banga, G., Druschel, P., and Mogul, J.C. "Resource Containers: A New Facility for Resource Management in Server Systems." In Proceedings of the 3rd USENIX Symposium on Operating Systems Design and Implementation (OSDI), 1999.
