# KAPHE: Kernel-Aware Performance Heuristic Extraction
## A Lightweight Statistical Methodology for Interpretable OS Kernel Configuration

---

## 1. Introduction

### 1.1 Context and Problem Statement

Modern Linux kernels expose thousands of tunable parameters across memory management (`vm.*`), I/O scheduling, process scheduling, and networking subsystems. These parameters significantly impact application performance, yet system administrators and developers typically rely on static heuristics or laborious trial-and-error to configure them. For example:

- `vm.swappiness` controls the aggressiveness of page reclaim, with recommended values ranging from 0 (low-latency workloads) to 60+ (batch processing)
- I/O scheduler selection (none, mq-deadline, bfq) depends on storage device type and access patterns
- Memory allocator choice (SLAB, SLUB, SLOB) must balance fragmentation vs. speed tradeoffs

Current approaches to kernel tuning fall into three categories, each with significant limitations:

1. **Expert-driven heuristics** rely on accumulated operational wisdom (e.g., "use noop scheduler for NVMe") but fail to capture workload-specific nuances and become outdated as kernel internals evolve.

2. **Machine learning methods** (e.g., OS-R1, 2025; MLKAPS, 2025) require extensive training data, suffer from poor interpretability, and struggle to generalize across diverse hardware and kernel versions. These approaches treat kernel tuning as a black-box optimization problem, obscuring *why* certain configurations are recommended.

3. **LLM-assisted frameworks** (e.g., AutoOS, 2024) leverage large language models for configuration generation but face challenges with hallucination, configuration validity checking, and require expensive LLM inference at deployment time.

### 1.2 Key Insight and Hypothesis

We observe that **workload sensitivity to OS kernel configuration follows predictable statistical patterns** that can be characterized through lightweight profiling and rule extraction. Rather than learning black-box models or relying on expensive LLM inference, we can extract interpretable heuristics of the form:

> "Workloads with high memory allocation churn (>1000 allocs/sec/thread), small working sets (<100MB), and frequent thread creation benefit from SLUB over SLAB with 15% better throughput and 8% lower latency."

Our central hypothesis is:

**H1**: A statistical methodology that profiles application workloads across kernel configuration dimensions and extracts decision rules based on workload characteristics can achieve performance within 10% of optimal with minimal overhead (<3%), while providing actionable, interpretable guidance that outperforms black-box ML approaches in explainability.

### 1.3 Proposed Approach

We propose **KAPHE** (Kernel-Aware Performance Heuristic Extraction), a three-phase methodology:

1. **Profiling Phase**: Execute targeted microbenchmarks that exercise specific kernel subsystems (memory allocation, I/O patterns, scheduling behavior) while varying runtime-tunable configuration parameters via `sysctl`.

2. **Characterization Phase**: Use statistical analysis (effect size measurement, correlation analysis, ANOVA) to identify which workload metrics predict performance under different configurations.

3. **Extraction Phase**: Generate decision rules using rule induction algorithms (RIPPER) that map workload characteristics to optimal configurations, producing human-readable IF-THEN rules with confidence metrics.

The output is a set of interpretable rules and an automated recommendation engine that requires no ML model or LLM at deployment time—only simple rule matching.

---

## 2. Related Work

### 2.1 ML-Based HPC Kernel Auto-Tuning

**MLKAPS** (Jam et al., 2025) uses decision trees and Gradient Boosting Decision Trees (GBDT) to map HPC kernel inputs to optimized design parameters. While MLKAPS shares the goal of input-to-configuration mapping, it differs fundamentally from KAPHE in three key aspects:

1. **Domain**: MLKAPS targets HPC compute kernels (e.g., Intel MKL matrix operations) with algorithmic parameters, while KAPHE targets the Linux OS kernel with runtime-tunable system parameters.

2. **Methodology**: MLKAPS uses computationally expensive GBDT surrogate models for exploration. KAPHE uses lightweight statistical analysis and rule induction with no runtime ML inference.

3. **Output**: MLKAPS generates decision tree code for embedding into HPC applications. KAPHE generates human-readable IF-THEN rules explaining *why* configurations are recommended.

### 2.2 Bayesian Optimization for Auto-Tuning

**GPTune** (Liu et al., 2021) uses multitask learning and Gaussian Process models for HPC application autotuning. Like MLKAPS, GPTune operates in the HPC domain, optimizing application-level parameters rather than OS kernel configurations. KAPHE differs by focusing on interpretability through explicit rule extraction rather than black-box surrogate models.

### 2.3 LLM-Based Kernel Tuning

**AutoOS** (Chen et al., 2024) applies Large Language Models to Linux kernel configuration optimization for AIoT devices, using a state machine-based traversal algorithm. While AutoOS addresses the same domain (OS kernel), it relies on expensive LLM inference and faces hallucination challenges. KAPHE provides deterministic, interpretable rules without requiring LLM deployment.

**OS-R1** (2025) uses reinforcement learning trained LLMs for kernel tuning. This approach requires extensive GPU resources for training and inference, making it impractical for production deployment. KAPHE requires no training and minimal runtime overhead.

### 2.4 Positioning Summary

| Approach | Domain | Method | Runtime Cost | Interpretability |
|----------|--------|--------|--------------|------------------|
| MLKAPS | HPC Kernels | GBDT + Decision Trees | Low (embedded C code) | Low (black-box model) |
| GPTune | HPC Applications | Gaussian Processes | High (BO iterations) | Low |
| AutoOS | OS Kernel | LLM + State Machine | Very High (LLM inference) | Medium (LLM reasoning) |
| OS-R1 | OS Kernel | RL-trained LLM | Very High (LLM inference) | Low |
| **KAPHE** | **OS Kernel** | **Statistical + Rule Induction** | **Very Low (<1ms lookup)** | **High (explicit rules)** |

KAPHE fills the gap for lightweight, interpretable, deployment-friendly kernel tuning.

---

## 3. Proposed Methodology

### 3.1 Overview

KAPHE operates in three phases with clearly defined interfaces:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Profiling      │───▶│ Characterization│───▶│  Rule Extraction│
│  Phase          │    │  Phase          │    │  Phase          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
     Workload             Sensitivity            Decision Rules
     Signatures          Analysis               & Recommendations
```

### 3.2 Phase 1: Profiling

**Subsystem Coverage**: We target three critical kernel subsystems with runtime-tunable parameters:

1. **Memory Management**: 
   - Reclaim aggressiveness (`vm.swappiness`, `vm.dirty_ratio`, `vm.dirty_background_ratio`)
   - Huge page settings (`vm.nr_hugepages`)
   - Page cache tuning (`vm.vfs_cache_pressure`)

2. **I/O Subsystem**:
   - Scheduler selection (`/sys/block/*/queue/scheduler`)
   - Queue depth (`/sys/block/*/queue/nr_requests`)
   - Read-ahead settings (`/sys/block/*/queue/read_ahead_kb`)

3. **Process Scheduling**:
   - CFS tunables (`kernel.sched_latency_ns`, `kernel.sched_min_granularity_ns`)
   - Load balancing (`kernel.sched_migration_cost_ns`)

**Workload Metrics**: For each execution, we collect:
- Memory: allocation rate, working set size, page fault rate
- I/O: operation mix (read/write ratio), sequentiality, queue depth
- CPU: IPC, context switch rate, cache miss rates
- System: syscall rate, interrupt frequency

**Implementation**: We use eBPF programs attached to kernel tracepoints to collect metrics with <2% overhead. Configurations are varied via `sysctl` and `/sys` interfaces without requiring kernel recompilation.

### 3.3 Phase 2: Characterization

**Sensitivity Analysis**: For each subsystem configuration, we compute:
- Effect size (Cohen's d) between different configurations
- Pearson correlation between workload metrics and performance improvement
- Interaction effects between subsystems using two-way ANOVA

**Statistical Methods**:
- ANOVA to identify significant configuration effects (p < 0.05)
- Pearson correlation for continuous workload metrics
- Eta-squared to measure effect size

**Output**: A sensitivity matrix that quantifies which workload characteristics predict benefit from specific configurations.

### 3.4 Phase 3: Rule Extraction

We use the RIPPER algorithm (Cohen, 1995) for rule induction, producing interpretable IF-THEN rules with quality metrics:

```
IF alloc_rate > 1000 AND working_set < 100MB AND thread_churn > 50/sec
THEN recommend swappiness=10, dirty_ratio=5
EXPECTED_IMPROVEMENT: 12-18%
CONFIDENCE: 0.87
COVERAGE: 23% of training workloads
```

Rules include:
- **Confidence**: Precision of the rule on training data
- **Coverage**: Percentage of workloads triggering the rule
- **Expected improvement**: Mean performance gain with confidence interval
- **Conditions**: Human-readable workload characteristics

### 3.5 Deployment

At runtime, KAPHE profiles the target workload for 30-60 seconds, computes its signature, matches against the rule database, and recommends configurations. The recommendation engine is simple rule matching with <1ms lookup time—no ML inference or LLM calls required.

---

## 4. Experimental Plan

### 4.1 Research Questions

**RQ1**: Can KAPHE accurately characterize workload sensitivity to kernel configurations across diverse application types?

**RQ2**: Do the extracted rules generalize to unseen workloads within the same category?

**RQ3**: How close to oracle-optimal performance do KAPHE recommendations achieve compared to ML-based approaches?

**RQ4**: What is the runtime overhead of the profiling and recommendation pipeline?

**RQ5**: How interpretable are KAPHE's rules compared to MLKAPS decision trees, as measured by established metrics?

### 4.2 Benchmark Workloads

We select 12 workloads across 4 categories (reduced from original 20 to fit 8-hour budget):

| Category | Workloads | Key Characteristics |
|----------|-----------|---------------------|
| In-memory DB | Redis, Memcached | High alloc rate, small objects |
| Analytics | TPC-H (PostgreSQL), ClickHouse | Large working sets, sequential I/O |
| Web Services | Nginx | Connection churn, mixed I/O |
| Build/Compile | Linux kernel compile | Fork-heavy, bursty I/O |

### 4.3 Evaluation Metrics

**Performance Metrics**:
- **Recommendation Accuracy**: % of workloads where KAPHE selects a configuration within 5% of optimal
- **Performance Improvement**: Speedup vs. default configuration, normalized by oracle-optimal
- **Profiling Overhead**: CPU and latency impact during profiling phase

**Interpretability Metrics** (following Brakke, 2024 and Fityah et al., 2025):
- **Rule Complexity**: Number of rules + average conditions per rule
- **Rule Coverage**: Percentage of test cases covered by top-N rules
- **Fidelity**: Agreement between rule-based recommendations and oracle on test set

### 4.4 Baselines

1. **Default**: Stock kernel configuration
2. **Oracle**: Grid search of configuration space (limited to feasible subset)
3. **MLKAPS-style**: Decision tree trained on same profiling data
4. **Expert Heuristics**: Rules from Linux tuning guides (Red Hat, Ubuntu performance docs)

*Note*: We exclude OS-R1 as it requires GPU resources and extensive training infrastructure not available in our environment.

### 4.5 Real Kernel Validation Experiment

To address concerns about simulation validity, we include a **real kernel validation subset**:

- Select 4 representative workloads (1 per category)
- Run on actual Linux kernel with `sysctl`-tunable parameters
- Measure actual performance vs. predicted optimal
- Validate rule predictions against ground truth

### 4.6 Expected Results

We hypothesize:
- KAPHE achieves within 10% of oracle performance on 80%+ of workloads
- Profiling overhead remains below 3% for all workloads
- Rule complexity stays below 50 total conditions across all rules
- Rule fidelity exceeds 85% on held-out test workloads
- Cross-workload generalization within categories exceeds 70% accuracy

### 4.7 Time Budget Allocation (8 hours)

| Task | Time |
|------|------|
| Profiling implementation | 2 hours |
| Workload execution and data collection | 3 hours |
| Rule extraction algorithm | 1.5 hours |
| Analysis and validation | 1.5 hours |

---

## 5. Success Criteria

### 5.1 Primary Success Criteria

The research succeeds if:

1. **Technical Criterion**: KAPHE demonstrates statistically significant performance improvements over default configurations (paired t-test, p<0.05) across diverse workloads.

2. **Generalization Criterion**: Rules extracted from one workload set generalize to unseen workloads with <20% performance degradation vs. workload-specific tuning.

3. **Practicality Criterion**: End-to-end profiling and recommendation completes within 5 minutes with <5% overhead during profiling.

4. **Interpretability Criterion**: Extracted rules achieve:
   - Average rule length ≤ 5 conditions
   - Total rule count ≤ 30 for full rule set
   - Fidelity ≥ 80% on test set

### 5.2 Failure Conditions

The research fails if:
- Workload sensitivity to kernel configurations is found to be essentially random (no predictable patterns)
- Rule extraction produces trivial rules (e.g., single condition rules) or unusable recommendations
- Overhead exceeds acceptable thresholds for production deployment (>5%)
- Rules are less interpretable than simple decision tree baselines

---

## 6. Contributions

1. **Methodology**: A lightweight, interpretable framework for characterizing and extracting decision rules for OS kernel configuration selection, filling the gap between black-box ML approaches and static expert heuristics.

2. **Empirical Findings**: A systematic sensitivity analysis of runtime-tunable Linux kernel parameters across memory, I/O, and scheduling subsystems for diverse workloads.

3. **Practical Tool**: An open-source implementation of KAPHE that provides actionable kernel tuning recommendations without requiring ML models or LLMs at deployment.

4. **Validation Dataset**: A curated benchmark suite with real kernel measurements enabling reproducible research in OS configuration optimization.

---

## 7. References

1. Jam, M., et al. (2025). MLKAPS: Machine Learning and Adaptive Sampling for HPC Kernel Auto-tuning. ACM Transactions on Architecture and Code Optimization. https://doi.org/10.1145/3774418

2. Chen, H., et al. (2024). AutoOS: Make Your OS More Powerful by Exploiting Large Language Models. Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235:7511-7525.

3. OS-R1 Authors. (2025). OS-R1: Agentic Operating System Kernel Tuning with Reinforcement Learning. arXiv:2508.12551.

4. Liu, Y., et al. (2021). GPTune: Multitask Learning for Autotuning Exascale Applications. Proceedings of the 26th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP), 234-246.

5. Matias, R., et al. (2015). An Experimental Comparison Analysis of Kernel-level Memory Allocators. Journal of Systems and Software, 103, 23-35.

6. Cohen, W.W. (1995). Fast Effective Rule Induction. Proceedings of the 12th International Conference on Machine Learning (ICML), 115-123.

7. Craun, M., et al. (2024). Eliminating eBPF Tracing Overhead on Untraced Processes. ACM SIGCOMM eBPF Workshop.

8. Lawall, J., et al. (2024). Should We Balance? Towards Formal Verification of the Linux Kernel Scheduler. Static Analysis Symposium (SAS).

9. Vernet, P. (2024). sched_ext: Extensible Scheduler Class. Linux Kernel Documentation.

10. Wang, X., et al. (2025). Mixture-of-Schedulers: An Adaptive Scheduling Agent as a Learned Router for Expert Policies. arXiv:2511.11628.

11. Brakke, L. (2024). Simplifying Random Forests Through Post-Hoc Rule Extraction. Master's Thesis, University of Georgia.

12. Fityah, F., Setiawan, N.A., & Anggrahini, D.W. (2025). Interpretability Evaluation of Rule-Based Classifier in Myocardial Infarction Classification. Communications in Science and Technology, 10(2), 460-466.
