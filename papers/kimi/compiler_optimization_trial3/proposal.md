# Research Proposal: Joint Compute and Layout Optimization via Hierarchical E-Graphs

## 1. Introduction

### 1.1 Context and Problem Statement

Modern compilers face a fundamental tension: they must explore vast optimization spaces while maintaining practical compilation times. The phase-ordering problem—where the sequence of optimization passes dramatically affects output quality—has plagued compilers for decades. Traditional solutions rely on fixed pass pipelines or machine learning models trained to predict pass sequences, but these approaches often miss optimal solutions hidden behind complex interaction effects.

Equality saturation has emerged as a promising alternative. By representing all equivalent program variants compactly in an e-graph and deferring optimization decisions to extraction time, it eliminates the phase-ordering problem entirely. However, three critical challenges prevent widespread adoption:

1. **Extraction is NP-hard**: Finding the optimal program from an e-graph requires solving an intractable problem (Zhang 2023; Stepp 2011).

2. **Memory bandwidth is the new bottleneck**: While equality saturation excels at finding algebraic simplifications, it largely ignores data layout and memory access patterns—yet on modern architectures, memory bandwidth often dominates execution time more than instruction count.

3. **Scalability concerns**: Monolithic e-graphs for entire programs lead to exponential blowup, limiting practical application.

### 1.2 Key Insight

Recent theoretical advances show that e-graph extraction becomes tractable for graphs with bounded treewidth. Goharshady et al. (OOPSLA 2024) demonstrated this empirically on real compiler e-graphs from Cranelift, showing that treewidth-aware extraction runs in fractions of the time needed by ILP solvers. Independently, Sun et al. (2024) established a theoretical connection between e-graphs and Boolean circuits, deriving a similar treewidth-parameterized algorithm.

We hypothesize that combining these insights with a novel hierarchical representation enables a new compiler architecture: **hierarchical e-graphs** that jointly optimize computation (via traditional rewrite rules) and data layout (via layout-aware extraction), using treewidth-aware algorithms for scalable near-optimal extraction.

**Our contribution differs from prior work**: While Goharshady et al. and Sun et al. focus on extraction algorithms alone, we introduce:
- A hierarchical e-graph representation that enables compositional optimization across loop nests, functions, and modules
- Joint optimization of computation and data layout within a unified e-graph framework
- Layout transformation rules with formal soundness guarantees

### 1.3 Hypothesis

**H1**: Real-world program e-graphs exhibit low treewidth (≤10) when rewrite rules are applied hierarchically, making treewidth-aware extraction practical.

**H2**: Joint optimization of computation and data layout via unified hierarchical e-graph representation achieves better memory bandwidth utilization than sequential optimization (first compute, then layout).

**H3**: The combination of hierarchical saturation and treewidth-aware extraction provides compile times competitive with greedy extraction while achieving solution quality within 10% of optimal ILP.

## 2. Proposed Approach

### 2.1 Overview

We propose **MemSat** (Memory-aware Equality Saturation), a compiler optimization framework with three core innovations:

1. **Hierarchical E-Graph Construction**: Rather than saturating an entire program at once, we build e-graphs at multiple granularities (loop nest → function → module), enabling compositional analysis.

2. **Layout-Aware Rewrite Rules**: We extend traditional rewrite rules with data layout transformations (Array-of-Structs ↔ Struct-of-Arrays, tiling, padding), embedding memory cost models directly into the e-graph.

3. **Treewidth-Guided Extraction**: We leverage existing treewidth-aware algorithms (Goharshady et al. 2024; Sun et al. 2024) for efficient near-optimal extraction.

### 2.2 Technical Details

#### 2.2.1 Hierarchical E-Graph Representation

Traditional equality saturation builds a monolithic e-graph for an entire function, leading to exponential blowup. MemSat instead uses a hierarchical representation:

- **Level 1 (Loop Nest)**: E-graphs for individual loop nests, capturing arithmetic optimizations and local data layout.
- **Level 2 (Function)**: Function-level e-graphs where e-classes contain e-nodes representing different loop nest implementations.
- **Level 3 (Module)**: Interprocedural e-graphs for global layout decisions.

Each level communicates via **interface summaries** that characterize memory access patterns without exposing full internal structure. This decomposition keeps treewidth bounded at each level.

#### 2.2.2 Layout-Aware Rewriting

We extend the e-graph with **layout e-nodes** that represent different data organization strategies:

```
// Original: Array-of-Structs
struct Point { float x, y, z; };
Point points[N];

// E-class contains equivalent layouts:
// - AoS: points[i].x, points[i].y, points[i].z
// - SoA: xs[i], ys[i], zs[i]
// - Tiled: tile-major organization
```

Each layout e-node has an associated **memory cost model**:
- *Spatial locality*: Consecutive accesses to same cache line
- *Temporal locality*: Reuse distance in cache lines
- *Stride patterns*: Regularity for prefetcher effectiveness

**Soundness of Layout Transformations**: All layout transformations are proven correct via structural equivalence. For each layout transformation rule (e.g., AoS→SoA), we provide:
1. A formal specification of the source and target memory layouts
2. A proof that access patterns to corresponding elements yield identical values
3. Verification that the transformation preserves data dependencies

We verify correctness through:
- **Static proofs**: For regular transformations (AoS↔SoA, tiling), we provide parametric proofs that work for any array dimensions
- **Symbolic execution**: For complex cases, we use symbolic execution to verify equivalence
- **Type system**: Layout types track memory organization invariants, ensuring only valid transformations are applied

#### 2.2.3 Treewidth-Aware Extraction

Our extraction algorithm exploits the observation that e-graphs from compiler optimizations typically have low treewidth because:
1. Rewrite rules are local (affecting small program fragments)
2. Program structure (control flow, nesting) limits connectivity
3. Hierarchical decomposition keeps cross-cutting edges sparse

We leverage existing treewidth-aware extraction algorithms:
- **Goharshady et al. (2024)**: Dynamic programming on tree decompositions, running in O(|G| × 2^O(k)) time for treewidth k
- **Sun et al. (2024)**: Circuit-based approach with simplification techniques that reduce e-graph size by 40-80%

For graphs with treewidth k ≤ 10, these algorithms provide optimal extraction in fractions of ILP time.

#### 2.2.4 Memory Cost Model

We develop an analytical model for memory bandwidth consumption:

```
Cost(layout, access_pattern) = 
    Σ (accesses_to_cache_line_i / reuse_distance_i) × 
    (1 - prefetcher_effectiveness(stride_i))
```

This model is calibrated against hardware performance counters but requires only static analysis to apply.

### 2.3 Key Innovations

1. **First joint optimization framework** combining equality saturation with data layout transformation in a unified representation
2. **Hierarchical e-graph representation** enabling compositional equality saturation with bounded treewidth
3. **Verified layout transformation rules** with formal soundness guarantees

## 3. Related Work

### 3.1 Equality Saturation and E-Graphs

Equality saturation was introduced by Joshi et al. (2002) and popularized for compiler optimization by Tate et al. (2009). The egg library (Willsey et al., POPL 2021) made equality saturation accessible, leading to applications in tensor compilers (TenSat), hardware synthesis (Lakeroad), and program synthesis.

**E-Graph Extraction**: The extraction problem is NP-hard (Zhang 2023; Stepp 2011). Current approaches use:
- ILP formulation (slow but optimal)
- Greedy heuristics (fast but suboptimal)
- Beam search (intermediate tradeoff)

**Treewidth-Aware Extraction**: 
- **Goharshady et al. (OOPSLA 2024)**: "Fast and Optimal Extraction for Sparse Equality Graphs" — demonstrates that real-world e-graphs from Cranelift have low treewidth and presents a parameterized algorithm achieving optimal extraction in O(|G| × 2^O(k)) time.

- **Sun et al. (arXiv 2024)**: "E-Graphs as Circuits, and Optimal Extraction via Treewidth" — independently establishes a connection between e-graphs and monotone Boolean circuits, deriving a similar treewidth-parameterized algorithm with additional circuit simplification techniques.

**Our Differentiation**: While both Goharshady et al. and Sun et al. focus on extraction algorithms in isolation, we apply treewidth-aware extraction to a novel hierarchical e-graph representation that enables joint optimization of computation and data layout. Our contribution is the system architecture, not the extraction algorithm itself.

### 3.2 Compiler Phase Ordering

The phase ordering problem has been studied for decades (Whitfield & Soffa 1997). Recent approaches include:

- **ML-guided pass ordering**: AutoPhase (Haj-Ali et al., MLSys 2020) uses reinforcement learning to find pass sequences. Coreset (Liang et al., 2023) uses GNNs. These require training data and don't generalize to unseen programs.

- **Search-based**: MiCOMP (Ashouri et al., 2017) uses neuro-evolution. Wang et al. (arXiv 2024) propose iterative bi-directional optimization to explore less efficient intermediate programs.

Equality saturation eliminates phase ordering by construction, but prior work (e.g., Diospyros) only applies it to small kernels due to scalability concerns.

### 3.3 Memory Bandwidth Optimization

Memory bandwidth is increasingly the bottleneck (Wulf & McKee 1995). Optimization approaches include:

- **Data layout transformation**: SoA vs. AoS conversion (Allen & Kennedy 2002), padding for alignment, tiling for cache. Traditionally applied manually or via simple heuristics.

- **Polyhedral compilation**: Pluto (Bondhugula et al., PLDI 2008), Polly (Grosser et al.) perform automatic loop transformations but don't consider layout changes that cross function boundaries.

- **Profile-guided layout**: Cao et al. (CGO 2024) use profiling to guide structure splitting. Requires representative inputs and adds profiling overhead.

MemSat differs by integrating layout optimization into the e-graph, enabling joint optimization with computation and exploiting semantic equivalences for layout alternatives.

### 3.4 Recent Work on E-Graphs in ML Compilers

Vohra et al. (OOPSLA 2025) present "Mind the Abstraction Gap: Bringing Equality Saturation to Real-World ML Compilers," addressing the integration of equality saturation with production MLIR-based compilers. Our work is complementary: they focus on bridging the gap between e-graph representations and production compiler infrastructures, while we focus on joint compute+layout optimization via hierarchical decomposition.

### 3.5 Positioning

| Approach | Phase Ordering | Memory-Aware | Scalable | Optimal |
|----------|---------------|--------------|----------|---------|
| Traditional compilers | ✗ | ✗ | ✓ | ✗ |
| ML-guided (AutoPhase) | Partial | ✗ | ✓ | ✗ |
| Superoptimization | ✓ | ✗ | ✗ | ✓ |
| Equality saturation (egg) | ✓ | ✗ | Partial | ✗ |
| Goharshady et al. 2024 | ✓ | ✗ | ✓ | Yes (bounded tw) |
| Sun et al. 2024 | ✓ | ✗ | ✓ | Yes (bounded tw) |
| **MemSat (ours)** | ✓ | ✓ | ✓ | Near |

## 4. Experiments

### 4.1 Research Questions

1. Do real-world compiler e-graphs exhibit low treewidth under hierarchical decomposition? (validates H1)
2. Does joint optimization outperform sequential approaches? (validates H2)
3. What is the compile time vs. solution quality tradeoff? (validates H3)

### 4.2 Experimental Setup (Narrowed Scope for 8-hour Timeline)

Given the 8-hour experiment time limit, we focus on a focused feasibility demonstration:

**Benchmarks** (reduced set):
- Polybench/C 4.2 (10 kernels: gemm, 2mm, 3mm, gemver, gesummv, mvt, syrk, syr2k, jacobi-2d, fdtd-2d)
- Focus on linear algebra and stencil kernels where memory bandwidth matters

**Baselines**:
1. LLVM -O3 (standard optimization)
2. Greedy equality saturation (egg with greedy extraction)
3. ILP-based extraction (optimal but slow, run on smaller instances)

**Hardware**: 
- Intel Xeon (CPU-only, 2 cores available, 128GB RAM)
- No GPU required—experiments are CPU-based

### 4.3 Evaluation Metrics

**Performance**: 
- Execution time (geometric mean across benchmarks)
- Memory bandwidth utilization (bytes accessed / useful work)
- Cache miss rates (measured with perf)

**Compilation**: 
- Compile time
- E-graph size (nodes, edges, e-classes)
- Treewidth of resulting e-graphs (measured with LibTW)
- Extraction time

### 4.4 Expected Results

**R1 (Treewidth)**: We expect 80%+ of loop-nest e-graphs to have treewidth ≤ 8, validating the hierarchical decomposition approach.

**R2 (Performance)**: MemSat should achieve:
- 10-20% speedup over LLVM -O3 on memory-bound benchmarks
- 5-10% speedup over greedy equality saturation

**R3 (Compile Time)**: 
- Hierarchical extraction: 50-200ms per function
- Within 3× of greedy extraction time
- Solution quality within 10% of ILP optimal (where feasible)

### 4.5 Ablation Studies

1. **Hierarchy levels**: Compare flat vs. 2-level hierarchy
2. **Extraction algorithms**: Greedy vs. treewidth-aware vs. ILP
3. **Layout rules**: With vs. without data layout transformations
4. **Cost models**: Simple instruction count vs. memory-aware cost

### 4.6 Fallback Plan

If treewidth-aware extraction proves too complex to implement within the time limit:
- **Primary fallback**: Use beam search extraction (easier to implement, intermediate quality)
- **Secondary fallback**: Demonstrate concept on smaller kernels only, showing treewidth measurements and cost model effectiveness without full extraction

## 5. Success Criteria

### 5.1 Confirmation

The hypothesis is confirmed if:
- At least 70% of benchmark e-graphs have treewidth ≤ 10 under hierarchical decomposition
- MemSat achieves ≥10% geomean speedup over LLVM -O3 on memory-bound workloads
- Extraction is ≤5× slower than greedy while achieving ≥90% of ILP solution quality

### 5.2 Refutation

The hypothesis is refuted if:
- Real-world e-graphs have treewidth > 15 under hierarchical decomposition
- Joint optimization shows no improvement over sequential (computation then layout)
- Hierarchical decomposition loses optimization opportunities vs. flat e-graphs

## 6. Implementation Plan

### 6.1 Prototype Architecture

MemSat is implemented as a standalone tool that processes LLVM IR:

1. **Frontend**: Parse LLVM IR, identify loop nests and functions
2. **E-graph Builder**: Construct hierarchical e-graphs using egg library
3. **Layout Generator**: Create layout e-nodes for array accesses
4. **Treewidth Analyzer**: Measure treewidth using LibTW or heuristics
5. **Extractor**: Treewidth-aware extraction (leveraging existing algorithms)
6. **Code Generator**: Convert extracted e-graph back to LLVM IR

### 6.2 Timeline (8-hour Budget)

| Hours | Task |
|-------|------|
| 0-1.5 | E-graph construction for Polybench kernels, treewidth measurement |
| 1.5-3 | Implement layout transformation rules with verification |
| 3-5 | Implement hierarchical extraction (or beam search fallback) |
| 5-6.5 | Performance evaluation on benchmarks |
| 6.5-7.5 | Ablation studies and data collection |
| 7.5-8 | Documentation and analysis |

## 7. References

### Key Papers

1. Willsey et al., "egg: Fast and Extensible Equality Saturation", POPL 2021
2. Goharshady et al., "Fast and Optimal Extraction for Sparse Equality Graphs", OOPSLA 2024
3. Sun et al., "E-Graphs as Circuits, and Optimal Extraction via Treewidth", arXiv 2024
4. Haj-Ali et al., "AutoPhase: Juggling HLS Phase Orderings in Random Forests with Deep Reinforcement Learning", MLSys 2020
5. Bondhugula et al., "A Practical Automatic Polyhedral Parallelizer and Locality Optimizer", PLDI 2008
6. Tate et al., "Equality Saturation: A New Approach to Optimization", POPL 2009
7. Wang et al., "Solving the Phase Ordering Problem ≠ Generating the Globally Optimal Code", arXiv 2024
8. Yihong Zhang, "The E-graph Extraction Problem is NP-complete", Blog post, June 2023
9. Vohra et al., "Mind the Abstraction Gap: Bringing Equality Saturation to Real-World ML Compilers", OOPSLA 2025
10. Yang et al., "Equality Saturation for Tensor Graph Superoptimization", MLSys 2021
