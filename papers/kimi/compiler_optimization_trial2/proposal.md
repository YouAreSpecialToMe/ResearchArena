# Research Proposal: LEOPARD — Lightweight Learned Guidance for Equality Saturation in Compiler Optimization

## 1. Introduction

### 1.1 Background and Problem Statement

The phase-ordering problem has plagued compiler optimization for decades. Traditional compilers apply optimization passes sequentially, where each pass destructively transforms the intermediate representation (IR). The order in which these passes are applied significantly impacts performance—different orders can produce vastly different code quality, and finding the optimal sequence is NP-hard [1, 2].

Recent work has explored machine learning approaches to predict good pass sequences [3, 4, 5], but these treat the compiler as a black box and require extensive training data. A more principled approach is **equality saturation** (ES), which uses e-graphs to represent multiple equivalent program versions simultaneously, avoiding destructive rewriting and thus eliminating the phase-ordering problem in theory [6, 7].

However, equality saturation faces a critical practical limitation: **memory explosion**. As rewrite rules are applied, the e-graph grows exponentially, quickly exhausting available memory. When memory limits are reached, the construction phase must stop prematurely, effectively reintroducing the phase-ordering problem—certain rewrites were applied while others were never considered [8].

### 1.2 Related Prior Work: Aurora

Aurora (Barbulescu et al., 2024) [9] demonstrated that learned guidance for equality saturation is feasible in the domain of relational query optimization. Aurora uses reinforcement learning with spatio-temporal graph neural networks (GNNs combined with RNNs) to guide e-graph construction for SQL queries. While groundbreaking, Aurora has several characteristics that limit its applicability to general compiler optimization:

1. **Heavyweight RL training**: Aurora requires 200K+ training steps per task with complex PPO-based training, making it expensive to adapt to new domains.
2. **Complex architecture**: The GNN+RNN architecture has substantial inference overhead, potentially adding latency to the compilation pipeline.
3. **Domain specificity**: Aurora is designed for SQL query plans, which have different structural properties than general-purpose compiler IRs.

### 1.3 Key Insight and Research Hypothesis

Our key insight is that **learned guidance for equality saturation can be effective with lightweight supervised learning rather than expensive reinforcement learning, enabling practical deployment in production compilers**. The patterns in beneficial rewrite sequences are learnable with simple models (small MLPs or gradient-boosted trees) and transferable across programs in general-purpose compiler optimization domains.

**Research Hypothesis**: A lightweight supervised learning model (small MLP or gradient-boosted trees) trained to predict the optimization potential of rewrite rules can guide e-graph construction to achieve within 80% of exhaustive equality saturation's code quality, while using significantly less memory (40-60% reduction) and with minimal inference overhead (<5% of compilation time).

### 1.4 Proposed Approach: LEOPARD (LEarned Optimization Potential for Adaptive Rewrite Direction)

We propose LEOPARD, a system that combines equality saturation with a lightweight learned predictor to guide e-graph construction for general-purpose compiler optimization:

1. **Lightweight Rule Scorer**: A small feedforward network or gradient-boosted tree model (~1K parameters) predicts the expected improvement from applying a rule in a given context. This is trained via supervised learning on data collected from random rule application.

2. **Adaptive E-graph Construction**: Rules are applied in order of predicted benefit until a memory budget is reached, with periodic re-scoring based on e-graph evolution.

3. **Graceful Degradation**: When prediction confidence is low, the system falls back to heuristics or exhaustive application, ensuring correctness and avoiding catastrophic failures.

4. **Efficient Extraction**: Once construction completes, we extract the optimal program using a learned cost model that estimates actual runtime performance.

## 2. Related Work

### 2.1 Phase Ordering Solutions

**Traditional Approaches**: Fixed pass pipelines (-O2, -O3) are widely used but suboptimal [1]. Iterative compilation searches for better sequences but is computationally prohibitive [2].

**Machine Learning for Pass Ordering**: AutoPhase [3] uses reinforcement learning to find pass sequences, but requires thousands of compilations per program. Coreset-NVP [4] improves sample efficiency but still treats the compiler as a black box. Cummins et al. [5] train LLMs to predict optimization sequences, but these lack correctness guarantees and require massive training data.

### 2.2 Equality Saturation

Tate et al. [6] introduced equality saturation using Program Expression Graphs (PEGs). Willsey et al. [7] developed `egg`, a fast and flexible equality saturation library that has enabled numerous applications.

**Guided Equality Saturation**: Hagedorn et al. [10] propose sketch-guided equality saturation where programmers provide intermediate goals as "sketches." While effective, this requires manual guidance. Our approach learns guidance automatically.

**Learned Guidance - Aurora**: Barbulescu et al. [9] demonstrated that reinforcement learning can guide equality saturation for relational queries. Aurora uses a spatio-temporal RL agent (GNN+RNN) trained with PPO. LEOPARD differs from Aurora in three key aspects:

| Aspect | Aurora [9] | LEOPARD (This Work) |
|--------|------------|---------------------|
| **Domain** | Relational query optimization | General-purpose compiler optimization |
| **Learning** | RL (PPO) with GNN+RNN | Supervised learning with small MLP/GBDT |
| **Focus** | Exploration quality (training cost acceptable) | Inference speed (production deployment) |
| **Training Cost** | 200K+ RL steps, GPU required | Minutes on CPU, ~1K compilations |
| **Model Size** | GNN+RNN (thousands of parameters) | Small MLP or GBDT (~1K parameters) |
| **Inference Overhead** | Moderate (GNN forward pass) | Minimal (<1ms per decision) |

**E-graph Construction Guidance**: Hartmann et al. [8] use Monte Carlo Tree Search (MCTS) to guide e-graph construction for tensor programs. While effective, MCTS requires many rollouts per program. LEOPARD learns transferable patterns that amortize this cost across programs.

### 2.3 E-graph Extraction

Extracting the optimal program from an e-graph is NP-hard [11]. Current approaches use either expensive ILP solvers or fast but suboptimal greedy algorithms [12]. Recent work explores learned extraction [13], but focuses on the extraction phase only. LEOPARD addresses both construction and extraction.

## 3. Proposed Method

### 3.1 System Architecture

```
Input Program → E-graph Initialization → Iterative Expansion (guided by Learned Scorer)
                                                            ↓
                                              Memory Budget Reached
                                                            ↓
                                    Learned Extraction ← E-graph Final State
                                                            ↓
                                                    Optimized Program
```

### 3.2 Learned Rule Scorer

The core of LEOPARD is a lightweight predictor that scores rewrite rules. Given:
- Current e-graph state G
- Candidate rewrite rule r
- Context features c (operator being rewritten, surrounding code patterns)

The model predicts: `score(G, r, c) = E[improvement | apply r to G]`

**Model Architecture Options**:
1. **Small MLP**: 2-3 layers, 32-64 hidden units, ReLU activations (~1K parameters)
2. **Gradient Boosted Trees**: 10-50 trees, max depth 3-5 (fast inference, interpretable)

Features include:
- E-graph statistics (number of e-classes, average e-class size, depth)
- Rule features (transformation type, loop-nest level, data dependency pattern)
- Context features (dominant operators in the e-class, presence of specific patterns like reductions)

**Training Data**: We collect data by running equality saturation on a diverse set of programs with a random rule selection policy. For each rule application, we record:
- The e-graph state before application
- The rule applied
- The eventual improvement in extracted program quality

This creates a supervised learning problem: predict eventual improvement from immediate context.

### 3.3 Adaptive E-graph Construction with Graceful Degradation

Instead of the standard saturation loop, LEOPARD uses:

```python
def adaptive_expansion(program, rules, budget, scorer, confidence_threshold=0.7):
    egraph = EGraph(program)
    applied = set()
    
    while not egraph.saturated() and memory < budget:
        # Score all pending rewrites
        candidates = get_matchable_rules(egraph, rules)
        scores, confidences = scorer.score_with_confidence(candidates, egraph)
        
        # Apply top-k rules with confidence filtering
        for i, rule in enumerate(top_k(candidates, scores)):
            if confidences[i] >= confidence_threshold:
                egraph.apply(rule)
            else:
                # Fall back to round-robin for low-confidence predictions
                egraph.apply_next_available(rules)
        
        # Rebuild e-graph invariants
        egraph.rebuild()
    
    return egraph
```

The key innovations are:
1. **Confidence-aware scoring**: The model provides both a score and a confidence estimate
2. **Graceful degradation**: When confidence is low, the system falls back to baseline behavior
3. **Periodic re-scoring**: The scorer is updated based on the evolving e-graph structure

### 3.4 Failure Modes and Mitigation Strategies

We explicitly address potential failure modes:

| Failure Mode | Cause | Mitigation Strategy |
|--------------|-------|---------------------|
| **Poor predictions** | Model trained on insufficient/diverse data | Confidence threshold triggers fallback to exhaustive search |
| **Overfitting** | Model memorizes training programs | Regularization, small model size, cross-validation |
| **Distribution shift** | Test programs structurally different from training | Domain-agnostic features, ensemble of models |
| **Scoring overhead** | Feature extraction too expensive | Cache features, incremental updates |
| **Local optima** | Greedy rule selection misses better sequences | Periodic re-scoring, stochastic top-k selection |

### 3.5 Learned Extraction

After construction, we extract the optimal program. Standard greedy extraction can be suboptimal [12]. We enhance it with a learned value function that estimates the true cost of selecting each e-node.

## 4. Experimental Plan

### 4.1 Research Questions

1. **RQ1**: Can LEOPARD achieve comparable code quality to exhaustive equality saturation with significantly less memory?
2. **RQ2**: Does the lightweight supervised learning approach transfer across different programs and program types?
3. **RQ3**: How does LEOPARD compare to existing baselines (MCTS-guided ES, Aurora-style RL guidance, traditional pass ordering)?
4. **RQ4**: How gracefully does the system degrade when the learned scorer makes poor predictions?

### 4.2 Experimental Setup

**Benchmarks**: We use:
- **PolyBench/C** [14]: 30 numerical kernels representing various compute patterns
- **LLVM Test Suite**: Real-world C/C++ programs
- **Custom microbenchmarks**: Targeted tests for specific optimization patterns

**Baselines**:
1. **LLVM -O3**: Production compiler baseline
2. **Exhaustive ES**: Standard equality saturation with full saturation (or memory limit)
3. **MCTS-ES**: Hartmann et al.'s approach [8] adapted to our setting
4. **Aurora-style RL**: Simplified RL guidance (single-layer GNN) for comparison
5. **Random Selection**: Baseline to measure learning benefit

**Metrics**:
- **Code quality**: Instruction count, runtime speedup (geometric mean)
- **Resource usage**: Peak memory, compilation time, number of rule applications
- **Transfer**: Performance on unseen programs after training on a held-out set
- **Robustness**: Performance under adversarial test cases, degradation curve

### 4.3 Expected Results

We expect LEOPARD to:
1. Achieve within 20% of exhaustive ES code quality (i.e., 80% of speedup) while using 40-60% less memory
2. Generalize across program types, with <15% performance drop on unseen benchmarks
3. Outperform MCTS-ES on compilation time (no per-program search) while matching code quality
4. Outperform Aurora-style RL in inference speed (>10x faster rule selection)
5. Degrade gracefully when predictions are uncertain, never performing worse than random selection

## 5. Success Criteria

### 5.1 Preliminary Justification for Thresholds

The success criteria thresholds (80% of speedup, 40-60% memory reduction) are based on the following reasoning:
- Aurora [9] achieved competitive results with RL-guided ES but did not report memory savings
- MCTS-ES [8] trades off search time for quality, typically achieving 70-85% of optimal
- A 40-60% memory reduction would enable ES on larger programs that currently exhaust memory
- The 80% speedup threshold balances practicality (memory savings) with quality (near-optimal)

### 5.2 Confirming the Hypothesis

The hypothesis is confirmed if:
- LEOPARD achieves ≥80% of the speedup of exhaustive ES with ≤60% of the memory usage
- The learned scorer achieves >65% accuracy in predicting beneficial rules (above random baseline of ~30-40%)
- LEOPARD generalizes to unseen programs with <15% performance degradation
- Inference overhead is <5% of total compilation time

### 5.3 Refuting the Hypothesis

The hypothesis is refuted if:
- Rule selection patterns are not transferable across programs (scorer accuracy <55% on test set)
- Memory constraints fundamentally prevent finding high-quality solutions regardless of guidance
- The overhead of scoring (even with lightweight models) outweighs benefits for realistic program sizes
- The system fails to degrade gracefully when predictions are poor (worse than random selection)

## 6. Feasibility Analysis

### 6.1 Computational Resources

The proposed experiments require:
- **Training**: ~1000 program compilations for data collection (can be parallelized)
- **Model training**: Minutes on CPU (small model, no GPU required)
- **Evaluation**: ~1000 test compilations

Total estimated time: **4-6 hours** on a 2-core CPU system with 128GB RAM, well within our constraints.

### 6.2 Implementation Feasibility

We build on:
- **egg** [7]: Open-source equality saturation library in Rust
- **LLVM/MLIR**: For IR handling and baseline comparisons
- **PolyBench**: Readily available benchmark suite

The implementation involves:
1. Instrumenting egg to log rule applications and outcomes
2. Implementing the rule scorer and adaptive expansion loop
3. Collecting training data and training the lightweight model
4. Implementing graceful degradation mechanisms
5. Implementing learned extraction

All components are algorithmic/analytical and require no GPU training.

## 7. Contributions

This research makes the following contributions:

1. **LEOPARD**, a lightweight learned guidance system for equality saturation that uses supervised learning instead of expensive RL, making it practical for production compiler deployment.

2. **Direct comparison with Aurora** [9], demonstrating that lightweight supervised learning can achieve comparable results to RL-based guidance at a fraction of the inference cost.

3. **Empirical analysis** of rule selection patterns across diverse compiler IR programs, showing transferability with simple models.

4. **Graceful degradation mechanisms** that ensure the system maintains correctness and avoids performance cliffs when predictions are uncertain.

5. **Open-source implementation** enabling future research in lightweight learned compiler optimization.

## 8. Conclusion

Equality saturation offers a principled solution to the phase-ordering problem, but its practical adoption has been limited by memory explosion. While Aurora [9] demonstrated that learned guidance is feasible using reinforcement learning, LEOPARD takes a different approach: lightweight supervised learning that prioritizes inference speed for production deployment. By learning to guide e-graph construction with simple, fast models, LEOPARD aims to make equality saturation practical for real-world compilers while explicitly addressing failure modes through graceful degradation.

## References

[1] Cooper, Keith D., et al. "Compiler transformations for high-performance computing." ACM Computing Surveys 27.4 (1995): 457-472.

[2] Almagor, L., et al. "Finding effective compilation sequences." ACM SIGPLAN Notices 39.7 (2004): 231-239.

[3] Haj-Ali, Ameer, et al. "AutoPhase: Compiler phase-ordering for high level synthesis with deep reinforcement learning." IEEE Micro 40.1 (2020): 46-54.

[4] Liang, Youwei, et al. "Learning compiler pass orders using coreset and normalized value prediction." Proceedings of the 36th ACM International Conference on Supercomputing. 2023.

[5] Cummins, Chris, et al. "Large language models for compiler optimization." arXiv:2309.07062 (2023).

[6] Tate, Ross, et al. "Equality saturation: a new approach to optimization." ACM SIGPLAN Notices 44.1 (2009): 264-276.

[7] Willsey, Max, et al. "egg: Fast and extensible equality saturation." Proceedings of the ACM on Programming Languages 5.POPL (2021): 1-29.

[8] Hartmann, Jakob, Guoliang He, and Eiko Yoneki. "Optimizing tensor computation graphs with equality saturation and monte carlo tree search." Proceedings of the 29th ACM International Conference on Parallel Architectures and Compilation Techniques. 2024.

[9] Barbulescu, George-Octavian, et al. "Learned graph rewriting with equality saturation: A new paradigm in relational query rewrite and beyond." arXiv:2407.12794 (2024).

[10] Hagedorn, Bastian, et al. "Guided equality saturation." Proceedings of the ACM on Programming Languages 8.POPL (2024): 1727-1755.

[11] Goharshady, Amir Kafshdar, Chun Kit Lam, and Lionel Parreaux. "Fast and optimal extraction for sparse equality graphs." Proceedings of the ACM on Programming Languages 8.OOPSLA2 (2024): 1614-1642.

[12] Wang, Yichen, et al. "Equality saturation for tensor graph superoptimization." Proceedings of Machine Learning and Systems 3 (2021): 255-268.

[13] Yang, Yichen, et al. "SmartEMU: An efficient emulation framework for smart contract transactions." Proceedings of the 30th ACM SIGSOFT International Symposium on Software Testing and Analysis. 2021.

[14] Pouchet, Louis-Noël, and Tomofumi Yuki. "PolyBench/C: A benchmark suite for polyhedral compilation." URL: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1 (2016).
