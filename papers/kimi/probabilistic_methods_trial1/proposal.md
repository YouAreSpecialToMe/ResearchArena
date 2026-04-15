# Research Proposal: Decaying HyperLogLog
## Continuous-Time Cardinality Estimation with Exponential Aging

---

## 1. Introduction

### 1.1 Background and Motivation

Estimating the number of distinct elements (cardinality) in massive data streams is a fundamental problem in databases, network monitoring, and real-time analytics. The HyperLogLog (HLL) algorithm [1] has become the de facto standard, achieving a relative standard error of approximately 1.04/√m using only m registers of O(log log n) bits each. Its mergeability—enabling distributed computation by combining sketches—makes it indispensable for large-scale systems.

However, standard HLL and its variants treat all historical data equally. In many real-world applications, recent data is more relevant than older data. For instance:
- **Network monitoring**: Detecting current DDoS attacks requires emphasizing recent packet flows
- **Real-time analytics**: Trending topics on social media should weight recent posts higher
- **Fraud detection**: Recent transaction patterns are more indicative of ongoing fraud

Existing approaches to handle temporal relevance fall into two categories:
1. **Sliding window models** [2]: Only the most recent W elements are considered, with elements outside the window having zero weight. This creates abrupt transitions and requires complex data structures like exponential histograms or lists of future possible maxima.
2. **Batch recomputation**: Periodically rebuild the sketch from recent data, which is computationally expensive and creates latency.

Neither approach provides **continuous exponential decay**, where the contribution of each element decays smoothly and exponentially with its age. Exponential decay is natural for modeling information aging and is widely used in frequency estimation (e.g., Count-Min Sketch with exponential histograms [3]), but has not been successfully integrated into cardinality estimation.

### 1.2 Problem Statement

**The core problem**: How can we design a cardinality estimator that:
1. Supports continuous exponential decay of element contributions over time?
2. Maintains mergeability for distributed computation?
3. Provides provable accuracy guarantees similar to standard HLL?
4. Remains computationally efficient with small space overhead?

### 1.3 Key Insight and Hypothesis

Our key insight is that the stochastic averaging principle in HLL can be extended to incorporate time decay by treating register values as decaying over time. Instead of storing the maximum observed rank for each bucket, we store a "decaying maximum" that combines both the hash rank and the element's timestamp.

**Hypothesis**: A cardinality estimator based on decaying registers can achieve a relative standard error of O(1/√m) while providing exponentially decayed cardinality estimates, with mergeability preserved through synchronized decay parameters.

---

## 2. Proposed Approach

### 2.1 The Decaying HyperLogLog (D-HLL) Algorithm

#### 2.1.1 Core Idea

Standard HLL maintains m registers M[1..m], where each register stores the maximum observed rank (position of the leftmost 1-bit in the hashed value) for elements mapping to that bucket:
```
M[j] = max{ρ(h(x)) : x ∈ S, h(x) mod m = j}
```

In D-HLL, each register stores a decaying maximum that decreases over time:
```
M[j] = max{e^(-λ(t - t_x)) · ρ(h(x)) : x ∈ S, h(x) mod m = j}
```

where λ is the decay rate parameter and t_x is the arrival time of element x.

#### 2.1.2 Update Procedure

When processing element x at time t:
1. Compute j = h(x) mod m (bucket index)
2. Compute r = ρ(h'(x)) (rank of leftmost 1-bit in remaining hash bits)
3. Compute decayed value: v = r · e^(-λ(t - t)) = r (since t_x = t for new elements)
4. Update: M[j] = max(M[j] · e^(-λ(t - t_last)), v)

where t_last is the timestamp of the last update to register j.

#### 2.1.3 Decay Estimation

The key challenge is that we cannot apply decay continuously without O(1) operations per unit time. We use **lazy decay**: each register stores its last update time, and decay is applied only when the register is accessed:

```
M[j] ← M[j] · e^(-λ(t_current - T[j]))
T[j] ← t_current
```

where T[j] stores the timestamp of the last access to register j.

#### 2.1.4 Cardinality Estimation

The decayed cardinality estimate at time t is derived from the harmonic mean of the registers. For a stream with decay rate λ, the effective "active" elements have exponentially decreasing influence.

The estimator takes the form:
```
Ĉ_λ(t) = α_m · m² · (∑_{j=1}^m 2^(-M[j]))^(-1)
```

where α_m is the standard HLL bias correction constant.

### 2.2 Mergeability

A key requirement for distributed computation is that sketches from different sources can be merged. Two D-HLL sketches with the same decay rate λ can be merged by:

1. Synchronizing timestamps (using a common reference)
2. For each register j: M[j] = max(M₁[j] · e^(-λ(t - T₁[j])), M₂[j] · e^(-λ(t - T₂[j])))

This operation is associative and commutative, preserving the mergeability property.

### 2.3 Theoretical Analysis

#### 2.3.1 Unbiasedness

We conjecture that D-HLL provides an asymptotically unbiased estimate of the decayed cardinality:

**Conjecture 1**: For a stream where each element x at time t_x contributes weight e^(-λ(t - t_x)) to the cardinality, D-HLL provides an estimate Ĉ_λ such that E[Ĉ_λ] → C_λ as C_λ → ∞, where C_λ = ∑_{x ∈ S_t} e^(-λ(t - t_x)).

#### 2.3.2 Variance Bound

**Conjecture 2**: The relative standard error of D-HLL is bounded by O(1/√m), similar to standard HLL.

#### 2.3.3 Space Complexity

Each register requires:
- O(log log n) bits for the decayed rank value
- O(log n) bits for the timestamp

Total space: O(m · (log log n + log n)) = O(m log n) bits, a factor of O(log n / log log n) increase over standard HLL.

### 2.4 Optimizations

#### 2.4.1 Quantized Timestamps

To reduce space overhead, we can quantize timestamps to a coarser granularity (e.g., every 100ms) without significantly affecting accuracy.

#### 2.4.2 Batch Decay

Instead of per-register timestamps, use a global timestamp and apply decay in batches, trading some accuracy for reduced space.

---

## 3. Related Work

### 3.1 Standard Cardinality Estimation

**HyperLogLog (Flajolet et al., 2007)** [1]: The foundation of our work. Uses stochastic averaging with harmonic mean to achieve 1.04/√m standard error with near-optimal space complexity.

**UltraLogLog (Ertl, 2024)** [4]: Recent improvement achieving better space efficiency (6 bits per register) with comparable accuracy.

**Huffman-Bucket Sketch (2026)** [5]: Lossless compression of HLL using Huffman coding, achieving optimal space O(m + log n) bits.

### 3.2 Sliding Window Cardinality

**Sliding HyperLogLog (Chabchoub et al., 2010)** [2]: Extends HLL to sliding windows using a "List of Future Possible Maxima" (LPFM). Stores pairs of (timestamp, rank) for elements that could be maxima in future windows. Requires O(log n) pairs per bucket in worst case.

**Exponential Histograms (Datar et al., 2002)** [6]: General technique for sliding window aggregates, applied to various sketches but not specifically optimized for cardinality estimation.

### 3.3 Time-Decay Models

**ECM-Sketch (Papapetrou et al., 2012)** [3]: Combines Count-Min Sketch with exponential histograms for frequency estimation under sliding windows. Does not support cardinality estimation.

**Time-Decayed Correlated Aggregates (Cormode et al., 2009)** [7]: Studies the hardness of correlated aggregates with various decay functions, showing exponential and sliding window decay are particularly challenging for relative error guarantees.

**Learning-Augmented Moment Estimation on Time-Decay Models (2026)** [8]: Recent work on frequency estimation with time decay using learning-augmented approaches.

### 3.4 How Our Work Differs

| Aspect | Sliding HLL | D-HLL (Proposed) |
|--------|-------------|------------------|
| Decay model | Hard sliding window | Continuous exponential |
| Space per bucket | O(log n) pairs | O(1) registers + timestamps |
| Query flexibility | Fixed window W | Any decay rate λ (parameter) |
| Mergeability | Complex | Preserved |
| Smoothness | Abrupt cutoff | Gradual decay |

Our work is the first to provide continuous exponential decay for cardinality estimation while maintaining mergeability and near-constant space per bucket.

---

## 4. Experimental Plan

### 4.1 Implementation

We will implement D-HLL in Python with C++ extensions for performance-critical operations, targeting:
- Core algorithm: ~500 lines of C++
- Estimators and analysis: ~300 lines of Python
- Experimental framework: ~400 lines of Python

### 4.2 Datasets

**Synthetic Data**:
1. Uniform random streams (10^7 - 10^9 elements)
2. Zipfian-distributed streams (skew parameter 1.0 - 2.0)
3. Bursty streams (simulating DDoS attacks)

**Real-world Data**:
1. CAIDA anonymized packet traces [9]
2. Wikipedia page view logs [10]

### 4.3 Baselines

1. **Standard HLL**: Landmark model (all elements weighted equally)
2. **Sliding HLL**: Implementation of [2]
3. **Naive approach**: Rebuild HLL every T time units
4. **Sample-and-estimate**: Sample elements with probability decay and apply standard HLL

### 4.4 Metrics

1. **Relative Error**: |Ĉ - C| / C, where C is the exact decayed cardinality
2. **Throughput**: Elements processed per second
3. **Memory Usage**: Bits per register/bucket
4. **Merge Error**: Accuracy after distributed merge vs. centralized computation

### 4.5 Experimental Scenarios

**Experiment 1: Accuracy vs. Decay Rate**
- Vary λ from 0.001 to 0.1 (corresponding to half-lives from ~7 to ~700 time units)
- Measure relative error for different true cardinalities
- Expected: Error remains bounded by O(1/√m) across decay rates

**Experiment 2: Comparison with Sliding Window**
- Compare D-HLL (λ tuned for equivalent "memory") vs. Sliding HLL
- Metrics: accuracy, throughput, memory usage
- Expected: D-HLL provides smoother estimates with less space

**Experiment 3: Mergeability**
- Distribute stream across 10 nodes, merge sketches
- Compare merged estimate vs. centralized computation
- Expected: Merge introduces minimal additional error (<5%)

**Experiment 4: Adaptivity to Concept Drift**
- Sudden cardinality changes (simulating attacks ending)
- Measure convergence time to new true cardinality
- Expected: D-HLL adapts faster than sliding window due to exponential forgetting

### 4.6 Expected Results

1. **Accuracy**: Relative standard error < 2% with m = 2048 registers across decay rates
2. **Performance**: Throughput > 10^6 elements/second on single core
3. **Space**: < 2x overhead compared to standard HLL with timestamp quantization
4. **Scalability**: Linear speedup with distributed merging

---

## 5. Success Criteria

### 5.1 Confirmation of Hypothesis

The hypothesis is confirmed if:
- D-HLL provides unbiased estimates of decayed cardinality (bias < 5%)
- Relative standard error scales as O(1/√m), matching theoretical predictions
- Mergeability is preserved with minimal error increase (<10%)

### 5.2 Refutation Scenarios

The hypothesis may be refuted if:
- Exponential decay introduces unbounded bias in cardinality estimation
- The decay mechanism fundamentally breaks the stochastic averaging properties
- Space or time overhead makes the approach impractical (>10x overhead)

### 5.3 Success Metrics for Publication

For the work to be publishable at a top venue (e.g., SIGMOD, VLDB, PODS), we aim to demonstrate:
1. Novel algorithm with clear theoretical intuition
2. Provable guarantees (or strong empirical evidence) for accuracy
3. Significant practical advantages over existing approaches
4. Comprehensive experimental evaluation on real and synthetic data

---

## 6. Timeline and Feasibility

### 6.1 Resource Requirements

- **Compute**: CPU-only (2 cores, 128GB RAM as provided)
- **Time**: ~8 hours for all experiments
- **Storage**: < 10GB for datasets and results

The proposed experiments are entirely analytical and algorithmic—no GPU computation or neural network training is required. This aligns perfectly with the CPU-only constraint.

### 6.2 Risk Mitigation

**Risk 1**: Decay mechanism introduces unbounded bias
- *Mitigation*: We have fallback approaches using bounded decay functions

**Risk 2**: Timestamp storage overhead is prohibitive
- *Mitigation*: Quantization and batch decay optimizations

**Risk 3**: Lazy decay causes accuracy degradation
- *Mitigation*: Periodic background decay sweeps

---

## 7. Impact and Significance

### 7.1 Scientific Contribution

1. **First decaying cardinality estimator**: Fills the gap between landmark sketches (all history) and sliding window sketches (abrupt cutoff)
2. **Theoretical framework**: Analysis of decay in probabilistic distinct counting
3. **Practical algorithm**: Mergeable, efficient, and easy to implement

### 7.2 Applications

1. **Network security**: Real-time detection of ongoing attacks with time-decayed flow counting
2. **Stream processing**: Apache Flink/Spark integration for time-aware distinct counts
3. **Database systems**: Time-decayed statistics for query optimization
4. **IoT and sensor networks**: Distributed counting with aging of stale readings

### 7.3 Future Directions

1. Extension to other decay functions (polynomial, step)
2. Integration with differential privacy for private decayed counting
3. Hardware-efficient implementations (FPGA, P4)
4. Application to graph streams (decaying neighborhood sizes)

---

## 8. References

[1] P. Flajolet, É. Fusy, O. Gandouet, and F. Meunier. "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm." In Proceedings of the 2007 Conference on Analysis of Algorithms (AofA), 2007.

[2] F. Chabchoub, C. Fricker, and H. Mohamed. "Sliding HyperLogLog: Estimating cardinality in a data stream over a sliding window." In IEEE International Conference on Communications (ICC), 2010.

[3] O. Papapetrou, M. Garofalakis, and A. Deligiannakis. "Sketch-based querying of distributed sliding-window data streams." In Proceedings of the VLDB Endowment, 2015.

[4] O. Ertl. "UltraLogLog: A practical and more space-efficient alternative to HyperLogLog for approximate distinct counting." In Proceedings of the VLDB Endowment, 17(7), 2024.

[5] M. Hehir and G. Cormode. "Huffman-Bucket Sketch: A Simple O(m) Algorithm for Cardinality Estimation." arXiv:2603.10930, 2026.

[6] M. Datar, A. Gionis, P. Indyk, and R. Motwani. "Maintaining stream statistics over sliding windows." SIAM Journal on Computing, 31(6), 2002.

[7] G. Cormode, V. Shkapenyuk, D. Srivastava, and B. Xu. "Forward decay: A practical time decay model for streaming systems." In IEEE International Conference on Data Engineering (ICDE), 2009.

[8] Y. Jiang, J. Chen, and Y. Li. "Learning-Augmented Moment Estimation on Time-Decay Models." arXiv:2603.02488, 2026.

[9] CAIDA. "The CAIDA UCSD Anonymized Internet Traces." https://www.caida.org/catalog/datasets/passive_dataset/

[10] Wikimedia Foundation. "Wikimedia Pageview Statistics." https://dumps.wikimedia.org/other/pageviews/
