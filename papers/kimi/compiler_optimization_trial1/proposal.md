# Research Proposal: Extending Learned Static Prediction to Data Layout Optimization

## Title
**From Branches to Bytes: Extending Learned Static Prediction to Data Layout Optimization**

## 1. Introduction

### 1.1 Context and Problem Statement

Data layout optimizations, such as structure splitting and field reordering, are critical for improving cache performance in modern applications. By reorganizing how data structures are laid out in memory, compilers can significantly reduce cache misses and improve spatial locality. However, identifying profitable data layout transformations is challenging because:

1. **Access patterns vary by code region**: The same data structure may be accessed differently in various parts of a program, making global layout decisions suboptimal.
2. **Static analysis limitations**: Precise identification of hot fields requires knowing runtime behavior, which static analysis cannot reliably predict.
3. **Profile overhead**: Profile-guided optimization (PGO) can identify profitable layouts but requires expensive profiling runs, instrumentation overhead, and representative input data.

Current production compilers like GCC and LLVM have largely abandoned automatic structure layout optimization due to these challenges. GCC's `-fipa-struct-reorg` was removed in version 4.8 because it "did not always work correctly" and lacked LTO support. This leaves a significant performance opportunity untapped.

### 1.2 Key Insight and Positioning

Rotem and Cummins (2021) demonstrated that lightweight machine learning models (XGBoost) trained on static LLVM IR features can effectively predict branch probabilities, achieving profile-guided optimization benefits without profiling overhead. Their work proves that "profile-guided optimization without profiles" is viable for control-flow decisions.

Our key insight is that this proven methodology can be extended to data layout optimization. While branch prediction and data layout prediction differ significantly in complexity (Section 2.3), the same fundamental approach—training lightweight ML models on static features to predict dynamic behavior—can guide structure layout decisions.

**Positioning**: This work directly extends Rotem & Cummins' methodology to a new domain (data layout), addressing the unique challenges that make data layout prediction harder than branch prediction. We do not claim to invent "learned static prediction"—rather, we apply and adapt this proven technique to data layout, where it has not been previously explored.

### 1.3 Why Data Layout is Harder Than Branch Prediction

Data layout optimization presents several challenges that make it inherently more difficult than branch prediction:

1. **Higher-dimensional output space**: Branch prediction is a binary or probability estimation task (taken vs. not taken). Data layout requires predicting which fields to split, their optimal ordering, and the profitability of transformation—combinatorially more complex.

2. **Context-dependent optimality**: A branch's probability is a property of the branch itself. A data layout's optimality depends on the surrounding code region, aliasing relationships, and loop structure—requiring richer contextual features.

3. **Legality constraints**: Branch prediction errors affect performance but not correctness. Data layout transformations must preserve correctness through alias analysis and type safety, adding complexity to the prediction task.

4. **Delayed feedback**: Branch mispredictions provide immediate feedback. Data layout benefits depend on cache behavior, which is harder to measure and attribute to specific transformations.

5. **Transformation synthesis**: Predicting branch probabilities only requires annotating metadata. Data layout requires synthesizing actual IR transformations (splitting structures, inserting copy code), which is more invasive.

### 1.4 Hypothesis

**Hypothesis**: A learned static predictor, trained on static program features from LLVM IR, can identify profitable structure layout transformations with accuracy within 20% of profile-guided approaches, while eliminating profiling overhead entirely.

## 2. Related Work

### 2.1 Learned Static Prediction (Foundational)

Rotem and Cummins (2021) demonstrated "Profile Guided Optimization without Profiles" using XGBoost models trained on LLVM IR features to predict branch probabilities. Their approach achieves 75% prediction accuracy for branch weight classes and a 1.6% geomean speedup over non-PGO compilation. This work establishes the viability of learned static prediction and provides our methodological foundation.

**Our extension**: We apply their proven XGBoost-on-LLVM-IR methodology to data layout optimization, extracting different features and predicting layout decisions rather than branch probabilities.

### 2.2 Static Heuristics for Branch Prediction

Ball and Larus (1993) introduced influential static heuristics for branch prediction (loop header, pointer comparison, etc.), achieving ~80% accuracy. Wu and Larus (1994) extended this to static branch frequency estimation. Calder et al. (1997) improved prediction using decision trees trained on program features.

**Distinction**: These works target control-flow prediction. We target data layout decisions, which require different features and prediction targets.

### 2.3 Data Layout Optimization with Profiling

Chilimbi et al. (1999) introduced cache-conscious structure definition and layout, using profiling to identify hot/cold fields. Zhong et al. (2004) used whole-program reference affinity from profile data to guide array regrouping and structure splitting.

**Distinction**: These approaches require runtime profiling. We eliminate profiling through static prediction, similar to how Rotem & Cummins eliminated profiling for branch prediction.

### 2.4 Static Data Layout Optimization

Rohwedder (2025) introduced RebaseDL, a region-based structure splitting and field reordering system using static data reuse analysis. RebaseDL identifies code regions with poor cache utilization and applies transformations without profiling, achieving up to 1.34× speedups in SPEC CPU benchmarks.

**Key distinction from RebaseDL**: 
- RebaseDL uses **static analysis** (cache utilization ratios, data reuse analysis) with manually-tuned thresholds
- Our approach uses **machine learning** to automatically learn prediction rules from training data
- ML can potentially discover patterns that static analysis misses, combine features in non-obvious ways, and generalize across program types
- We view our approach as complementary: ML-based prediction may identify opportunities that RebaseDL's conservative analysis misses

### 2.5 Machine Learning in Compilers

MLGO (Trofin et al., 2021) uses reinforcement learning for inlining and register allocation. Cummins et al. (2021) developed CompilerGym for ML-based compiler optimization research. LOOPer (Merouani et al., 2024) applies deep learning to polyhedral loop optimization.

**Distinction**: These works target different optimization problems (inlining, register allocation, loop optimization). No prior work applies learned static prediction specifically to data layout optimization.

## 3. Proposed Approach

### 3.1 Overview

We propose **LayoutLearner**, a system that extends Rotem & Cummins' methodology to data layout optimization:

1. **Feature extraction**: Extract static features from LLVM IR representing structure access patterns (different features than branch prediction)
2. **Model training**: Train lightweight ML models (XGBoost) to predict profitable layout transformations
3. **Transformation synthesis**: Generate structure splitting and field reordering based on model predictions
4. **Validation**: Ensure correctness through alias analysis

### 3.2 Feature Engineering

Unlike branch prediction (which uses control-flow features), data layout requires memory-access-focused features:

**Structural Features**:
- Number of fields in the structure
- Size of each field and total structure size
- Field type categories (pointer, primitive, nested struct)
- Alignment requirements

**Access Pattern Features**:
- Loop nesting depth at access sites
- Number of access sites per field
- Control flow dominance relationships between accesses
- Pointer arithmetic patterns (sequential vs. random access)
- Field access co-occurrence within loops

**Context Features**:
- Function call frequency estimates (using static heuristics like Wu & Larus)
- Memory allocation patterns (array vs. single object)
- Alias information (may-alias vs. must-alias)
- Loop trip count estimates (where statically available)

### 3.3 Model Architecture

Following Rotem & Cummins' proven approach:

- **Primary Model**: Gradient Boosted Decision Trees (XGBoost)
  - Fast inference (~microseconds per prediction)
  - Interpretable feature importance
  - Robust to feature scaling issues
  - Can be compiled into C++ for LLVM integration
  
- **Prediction Target**: Instead of branch probabilities, we predict:
  - Binary: Is splitting/reordering profitable for this region?
  - Ranking: Which fields should be grouped together?
  - Expected cache utilization improvement

### 3.4 Transformation Synthesis

The model predicts:
1. **Split candidates**: Which fields should be separated into distinct structures
2. **Reorder candidates**: Optimal field ordering based on access affinity
3. **Transformation profitability**: Expected cache miss reduction

Transformations are synthesized as LLVM IR transformations and validated for correctness using alias analysis. We adopt RebaseDL's region-based approach (local transformations with copying) to simplify legality checking.

### 3.5 Training Methodology

**Dataset Construction**:
1. Compile benchmarks (PolyBench, small SPEC CPU subsets) with different structure layouts
2. Measure actual cache performance using hardware counters
3. Label optimal transformations as ground truth
4. Extract static features from unoptimized IR

**Model Training**:
- Cross-validation across benchmark suites
- Hyperparameter tuning for compilation-time constraints
- Feature selection to minimize extraction overhead

## 4. Experimental Plan

### 4.1 Research Questions

1. **RQ1 (Accuracy)**: How accurately can static features predict profitable layout transformations compared to profile-guided approaches?
2. **RQ2 (Performance)**: What speedups does LayoutLearner achieve compared to baseline -O3 and RebaseDL?
3. **RQ3 (Overhead)**: What is the compilation-time overhead of LayoutLearner inference?
4. **RQ4 (Comparison to RebaseDL)**: Does ML-based prediction identify opportunities missed by static analysis?

### 4.2 Benchmarks

Given 8-hour time constraint and CPU-only resources:
- **PolyBench**: Kernel-focused benchmarks amenable to layout optimization
- **Small SPEC CPU 2017 subset**: Selected benchmarks with heavy struct usage (e.g., 505.mcf, 554.roms)
- **Synthetic benchmarks**: Hand-crafted microbenchmarks with known optimal layouts

### 4.3 Baselines

1. **LLVM -O3**: Production compiler without layout optimization
2. **RebaseDL**: Static analysis-based approach (reproduced or using published results)
3. **Profile-guided layout**: Layout optimization using perfect profile information (upper bound)
4. **Rotem & Cummins-style heuristics**: Static heuristics adapted for layout (e.g., "fields in hot loops are hot")

### 4.4 Metrics

- **Prediction Accuracy**: Precision/recall of profitable transformations
- **Performance Speedup**: Execution time improvement over -O3
- **Cache Miss Reduction**: L1/L2 cache miss rate improvement
- **Compilation Time**: Overhead of feature extraction and model inference
- **Comparison to RebaseDL**: Number of additional opportunities identified

### 4.5 Expected Results

We expect LayoutLearner to:
- Achieve 60-80% of the performance gains of profile-guided layout optimization (vs. 20% target in hypothesis, acknowledging the harder problem)
- Identify some optimization opportunities missed by RebaseDL's static analysis
- Introduce less than 5% compilation-time overhead

## 5. Success Criteria

### 5.1 Confirmation

The hypothesis is confirmed if:
- LayoutLearner achieves within 20% of profile-guided layout optimization performance
- Compilation-time overhead remains below 5%
- The approach identifies profitable transformations on at least 40% of benchmark programs

### 5.2 Refutation

The hypothesis is refuted if:
- Static features cannot reliably predict profitable layouts (prediction accuracy < 50%)
- Performance gains are statistically insignificant (< 2% geomean speedup)
- Compilation-time overhead exceeds acceptable thresholds (> 10%)

## 6. Timeline and Milestones

Given the 8-hour total time constraint, we adopt a **simulation-based approach** rather than full LLVM pass implementation:

- **Hour 1-2**: Implement feature extraction from LLVM IR (Python-based, processing bitcode)
- **Hour 3-4**: Generate training dataset using synthetic benchmarks with known optimal layouts
- **Hour 5-6**: Train XGBoost models and evaluate prediction accuracy
- **Hour 7**: Compare against static heuristics and analyze feature importance
- **Hour 8**: Document results and validate hypothesis

**Note**: We do not implement full transformation synthesis in LLVM within 8 hours. Instead, we demonstrate that learned models can predict profitable layouts, leaving full integration for future work.

## 7. References

[1] Nadav Rotem and Chris Cummins. "Profile Guided Optimization without Profiles: A Machine Learning Approach." arXiv:2112.14679, 2021.

[2] Thomas Ball and James R. Larus. "Branch Prediction for Free." PLDI 1993.

[3] Youfeng Wu and James R. Larus. "Static Branch Frequency and Program Profile Analysis." MICRO 1994.

[4] Brad Calder, Dirk Grunwald, Michael Jones, Donald Lindsay, James Martin, Michael Mozer, and Benjamin Zorn. "Evidence-based Static Branch Prediction using Machine Learning." ACM TOPLAS 19(1), 1997.

[5] Trishul M. Chilimbi, Bob Davidson, and James R. Larus. "Cache-conscious Structure Definition." PLDI 1999.

[6] Trishul M. Chilimbi, Mark D. Hill, and James R. Larus. "Cache-conscious Structure Layout." PLDI 1999.

[7] Yutao Zhong, Maksim Orlovich, Xipeng Shen, and Chen Ding. "Array Regrouping and Structure Splitting using Whole-program Reference Affinity." PLDI 2004.

[8] Caio L. Rohwedder. "Region-Based Data Layout Transformations." PhD Thesis, University of Alberta, 2025.

[9] Caio S. Rohwedder and João P. L. De Carvalho and J. Nelson Amaral. "Region-Based Data Layout via Data Reuse Analysis." CC 2024.

[10] Mircea Trofin et al. "MLGO: a Machine Learning Guided Compiler Optimizations Framework." arXiv:2101.04808, 2021.

[11] Chris Cummins et al. "CompilerGym: Robust, Performant Compiler Optimization Environments for AI Research." arXiv:2109.08267, 2021.

[12] Massinissa Merouani et al. "LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers." arXiv:2405.17712, 2024.
