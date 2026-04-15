# PopBench: Population-Aware Hierarchical Adaptive Evaluation for Large Language Models

## 1. Introduction

### 1.1 Problem Statement

Evaluating large language models (LLMs) has become a computational bottleneck in the AI research pipeline. With thousands of models released annually—from scaling model families (Qwen, LLaMA, Gemma) to countless fine-tuned variants—exhaustive evaluation on benchmarks like MMLU (14K items), MMLU-Pro (12K items), and LiveBench is prohibitively expensive. Current adaptive evaluation methods reduce cost but still treat each model as an independent "test-taker," ignoring a crucial fact: **models form a highly structured population**.

The correlations between models are extraordinarily strong. Models within the same family (e.g., Qwen3-0.6B through Qwen3-235B) exhibit near-perfect correlation in error patterns; instruction-tuned variants maintain high similarity with their base models; and scaling laws predict performance relationships across parameter counts. Yet existing approaches—whether static benchmarks (MMLU), unidimensional adaptive testing (ATLAS), multi-modal IRT (M3IRT), or deep RL-based CAT (Deep-CAT)—evaluate each model in isolation, discarding this structure.

This leads to three critical inefficiencies:

1. **Redundant evaluation**: Related models are evaluated from scratch despite predictable performance relationships
2. **Cold-start inefficiency**: New models require dozens of items before achieving reliable estimates
3. **Missed information transfer**: Knowledge about model families and scaling relationships is not exploited during evaluation

### 1.2 Key Insight and Hypothesis

**Key Insight**: Models can be viewed as samples from a population distribution with learned covariance structure. By modeling this population hierarchy—where individual model capabilities are draws from family-level distributions, which are themselves draws from a global model population—we can enable **zero-shot evaluation** (estimating a model's full capabilities without observing any responses) and **information sharing** (responses from one model inform estimates for related models).

**Core Hypothesis**: A population-aware hierarchical Bayesian IRT framework can achieve 90%+ correlation with full-benchmark rankings using **<5% of items** for new models from known families, and can provide meaningful zero-shot predictions (correlation >0.7) before observing any responses.

### 1.3 Proposed Solution

We propose **PopBench**, a hierarchical adaptive evaluation framework that:

1. **Models the LLM population structure**: Uses a multi-level Bayesian model where individual models are grouped by family, with learned covariance matrices capturing relationships across dimensions (reasoning, knowledge, coding)
2. **Enables zero-shot evaluation**: For new models with metadata (family, size, architecture), provides initial capability estimates using only the population prior—no items required
3. **Implements adaptive information sharing**: During evaluation, responses from the target model are combined with information from related models through the hierarchical posterior
4. **Selects items for population information gain**: Chooses items that are maximally informative for distinguishing the model's position within the population, not just reducing individual uncertainty

## 2. Related Work

### 2.1 Static Benchmarks and Efficient Evaluation

Traditional LLM evaluation relies on static benchmarks (MMLU, HumanEval, GSM8K) with aggregate accuracy metrics. Recent work has explored efficiency through static subset selection (TinyBenchmarks, MetaBench), but these cannot adapt to individual models.

### 2.2 Adaptive Testing for LLM Evaluation

**ATLAS** (Li et al., 2025) applies unidimensional IRT with Fisher information-based item selection to reduce evaluation cost. While effective for scalar ability estimation, ATLAS cannot model multi-dimensional capabilities or share information across models.

**M3IRT** (Uebayashi et al., 2026) extends IRT to multi-modal settings, decomposing model ability into image-only, text-only, and cross-modal components. This is complementary to our work—M3IRT addresses multimodal evaluation while PopBench addresses population structure in model evaluation. M3IRT does not model cross-model correlations or enable zero-shot evaluation.

**Deep-CAT** (Li et al., 2025) uses deep reinforcement learning for multi-dimensional adaptive testing. Their approach learns a Q-network for item selection but treats each examinee independently. PopBench differs by using hierarchical Bayesian modeling rather than RL, explicitly modeling population structure to enable information sharing and zero-shot inference.

### 2.3 Multi-Dimensional IRT

**Segall (1996)** and **Reckase (2009)** established foundational theory for multi-dimensional adaptive testing (MAT). However, classic MAT assumes independent examinees and pre-calibrated item parameters. PopBench extends MAT with:
- Hierarchical population priors on model abilities
- Online joint estimation of item and model parameters
- Zero-shot capability prediction from model metadata

**PSN-IRT** (Zhou et al., 2025) uses pseudo-siamese networks for neural IRT parameter estimation but focuses on benchmark analysis rather than adaptive evaluation.

### 2.4 Hierarchical Bayesian IRT

**Huo et al. (2015)** proposed hierarchical multi-unidimensional IRT for integrative data analysis across multiple human studies. Their approach shares information across groups but requires pre-specified group structures and does not support adaptive testing or zero-shot prediction. PopBench adapts hierarchical IRT principles to the ML setting with:
- Learned (not pre-specified) population structure
- Adaptive item selection based on hierarchical posteriors
- Zero-shot prediction from model metadata using learned population parameters

### 2.5 How PopBench Differs

| Approach | Population Structure | Zero-Shot | Info Sharing | Adaptive | Multi-Dim |
|----------|---------------------|-----------|--------------|----------|-----------|
| ATLAS | ✗ | ✗ | ✗ | ✓ | ✗ |
| M3IRT | ✗ | ✗ | ✗ | ✗ | ✓ |
| Deep-CAT | ✗ | ✗ | ✗ | ✓ | ✓ |
| Segall (1996) | ✗ | ✗ | ✗ | ✓ | ✓ |
| Huo et al. (2015) | ✓ (pre-specified) | ✗ | ✓ | ✗ | ✓ |
| **PopBench** | **✓ (learned)** | **✓** | **✓** | **✓** | **✓** |

## 3. Proposed Approach

### 3.1 Overview

PopBench operates in two phases:

**Phase 1: Population Learning** (offline)
- Collect historical evaluation data from diverse models
- Fit hierarchical Bayesian MIRT to learn population structure
- Output: item parameters, population covariance matrices, family-level distributions

**Phase 2: Adaptive Evaluation** (online)
- For a new model, initialize beliefs using population prior + metadata
- Adaptively select items using hierarchical information gain
- Update hierarchical posterior after each response
- Terminate when uncertainty falls below threshold

### 3.2 Hierarchical Bayesian MIRT Model

We model the probability that model $m$ answers item $i$ correctly as:

$$P(y_{mi} = 1 | \boldsymbol{\theta}_m, \mathbf{a}_i, b_i) = \sigma\left(\sum_{d=1}^D a_{id} \theta_{md} - b_i\right)$$

**Level 1 (Individual Model)**: Each model $m$ has D-dimensional capability vector $\boldsymbol{\theta}_m$ drawn from its family distribution:
$$\boldsymbol{\theta}_m \sim \mathcal{N}(\boldsymbol{\mu}_{f(m)}, \boldsymbol{\Sigma}_{f(m)})$$

**Level 2 (Model Family)**: Each family $f$ has mean and covariance drawn from population hyperpriors:
$$\boldsymbol{\mu}_f \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$$
$$\boldsymbol{\Sigma}_f \sim \text{Inverse-Wishart}(\Psi, \nu)$$

**Level 3 (Global Population)**: Hyperparameters $\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0, \Psi, \nu$ are learned from data or given weak priors.

**Metadata-Conditioned Prior**: For zero-shot prediction, we condition the family distribution on model metadata $x_m$ (parameter count, architecture, training tokens):
$$\boldsymbol{\mu}_{f(m)} = g_\phi(x_m)$$
where $g_\phi$ is a learned neural network (e.g., small MLP predicting mean capability from scaling laws).

### 3.3 Zero-Shot Capability Prediction

For a new model with metadata $x_m$ but no observed responses:

1. Use learned metadata network: $\hat{\boldsymbol{\mu}} = g_\phi(x_m)$
2. Sample from family covariance: $\boldsymbol{\theta}_m^{(0)} \sim \mathcal{N}(\hat{\boldsymbol{\mu}}, \boldsymbol{\Sigma}_f)$
3. Predict full-benchmark performance via item parameters:
$$\hat{p}_i = \sigma(\mathbf{a}_i^\top \boldsymbol{\theta}_m^{(0)} - b_i)$$

This enables **zero-shot ranking** before evaluating any items.

### 3.4 Population-Aware Adaptive Item Selection

At each step $t$, we maintain a hierarchical posterior $p(\boldsymbol{\theta}_m, \boldsymbol{\mu}_f, \boldsymbol{\Sigma}_f | \mathcal{D}_t)$.

**Standard EIG** (for independent models):
$$i^*_{ind} = \arg\max_i \mathbb{E}_{y} [\text{KL}(p(\boldsymbol{\theta}_m | \mathcal{D}_t, y) \| p(\boldsymbol{\theta}_m | \mathcal{D}_t))]$$

**Population EIG** (PopBench):
$$i^*_{pop} = \arg\max_i \mathbb{E}_{y} [\text{KL}(p(\boldsymbol{\theta}_m, \boldsymbol{\mu}_f | \mathcal{D}_t, y) \| p(\boldsymbol{\theta}_m, \boldsymbol{\mu}_f | \mathcal{D}_t))]$$

The population EIG captures not just information about the individual model, but about the model's position in the population structure—enabling better discrimination between models with similar aggregate performance but different capability profiles.

**Efficient approximation**: We use variational inference to approximate the hierarchical posterior, computing EIG via reduction in total uncertainty (individual + family + population levels).

### 3.5 Cross-Model Information Sharing

When evaluating multiple models from the same family (e.g., scaling sweep), PopBench jointly infers the family parameters:

$$p(\boldsymbol{\mu}_f, \boldsymbol{\Sigma}_f | \{\mathcal{D}_m\}_{m \in f}) \propto p(\boldsymbol{\mu}_f, \boldsymbol{\Sigma}_f) \prod_{m \in f} \int p(\mathcal{D}_m | \boldsymbol{\theta}_m) p(\boldsymbol{\theta}_m | \boldsymbol{\mu}_f, \boldsymbol{\Sigma}_f) d\boldsymbol{\theta}_m$$

This enables:
- **Warm start**: Smaller models inform priors for larger models in the same family
- **Joint termination**: Stop when family-level uncertainty is sufficiently reduced
- **Outlier detection**: Identify models that deviate from expected family behavior

### 3.6 Uncertainty-Aware Stopping

We propose three stopping criteria:

1. **Individual precision**: $\text{Tr}(\text{Cov}(\boldsymbol{\theta}_m)) < \epsilon_{ind}$
2. **Population precision**: Family covariance determinant $|\boldsymbol{\Sigma}_f| < \epsilon_{pop}$
3. **Decision certainty**: For ranking tasks, stop when pairwise comparison confidence exceeds threshold

## 4. Experiments

### 4.1 Research Questions

1. **Zero-shot accuracy**: How well can PopBench predict model rankings without observing any responses?
2. **Adaptive efficiency**: How many items are needed to achieve reliable estimates compared to independent-model baselines?
3. **Information sharing**: Does evaluating multiple related models jointly improve efficiency?
4. **Scaling law alignment**: Do learned population parameters capture known scaling relationships?

### 4.2 Benchmark Datasets

- **MMLU** (Hendrycks et al., 2020): 14K items across 57 subjects
- **MMLU-Pro** (Wang et al., 2024): 12K professionally validated items
- **LiveBench** (White et al., 2024): 2,400 items with contamination controls
- **HumanEval** (Chen et al., 2021): 164 coding problems

### 4.3 Models Evaluated

**Training population**: 100+ models from public leaderboards spanning:
- Scaling families: Qwen2/3 (8 sizes), LLaMA-2/3 (4 sizes), Gemma (3 sizes), Mistral (4 sizes)
- Instruction variants: Base, chat, RLHF versions
- Architecture diversity: Dense, MoE, RNN hybrids

**Test models**: 20 held-out models including:
- New sizes in known families (e.g., Qwen3-14B if trained on others)
- Fine-tuned variants (e.g., LLaMA-3-8B-Instruct vs base)
- Cross-family models for generalization testing

### 4.4 Baselines

1. **Random sampling**: Random item selection
2. **ATLAS** (Li et al., 2025): Unidimensional CAT with Fisher information
3. **M3IRT** (Uebayashi et al., 2026): Multi-modal IRT with D-optimality (adapted to text-only)
4. **Deep-CAT** (Li et al., 2025): Deep RL-based selection
5. **Independent MAT**: Multi-dimensional IRT without population structure (Segall-style)
6. **Full evaluation**: All items (gold standard)

### 4.5 Metrics

- **Zero-shot correlation**: Spearman $\rho$ between predicted and true rankings (no items observed)
- **Items to target precision**: Number of items to reach MAE < 0.05 on full-benchmark estimate
- **Rank correlation**: Kendall's $\tau$ with full-benchmark ranking
- **Calibration**: Reliability of uncertainty estimates
- **Family prediction error**: RMSE of predicted vs actual performance within scaling families

### 4.6 Expected Results

**Hypothesis 1 (Zero-shot)**: PopBench achieves Spearman $\rho > 0.75$ with ground-truth rankings without observing any model responses, using only metadata and learned population structure. This outperforms naive baselines (random guessing: $\rho \approx 0$; scaling law regression: $\rho \approx 0.6$).

**Hypothesis 2 (Adaptive efficiency)**: PopBench achieves MAE < 0.05 using 60-70% fewer items than independent-model MAT and 80% fewer than ATLAS. Target: <10% of benchmark items on average.

**Hypothesis 3 (Information sharing)**: Joint evaluation of 5 models from the same family requires 40% fewer total items than independent evaluation, as family parameters are shared.

**Hypothesis 4 (Scaling law recovery)**: Learned metadata network $g_\phi$ recovers power-law scaling relationships with $R^2 > 0.9$ for compute-optimal models.

## 5. Success Criteria

The project succeeds if:

1. **Zero-shot prediction**: Spearman correlation > 0.7 with ground-truth rankings without any observed responses
2. **Adaptive efficiency**: <10% of items needed for MAE < 0.05 on held-out models from known families
3. **Calibration**: 90% credible intervals contain true performance 85-95% of the time
4. **Information sharing**: Joint evaluation of 4+ related models shows >30% item reduction vs independent evaluation

Failure modes to monitor:
- Overfitting to popular model families (LLaMA, Qwen) with poor generalization to novel architectures
- Underestimating uncertainty for outlier models that deviate from population patterns
- Item parameter drift as model capabilities evolve beyond historical training distribution

## 6. Impact and Significance

### 6.1 Scientific Contribution

PopBench introduces a paradigm shift from **independent model evaluation** to **population-aware evaluation**. This challenges the assumption that each model must be evaluated as a unique entity, instead leveraging the rich structure in the model ecosystem. The framework bridges hierarchical Bayesian modeling, psychometrics, and ML evaluation.

### 6.2 Practical Impact

- **Dramatic cost reduction**: 90%+ reduction in evaluation compute for model developers
- **Instant leaderboard updates**: Zero-shot estimates for new model releases within minutes
- **Rapid scaling analysis**: Evaluate entire model families jointly, understanding capability emergence across scales
- **Quality assurance**: Identify anomalous models that deviate from expected family behavior

### 6.3 Connection to AI Safety

Population-aware evaluation enables monitoring of capability trends across the model ecosystem, detecting unexpected jumps in specific capabilities that might indicate dangerous emergent behaviors.

## 7. Timeline and Feasibility

**Week 1 (8 hours)**:
- Collect historical evaluation data from public leaderboards (HuggingFace, OpenCompass)
- Implement hierarchical Bayesian MIRT with variational inference
- Train population model on historical data

**Week 2 (8 hours)**:
- Implement metadata-conditioned prior network
- Implement population-aware adaptive item selection
- Implement baselines (ATLAS, independent MAT)

**Week 3 (8 hours)**:
- Run experiments on 3-4 benchmarks with 20+ test models
- Analyze zero-shot prediction accuracy
- Measure information sharing gains

**Week 4 (8 hours)**:
- Paper writing and refinement
- Visualization of population structure and scaling laws
- Prepare reproducibility package

**Compute Requirements**:
- Population model training: ~3 hours on A6000 (48GB)
- Evaluation: Uses API calls or local models (7B-13B)
- Well within 8-hour total budget

## References

1. Uebayashi, S., Masui, K., Atarashi, K., et al. (2026). Evaluating Cross-Modal Reasoning Ability with Multimodal Item Response Theory. arXiv:2603.02663.
2. Li, J., et al. (2025). Deep Computerized Adaptive Testing. arXiv:2502.19275.
3. Li, P., Tang, X., Chen, S., et al. (2025). ATLAS: Adaptive Testing for LLM Evaluation. arXiv:2511.04689.
4. Segall, D.O. (1996). Multidimensional Adaptive Testing. Psychometrika, 61(2), 331-354.
5. Reckase, M.D. (2009). Multidimensional Item Response Theory. Springer.
6. Zhou, Y., et al. (2025). Lost in Benchmarks? Rethinking LLM Benchmarking with IRT. arXiv:2505.15055.
7. Huo, Y., et al. (2015). A Hierarchical Multi-Unidimensional IRT Approach for Analyzing Sparse, Multi-Group Data. Psychometrika, 80(3), 765-783.
8. Hendrycks, D., et al. (2020). Measuring Massive Multitask Language Understanding. arXiv:2009.03300.
9. White, C., et al. (2024). LiveBench: A Challenging, Contamination-Free LLM Benchmark. arXiv:2406.19314.
10. Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code. arXiv:2107.03374.
