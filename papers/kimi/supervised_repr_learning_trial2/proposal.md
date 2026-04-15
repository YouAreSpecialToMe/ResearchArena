# Research Proposal: LASER-SCL — Learning-dynamics Aware Sample weighting with Expected-loss for Supervised Contrastive Learning

## 1. Introduction

### 1.1 Context and Problem Statement

Supervised contrastive learning (SupCon) [Khosla et al., 2020] has emerged as a powerful paradigm for representation learning, consistently outperforming cross-entropy-based training in generalization and robustness. However, **SupCon treats all training samples uniformly**, leading to suboptimal representations when data contains label noise or varying sample difficulty.

Recent work has addressed this through different approaches:
- **MW-Net** [Shu et al., 2019] and **CMW-Net** [Shu et al., 2022] learn adaptive weighting via meta-learning but **require clean validation data**, which is often unavailable.
- **UNICON** [Karim et al., 2022] uses uniform selection with Jensen-Shannon divergence and contrastive learning for noisy labels, but does not explicitly model sample difficulty dynamics.
- **Han et al. [2022]** uses a weight prediction network learned via meta-learning for dynamic early-exiting networks.
- **Zhuang et al. [2024]** applies curriculum to self-supervised contrastive learning using embedding similarity.

A critical gap remains: **no existing method provides adaptive, trajectory-based sample weighting for supervised contrastive learning without requiring meta-learning, clean meta-data, or multiple networks.**

### 1.2 Key Insight and Hypothesis

**Key Insight**: The difficulty of a sample is not static—it evolves as training progresses. By tracking how sample losses evolve over time (learning dynamics), we can distinguish learnable hard samples from noise without external supervision. Specifically:
- Samples with consistently decreasing loss trends are learnable (regardless of absolute loss)
- Samples with flat/increasing loss trends are likely noisy or memorized
- Using **predicted future loss** (based on trajectory) rather than current loss enables better curriculum scheduling

**Core Hypothesis**: A sample weighting mechanism that uses Expected Learning Progress (ELP)—the trend of loss decrease over a sliding window—can more accurately distinguish learnable samples from noise than static loss-based reweighting in supervised contrastive learning, without requiring meta-learning, clean validation data, or multiple networks.

### 1.3 Proposed Solution Overview

We propose **LASER-SCL**, a focused framework for adaptive sample weighting in supervised contrastive learning:

1. **Learning Dynamics Tracking**: For each sample, track loss values over a sliding window and fit a linear trend to compute Expected Learning Progress (ELP).

2. **Difficulty Estimation**: Compute difficulty using predicted future loss (based on the fitted trajectory) rather than current loss.

3. **Adaptive Curriculum**: Apply a curriculum schedule that gradually shifts weight from easy to hard samples based on the predicted difficulty.

**Scope Note**: This proposal focuses specifically on the core problem of sample weighting under label noise. Class imbalance and hard negative mining are secondary considerations that can be addressed in future extensions.

---

## 2. Related Work and Key Distinctions

### 2.1 Comparison with Han et al. [2022]

Han et al. [2022] proposed a weight prediction network (WPN) for dynamic early-exiting networks. Key differences:

| Aspect | Han et al. [2022] | LASER-SCL |
|--------|-------------------|-----------|
| **Core mechanism** | MLP-based weight prediction network | Trajectory-based ELP computation |
| **Learning paradigm** | Meta-learning with pseudo-updates | Direct optimization, no meta-learning |
| **Input to weighting** | Loss values from multiple exits | Loss trajectory trend over time |
| **Application** | Dynamic early-exiting networks | Supervised contrastive learning |
| **Weight interpretation** | Learned black-box weights | Interpretable difficulty scores |

**Fundamental distinction**: Han et al. learn *how to weight* via meta-learning; LASER-SCL computes *what to weight* via learning dynamics analysis. Their method requires optimizing an auxiliary network through pseudo-updates on meta-batches, while ours directly calculates weights from observed training dynamics.

### 2.2 Comparison with UNICON [Karim et al., 2022]

UNICON addresses label noise through:
- Jensen-Shannon divergence-based uniform selection
- Two-network co-teaching setup
- Contrastive learning for unlabeled samples

Key differences from LASER-SCL:
- UNICON uses **binary selection** (clean vs. noisy); LASER-SCL uses **continuous weighting** (fine-grained importance)
- UNICON requires **two networks** with cross-training; LASER-SCL uses **single network**
- UNICON's selection is based on **divergence from labels**; LASER-SCL is based on **learning trajectory**

### 2.3 Comparison with Zhuang et al. [2024]

Zhuang et al. applies curriculum to **self-supervised** contrastive learning using embedding similarity as difficulty measure.

LASER-SCL differs:
- Designed for **supervised** contrastive learning with label-aware positives/negatives
- Uses **learning dynamics** (loss trajectory) rather than embedding similarity
- Emphasizes **noise detection** through trajectory analysis

---

## 3. Proposed Approach

### 3.1 Preliminaries: Supervised Contrastive Learning

Given a batch of $N$ samples, the SupCon loss for sample $i$ is:

$$\mathcal{L}_i^{\text{SupCon}} = -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}$$

where $P(i)$ are positive samples (same class) and $A(i)$ are all other samples.

### 3.2 Expected Learning Progress (ELP)

For each sample $i$, maintain loss history $\mathcal{H}_i^{(t)} = \{l_i^{(t-W+1)}, ..., l_i^{(t)}\}$ over window $W$.

**Expected Learning Progress** measures the trend of loss decrease:

$$\text{ELP}_i^{(t)} = -\frac{\text{Cov}(\mathcal{H}_i^{(t)}, \mathbf{s})}{\text{Var}(\mathbf{s})}$$

where $\mathbf{s} = [1, 2, ..., W]$ is the step index vector. Positive ELP indicates decreasing loss (sample is being learned); near-zero ELP suggests noise or memorization.

### 3.3 Predicted Future Loss

Rather than using current loss for difficulty, we use **predicted future loss** based on the trajectory:

$$\hat{l}_i^{(t+\Delta)} = \bar{l}_i^{(t)} - \text{ELP}_i^{(t)} \cdot \Delta$$

where $\bar{l}_i^{(t)}$ is the mean loss over the window and $\Delta$ is the prediction horizon (typically $W/2$).

This formulation distinguishes:
- High current loss + positive ELP = learnable hard sample (weight should increase)
- High current loss + near-zero ELP = likely noise (weight should decrease)

### 3.4 Curriculum-Based Weighting

Compute sample weights using a sigmoid curriculum:

$$w_i^{(t)} = \frac{1}{1 + \exp(k \cdot (\hat{l}_i^{(t+\Delta)} - \mu^{(t)}))}$$

where $\mu^{(t)}$ is a dynamic threshold implementing the curriculum:

$$\mu^{(t)} = \mu_{\min} + (\mu_{\max} - \mu_{\min}) \cdot \left(\frac{t}{T}\right)^{\rho}$$

The curriculum gradually shifts from easy (low predicted loss) to hard (high predicted loss) samples.

### 3.5 Final Weighted Loss

$$\mathcal{L}_{\text{LASER-SCL}} = \sum_{i=1}^{N} w_i^{(t)} \mathcal{L}_i^{\text{SupCon}}$$

---

## 4. Preliminary Evidence for ELP

Prior work provides evidence that learning dynamics offer benefits over static loss-based reweighting:

1. **Memorization Effect** [Arpit et al., 2017]: DNNs learn clean patterns before memorizing noise. ELP captures this by identifying samples with flat/increasing loss trends (being memorized).

2. **Loss Trajectory Analysis** [Arazo et al., 2019]: The "Area Under the Margin" (AUM) statistic uses loss differences between samples to detect mislabeled data, demonstrating that trajectory information outperforms single-loss thresholds.

3. **Curriculum Learning Theory** [Bengio et al., 2009]: Gradual exposure to harder examples improves convergence. Using predicted future loss enables better scheduling than current loss.

We will empirically validate that ELP outperforms simple loss-based reweighting through synthetic experiments on CIFAR-10 with controlled noise patterns.

---

## 5. Experiments

### 5.1 Datasets

| Dataset | Size | Purpose | Compute Estimate |
|---------|------|---------|------------------|
| CIFAR-10 | 50K train | Primary evaluation with synthetic noise | ~30 min/run |
| CIFAR-100 | 50K train | Higher complexity validation | ~45 min/run |
| CIFAR-100-N | 50K train | Real-world human annotation noise | ~45 min/run |

**Removed**: Clothing1M (dataset availability issues, time constraints)

### 5.2 Baselines

**Simple reweighting baselines** (critical for validating unique value):
1. **SupCon + Loss Reweighting (LR)**: Weight $\propto 1 / \text{loss}$ (high loss = low weight)
2. **SupCon + Inverse Loss (IL)**: Weight $\propto \text{loss}$ (high loss = high weight)
3. **SupCon + Focal Reweighting**: Weight = $(1 - p_i)^\gamma$ based on prediction confidence

**State-of-the-art methods**:
4. **SupCon** [Khosla et al., 2020]: Vanilla supervised contrastive learning
5. **UNICON** [Karim et al., 2022]: For label noise comparison
6. **DivideMix** [Li et al., 2020]: Semi-supervised approach for noisy labels

**Note**: MW-Net excluded as it requires clean meta-data which is often unavailable.

### 5.3 Evaluation Protocol

1. **Representation Learning**: Train encoder with contrastive method
2. **Linear Evaluation**: Freeze encoder, train linear classifier
3. **Metrics**: Top-1 accuracy under varying noise rates

### 5.4 Ablation Studies

| Variant | Description | Purpose |
|---------|-------------|---------|
| LASER-SCL (Full) | Complete method with curriculum + ELP | Main result |
| LASER-SCL (No Curriculum) | Static threshold $\mu^{(t)} = \mu_{\text{fixed}}$ | Test curriculum value |
| LASER-SCL (Current Loss) | Use $\bar{l}_i$ instead of $\hat{l}_i^{(t+\Delta)}$ | Test ELP value |
| LASER-SCL (Fixed Window) | No curriculum, no prediction | Compare to simple baseline |

### 5.5 Synthetic Validation Experiments

To validate ELP before full implementation, we will conduct synthetic experiments:

1. **Controlled Noise Patterns**: Inject 20%, 40%, 60% symmetric label noise on CIFAR-10
2. **Trajectory Visualization**: Plot loss trajectories for clean vs. noisy samples
3. **ELP Correlation**: Measure correlation between ELP scores and true label correctness
4. **Comparison**: ELP-based selection vs. simple loss-based selection

### 5.6 Expected Results

**Hypothesis 1**: LASER-SCL outperforms SupCon+LR (loss-based reweighting) by at least 2% absolute accuracy under 40% label noise on CIFAR-100.

**Hypothesis 2**: Ablation shows ELP (predicted future loss) provides ≥1% improvement over using current loss alone.

**Hypothesis 3**: Ablation shows curriculum scheduling provides ≥1% improvement over static threshold.

**Hypothesis 4**: LASER-SCL achieves competitive or better performance than UNICON without requiring two networks.

### 5.7 Implementation Details

- **Architecture**: ResNet-18 for all experiments
- **Training**: 500 epochs, batch size 256
- **Optimizer**: SGD with cosine annealing, initial lr=0.5
- **Hyperparameters**: $\tau=0.1$, $W=10$, $\Delta=5$, $(\mu_{\min}, \mu_{\max}, \rho) = (0.3, 0.7, 2)$, $k=10$
- **Noise types**: Symmetric (uniform random), Asymmetric (class-dependent flips)

### 5.8 Computational Budget

| Experiment | Time/Run | Total (3 seeds) |
|------------|----------|-----------------|
| CIFAR-10 synthetic | 30 min | 4.5 hours |
| CIFAR-100 synthetic | 45 min | 2.25 hours |
| CIFAR-100-N | 45 min | 0.75 hours |
| Ablations | - | 1.5 hours |
| **Total** | | **~9 hours** |

Adjusted to fit within constraints by removing Clothing1M and ImageNet-100 experiments.

---

## 6. Success Criteria

### 6.1 Confirmation Criteria

The hypothesis is **confirmed** if:

1. **Primary**: LASER-SCL outperforms SupCon+LR by ≥2% under 40% label noise on CIFAR-100.

2. **Component Validation**: Each component (curriculum, ELP) contributes ≥0.5% when ablated.

3. **ELP Validation**: ELP-based selection achieves higher precision in identifying clean samples than loss-based selection (measured by AUC-ROC).

4. **Efficiency**: Computational overhead <20% compared to vanilla SupCon.

### 6.2 Refutation Criteria

The hypothesis is **refuted** if:

1. LASER-SCL performs worse than SupCon+LR across all noise conditions.
2. The learning dynamics tracking provides no improvement over using just current loss.
3. ELP scores show no correlation with true label correctness.

---

## 7. References

1. Prannay Khosla, et al. "Supervised Contrastive Learning." NeurIPS 2020.

2. Jun Shu, et al. "Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting." NeurIPS 2019.

3. Jun Shu, et al. "CMW-Net: Learning a Class-Aware Sample Weighting Mapping for Robust Deep Learning." TPAMI 2022.

4. Nazmul Karim, et al. "UNICON: Combating Label Noise Through Uniform Selection and Contrastive Learning." CVPR 2022.

5. Yizeng Han, et al. "Learning to Weight Samples for Dynamic Early-exiting Networks." ECCV 2022.

6. Jin Zhuang, et al. "Mining negative samples on contrastive learning via curricular weighting strategy." Information Sciences 2024.

7. Jingyi Cui, et al. "An Inclusive Theoretical Framework of Robust Supervised Contrastive Loss against Label Noise." arXiv:2501.01130, 2025.

8. Devansh Arpit, et al. "A Closer Look at Memorization in Deep Networks." ICML 2017.

9. Eric Arazo, et al. "Unsupervised Label Noise Modeling and Loss Correction." ICML 2019.

10. Yoshua Bengio, et al. "Curriculum Learning." ICML 2009.

11. Junnan Li, et al. "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." ICLR 2020.
