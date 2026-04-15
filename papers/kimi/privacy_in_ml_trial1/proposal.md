# Post-Hoc Compression-Aware Differential Privacy: Optimizing DP Training for Deployed Compressed Models

## 1. Introduction

### 1.1 Background and Problem Statement

Differential Privacy (DP) has emerged as the gold standard for protecting training data privacy in machine learning, with DP-SGD being the most widely adopted algorithm for training private deep neural networks. However, DP-SGD suffers from a fundamental privacy-utility trade-off: stronger privacy guarantees (lower ε) typically result in significant accuracy degradation. This challenge is particularly acute when deploying models on resource-constrained devices, where model compression (via pruning, quantization, or knowledge distillation) is essential.

Recent work by Li et al. (CompLeak, 2025) revealed a critical but overlooked phenomenon: model compression operations—including pruning, quantization, and weight clustering—exacerbate privacy leakage when adversaries have access to multiple compressed versions of a model. Their experiments demonstrate that compression affects member and non-member samples differently, creating new vulnerabilities that compound existing privacy risks.

This creates a paradox: DP training reduces model capacity through noise injection, then compression further reduces capacity, yet the combination hasn't been systematically studied for the common deployment scenario where **compression is applied after training** (post-hoc compression). Practitioners currently train with DP-SGD assuming the model will be deployed as-is, then compress it as a separate post-processing step—potentially undermining privacy guarantees in ways that are not accounted for in the privacy budget.

### 1.2 Key Insight and Hypothesis

**Key Insight:** When post-hoc compression is applied after DP training, the interaction between DP noise and compression-induced information loss creates a complex privacy-utility landscape. Current DP training is compression-oblivious, allocating privacy budget uniformly across all parameters regardless of whether they will survive compression. This is suboptimal: parameters that will be pruned need not receive the same privacy protection investment as those that will be retained.

**Core Hypothesis:** By incorporating knowledge of the target **post-hoc compression strategy** (type, degree, and timing) into the DP-SGD training process, we can achieve better privacy-utility trade-offs compared to both (1) standard DP-SGD followed by post-hoc compression, and (2) training-time pruning approaches that require public data for parameter selection. Specifically, we hypothesize that:

1. DP-SGD with compression-aware per-parameter clipping and noise calibration can produce models that, after post-hoc compression, achieve higher accuracy under the same privacy budget compared to compression-agnostic DP training.

2. Unlike pre-pruning approaches (Adamczewski et al., 2023) that require public data to select which parameters to retain, our method works without any public data by using estimated survival probabilities.

3. The specific formulation of per-parameter clipping weights based on parameter survival probability (not just binary pruning masks) enables finer-grained privacy budget allocation than existing importance-based masking approaches (AdaDPIGU, Zhang & Xie, 2025).

### 1.3 Research Questions

1. How does the combination of DP training and subsequent **post-hoc model compression** affect membership inference vulnerability compared to either technique alone?

2. Can we formulate a compression-aware DP-SGD algorithm that optimizes privacy parameters based on anticipated post-hoc compression settings, without requiring public data for parameter selection?

3. How does per-parameter clipping based on continuous survival probabilities compare to binary importance-based masking approaches (AdaDPIGU) and pre-pruning methods (Adamczewski et al., 2023)?

4. What is the theoretical relationship between DP guarantees and post-hoc compression operations, and can we derive composition theorems that account for both?

## 2. Proposed Approach

### 2.1 Method Overview

We propose **Post-Hoc Compression-Aware DP-SGD (PHCA-DP-SGD)**, a training algorithm that jointly optimizes for differential privacy and **post-hoc model compression**. The key innovation is treating post-hoc compression not as an external post-processing step but as an anticipated operation that should inform how privacy budget is allocated during training.

**Standard DP-SGD Pipeline:**
```
Private Data → DP-SGD Training → Model → Compression → Deployed Model
      ↑___________________________________________↓
                    (Privacy budget computed here, 
                     ignoring compression effects)
```

**Adamczewski et al. (2023) Pre-Pruning Pipeline:**
```
Public Data → Select Parameters → Pre-Prune → DP-SGD on Subnetwork → Deployed Model
                                     ↑
                    (Requires public data for parameter selection)
```

**PHCA-DP-SGD Pipeline:**
```
Private Data → PHCA-DP-SGD Training (compression-aware) → Compression → Deployed Model
      ↑__________________________________________________________↓
          (Privacy budget allocated based on survival probability;
           NO public data required)
```

### 2.2 Technical Details

#### 2.2.1 Problem Formulation

Let:
- $\mathcal{D}$ be the private training dataset
- $f(\cdot; \theta)$ be a neural network with parameters $\theta \in \mathbb{R}^d$
- $C(\cdot; \kappa)$ be a post-hoc compression operator with configuration $\kappa$ (e.g., sparsity ratio for pruning, bit-width for quantization) applied **after training**
- $\mathcal{M}(\cdot; \epsilon, \delta)$ be $(\epsilon, \delta)$-DP mechanism

**Standard approach** optimizes:
$$\min_{\theta} \mathbb{E}_{(x,y) \sim \mathcal{D}}[\mathcal{L}(f(x; \theta), y)] \quad \text{s.t.} \quad \theta \sim \mathcal{M}(\cdot; \epsilon, \delta)$$

then applies:
$$\theta_{\text{deployed}} = C(\theta; \kappa)$$

**Adamczewski et al. (2023) pre-pruning** first uses public data to select a subnetwork $S \subseteq [d]$, then trains only $\theta_S$ with DP-SGD.

**Our approach** optimizes:
$$\min_{\theta} \mathbb{E}_{(x,y) \sim \mathcal{D}}[\mathcal{L}(f(x; C(\theta; \kappa)), y)] \quad \text{s.t.} \quad \theta \sim \mathcal{M}(\cdot; \epsilon', \delta)$$

where $\epsilon' \leq \epsilon$ (we aim for same or better privacy with better utility), and critically, **no public data is required** for parameter selection.

#### 2.2.2 Compression-Aware Per-Parameter Gradient Clipping

Standard DP-SGD clips per-sample gradients uniformly:
$$\bar{g}_i = g_i / \max\left(1, \frac{\|g_i\|_2}{C}\right)$$

AdaDPIGU (Zhang & Xie, 2025) uses binary importance-based masking based on public data pretraining.

**PHCA-DP-SGD uses continuous per-parameter clipping weights** based on estimated survival probability under compression:

$$\bar{g}_i^{(j)} = g_i^{(j)} / \max\left(1, \frac{|g_i^{(j)}|}{C \cdot w_j}\right)$$

where $w_j \in (0, 1]$ is a per-parameter weight:

$$w_j = f_{\text{survival}}(p_j(\kappa))$$

For magnitude pruning, $p_j(\kappa)$ is the probability that parameter $j$ survives given compression ratio $\kappa$. Unlike binary masking (AdaDPIGU), we use **continuous weights** based on survival probability:

$$w_j = p_j(\kappa) + \alpha(1 - p_j(\kappa))$$

where $\alpha \in [0, 1]$ controls how aggressively we discount parameters likely to be pruned. When $\alpha = 0$, parameters with $p_j \approx 0$ receive minimal clipping budget; when $\alpha = 1$, we recover standard uniform clipping.

**Key difference from AdaDPIGU:** AdaDPIGU uses binary masks (update or don't update each parameter) based on importance scores from DP pretraining. PHCA-DP-SGD uses continuous per-parameter clipping weights based on survival probability, enabling finer-grained privacy budget allocation.

#### 2.2.3 Compression-Aware Noise Calibration

Standard DP-SGD adds isotropic Gaussian noise:
$$\theta_{t+1} = \theta_t - \eta \left(\frac{1}{B}\sum_{i=1}^B \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I})\right)$$

In PHCA-DP-SGD, we calibrate noise based on anticipated compression:

$$\theta_{t+1} = \theta_t - \eta \left(\frac{1}{B}\sum_{i=1}^B \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 \mathbf{\Sigma})\right)$$

where $\mathbf{\Sigma}$ is a diagonal covariance matrix with entries:

$$\Sigma_{jj} = (1 - p_j(\kappa))^\beta$$

where $\beta \geq 0$ controls the noise reduction rate. Parameters with low survival probability receive less noise.

#### 2.2.4 Estimating Survival Probabilities Without Public Data

**Key innovation:** Unlike Adamczewski et al. (2023) who require public data for parameter selection, PHCA-DP-SGD estimates survival probabilities from private training dynamics.

At training step $t$, we maintain an exponential moving average of per-parameter gradient magnitudes:

$$s_j^{(t)} = \gamma s_j^{(t-1)} + (1-\gamma) |\bar{g}_j^{(t)}|$$

For magnitude pruning with target sparsity $\kappa$, the survival probability is:

$$p_j(\kappa) = \mathbb{P}(|\theta_j| \geq \text{threshold}(\kappa)) \approx \mathbb{P}(s_j \geq \text{percentile}(s, \kappa))$$

This estimation is performed on clipped gradients, so it incurs no additional privacy cost beyond the standard DP-SGD privacy budget.

### 2.3 Privacy Analysis

#### 2.3.1 Theoretical Guarantees

**Theorem (Privacy of PHCA-DP-SGD):** If standard DP-SGD with clipping threshold $C$ and noise multiplier $\sigma$ satisfies $(\epsilon, \delta)$-DP, then PHCA-DP-SGD with per-parameter weights $w_j \in [w_{\min}, 1]$ and noise covariance $\Sigma_{jj} \leq 1$ satisfies $(\epsilon', \delta)$-DP where:

$$\epsilon' \leq \epsilon \cdot \frac{1}{w_{\min}}$$

By setting $w_{\min} = \alpha$ (the discount factor for likely-pruned parameters), we can calibrate the base noise to achieve the target privacy guarantee.

**Key advantage over pre-pruning:** Adamczewski et al. (2023) require public data for parameter selection to avoid privacy cost. PHCA-DP-SGD achieves similar benefits without public data by using differentially private estimates of parameter importance.

### 2.4 Key Innovations and Differentiation

1. **Targets post-hoc compression**, not training-time pruning: Unlike Adamczewski et al. (2023) who pre-prune before training, we train the full model anticipating post-hoc compression, enabling better optimization of the final compressed model.

2. **No public data required:** Unlike pre-pruning approaches that rely on public data for parameter selection, PHCA-DP-SGD estimates survival probabilities from private training dynamics.

3. **Continuous per-parameter clipping weights based on survival probability:** Unlike AdaDPIGU's binary importance-based masking, we use continuous weights derived from survival probability, enabling finer-grained privacy budget allocation.

4. **Compression-aware noise calibration:** We add differentially-private noise scaled by anticipated compression impact, not uniformly across all parameters.

## 3. Related Work

### 3.1 Differential Privacy in Deep Learning

Dwork et al. (2006) introduced differential privacy as a formal privacy framework. Abadi et al. (2016) proposed DP-SGD, which has become the standard algorithm for training private deep neural networks. Recent advances include privacy amplification via subsampling (Balle et al., 2018), tighter accounting via Rényi DP (Mironov, 2017), and adaptations for large models (Li et al., 2022; Yu et al., 2022).

### 3.2 Pre-Pruning and DP Training

**Adamczewski et al. (2023)** ("Pre-Pruning and Gradient-Dropping Improve Differentially Private Image Classification", arXiv:2306.11754) explicitly study the interaction between DP and pruning. They propose:
- **Pre-pruning:** Using public data to select a subnetwork before DP training
- **Gradient-dropping:** During training, updating only selected parameters based on public data

**Key differences from our work:**
1. Adamczewski et al. require **public data** for parameter selection and gradient masking. PHCA-DP-SGD requires **no public data**.
2. Adamczewski et al. focus on **training-time pruning** (reducing trainable parameters). We target **post-hoc compression** (training full model, compressing after).
3. Adamczewski et al. use **binary** parameter selection (include/exclude). We use **continuous** per-parameter clipping weights.
4. Adamczewski et al. (2023) and Adamczewski & Park (2023) ("Differential Privacy Meets Neural Network Pruning") study pre-pruning with public data; we study post-hoc compression without public data.

### 3.3 Importance-Based Gradient Selection

**AdaDPIGU (Zhang & Xie, 2025)** (arXiv:2507.06525) proposes:
- Importance-based masking with progressive unfreezing
- Adaptive clipping for DP-SGD
- Gradient sparsification based on parameter importance

**Key differences from our work:**
1. AdaDPIGU uses **binary masks** (0/1) for gradient selection based on importance scores. We use **continuous clipping weights** based on survival probability.
2. AdaDPIGU's importance estimation requires **DP pretraining** on private data, consuming privacy budget. Our survival probability estimation is derived from **training dynamics** without additional privacy cost.
3. AdaDPIGU focuses on **training efficiency** (reducing dimensions for noise injection). We focus on **post-hoc compression optimization**.
4. AdaDPIGU's progressive unfreezing gradually increases retention ratio. Our method progressively adjusts compression awareness based on estimated survival probabilities.

### 3.4 Privacy-Preserving Compression and Attacks

**CompLeak (Li et al., 2025)** demonstrated that model compression exacerbates privacy leakage through membership inference attacks. They identified the problem but did not propose DP-based solutions.

**PAST (Hu et al., 2024)** proposes privacy-aware sparsity tuning using adaptive L1 regularization based on privacy sensitivity. However, PAST does not use DP training and focuses on empirical privacy (attack resistance) rather than certified privacy guarantees.

## 4. Experimental Plan

### 4.1 Datasets and Models

**Datasets:**
- CIFAR-10 (60K images, 10 classes) - primary evaluation
- CIFAR-100 (60K images, 100 classes) - generalization test
- Purchase-100 (synthetic purchase records) - tabular data evaluation

**Models:**
- ResNet-18 (standard architecture for DP research)
- ConvNet (4-layer CNN, standard for privacy benchmarking)

### 4.2 Compression Strategies

**Post-hoc Pruning (applied after training):**
- Magnitude pruning (unstructured)
- Global magnitude pruning

**Quantization:**
- Post-training quantization (8-bit, 4-bit)

### 4.3 Baselines

1. **Standard DP-SGD + Post-hoc Compression:** Train with standard DP-SGD, compress after training (current practice)
2. **Adamczewski et al. (2023) Pre-Pruning:** Use public data to pre-prune, train subnetwork with DP-SGD
3. **AdaDPIGU (Zhang & Xie, 2025):** Importance-based masking with adaptive clipping
4. **DP-SGD without Compression:** Standard DP-SGD, no compression (upper bound on utility)
5. **Non-private + Compression:** No DP, compress after training (no privacy guarantee)

**Note on public data:** For Adamczewski et al. baseline, we will simulate the public data requirement by using a held-out subset (not counted in privacy budget) for parameter selection, following their methodology.

### 4.4 Evaluation Metrics

**Utility Metrics:**
- Test accuracy (%)
- Compression ratio (parameters removed / total parameters)

**Privacy Metrics:**
- Certified privacy: $(\epsilon, \delta)$ values from accounting
- Empirical privacy:
  - MIA accuracy (lower is better)
  - MIA AUC (lower is better)

**Combined Metrics:**
- Privacy-utility curve: Test accuracy vs. MIA accuracy trade-off
- Accuracy at fixed $\epsilon$ and compression ratio

### 4.5 Expected Results

**Hypothesis 1:** PHCA-DP-SGD achieves 3-5% higher test accuracy than standard DP-SGD + post-hoc compression at the same privacy budget ($\epsilon = 3, \delta = 10^{-5}$) and compression ratio (70% sparsity).

**Hypothesis 2:** PHCA-DP-SGD matches or exceeds Adamczewski et al. (2023) pre-pruning performance **without requiring public data** for parameter selection.

**Hypothesis 3:** PHCA-DP-SGD outperforms AdaDPIGU on post-hoc compression scenarios due to continuous per-parameter clipping weights vs. binary masking.

**Hypothesis 4:** The optimal per-parameter clipping weights ($w_j$) correlate with final parameter magnitudes after training, validating our survival probability estimation approach.

## 5. Success Criteria

### 5.1 Confirming the Hypothesis

The research hypothesis is confirmed if:

1. **Primary:** PHCA-DP-SGD achieves statistically significantly better test accuracy (p < 0.05, paired t-test) than standard DP-SGD + post-hoc compression at the same certified privacy level ($\epsilon$) and compression ratio, averaged across at least 3 datasets.

2. **Secondary:** PHCA-DP-SGD achieves comparable or better accuracy than Adamczewski et al. (2023) pre-pruning approach **without using public data** for parameter selection.

3. **Tertiary:** PHCA-DP-SGD shows improved performance over AdaDPIGU on post-hoc compression tasks, demonstrating the value of continuous per-parameter clipping weights over binary masking.

4. **Theoretical:** We derive a theoretical framework showing how post-hoc compression interacts with DP guarantees, validated empirically.

### 5.2 Refuting the Hypothesis

The hypothesis is refuted if:

1. Standard DP-SGD + compression achieves accuracy within 1% of PHCA-DP-SGD across all tested configurations, suggesting compression-awareness provides no benefit.

2. PHCA-DP-SGD cannot match Adamczewski et al. (2023) performance without public data, indicating the public data requirement is essential.

3. Binary masking (AdaDPIGU) matches or exceeds continuous per-parameter clipping performance.

### 5.3 Scope and Limitations

**In Scope:**
- Image classification tasks
- Post-hoc compression (pruning, quantization)
- Standard DP-SGD framework
- Centralized training (not federated)

**Out of Scope:**
- Generative models (diffusion models, GANs)
- Large language models (due to computational constraints)
- Federated learning settings
- Training-time pruning (addressed by Adamczewski et al., 2023)

## 6. Broader Impact

### 6.1 Significance

This research addresses a critical gap in deploying privacy-preserving machine learning systems. Current practice treats privacy (via DP) and efficiency (via compression) as separate concerns, but our work shows they are deeply coupled. By providing a principled framework for joint optimization that **does not require public data**, we enable:

1. **More accurate private models:** Better privacy-utility trade-offs mean DP can be deployed in scenarios where public data is unavailable.

2. **Safer model deployment:** Understanding how post-hoc compression affects privacy leakage allows practitioners to make informed decisions.

3. **Resource-constrained privacy:** Enabling accurate private models on edge devices expands the scope of privacy-preserving ML.

### 6.2 Potential Risks and Mitigations

**Risk:** Improved utility might encourage deployment of private models in sensitive domains without adequate privacy review.

**Mitigation:** We will emphasize that our work improves the privacy-utility trade-off but does not eliminate privacy risks. All deployments should undergo thorough privacy auditing.

**Risk:** Misinterpretation of compression as a privacy guarantee.

**Mitigation:** Clear distinction between certified privacy (DP) and empirical privacy (attack resistance). Compression alone does not provide formal privacy guarantees.

## 7. References

- Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS).

- Adamczewski, K., He, Y., & Park, M. (2023). Pre-Pruning and Gradient-Dropping Improve Differentially Private Image Classification. arXiv preprint arXiv:2306.11754.

- Adamczewski, K., & Park, M. (2023). Differential Privacy Meets Neural Network Pruning. arXiv preprint arXiv:2303.04612.

- Balle, B., Barthe, G., & Gaboardi, M. (2018). Privacy Amplification by Subsampling in Tight Analyses of Differentially Private Stochastic Gradient Descent. Proceedings of Machine Learning Research.

- Carlini, N., Liu, C., Kos, J., Úlfar, E., & Song, D. (2019). The Secret Sharer: Measuring Unintended Neural Network Memorization. In USENIX Security Symposium.

- Dong, J., Roth, A., & Su, W. J. (2019). Gaussian Differential Privacy. arXiv preprint arXiv:1905.02383.

- Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. Foundations and Trends in Theoretical Computer Science.

- Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating Noise to Sensitivity in Private Data Analysis. In Theory of Cryptography Conference.

- Erlingsson, Ú., Feldman, V., Mironov, I., Raghunathan, A., Talwar, K., & Thakurta, A. (2019). Amplification by Shuffling. In IEEE Symposium on Security and Privacy.

- Hu, Q., Zhang, H., & Wei, H. (2024). Defending Membership Inference Attacks via Privacy-aware Sparsity Tuning. arXiv preprint arXiv:2410.06814.

- Li, T., et al. (2025). CompLeak: Deep Learning Model Compression Exacerbates Privacy Leakage. arXiv preprint.

- Mironov, I. (2017). Rényi Differential Privacy. In IEEE Computer Security Foundations Symposium (CSF).

- Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership Inference Attacks Against Machine Learning Models. In IEEE Symposium on Security and Privacy.

- Zhang, H., & Xie, F. (2025). AdaDPIGU: Differentially Private SGD with Adaptive Clipping and Importance-Based Gradient Updates for Deep Neural Networks. arXiv preprint arXiv:2507.06525.
