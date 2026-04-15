# Research Proposal: G3P - Gradient-Guided Privacy-Preserving Pruning via Train-Test Gradient Saliency

## 1. Introduction

### 1.1 Context and Problem Statement

Neural networks face significant privacy vulnerabilities, particularly from **Membership Inference Attacks (MIAs)** that determine whether a specific sample was part of the training set. While various privacy-preserving techniques exist, they typically suffer from severe privacy-utility trade-offs. Differential Privacy (DP) provides strong theoretical guarantees but often results in substantial accuracy degradation.

Model pruning, widely used for efficient deployment, has shown mixed effects on privacy. Wang et al. (IJCAI 2021) demonstrated that weight pruning can reduce MIA attack accuracy by 10-13.6%, suggesting pruning's potential as a privacy defense. However, Yuan & Zhang (USENIX Security 2022) showed that pruning can actually *increase* memorization and privacy risks in certain scenarios. Meanwhile, recent work on spurious privacy leakage (Zhang et al., TMLR 2025) reveals that certain subpopulations exhibit disproportionately higher privacy vulnerabilities—up to 100× more vulnerable than others.

This raises a critical question: **Can we design pruning strategies that consistently improve privacy by explicitly targeting privacy-leaking neurons, rather than relying on incidental effects?**

### 1.2 Key Insight and Hypothesis

**Key Insight**: Not all neurons contribute equally to privacy leakage. We hypothesize that neurons exhibiting significantly different gradient behaviors between training and test data are the primary contributors to membership inference vulnerability. The intuition is that these neurons "memorize" training-specific patterns, producing distinct gradient signals for member vs. non-member data—precisely the signal exploited by MIAs.

**Novelty of Our Formulation**: While gradient-based importance has been used for task-oriented pruning (Molchanov et al., 2019), the use of **train-test gradient magnitude differences as a privacy-specific saliency metric** is novel. Prior work uses gradient information to estimate impact on task loss; we use the *divergence* between train and test gradients to estimate privacy leakage.

**Hypothesis**: A saliency-guided pruning approach that identifies and removes neurons based on train-test gradient magnitude differences can achieve better and more consistent privacy-utility-efficiency trade-offs than magnitude-based pruning. Specifically, gradient-based neuron selection provides privacy benefits beyond what can be achieved by combining magnitude pruning with KL regularization.

## 2. Proposed Approach

### 2.1 Overview

We propose **G3P (Gradient-Guided Privacy-Preserving Pruning)**, a method that:
1. **Quantifies neuron-level privacy contribution** using the absolute difference in gradient magnitudes between training and test distributions
2. **Combines task importance and privacy risk** into a unified saliency score
3. **Progressively prunes privacy-vulnerable neurons** while preserving task-critical ones
4. **Fine-tunes with privacy-aware regularization** to further reduce leakage

### 2.2 Method Details

#### 2.2.1 Privacy Saliency Metric

For each neuron $i$, we define a privacy saliency score:

$$S_{privacy}(i) = \left| \mathbb{E}_{(x,y) \sim \mathcal{D}_{train}} \left[ \left| \frac{\partial \mathcal{L}(f_\theta(x), y)}{\partial h_i} \right| \right] - \mathbb{E}_{(x,y) \sim \mathcal{D}_{test}} \left[ \left| \frac{\partial \mathcal{L}(f_\theta(x), y)}{\partial h_i} \right| \right] \right|$$

where $h_i$ is the activation of neuron $i$. The intuition is that neurons with significantly different gradient magnitudes between training and test data contribute to the membership signal exploited by MIAs.

#### 2.2.2 Task Saliency Metric

Following Molchanov et al. (2019), we estimate task importance using first-order Taylor expansion:

$$S_{task}(i) = \left| \mathbb{E}_{(x,y) \sim \mathcal{D}_{train}} \left[ \frac{\partial \mathcal{L}}{\partial h_i} \cdot h_i \right] \right|$$

This approximates the change in loss if neuron $i$ were removed.

#### 2.2.3 Combined Saliency Score

We combine task and privacy saliency:

$$S_{combined}(i) = \alpha \cdot S_{task}(i) - \beta \cdot S_{privacy}(i)$$

where $\alpha, \beta$ are hyperparameters controlling the privacy-utility trade-off. Neurons with lowest $S_{combined}$ are pruned first.

#### 2.2.4 Progressive Privacy-Aware Pruning

1. **Warmup**: Train the model to convergence
2. **Privacy Audit**: Compute privacy saliency scores using a small held-out validation set
3. **Structured Pruning**: Remove channels with lowest $S_{combined}(i)$
4. **Fine-tuning with Privacy Regularization**: 
   - Minimize the KL divergence between predictions on training and test distributions
5. **Iteration**: Repeat steps 2-4 until target sparsity is reached

### 2.3 Key Differentiations

#### Differentiation from Molchanov et al. (CVPR 2019)
- Molchanov et al. use gradient-based importance for **task performance** via Taylor expansion
- Our work introduces a **privacy-specific** gradient metric: train-test gradient differences
- We combine task and privacy saliency for targeted privacy-preserving pruning

#### Differentiation from Wang et al. (IJCAI 2021)
- Wang et al. showed that magnitude-based pruning can reduce MIA vulnerability as a **side effect**
- Our approach *explicitly targets* privacy-leaking neurons through privacy-specific gradient saliency
- We evaluate on modern attack methods (LiRA) beyond the simpler attacks used in 2021

#### Differentiation from Yuan & Zhang (USENIX Security 2022)
This is a **critical distinction** that addresses a key limitation in our original proposal:

- **Yuan & Zhang** proposed KL-divergence regularization to align training and test predictions during pruning
- **Their approach**: Applies KL regularization uniformly to all neurons during fine-tuning without modifying the pruning criterion
- **Our approach**: Uses train-test gradient differences to **SELECT** which neurons to prune before applying KL regularization
- **Key Difference**: G3P identifies privacy-vulnerable neurons through gradient saliency and removes them structurally, whereas Yuan & Zhang keep all neurons but regularize their outputs

To isolate the value of our gradient-based selection, we will include a **hybrid baseline** that combines magnitude pruning with KL regularization—directly testing whether gradient-based neuron selection provides benefits beyond the combination of magnitude pruning and KL regularization.

#### Differentiation from PriPrune (Chu et al., ACM TOMPECS 2024)
- PriPrune focuses on **federated learning** with gradient protection against gradient inversion attacks
- Our method targets general membership inference attacks in **centralized settings**
- PriPrune uses pseudo-pruning masks for local training; we use gradient-based saliency for structured channel pruning

## 3. Related Work

### 3.1 Privacy Attacks in Machine Learning

**Membership Inference Attacks**: Shokri et al. (2017) introduced the shadow model technique. Salem et al. (2019) proposed low-cost attack variants. Carlini et al. (2022) developed LiRA, a likelihood ratio-based attack achieving state-of-the-art performance by evaluating true-positive rate at low (≤0.1%) false-positive rates.

**Privacy Vulnerability Disparities**: Zhang et al. (TMLR 2025) identified spurious privacy leakage where groups with spurious correlations are significantly more vulnerable to MIAs than non-spurious groups, creating privacy disparities that aggregate metrics hide.

### 3.2 Privacy Defenses

**Differential Privacy**: DP-SGD (Abadi et al., 2016) provides strong privacy guarantees but suffers from significant utility degradation, often 5-15% accuracy drop for meaningful privacy budgets (ε ≤ 4).

**Regularization-based Defenses**: RelaxLoss (Chen et al., 2022) aligns training and test loss distributions. Early stopping and L2 regularization reduce overfitting but provide no formal guarantees.

### 3.3 Model Pruning and Privacy

**Gradient-Based Pruning for Task Performance**: Molchanov et al. (2019) introduced gradient-based importance estimation using first and second-order Taylor expansions to approximate a filter's contribution to task loss. This work focuses exclusively on task performance, not privacy.

**Wang et al. (IJCAI 2021)** demonstrated that magnitude-based pruning can reduce MIA attack accuracy by 10-13.6% compared to undefended models. They provided theoretical analysis connecting pruning's regularization effect to privacy improvement, but the privacy gains were incidental rather than targeted.

**Yuan & Zhang (USENIX Security 2022)** analyzed privacy risks in neural network pruning, showing that pruning can increase memorization and privacy risks in some scenarios. They proposed KL-divergence regularization as a defense but did not explore pruning criteria that explicitly account for privacy at the neuron level. Our work differs by introducing neuron-level privacy saliency as a pruning criterion.

**PriPrune** (Chu et al., ACM TOMPECS 2024) proposed privacy-aware pruning for federated learning using pseudo-pruning techniques to protect gradients from inversion attacks in FL settings.

**Key Gap**: While prior work has shown that pruning *can* improve privacy, no existing method explicitly identifies and removes privacy-leaking neurons at the granularity level through **privacy-specific gradient-based saliency metrics** (i.e., train-test gradient differences). Our contribution is a principled criterion for selecting which neurons to prune, not the general idea of using pruning for privacy.

## 4. Experiments

### 4.1 Experimental Setup

**Datasets**:
- CIFAR-10 and CIFAR-100 for image classification (primary focus)
- Purchase-100 for tabular data validation

**Models**:
- ResNet-18 (primary)
- VGG-16 (for architecture comparison)

**Baselines**:
1. **Standard Magnitude Pruning** (Han et al., 2015) - ℓ1-norm based
2. **Taylor Importance Pruning** (Molchanov et al., 2019) - task-only gradient saliency
3. **Hybrid: Magnitude + KL** - magnitude pruning with KL-divergence regularization (NEW: isolates value of gradient-based selection)
4. **Wang et al. (IJCAI 2021)** - magnitude pruning defense baseline
5. **Yuan & Zhang (USENIX 2022)** - KL-regularized pruning
6. **DP-SGD** (Abadi et al., 2016) - privacy baseline with ε=4
7. **RelaxLoss** (Chen et al., 2022) - privacy defense baseline

**Attack Methods**:
- Threshold-based attack (fast evaluation, ~5 minutes per model)
- LiRA (Carlini et al., 2022) - final validation only, ~1.5 hours per evaluation

**Metrics**:
- **Utility**: Test accuracy
- **Efficiency**: FLOPs reduction, model size reduction
- **Privacy**: MIA AUC (lower is better), Attack Advantage (TPR@0.1% FPR)
- **Combined**: Privacy-utility-efficiency frontier

### 4.2 Expected Results

**Primary Hypothesis**: G3P will achieve better and more consistent privacy-utility trade-offs than magnitude-based pruning.

**Specific Predictions**:
1. At 50% sparsity, G3P will reduce MIA AUC by 10-15% compared to magnitude pruning while maintaining comparable task accuracy
2. **G3P will outperform the hybrid baseline** (magnitude + KL), demonstrating that gradient-based neuron selection provides privacy benefits beyond simply combining magnitude pruning with KL regularization
3. G3P will achieve similar privacy protection (MIA AUC ≈ 0.55-0.60) as DP-SGD with ε=4 but with 5-8% higher task accuracy
4. G3P will show more consistent privacy improvements across different attack methods compared to magnitude pruning

**Ablation Studies**:
- Impact of $\alpha$ and $\beta$ hyperparameters on privacy-utility trade-off
- Comparison of pruning-only vs. pruning + KL regularization
- Analysis of which layers contribute most to privacy leakage
- Correlation between train-test gradient difference and actual MIA vulnerability

### 4.3 Computational Requirements

Given the resource constraints (1× A6000 48GB, 60GB RAM, 4 cores, 8 hours):

**Streamlined Experiment Plan**:
- Focus on CIFAR-10 with ResNet-18 (primary configuration)
- Use threshold-based attack for intermediate evaluations (~5 minutes per model)
- Reserve LiRA evaluation (16 shadow models, ~1.5 hours) for final validation only
- Total estimated time: 6-8 hours

**Time Breakdown**:
- Training baseline models: ~2 hours (4 models × 30 min)
- G3P pruning experiments: ~2.5 hours (various sparsity levels, α/β tuning)
- Baseline comparisons (including hybrid): ~1.5 hours
- Final LiRA validation: ~1.5 hours
- **Total: ~7.5 hours** (feasible within constraints)

## 5. Success Criteria

### 5.1 Confirmation

The hypothesis is confirmed if:
1. G3P achieves statistically significantly lower MIA AUC than magnitude pruning at the same sparsity level (p < 0.05, paired t-test)
2. **G3P outperforms the hybrid baseline** (magnitude pruning + KL regularization), confirming that gradient-based selection provides value beyond combining magnitude pruning with KL
3. The privacy improvement from G3P is more consistent across different attack methods compared to magnitude pruning
4. G3P maintains within 2% of the unpruned model's accuracy at moderate sparsity (30-50%)

### 5.2 Refutation

The hypothesis is refuted if:
1. Train-test gradient difference saliency does not correlate with actual membership inference vulnerability better than magnitude-based importance
2. The hybrid baseline (magnitude + KL) matches G3P performance, suggesting gradient-based selection adds no value beyond KL regularization
3. Pruning privacy-vulnerable neurons harms task accuracy disproportionately (>5% drop at 30% sparsity)

## 6. Impact and Significance

### 6.1 Scientific Contribution

This work contributes a principled, neuron-level criterion for privacy-preserving pruning. Rather than treating privacy as a side effect of compression, we explicitly identify which architectural components leak membership information through a novel privacy-specific formulation (train-test gradient differences). This opens new directions for:
- Understanding the connection between gradient behavior and privacy leakage
- Hardware-efficient privacy-preserving ML through structured pruning
- Practical deployment of privacy-preserving models on resource-constrained devices

### 6.2 Practical Impact

- **Edge Deployment**: Enables deployment of efficient, privacy-preserving models on mobile and IoT devices
- **Healthcare**: Protects patient privacy in medical imaging models without severe accuracy degradation
- **Privacy Auditing**: Provides a method to identify privacy-vulnerable components in trained models

### 6.3 Future Directions

1. Extension to transformers and attention head pruning
2. Theoretical analysis connecting train-test gradient divergence to formal privacy bounds
3. Integration with differential privacy for hybrid privacy guarantees
4. Application to spurious privacy leakage mitigation in biased datasets

## 7. References

1. Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep learning with differential privacy. ACM CCS.
2. Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., ... & Raffel, C. (2021). Extracting training data from large language models. USENIX Security.
3. Carlini, N., Choquette-Choo, C. A., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022). Membership inference attacks from first principles. IEEE S&P.
4. Chen, D., Yu, N., & Fritz, M. (2022). Relaxloss: Defending membership inference attacks without losing utility. ICLR.
5. Chu, T., Yang, M., Laoutaris, N., & Markopoulou, A. (2024). PriPrune: Quantifying and preserving privacy in pruned federated learning. ACM TOMPECS.
6. Frankle, J., & Carbin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable neural networks. ICLR.
7. Han, S., Mao, H., & Dally, W. (2016). Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. ICLR.
8. Molchanov, P., Mallya, A., Tyree, S., Frosio, I., & Kautz, J. (2019). Importance estimation for neural network pruning. CVPR.
9. Salem, A., Zhang, Y., Humbert, M., Berrang, P., Fritz, M., & Backes, M. (2019). Ml-leaks: Model and data independent membership inference attacks and defenses on machine learning models. NDSS.
10. Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership inference attacks against machine learning models. IEEE S&P.
11. Wang, Y., Wang, C., Wang, Z., Zhou, S., Liu, H., Bi, J., Ding, C., & Rajasekaran, S. (2021). Against membership inference attack: Pruning is all you need. IJCAI.
12. Yuan, X., & Zhang, L. (2022). Membership inference attacks and defenses in neural network pruning. USENIX Security.
13. Zhang, C., Pang, J., & Mauw, S. (2025). Spurious privacy leakage in neural networks. TMLR.
