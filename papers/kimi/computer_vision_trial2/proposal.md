# Adaptive Prototype-Aware Consistency with Learnable Augmentation Policies for Single-Image Test-Time Adaptation

## Research Proposal

### 1. Introduction

#### 1.1 Problem Statement

Deep neural networks achieve remarkable performance when training and test data are drawn from the same distribution. However, in real-world deployment, models frequently encounter distribution shifts caused by environmental changes, sensor variations, or image corruptions. These shifts can severely degrade model performance—for example, a ResNet-50 trained on clean ImageNet images drops from 76.7% to 23.9% accuracy when tested on severely corrupted images (ImageNet-C severity level 5).

Test-Time Adaptation (TTA) addresses this challenge by adapting model parameters during inference using unlabeled test samples. While promising, existing TTA methods face three critical limitations:

1. **Confirmation Bias in Entropy-Based Methods**: Methods like TENT, EATA, and MEMO rely on entropy minimization or pseudo-labeling, which can spiral into incorrect predictions when the model is initially confident but wrong—a common occurrence under severe distribution shifts.

2. **Fixed Augmentation Strategies**: Existing augmentation-based methods (MEMO, TPT) use a fixed set of augmentations regardless of the test image characteristics. This is suboptimal because different corruptions require different augmentation strategies—heavy blur requires different augmentations than noise or contrast changes.

3. **Prototype Underutilization**: While recent methods like SWR+NSP and DPL use prototypes, they either weight all prototypes equally or focus on contrastive learning without addressing how to optimally select augmentations based on prototype information.

#### 1.2 Key Insight and Hypothesis

We propose that **learnable, prototype-aware augmentation selection combined with confidence-weighted consistency to source prototypes can significantly improve single-image test-time adaptation**. Our key insight is threefold:

1. **Prototype-aware augmentation**: Different test images should use different augmentation strategies based on their proximity to source prototypes. Images near class boundaries should use milder augmentations, while images in high-density regions can use stronger augmentations.

2. **Confidence-weighted consistency**: Not all source prototypes are equally reliable for a given test image. We should weight prototype guidance by the confidence/stability of each prototype.

3. **Learnable augmentation policy**: Instead of manually designing augmentations or using simple heuristics, we can learn a lightweight policy network that predicts which augmentations will be most informative for a given test image based on its feature similarity to source prototypes.

Our key hypothesis is:
> **Adaptive selection of augmentations based on prototype similarity, combined with confidence-weighted consistency to source prototypes, provides more robust and effective single-image test-time adaptation than fixed augmentation strategies, heuristic augmentation selection, or entropy-based methods, particularly under severe and mixed corruptions.**

#### 1.3 Proposed Approach

We introduce **Adaptive Prototype-Aware Consistency with Learnable Augmentation Policies for Test-Time Adaptation (APAC-TTA)**, a method with three novel components:

1. **Meta-Augmentation Policy Network (Meta-APN)**: A lightweight meta-network that predicts augmentation parameters (operation types, severities, mixing weights) based on the test image's feature distance to source prototypes. This network is trained offline on source data to learn which augmentations are most informative for different image-prototype relationships.

2. **Prototype Confidence Scoring**: We compute a confidence score for each source prototype based on its feature consistency across multiple augmented views of the test image. Prototypes with higher consistency receive higher weights in the consistency loss.

3. **Adaptive Prototype-Driven Consistency Loss**: We adapt model parameters to maximize agreement between augmented view predictions and their weighted nearest prototypes, where weights are determined by prototype confidence scores.

### 2. Related Work

#### 2.1 Test-Time Adaptation with Entropy Minimization

**TENT** (Wang et al., 2021) proposed fully test-time adaptation by minimizing prediction entropy through updates to batch normalization statistics. While effective for batches, TENT requires sufficient batch size for reliable gradient estimation and can suffer from confirmation bias.

**EATA** (Niu et al., 2022) improved upon TENT by filtering high-entropy samples and applying Fisher regularization to prevent forgetting. However, it still relies on entropy minimization and requires multiple samples.

**MEMO** (Zhang et al., 2022) enables single-sample adaptation by creating a virtual batch through test-time augmentation and minimizing marginal entropy. Unlike our approach, MEMO uses: (1) a fixed set of augmentations regardless of the test image, and (2) entropy minimization rather than prototype-guided consistency with learned augmentation policies.

#### 2.2 Prototype-Based Test-Time Adaptation

**T3A** (Iwasawa & Matsuo, 2021) adapts the classifier using online feature means computed from test batches. Unlike our approach, T3A does not use pre-computed source prototypes or learnable augmentation policies.

**SWR+NSP** (Choi et al., 2022) uses shift-agnostic weight regularization and nearest source prototypes. **Important distinction**: SWR+NSP requires access to source data *before* deployment (offline) for generating prototypes and identifying shift-agnostic parameters, but does NOT require source data *at test time*. This is a one-time pre-deployment cost. Key differences from our method: (1) SWR+NSP uses fixed augmentations, (2) does not learn augmentation policies based on prototype similarity, and (3) uses uniform prototype weighting rather than confidence-weighted guidance.

**PROGRAM** (Sun et al., ICLR 2024) constructs a prototype graph model for reliable pseudo-label generation. Unlike our approach, PROGRAM focuses on pseudo-label denoising via graph-based message passing rather than learnable augmentation policies based on prototype similarity.

**CPA** (Lee et al., 2024) uses pre-trained model weights as source prototypes and performs class-wise alignment. Unlike APAC-TTA: (1) CPA uses fixed pseudo-labeling rather than adaptive augmentation selection, (2) does not learn augmentation policies, (3) uses simple consistency rather than confidence-weighted prototype guidance.

#### 2.3 Recent Prototype-Based Methods (2024)

**DPL (Decoupled Prototype Learning)** (Wang et al., arXiv 2024) proposes a prototype-centric loss computation that decouples the optimization of class prototypes. For each prototype, DPL reduces its distance with positive samples and enlarges its distance with negative samples in a contrastive manner, preventing overfitting to noisy pseudo-labels. DPL also introduces a memory-based strategy for robustness with small batch sizes and uses style transfer for unconfident pseudo-labels.

**Acknowledgment of DPL**: DPL makes significant contributions to prototype-centric learning for TTA, demonstrating that treating prototypes as first-class optimization targets (rather than sample-centric cross-entropy) improves robustness to noisy pseudo-labels. DPL's prototype-centric approach shares conceptual ground with our use of prototypes for guidance. However, our work addresses a different problem: **augmentation selection**. DPL uses fixed augmentations with consistency regularization and focuses on contrastive prototype learning—we focus on learning corruption-specific augmentation strategies based on prototype similarity.

Key differences from our APAC-TTA:
- **Core innovation**: DPL focuses on contrastive prototype optimization; we focus on learnable augmentation policies based on prototype similarity
- **Augmentations**: DPL uses fixed augmentations for consistency; we learn adaptive policies
- **Prototype source**: DPL updates prototypes online; we use stable source prototypes with confidence weighting
- **Adaptation signal**: DPL uses contrastive learning; we use confidence-weighted prototype consistency
- **Target setting**: DPL is primarily designed for batch TTA; we target single-image TTA

**DPE (Dual Prototype Evolving)** (Zhang et al., NeurIPS 2024) introduces dual prototypes (textual and visual) for vision-language models. DPE evolves prototypes over time for VLMs using multi-modal consistency. Key differences: (1) DPE targets vision-language models (CLIP) while we target pure vision models, (2) DPE uses fixed augmentations, (3) DPE maintains evolving prototype queues while we use stable source prototypes with confidence weighting.

#### 2.4 Adaptive Test-Time Augmentation

**Adaptive-TTA** (Conde & Premebida, BMVC 2022) proposes learning weights for test-time augmentations to improve uncertainty calibration. Their method optimizes augmentation weights on a validation set to minimize Expected Calibration Error (ECE).

**Critical distinction from Adaptive-TTA**: Adaptive-TTA fundamentally differs from our approach in design goals and mechanisms:

1. **Primary Goal**: Adaptive-TTA explicitly focuses on **uncertainty calibration** (Brier score, ECE), not classification accuracy. As stated in their paper: "does not alter the class predicted by the model...can be specifically optimized to tackle problems regarding uncertainty calibration, without the concern of corrupting the model's accuracy." This means Adaptive-TTA **guarantees the predicted class remains unchanged** from the original model—explicitly trading potential accuracy improvements for calibration. APAC-TTA **explicitly targets accuracy improvement** on corrupted data and allows class predictions to change when evidence from augmented views supports a different class.

2. **Mechanism**: Adaptive-TTA uses fixed, pre-defined augmentation policies (e.g., AutoAugment) with learned **scalar aggregation weights** for prediction averaging. The augmentation policy itself is fixed; only the weights for aggregating predictions are learned. APAC-TTA uses a meta-network that predicts **augmentation parameters dynamically** (which operations to apply, their severity, how many) based on prototype similarity, enabling corruption-specific strategies.

3. **Prototype guidance**: Adaptive-TTA does not use prototypes or prototype-aware selection. APAC-TTA's core contribution is using prototype similarity to inform augmentation selection.

4. **Adaptation**: Adaptive-TTA does not adapt model parameters—it only aggregates predictions differently. APAC-TTA adapts BatchNorm parameters using confidence-weighted prototype consistency.

**Why APAC-TTA can succeed where Adaptive-TTA trades accuracy for calibration**: Adaptive-TTA's constraint of preserving the original prediction fundamentally limits its ability to correct misclassifications caused by distribution shift. In contrast, APAC-TTA uses prototype-guided consistency that can shift predictions toward more reliable classes when augmented views consistently support a different label. The confidence weighting mechanism allows the model to trust prototype guidance more when augmented views show high consistency, enabling genuine prediction correction rather than just calibration of confidence scores.

#### 2.5 Vision-Language Model Adaptation

**TPT** (Shu et al., 2022) performs test-time prompt tuning for CLIP by minimizing entropy across augmented views. While conceptually similar, TPT: (1) is designed specifically for vision-language models, (2) uses simple augmentations (random crops/flips) rather than learned policies, (3) adapts text prompts rather than model parameters with prototype guidance.

#### 2.6 How Our Work Differs

Our APAC-TTA differs from all prior work in four key aspects:

| Aspect | MEMO | DPE | Adaptive-TTA | SWR+NSP | DPL | APAC-TTA (Ours) |
|--------|------|-----|--------------|---------|-----|-----------------|
| Target Models | Pure Vision | Vision-Language | Pure Vision | Pure Vision | Pure Vision | Pure Vision |
| Augmentations | Fixed | Fixed | Fixed weights | Fixed | Fixed | **Learnable policy** |
| Prototype Usage | None | Dual evolving | None | Nearest source | Contrastive learning | **Confidence-weighted** |
| Adaptation Signal | Marginal entropy | Multi-modal consistency | None (aggregation) | Pseudo-labels | Contrastive prototype | **Prototype-aware consistency** |
| Source Data at Test | No | No | No | No (pre-deployment only) | No | **No** |
| Goal | Accuracy | Accuracy | **Calibration** | Accuracy | Accuracy | **Accuracy** |

**Key Differentiators:**

1. **Learnable Augmentation Policy**: We introduce a meta-learned augmentation policy network for test-time adaptation that selects augmentations based on prototype similarity. This differs from fixed augmentation strategies in MEMO, TPT, DPE, and DPL, and from the fixed-policy scalar weight learning in Adaptive-TTA (which explicitly avoids accuracy improvements to preserve calibration).

2. **Prototype Confidence Weighting**: Unlike SWR+NSP, T3A, and DPL that weight prototypes uniformly or via contrastive learning, we compute confidence scores for prototypes based on their consistency across augmented views, allowing adaptive weighting for each test sample.

3. **Pure Vision Focus**: Unlike DPE and TPT which require vision-language models, our method works with any standard vision model (ResNet, ViT, etc.).

4. **No Source Data Required**: Our method only uses compact source prototypes (C × d dimensions) computed during training. No source data is needed at test time or during deployment.

### 3. Proposed Method

#### 3.1 Preliminary: Source Prototype and Meta-Policy Network Training

**Source Prototype Computation:**
```
For each class c in {1, ..., C}:
    prototype_c = mean(feature_extractor(x) for x in class_c_training_data)
```

**Meta-Augmentation Policy Network (Meta-APN):**
We train a lightweight meta-network (2-layer MLP, ~10K parameters) on source data to predict augmentation policies:

```
Input: Feature vector z = feature_extractor(x)
       Prototype distances d = [||z - prototype_c|| for c in 1..C]
       
Output: Augmentation policy π = {
    operation_logits: [l1, l2, ..., l_K] for K operations,
    severity_scale: s ∈ [0.5, 1.5],
    num_augmentations: M ∈ {4, 8, 16}
}

Training objective: Maximize prediction accuracy on held-out source data
                    after applying augmentations sampled from π
```

**Handling Discrete Augmentation Selection:**
The augmentation selection involves discrete choices (which operation to apply, how many augmentations to use), which presents optimization challenges. We address this through:

1. **Gumbel-Softmax for Operation Selection**: We use the Gumbel-Softmax trick to make the operation selection differentiable:
   ```
   w_i = exp((log(π_i) + g_i) / τ) / Σ_j exp((log(π_j) + g_j) / τ)
   ```
   where g_i ~ Gumbel(0,1) and τ is a temperature parameter annealed during training.

2. **Continuous Relaxation for Severity**: Severity scales are predicted as continuous values and applied through interpolation.

3. **Proxy Task Training**: Since directly optimizing for "accuracy after augmentation" is non-differentiable, we use a proxy objective that maximizes prediction confidence on the correct class after applying the learned augmentation:
   ```
   L_meta = -E[log p(y_true | aug(x; π))]
   ```
   This provides gradients that guide the policy toward augmentations that preserve discriminative features.

4. **Straight-Through Estimator (STE)**: For the number of augmentations M, we use a straight-through estimator that rounds during the forward pass but passes gradients through during backpropagation.

5. **REINFORCE as Fallback**: If Gumbel-Softmax training becomes unstable, we fall back to REINFORCE (policy gradient) with a baseline:
   ```
   ∇J = E[(R - b) ∇log π(a|s)]
   ```
   where R is the post-augmentation accuracy and b is a running average baseline.

#### 3.2 Test-Time Adaptation Procedure

For each test image x_t arriving at time t:

**Step 1: Compute Prototype Similarities**
```
z = feature_encoder(x_t)
prototype_distances = [cosine_distance(z, prototype_c) for c in 1..C]
nearest_prototype_idx = argmin(prototype_distances)
min_distance = min(prototype_distances)
confidence_proxy = 1 - (min_distance / max(prototype_distances))
```

**Step 2: Generate Augmentations via Learned Policy**
```
# Use Meta-APN to predict augmentation policy
policy = meta_apn(z, prototype_distances)

# Sample M augmentations based on policy using Gumbel-Softmax
gumbel_noise = -log(-log(uniform(0,1)))
operation_probs = softmax((policy.operation_logits + gumbel_noise) / τ_test)

augmentations = []
for i in range(policy.M):
    # Sample operation based on policy weights
    op_idx = argmax(operation_probs)  # During inference, can use argmax
    op = AUGMENTATION_OPS[op_idx]
    severity = policy.severity_scale * base_severity[op_idx]
    x_aug = apply_augmentation(x_t, op, severity)
    augmentations.append(x_aug)
```

**Step 3: Compute Prototype Confidence Scores**
```
# For each prototype, compute consistency across augmentations
prototype_confidences = {}
for c in 1..C:
    similarities = [cosine_similarity(feature_encoder(x_aug), prototype_c) 
                   for x_aug in augmentations]
    # Confidence = consistency (low variance = high confidence)
    consistency = 1 / (1 + variance(similarities))
    # Also incorporate mean similarity
    mean_sim = mean(similarities)
    prototype_confidences[c] = consistency * mean_sim
```

**Step 4: Adaptive Consistency Loss**
```
L_total = 0
for x_aug in augmentations:
    z_aug = feature_encoder(x_aug)
    p_aug = softmax(classifier(z_aug))
    
    # Find weighted nearest prototypes
    weighted_distances = [cosine_distance(z_aug, prototype_c) / prototype_confidences[c]
                         for c in 1..C]
    nearest_c = argmin(weighted_distances)
    
    # Create target distribution centered on prototype with temperature
    target_dist = softmax(prototype_logits[nearest_c] / temperature)
    
    # Jensen-Shannon divergence for consistency
    L_total += JS_divergence(p_aug, target_dist)

L_total = L_total / len(augmentations)
```

**Step 5: Selective Adaptation**
```
# Only adapt if test image is reasonably close to source distribution
if min_distance < adaptation_threshold:
    # Update BatchNorm statistics and affine parameters
    update_parameters(L_total, lr=learning_rate)
else:
    # Use source model without adaptation
    pass
```

#### 3.3 Heuristic Baseline for Comparison

To justify the complexity of Meta-APN with Gumbel-Softmax/REINFORCE, we implement a simpler heuristic baseline:

**Distance-Based Severity Selection (Heuristic-APN)**: Instead of learning a policy network, use a simple heuristic:
- Compute prototype distance d = min_c ||z - prototype_c||
- Augmentation severity scales inversely with prototype distance:
  ```
  severity = max_severity * (1 - exp(-λ * d))
  ```
- Augmentation operations selected randomly from a pre-defined set
- Fixed number of augmentations (e.g., M = 8)

This baseline tests whether learned policies provide benefits beyond simple distance-based heuristics. If Meta-APN significantly outperforms Heuristic-APN, it justifies the additional complexity.

#### 3.4 Implementation Details

- **Adaptable Parameters**: BatchNorm running statistics, scale, and shift parameters (following TENT)
- **Meta-APN Architecture**: 2-layer MLP with hidden dimension 64, ReLU activation
- **Number of Augmentations**: Dynamically selected by policy (typically M = 8)
- **Temperature**: τ = 0.5 for prototype target distribution, annealed from 1.0 during Meta-APN training
- **Learning Rate**: 1e-3 for CIFAR, 1e-4 for ImageNet
- **Threshold**: μ + 2σ of training set prototype distances

### 4. Experiments

#### 4.1 Datasets

We evaluate on comprehensive corruption benchmarks:

1. **CIFAR-10-C** (Hendrycks & Dietterich, 2019): 10,000 test images with 15 corruption types at 5 severity levels
2. **CIFAR-100-C**: Same structure with 100 classes
3. **ImageNet-C** (Hendrycks & Dietterich, 2019): 5,000 test images with 15 corruption types at 5 severity levels (using ResNet-50 pretrained on ImageNet-1K)
4. **CIFAR-10.1** (Recht et al., 2018): Natural distribution shift with 2,000 test images
5. **ImageNet-V2** (Recht et al., 2019): Natural distribution shift, matched frequency subset (10,000 images)

#### 4.2 Baselines

We compare against state-of-the-art methods across three categories:

| Method | Type | Key Characteristic |
|--------|------|-------------------|
| Source | None | No adaptation |
| BN Adapt (Nado et al., 2020) | Batch statistic | Updates BN statistics from test batch |
| TENT (Wang et al., 2021) | Entropy minimization | Minimizes prediction entropy |
| EATA (Niu et al., 2022) | Entropy + filtering | Filters high-entropy samples |
| MEMO (Zhang et al., 2022) | Single-image + aug | Marginal entropy on augmented batch |
| T3A (Iwasawa & Matsuo, 2021) | Prototype | Online feature prototype adaptation |
| SWR+NSP (Choi et al., 2022) | Prototype + SWR | Nearest source prototype with regularization |
| DPL (Wang et al., 2024) | Prototype | Decoupled contrastive prototype learning |
| CoTTA (Wang et al., 2022) | Continual | Weight-averaged pseudo-labels |
| DPE (Zhang et al., 2024) | Prototype (VLM) | Dual prototype evolving for VLMs |

**Heuristic Baseline**: We additionally implement **Heuristic-APN**, a simple distance-based severity selection baseline (Section 3.3), to validate whether the learned Meta-APN provides benefits beyond simple heuristics.

**Note on SWR+NSP comparison**: SWR+NSP requires access to source data *before* deployment (offline) for generating prototypes and identifying shift-agnostic parameters, but does NOT require source data *at test time*. Our method uses the same source prototypes but learns augmentation policies. The comparison is fair on performance metrics (accuracy, mCE) since both methods operate under similar test-time constraints (no source data during inference).

#### 4.3 Evaluation Metrics

- **Mean Corruption Error (mCE)**: Error relative to AlexNet baseline (lower is better)
- **Top-1 Accuracy**: Direct classification accuracy
- **Average Inference Time**: Per-image adaptation time in milliseconds
- **Memory Overhead**: Additional parameters/storage required

#### 4.4 Statistical Significance

All experiments are run with **5 independent random seeds** (2022, 2023, 2024, 2025, 2026) for robust statistical testing, following best practices in TTA evaluation (Zhao et al., 2023). We report mean ± standard error and use paired t-tests with **p < 0.05** as the threshold for statistical significance.

#### 4.5 Expected Results

We expect APAC-TTA to:

1. **Outperform fixed-augmentation methods** (MEMO, T3A) by 2-4% on severe corruptions due to learned augmentation policies
2. **Exceed entropy-based methods** (TENT, EATA) by 3-5% by avoiding confirmation bias through prototype guidance
3. **Outperform heuristic baseline** (Heuristic-APN) by >1% justifying the complexity of learned policies
4. **Achieve competitive performance** on natural distribution shifts (CIFAR-10.1, ImageNet-V2)
5. **Scale to ImageNet-C** demonstrating applicability to high-resolution images
6. **Maintain efficiency** with <100ms per-image adaptation time

**Expected results on CIFAR-10-C (severity 5 average) will be reported as follows:**

- Improvement in mCE over MEMO baseline
- Comparison with TENT and EATA on entropy-based methods
- Performance comparison with SWR+NSP and DPL on prototype-based methods
- Comparison with Heuristic-APN to justify learned policy complexity
- Ablation studies showing contribution of learned policy vs. fixed policy
- Analysis of confidence weighting effectiveness

*Note: Specific numerical values will be obtained through actual experiments. The table format will follow standard TTA evaluation protocols from prior work (Wang et al., 2021; Zhang et al., 2022).*

### 5. Success Criteria

**Statistical Significance**: All performance comparisons will use paired t-tests with **p < 0.05** as the threshold for statistical significance. Results will be reported as mean ± standard error over **5 independent runs**.

**Primary Success (Confirming Hypothesis):**
- APAC-TTA achieves statistically significant improvement (p < 0.05) in mCE over MEMO on CIFAR-10-C severity 5 with effect size > 2% absolute improvement
- APAC-TTA achieves statistically significant improvement (p < 0.05) over Heuristic-APN by >1% mCE (justifying learned policy complexity)
- APAC-TTA achieves statistically significant improvement (p < 0.05) over SWR+NSP by >1.5% mCE (fair comparison; both use source prototypes but SWR+NSP requires additional offline source data access)
- APAC-TTA successfully scales to ImageNet-C with consistent improvements over MEMO baseline

**Secondary Success:**
- Clean accuracy maintained within 1% of source model on CIFAR-10.1 and ImageNet-V2
- Average inference time per image < 2× MEMO baseline
- Ablation study shows learned augmentation policy improves over fixed policy by >1% with statistical significance (p < 0.05)
- Confidence weighting improves performance over uniform weighting by >0.5% with statistical significance (p < 0.05)

**Failure Conditions** (would indicate hypothesis is wrong):
- APAC-TTA performs worse than simple MEMO baseline with statistical significance
- APAC-TTA shows no improvement over Heuristic-APN (learned policy unnecessary)
- Learned augmentation policy provides no benefit over random selection
- Prototype confidence weighting degrades performance

### 6. Resource Requirements and Timeline

Given computational constraints (1× A6000 48GB, ~8 hours):

| Phase | Time | Description |
|-------|------|-------------|
| Setup | 0.5h | Install dependencies, download datasets |
| Meta-APN Training | 1h | Train lightweight policy network on source data |
| Baseline Implementation | 1.5h | Implement MEMO, T3A, TENT baselines |
| APAC-TTA Implementation | 2h | Implement Meta-APN and adaptation loop |
| CIFAR-10/100-C Evaluation | 2.5h | Run all experiments on CIFAR benchmarks (5 seeds) |
| ImageNet-C Evaluation | 1h | Evaluate on ImageNet-C subset (5 corruption types × 5 seeds) |
| Analysis & Visualization | 0.5h | Generate figures, ablation studies, result tables |
| Contingency Buffer | 0.5h | Buffer for unexpected issues |

**Total Estimated Time**: ~9.5 hours (with 0.5-hour buffer)

All experiments use pre-trained models (no training from scratch), making this timeline feasible. ImageNet-C evaluation focuses on 5 representative corruption types for computational efficiency: Gaussian noise, shot noise, defocus blur, pixelate, and JPEG compression.

### 7. Contingency Plans

**If Meta-APN training fails to converge or provides no benefit:**

1. **Fallback to heuristic policy**: Use prototype distance to heuristically select augmentation severity (closer prototypes → milder augmentations, farther prototypes → stronger augmentations) without learned weights.

2. **Fixed corruption-specific policies**: Pre-define augmentation policies for common corruption types (noise, blur, weather) and select based on simple feature statistics.

3. **Simplified confidence weighting only**: Proceed with fixed augmentations but keep the confidence-weighted prototype consistency as the primary contribution.

4. **REINFORCE for discrete optimization**: If Gumbel-Softmax proves unstable, switch to REINFORCE with a learned baseline for policy optimization.

5. **Reduced scope**: Focus evaluation on CIFAR-10-C only, dropping CIFAR-100-C and ImageNet-C if time-constrained.

**If prototype confidence weighting degrades performance:**
- Investigate alternative confidence metrics (e.g., prediction margin, entropy of similarity distribution)
- Apply temperature scaling to confidence scores
- Use hard selection (top-k prototypes) instead of soft weighting

**If results do not meet success criteria:**
- Analyze failure cases to identify corruption types where the method struggles
- Report negative results with analysis of why prototype-aware augmentation did not help
- Discuss potential improvements for future work

### 8. Contributions

Our key contributions are:

1. **Learnable Augmentation Policies for TTA**: We introduce a meta-learned augmentation policy network for test-time adaptation that selects augmentations based on prototype similarity, enabling corruption-specific adaptation strategies. This differs from prior work using fixed augmentations (MEMO, DPE, DPL) or fixed-policy weight learning for calibration (Adaptive-TTA).

2. **Confidence-Weighted Prototype Guidance**: We propose a novel confidence scoring mechanism for source prototypes based on consistency across augmented views, enabling more reliable adaptation signals than equal prototype weighting.

3. **Heuristic Baseline Justification**: We implement and compare against a simple heuristic baseline (distance-based severity selection) to rigorously justify the complexity of learned policies.

4. **Comprehensive Evaluation**: We evaluate on standard benchmarks (CIFAR-10/100-C, ImageNet-C) and natural distribution shifts (CIFAR-10.1, ImageNet-V2), with thorough ablations validating each component, including 5 random seeds for statistical robustness.

5. **Practical Design**: Our method works with any pre-trained vision model without modification, requiring only lightweight meta-network training (~10K parameters) and compact prototype storage.

### 9. Limitations and Future Work

**Limitations:**
- Requires training Meta-APN on source data (one-time cost)
- Meta-APN may need dataset-specific tuning for very different domains
- Single-step adaptation may not capture complex multi-modal distribution shifts
- Discrete augmentation selection requires approximate gradients (Gumbel-Softmax or REINFORCE)

**Future Directions:**
- Extend to continual TTA setting with prototype updating
- Apply to semantic segmentation with spatial prototype representations
- Explore transformer-based architectures for the Meta-APN
- Investigate zero-shot transfer of learned augmentation policies across datasets
- Combine with contrastive learning approaches like DPL for enhanced prototype reliability

### 10. References

1. Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully test-time adaptation by entropy minimization. *ICLR*.

2. Niu, S., Wu, J., Zhang, Y., Wen, Y., Chen, Y., Zhao, P., & Tan, M. (2022). Efficient test-time adaptation with sample selection and regularization. *CVPR*.

3. Zhang, M., Levine, S., & Finn, C. (2022). Memo: Test time robustness via adaptation and augmentation. *NeurIPS*.

4. Iwasawa, Y., & Matsuo, Y. (2021). Test-time classifier adjustment module for model-agnostic domain generalization. *NeurIPS*.

5. **Choi, S., Yang, S., Choi, S., & Yun, S. (2022). Improving test-time adaptation via shift-agnostic weight regularization and nearest source prototypes. *ECCV***.

6. Zhang, C., Stepputtis, S., Sycara, K., & Xie, Y. (2024). Dual prototype evolving for test-time generalization of vision-language models. *NeurIPS*.

7. Sun, Y., et al. (2024). Program: Prototype graph model based pseudo-label learning for test-time adaptation. *ICLR*.

8. Lee, H., Lee, S., Jung, I., & Hong, S. (2024). Prototypical class-wise test-time adaptation. *Pattern Recognition Letters*.

9. Shu, M., Nie, W., Huang, D., Yu, Z., Goldstein, T., Anandkumar, A., & Xiao, C. (2022). Test-time prompt tuning for zero-shot generalization in vision-language models. *NeurIPS*.

10. Wang, Q., Fink, O., Van Gool, L., & Dai, D. (2022). Continual test-time domain adaptation. *CVPR*.

11. Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. *ICLR*.

12. Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). Do imagenet classifiers generalize to imagenet? *ICML*.

13. Nado, Z., Padhy, S., Sculley, D., D'Amour, A., Lakshminarayanan, B., & Snoek, J. (2020). Evaluating prediction-time batch normalization for robustness under covariate shift. *arXiv*.

14. Conde, P., & Premebida, C. (2022). Adaptive-TTA: accuracy-consistent weighted test time augmentation method for the uncertainty calibration of deep learning classifiers. *BMVC*.

15. **Wang, G., Ding, C., Tan, W., & Tan, M. (2024). Decoupled prototype learning for reliable test-time adaptation. *arXiv:2401.08703***.

16. Zhao, H., et al. (2023). On pitfalls of test-time adaptation. *ICML*.
