# Attention Tells When It Doesn't Know: Layerwise Entropy Profiles for Training-Free OOD Detection and Calibration in Vision Transformers

## Introduction

### Context and Problem Statement

Vision Transformers (ViTs) have become the dominant architecture for image classification, achieving state-of-the-art results across a wide range of benchmarks. However, deploying ViTs in safety-critical applications (autonomous driving, medical imaging, industrial inspection) requires reliable confidence estimates — the model must know when it doesn't know. Two critical aspects of reliability are:

1. **Out-of-Distribution (OOD) Detection**: Identifying inputs that differ substantially from the training distribution, where the model's predictions are unreliable.
2. **Confidence Calibration**: Ensuring that predicted probabilities align with actual correctness likelihoods, especially under distribution shift.

Current OOD detection methods for ViTs predominantly rely on output-level signals — softmax probabilities (MSP), energy scores, or penultimate-layer features (Mahalanobis distance, ViM, KNN). These methods treat the transformer as a black box and ignore the rich intermediate representations computed during inference. In particular, they completely overlook the **self-attention maps**, which encode how the model distributes its computational focus across spatial tokens at each layer.

Meanwhile, post-hoc calibration methods (temperature scaling, histogram binning) are typically fitted on in-distribution validation data and **degrade significantly under distribution shift** — precisely when reliable calibration is most needed (Minderer et al., 2021; recent findings in "Beyond Overconfidence," 2025).

### Key Insight

We observe that the **self-attention entropy profile** — a vector capturing how attention entropy varies across layers and heads — serves as a direct, interpretable signature of how a ViT processes an input. For in-distribution images, the model follows a characteristic processing trajectory: early layers attend broadly (high entropy), while later layers progressively focus on discriminative regions (decreasing entropy). Under distribution shift or for OOD inputs, this trajectory is disrupted: the model either fails to focus (persistently high entropy) or focuses on irrelevant features (anomalous entropy patterns).

### Hypothesis

We hypothesize that deviations in the layerwise attention entropy profile from in-distribution statistics provide a powerful, training-free signal for both OOD detection and confidence calibration in Vision Transformers. Specifically:

1. The Mahalanobis distance of a test input's attention entropy profile from in-distribution statistics achieves competitive OOD detection performance compared to state-of-the-art post-hoc methods.
2. Using this distance to modulate predicted confidence yields better calibration under distribution shift than standard post-hoc calibration methods.
3. The attention entropy signal is **complementary** to existing logit/feature-based OOD scores, and combining them improves performance.

## Proposed Approach

### Overview

We propose **AEP (Attention Entropy Profiling)**, a training-free, post-hoc method that extracts OOD detection scores and calibration signals from the self-attention maps of any pretrained Vision Transformer. AEP requires only a small in-distribution calibration set (e.g., 1,000 images) to compute reference statistics and adds zero computational overhead at inference time, since attention maps are already computed during the standard forward pass.

### Method Details

#### Step 1: Attention Entropy Extraction

Given a pretrained ViT with $L$ layers and $H$ attention heads per layer, for each input image $x$ we extract the attention map $A_{l,h} \in \mathbb{R}^{N \times N}$ at layer $l$ and head $h$, where $N$ is the number of tokens (patches + [CLS]).

For each attention map, we compute the following statistics:

1. **CLS Attention Entropy**: The entropy of the attention distribution from the [CLS] token to all patch tokens:
   $$E^{\text{cls}}_{l,h} = -\sum_{j=1}^{N-1} A_{l,h}[0,j] \log A_{l,h}[0,j]$$
   This measures how focused the model's classification-relevant attention is.

2. **Average Token Entropy**: The mean entropy across all token-to-token attention distributions:
   $$E^{\text{avg}}_{l,h} = \frac{1}{N}\sum_{i=0}^{N-1}\left(-\sum_{j=0}^{N-1} A_{l,h}[i,j] \log A_{l,h}[i,j]\right)$$

3. **Attention Concentration Ratio**: The fraction of total attention mass captured by the top-$k$ most-attended tokens (averaged over query tokens):
   $$C_{l,h} = \frac{1}{N}\sum_{i=0}^{N-1} \sum_{j \in \text{top-}k(A_{l,h}[i,:])} A_{l,h}[i,j]$$

4. **Head Agreement Score**: The average pairwise cosine similarity between attention distributions of different heads at the same layer:
   $$S_l = \frac{2}{H(H-1)}\sum_{h<h'} \cos(A_{l,h}[0,:], A_{l,h'}[0,:])$$

#### Step 2: Entropy Profile Construction

We aggregate the per-layer, per-head statistics into a compact **Attention Entropy Profile (AEP)** vector:

$$\mathbf{p}(x) = \left[\bar{E}^{\text{cls}}_1, \sigma(E^{\text{cls}}_1), \bar{E}^{\text{avg}}_1, \bar{C}_1, S_1, \ldots, \bar{E}^{\text{cls}}_L, \sigma(E^{\text{cls}}_L), \bar{E}^{\text{avg}}_L, \bar{C}_L, S_L\right]$$

where $\bar{\cdot}$ and $\sigma(\cdot)$ denote the mean and standard deviation over heads at each layer. This yields a profile of dimensionality $5L$ (e.g., 60 for a 12-layer ViT).

#### Step 3: In-Distribution Reference Statistics

Using a small calibration set $\mathcal{D}_{\text{cal}}$ of in-distribution images (e.g., 1,000 random samples from the training set), we compute:

- The mean profile: $\boldsymbol{\mu} = \frac{1}{|\mathcal{D}_{\text{cal}}|}\sum_{x \in \mathcal{D}_{\text{cal}}} \mathbf{p}(x)$
- The covariance matrix: $\boldsymbol{\Sigma} = \text{Cov}(\{\mathbf{p}(x)\}_{x \in \mathcal{D}_{\text{cal}}})$

Optionally, we also compute class-conditional statistics $(\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$ for each class $c$.

#### Step 4: OOD Scoring

For a test input $x_{\text{test}}$, the OOD score is the Mahalanobis distance of its AEP from the ID distribution:

$$\text{AEP-Score}(x_{\text{test}}) = (\mathbf{p}(x_{\text{test}}) - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{p}(x_{\text{test}}) - \boldsymbol{\mu})$$

Higher scores indicate greater deviation from in-distribution processing patterns, signaling potential OOD inputs.

#### Step 5: Confidence Calibration

We use the AEP score to modulate predicted confidence:

$$\hat{p}_{\text{cal}}(y|x) = \frac{\hat{p}(y|x)^{1/T(x)}}{\sum_{y'} \hat{p}(y'|x)^{1/T(x)}}$$

where the input-adaptive temperature is:

$$T(x) = T_0 + \alpha \cdot \text{AEP-Score}(x)$$

The parameters $T_0$ and $\alpha$ are fitted on the calibration set to minimize ECE. The key advantage over standard temperature scaling is that $T(x)$ is **input-dependent**: inputs with anomalous attention patterns (high AEP score) receive higher temperature (softer predictions), while normal inputs retain sharper predictions.

#### Step 6: Score Fusion (Optional)

Since AEP captures attention-level information while existing methods capture logit/feature-level information, they are complementary. We propose a simple fusion:

$$\text{Fused-Score}(x) = \beta \cdot \text{AEP-Score}(x) + (1-\beta) \cdot \text{Baseline-Score}(x)$$

where Baseline-Score is any existing OOD score (e.g., Energy, ViM) and $\beta$ is tuned on the calibration set.

### Key Innovations

1. **First systematic use of layerwise attention entropy profiles for OOD detection in ViTs.** While attention entropy has been studied for LLM correctness prediction (Head Entropy, 2025) and interpretability (Entropy-Lens, 2025), it has not been applied to OOD detection or calibration in vision transformers.

2. **Training-free and zero-overhead.** Unlike CalAttn (2026), which requires learning additional parameters, AEP uses only statistics of already-computed attention maps. Unlike Mahalanobis distance on features, AEP captures the model's *processing dynamics* rather than just the final representation.

3. **Input-adaptive calibration that is robust to distribution shift.** Standard post-hoc calibration uses a global temperature; AEP scales temperature per-input based on how the model's attention behaves, making it naturally adaptive to distribution shift.

4. **Complementary to existing methods.** AEP captures an orthogonal information source (attention structure vs. logit/feature magnitudes), enabling effective score fusion.

## Related Work

### Post-hoc OOD Detection

The dominant paradigm for OOD detection is post-hoc scoring on pretrained classifiers. **MSP** (Hendrycks & Gimpel, 2017) uses the maximum softmax probability. **ODIN** (Liang et al., 2018) adds temperature scaling and input perturbation. **Energy Score** (Liu et al., 2020) uses the log-sum-exp of logits. **Mahalanobis Distance** (Lee et al., 2018) and its recent improvement **Mahalanobis++** (2025) use feature-space distances. **ViM** (Wang et al., 2022) projects features onto a virtual logit dimension. **KNN** (Sun et al., 2022) uses k-nearest neighbor distances in feature space. **ASH** (Djurisic et al., 2023) prunes and scales activations. **GradNorm** (Huang et al., 2021) uses gradient magnitudes.

All of these methods operate on output logits or penultimate-layer features. **None leverage the self-attention maps** that are unique to transformer architectures. Our AEP method is the first to exploit layerwise attention statistics for OOD detection in ViTs, providing a signal that is complementary to all existing approaches.

### Confidence Calibration

**Temperature Scaling** (Guo et al., 2017) fits a single scalar temperature on validation data. **Platt Scaling** (Platt, 1999) fits a logistic regression on logits. Recent work has shown that these methods degrade under distribution shift (Ovadia et al., 2019). **CalAttn** (2026) learns a representation-aware calibration module that predicts instance-specific temperatures from the [CLS] token — the closest work to ours in spirit, but requiring supervised training. Our method achieves input-adaptive calibration without any training.

### Attention Analysis in Transformers

**Entropy-Lens** (2025) uses the information-theoretic signature of intermediate predictions in LLMs for interpretability. **Head Entropy** (2025) shows that attention head entropy predicts answer correctness in LLMs. **SVDA** (2025) introduces spectral analysis of ViT attention for interpretability. These works analyze attention for understanding but do not use it as a practical OOD detection or calibration signal in vision models.

### Token Pruning and Efficiency

**ToMe** (Bolya et al., 2023), **A-ViT** (Yin et al., 2022), and **TCA** (Wang et al., ICCV 2025) use token importance metrics for efficient inference. TCA notably uses token condensation for test-time adaptation, which is related but serves a different purpose (accuracy under shift vs. OOD detection and calibration). Our work complements these by showing that attention statistics useful for efficiency also encode uncertainty information.

## Experiments

### Experimental Setup

**Models** (pretrained, no training required):
- DeiT-Small (22M params, 12 layers, 6 heads)
- DeiT-Base (86M params, 12 layers, 12 heads)
- ViT-Base/16 (86M params, 12 layers, 12 heads)
- Swin-Tiny (28M params, adapted for windowed attention)

All models are loaded from `timm` with ImageNet-1K pretrained weights.

**In-Distribution Data**: ImageNet-1K validation set (50,000 images). We use 1,000 randomly selected images for calibration statistics and report results on the remaining 49,000.

**OOD Detection Benchmarks** (following OpenOOD v1.5):
- Near-OOD: SSB-hard, NINCO
- Far-OOD: iNaturalist, Textures (DTD), OpenImage-O
- Covariate shift: ImageNet-C (19 corruptions × 5 severity levels), ImageNet-R, ImageNet-Sketch

**Calibration Benchmarks**:
- ImageNet-1K (in-distribution)
- ImageNet-C at severity 1, 3, 5 (increasing corruption)
- ImageNet-R (renditions)

**OOD Detection Metrics**: AUROC, AUPR (in and out), FPR@95% TPR

**Calibration Metrics**: Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score

### Baselines

**OOD Detection**: MSP, ODIN, Energy Score, Mahalanobis Distance, ViM, KNN, ASH, GradNorm

**Calibration**: No calibration (raw softmax), Temperature Scaling, Histogram Binning, Isotonic Regression

### Experiments

1. **Main OOD Detection Results**: Compare AEP-Score against all baselines on near-OOD and far-OOD benchmarks across all four model architectures. Report AUROC, AUPR, FPR@95.

2. **Calibration Under Distribution Shift**: Compare AEP-based adaptive temperature against global temperature scaling on ImageNet-C at multiple severity levels and ImageNet-R. Report ECE and Brier Score.

3. **Score Fusion**: Combine AEP-Score with the best-performing logit/feature-based method and show improved OOD detection.

4. **Ablation Studies**:
   - Which AEP components matter most? (CLS entropy, average entropy, concentration, head agreement)
   - Which layers are most informative? (early, middle, late, or all)
   - How many calibration samples are needed? (100, 500, 1000, 5000)
   - Class-conditional vs. class-agnostic statistics
   - Sensitivity to the number of top-k tokens for concentration ratio

5. **Attention Entropy Visualization**: Visualize how the layerwise entropy profile differs between ID, near-OOD, and far-OOD inputs to provide interpretability and validate the core hypothesis.

6. **Computational Cost Analysis**: Measure the wall-clock overhead of AEP extraction (expected to be <1% of forward pass time since attention maps are already computed).

### Expected Results

Based on our hypothesis and the observed properties of attention in ViTs:

- **Far-OOD**: AEP should perform well because semantically different inputs (textures, iNaturalist) will produce fundamentally different attention patterns.
- **Near-OOD**: AEP may be weaker alone but should provide complementary signal when fused with feature-based methods.
- **Calibration**: AEP-based adaptive temperature should outperform global temperature scaling under distribution shift, since it adjusts confidence per-input.
- **Layers**: We expect middle-to-late layers to be most informative, consistent with findings from Head Entropy in LLMs.

## Success Criteria

### Confirms Hypothesis
- AEP-Score achieves AUROC within 2% of the best baseline on at least 2 of 5 OOD benchmarks
- AEP-Score + fusion outperforms the best individual baseline on average across benchmarks
- AEP-based calibration reduces ECE by ≥10% relative to temperature scaling on ImageNet-C severity 5
- Attention entropy profiles show statistically significant differences between ID and OOD distributions (p < 0.01, two-sample t-test)

### Refutes Hypothesis
- AEP-Score AUROC is >5% below the best baseline consistently across all benchmarks
- No improvement from fusion with existing methods
- Entropy profiles do not significantly differ between ID and OOD inputs

## Computational Feasibility

| Component | Estimated Time | GPU Memory |
|-----------|---------------|------------|
| ID statistics (1K images, 4 models) | ~15 min | <8 GB |
| OOD detection eval (50K ID + ~100K OOD) | ~3 hours | <8 GB |
| Calibration eval (ImageNet-C, 5 severities) | ~2 hours | <8 GB |
| Ablation studies | ~1.5 hours | <8 GB |
| Visualization and analysis | ~30 min | <4 GB |
| **Total** | **~7.5 hours** | **<8 GB peak** |

All experiments involve only forward passes through pretrained models — no training, no backpropagation. The A6000 (48 GB) provides ample memory. The experiments are highly parallelizable if needed.

## References

1. Hendrycks, D. & Gimpel, K. "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks." ICLR, 2017.
2. Liang, S., Li, Y., & Srikant, R. "Enhancing the Reliability of Out-of-distribution Image Detection in Neural Networks." ICLR, 2018.
3. Liu, W., Wang, X., Owens, J., & Li, Y. "Energy-based Out-of-distribution Detection." NeurIPS, 2020.
4. Lee, K., Lee, K., Lee, H., & Shin, J. "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks." NeurIPS, 2018.
5. Wang, H., Li, Z., Feng, L., & Zhang, W. "ViM: Out-Of-Distribution with Virtual-logit Matching." CVPR, 2022.
6. Sun, Y., Ming, Y., Zhu, X., & Li, Y. "Out-of-distribution Detection with Deep Nearest Neighbors." ICML, 2022.
7. Djurisic, A., Bae, G., Lie, W.S., & Liu, Z. "Extremely Simple Activation Shaping for Out-of-Distribution Detection." ICLR, 2023.
8. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. "On Calibration of Modern Neural Networks." ICML, 2017.
9. Ovadia, Y., Fertig, E., Ren, J., et al. "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift." NeurIPS, 2019.
10. Minderer, M., Djolonga, J., Romijnders, R., et al. "Revisiting the Calibration of Modern Neural Networks." NeurIPS, 2021.
11. Zhang, J., Yang, J., Shao, S., & Li, Z. "OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection." DMLR, 2024.
12. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR, 2021.
13. Touvron, H., Cord, M., Douze, M., et al. "Training data-efficient image transformers & distillation through attention." ICML, 2021.
14. Liu, Z., Lin, Y., Cao, Y., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV, 2021.
15. Bolya, D., Fu, C.Y., Dai, X., et al. "Token Merging: Your ViT But Faster." ICLR, 2023.
16. Wang, Z., Gong, D., Wang, S., Huang, Z., & Luo, Y. "Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation." ICCV, 2025.
17. Yoon, H.S., et al. "C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion." ICLR, 2024.
18. Sharifdeen, et al. "O-TPT: Orthogonality Constraints for Calibrating Test-time Prompt Tuning in Vision-Language Models." CVPR, 2025.
19. Yin, H. & Vahdat, A. "A-ViT: Adaptive Tokens for Efficient Vision Transformer." CVPR, 2022.
20. Hendrycks, D. & Dietterich, T. "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations." ICLR, 2019.
21. Huang, R., Geng, A., & Li, Y. "On the Importance of Gradients for Detecting Distributional Shifts in Visual Tasks." NeurIPS, 2021.
22. Hendrycks, D., Basart, S., Mu, N., et al. "The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization." ICCV, 2021.
