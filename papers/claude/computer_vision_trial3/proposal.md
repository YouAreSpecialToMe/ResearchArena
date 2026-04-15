# Spectral Token Gating: Training-Free Corruption Robustness for Vision Transformers via Frequency-Domain Token Analysis

## Introduction

### Context and Problem Statement

Vision Transformers (ViTs) have established themselves as the dominant architecture in computer vision, achieving state-of-the-art performance on image classification, object detection, and segmentation tasks (Dosovitskiy et al., 2021; Liu et al., 2021). However, despite their success on clean benchmarks, ViTs remain notably vulnerable to common image corruptions such as Gaussian noise, motion blur, fog, and JPEG compression artifacts (Hendrycks & Dietterich, 2019; Guo et al., 2023a). This vulnerability poses a significant barrier to deploying ViTs in safety-critical applications such as autonomous driving, medical imaging, and surveillance, where input images are routinely degraded by environmental conditions, sensor noise, or compression.

Recent work has identified a key mechanism underlying this vulnerability: **token overfocusing** (Guo et al., 2023b). The self-attention mechanism in ViTs tends to concentrate on a small number of important tokens, and these tokens are disproportionately affected by corruptions, leading to dramatically diverging attention patterns. Existing solutions to this problem require either retraining with specialized augmentation strategies (Wang et al., 2021; Guo et al., 2023a), architectural modifications (Zhou et al., 2022), or parameter updates at test time (Wang et al., 2021b). These approaches have significant practical limitations: retraining is expensive, architectural changes break compatibility with pretrained checkpoints, and test-time parameter updates can be unstable and require careful hyperparameter tuning.

### Key Insight

We observe that different types of image corruptions leave **distinct and detectable frequency-domain signatures** in the patch token embeddings of Vision Transformers. Noise corruptions elevate high-frequency energy in token embeddings, blur corruptions suppress it, and weather/digital corruptions create characteristic mid-frequency perturbation patterns. Crucially, these spectral signatures are systematic and predictable — they arise from the known frequency characteristics of the corruptions themselves as they propagate through the patch embedding projection.

This insight connects two previously separate lines of research: (1) the spectral analysis of ViT robustness (Kim & Lee, 2024), which showed ViTs rely predominantly on low-frequency and phase information; and (2) token reduction methods (Bolya et al., 2023), which demonstrated that selectively processing tokens can maintain accuracy with significant computational savings.

### Hypothesis

We hypothesize that by analyzing the frequency spectrum of patch token embeddings at inference time and comparing against clean-image reference statistics, we can identify tokens whose spectral profiles are anomalously distorted by corruption. By gating (downweighting) these corrupted tokens' contributions to self-attention, we can significantly improve corruption robustness without any retraining or parameter updates.

## Proposed Approach

### Overview

We propose **Spectral Token Gating (STG)**, a training-free, plug-and-play method that improves the corruption robustness of pretrained Vision Transformers. STG operates by:

1. **Spectral Analysis**: Computing the discrete Fourier transform (DFT) of each patch token's embedding vector at selected transformer layers
2. **Anomaly Detection**: Comparing per-token spectral energy distributions against reference statistics collected from a small calibration set of clean images
3. **Adaptive Gating**: Generating per-token gating scores that downweight tokens with anomalous spectral profiles
4. **Attention Modulation**: Applying the gating scores to modulate the value vectors in self-attention, suppressing corrupted tokens' influence on the output

### Method Details

#### Step 1: Calibration (Offline, One-Time)

Given a pretrained ViT and a small calibration set of clean images (e.g., 1000 images from the training set), we perform a single forward pass to collect reference spectral statistics:

- For each transformer layer $l$ and each token position $i$, compute the 1D-DFT of the token embedding $\mathbf{x}_i^l \in \mathbb{R}^d$:
  $$\hat{\mathbf{X}}_i^l = \text{DFT}(\mathbf{x}_i^l)$$
- Partition the frequency spectrum into $K$ bands (e.g., $K=3$: low, mid, high frequency)
- For each band $k$, compute the spectral energy: $E_k = \sum_{f \in \text{band}_k} |\hat{X}_f|^2$
- Compute the spectral energy ratio vector: $\mathbf{r}_i^l = [E_1/E_{\text{total}}, E_2/E_{\text{total}}, ..., E_K/E_{\text{total}}]$
- Aggregate statistics across the calibration set: mean $\boldsymbol{\mu}_{\text{ref}}^l$ and covariance $\boldsymbol{\Sigma}_{\text{ref}}^l$ of the spectral energy ratios at each layer

#### Step 2: Test-Time Spectral Analysis

At inference time, for each input image:
- At selected layers $l \in \mathcal{L}$, compute the spectral energy ratio $\mathbf{r}_i^l$ for each token $i$
- Compute the Mahalanobis distance from the clean reference:
  $$d_i^l = \sqrt{(\mathbf{r}_i^l - \boldsymbol{\mu}_{\text{ref}}^l)^\top (\boldsymbol{\Sigma}_{\text{ref}}^l)^{-1} (\mathbf{r}_i^l - \boldsymbol{\mu}_{\text{ref}}^l)}$$

#### Step 3: Gating Score Computation

Convert anomaly distances to gating scores using a soft sigmoid function:
$$g_i^l = \sigma(-\alpha(d_i^l - \tau))$$

where $\alpha$ controls the sharpness of the gating and $\tau$ is a threshold parameter (set based on the calibration set's distance distribution, e.g., the 95th percentile).

- Tokens with $d_i^l \ll \tau$ (spectrally normal) receive $g_i^l \approx 1$ (passed through)
- Tokens with $d_i^l \gg \tau$ (spectrally anomalous) receive $g_i^l \approx 0$ (suppressed)

#### Step 4: Attention Modulation

In the standard self-attention computation:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

We modulate the value vectors with the gating scores:
$$\text{STG-Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) \cdot \text{diag}(\mathbf{g}^l) \cdot V$$

This ensures that corrupted tokens contribute less to the attention output while preserving the attention structure for clean tokens. The CLS token is never gated.

### Key Innovations

1. **Training-free**: STG requires only a brief calibration pass on clean images (no gradient computation, no parameter updates)
2. **Plug-and-play**: Can be applied to any pretrained ViT without architectural modifications
3. **Frequency-domain corruption detection**: Exploits the fact that corruptions have characteristic spectral signatures in embedding space
4. **Selective gating**: Operates at the token level, preserving clean information while suppressing corrupted signals
5. **Minimal overhead**: The DFT computation and gating add less than 2% to inference FLOPs

## Related Work

### Vision Transformer Robustness

**FAN** (Zhou et al., 2022) proposed Fully Attentional Networks with channel self-attention for robustness, achieving 35.8% mCE on ImageNet-C. However, FAN requires training a new architecture from scratch. **RSPC** (Guo et al., 2023a) improves robustness by identifying vulnerable patches and aligning features between clean and corrupted images during training. **Robustifying Token Attention** (Guo et al., 2023b) addresses token overfocusing through Token-aware Average Pooling and Attention Diversification Loss. Both RSPC and the token attention work require retraining. Our method differs fundamentally by operating at test time without any parameter updates or retraining.

### Test-Time Adaptation

**Tent** (Wang et al., 2021b) adapts models at test time by minimizing prediction entropy, updating batch normalization parameters. While effective, Tent requires careful batch handling, can be unstable with small batches, and may catastrophically fail under certain distribution shifts. Our method avoids parameter updates entirely, instead using a statistics-based approach that is inherently more stable.

### Token Reduction and Efficiency

**ToMe** (Bolya et al., 2023) merges similar tokens for efficiency, achieving 2x throughput with minimal accuracy loss. **PiToMe** (Tran et al., 2024) extends this with spectrum-preserving merging. These methods focus on efficiency for clean images, not robustness. We draw inspiration from the token selection literature but apply it to a fundamentally different problem: identifying and suppressing corrupted (not redundant) tokens.

### Spectral Analysis of Vision Transformers

**Kim & Lee (2024)** explored adversarial robustness of ViTs in the spectral domain, finding that ViTs rely more on low-frequency and phase information. **SpectFormer** (Patro et al., 2025) combines spectral layers with attention layers for improved classification. These works analyze or leverage frequency properties but do not use frequency-domain token embedding analysis for corruption detection and mitigation at test time.

### Data Augmentation for Robustness

**AugMax** (Wang et al., 2021a) proposes adversarial composition of random augmentations with DuBIN normalization. While highly effective, it requires complete retraining and specialized normalization layers. Our approach is complementary — STG can be applied on top of models trained with any augmentation strategy.

## Experiments

### Planned Setup

**Models**: We evaluate STG on three pretrained ViT architectures from the timm library:
- DeiT-Small (22M params, patch size 16)
- DeiT-Base (86M params, patch size 16)
- Swin-Tiny (28M params, window size 7)

**Benchmarks**:
- **ImageNet-C** (Hendrycks & Dietterich, 2019): 15 corruption types at 5 severity levels, measuring mean Corruption Error (mCE)
- **ImageNet-P** (Hendrycks & Dietterich, 2019): Perturbation robustness, measuring mean Flip Probability (mFP)
- **Clean ImageNet**: Verify that STG does not degrade clean accuracy

**Baselines**:
- Vanilla (unmodified pretrained models)
- Tent (Wang et al., 2021b): Test-time entropy minimization
- FAN (Zhou et al., 2022): Trained robust architecture (for reference, not apples-to-apples comparison)
- AugMax (Wang et al., 2021a): Training-time augmentation (for reference)

**Calibration**: 1000 randomly sampled clean ImageNet training images

### Metrics
- **mCE** (mean Corruption Error): Primary metric on ImageNet-C, lower is better
- **Clean Top-1 Accuracy**: Verify no degradation on clean images
- **Per-corruption accuracy**: Analyze which corruption types benefit most from STG
- **mFP** (mean Flip Probability): Perturbation consistency on ImageNet-P

### Planned Experiments

1. **Main Results Table**: mCE on ImageNet-C across all models and baselines. Hypothesis: STG reduces mCE by 3-8% relative to vanilla models.

2. **Per-Corruption Analysis**: Accuracy per corruption type and severity. Hypothesis: STG shows largest gains on noise-type corruptions (which have the most distinctive spectral signatures) and moderate gains on blur/weather/digital.

3. **Spectral Signature Analysis**: Visualize the spectral energy distributions of token embeddings under different corruptions, confirming that corruptions create detectable spectral anomalies. This analysis itself is a contribution.

4. **Ablation Studies**:
   - Number of frequency bands ($K = 2, 3, 4, 5$)
   - Which layers to apply STG (early, middle, late, all)
   - Gating function (sigmoid, hard threshold, linear)
   - Calibration set size (100, 500, 1000, 5000)
   - Threshold $\tau$ sensitivity

5. **Gating Visualization**: Show attention maps and gating scores for corrupted images, demonstrating that STG correctly identifies and suppresses corrupted regions.

6. **Computational Overhead**: Measure FLOPs and wall-clock time overhead of STG.

7. **Combination with Training-Time Methods**: Apply STG on top of FAN and AugMax-trained models to show complementary improvements.

### Computational Feasibility

All experiments are feasible on a single NVIDIA RTX A6000 (48GB VRAM) within 8 hours:
- Calibration: ~5 minutes per model (1000 images, single forward pass)
- ImageNet-C evaluation: ~2 hours per model (75 corruption variants, forward passes only)
- Ablations: ~2 hours (subset evaluations)
- Analysis and visualization: ~30 minutes
- Total: ~3 models × 2h + 2.5h ablations + 0.5h analysis ≈ 9h → feasible with efficient parallelization and subset evaluation for ablations

## Success Criteria

### Primary Success (hypothesis confirmed)
- STG improves mCE by ≥3% relative to vanilla models on at least 2 of 3 architectures
- STG does not degrade clean accuracy by more than 0.5%
- Spectral analysis clearly shows distinct frequency signatures per corruption type

### Secondary Success
- STG shows complementary gains when combined with training-time methods (FAN, AugMax)
- The method generalizes across ViT architectures (both columnar DeiT and hierarchical Swin)
- Gating visualizations qualitatively demonstrate that STG correctly identifies corrupted regions

### Hypothesis Refutation Criteria
- If spectral signatures of corruptions are not reliably detectable in token embedding space (high overlap between clean and corrupted distributions), the core premise fails
- If gating corrupted tokens hurts accuracy even when corruption is correctly detected (because the model relies on corrupted features for classification), the method fails
- If clean-accuracy degradation exceeds 1%, the approach is too aggressive

## References

1. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

2. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jegou, H. (2021). Training data-efficient image transformers & distillation through attention. *ICML 2021*.

3. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV 2021*.

4. Hendrycks, D. & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. *ICLR 2019*.

5. Zhou, D., Yu, Z., Xie, E., Xiao, C., Anandkumar, A., Feng, J., & Alvarez, J.M. (2022). Understanding The Robustness in Vision Transformers. *ICML 2022*.

6. Guo, Y., Stutz, D., & Schiele, B. (2023a). Improving Robustness of Vision Transformers by Reducing Sensitivity to Patch Corruptions. *CVPR 2023*.

7. Guo, Y., Stutz, D., & Schiele, B. (2023b). Robustifying Token Attention for Vision Transformers. *ICCV 2023*.

8. Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021b). Tent: Fully Test-Time Adaptation by Entropy Minimization. *ICLR 2021*.

9. Wang, H., Xiao, C., Kossaifi, J., Yu, Z., Anandkumar, A., & Wang, Z. (2021a). AugMax: Adversarial Composition of Random Augmentations for Robust Training. *NeurIPS 2021*.

10. Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., & Hoffman, J. (2023). Token Merging: Your ViT But Faster. *ICLR 2023*.

11. Tran, H.-C., Nguyen, D.M.H., Nguyen, D.M., Nguyen, T., Le, N., Xie, P., Sonntag, D., Zou, J., Nguyen, B.T., & Niepert, M. (2024). Accelerating Transformers with Spectrum-Preserving Token Merging. *NeurIPS 2024*.

12. Kim, G. & Lee, J.-S. (2024). Exploring Adversarial Robustness of Vision Transformers in the Spectral Perspective. *WACV 2024*.

13. Patro, B.N., Agneeswaran, V.S., & Namboodiri, V.P. (2025). SpectFormer: Frequency and Attention is What You Need in a Vision Transformer. *WACV 2025*.
