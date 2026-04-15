# Consistency-Driven Adaptive Depth for Diffusion Transformers

## Research Proposal

---

## 1. Introduction

### 1.1 Context and Motivation

Diffusion Transformers (DiTs) have emerged as the dominant architecture for high-quality image generation, but their computational cost remains prohibitive for real-world deployment. A single image generation requires 20-50 denoising steps, each involving a full forward pass through 20+ transformer blocks. This creates a critical efficiency bottleneck.

Current acceleration approaches fall into two categories:
1. **Training-free methods** (DeepCache, Δ-DiT, TeaCache): Apply uniform strategies—caching features, skipping blocks on fixed schedules, or reducing steps. These treat all spatial positions identically, missing opportunities for input-dependent optimization.
2. **Training-based methods** (DiffRatio-MoD, E-DiT): Learn routers to dynamically allocate computation, but require expensive fine-tuning on large datasets.

A critical gap remains: **there is no training-free method that adapts computation depth at the per-token level based on input content.**

### 1.2 Key Insight

We observe that different image regions exhibit different convergence behaviors *within* transformer layers at each denoising step. Simple backgrounds (e.g., blue sky) reach stable predictions after just a few transformer layers, while complex textures (e.g., fur, hair) require deeper processing. The model's own prediction consistency—how much a token's output changes across consecutive layers—serves as a natural signal for convergence.

**Our insight**: We can use the model's prediction consistency as a training-free signal to dynamically determine when each spatial token has received sufficient computation. This enables spatially-varying adaptive depth without any learned components.

### 1.3 Problem Statement

Existing training-free methods cannot allocate computation adaptively across spatial positions:
- **Global early exit** (DeeDiff, ASE) makes the same depth decision for all tokens
- **Token pruning** (ToCa, RAS) eliminates tokens entirely, losing information
- **Fixed schedules** (Δ-DiT) apply uniform layer skipping regardless of content

Meanwhile, training-based methods like DiffRatio-MoD achieve spatial adaptation through learned routers, but require:
- Joint fine-tuning with the base model
- Additional training data and compute
- Modification of model weights

**We ask**: Can we achieve training-free, per-token adaptive depth allocation using only signals already present in the model's forward pass?

### 1.4 Hypothesis

> *A training-free consistency metric—measuring prediction stability across consecutive transformer layers—can serve as a reliable signal for per-token adaptive depth in Diffusion Transformers, enabling spatially-heterogeneous computation that reduces per-step FLOPs by 30-50% while maintaining generation quality.*

---

## 2. Proposed Approach

### 2.1 Overview

We propose **CAD-DiT (Consistency-Adaptive Depth for Diffusion Transformers)**, a training-free acceleration framework that enables per-token adaptive depth using only the model's own prediction signals. CAD-DiT introduces:

1. **Prediction Consistency Metric (PCM)**: A training-free measure of token-level convergence based on prediction stability across layers
2. **Token-Conditional Forward Pass**: A modified inference loop where each token conditionally continues or exits based on its PCM score
3. **Timestep-Aware Thresholding**: Dynamic consistency thresholds that adapt to denoising progress

### 2.2 Method Details

#### 2.2.1 Prediction Consistency Metric (PCM)

For each token $i$ at layer $l$ of denoising step $t$, we compute consistency by comparing predictions across consecutive layers:

$$\text{PCM}_i^{l,t} = 1 - \frac{\|\hat{\epsilon}_i^l - \hat{\epsilon}_i^{l-1}\|}{\|\hat{\epsilon}_i^{l-1}\| + \delta}$$

where:
- $\hat{\epsilon}_i^l$ is the predicted noise/velocity for token $i$ from layer $l$
- $\delta$ is a small constant for numerical stability
- Higher PCM indicates the token's prediction has stabilized

**Why this works**: As a token passes through transformer layers, its prediction initially changes rapidly (low consistency) and then stabilizes (high consistency). The layer at which stabilization occurs varies by token complexity.

#### 2.2.2 Token-Conditional Forward Pass

Standard DiT processes all tokens through all $L$ layers. CAD-DiT modifies this:

```
For each timestep t:
  Initialize active_set = all tokens
  For layer l = 1 to L:
    Process active_set through layer l
    For each token i in active_set:
      Compute PCM_i^l
      If PCM_i^l > τ_l(t):
        Mark token i as "converged"
        Freeze its prediction at current value
        Remove from active_set
    If active_set is empty: break
  Aggregate predictions from all tokens
```

Converged tokens are excluded from subsequent layer computations, reducing FLOPs.

#### 2.2.3 Timestep-Aware Thresholding

The consistency threshold $\tau_l(t)$ varies by timestep:
- **Early timesteps** ($t \approx T$): Lower threshold (more aggressive exit) since noise dominates
- **Late timesteps** ($t \approx 0$): Higher threshold (more conservative) for detail refinement

$$\tau_l(t) = \tau_{\text{base}} \cdot \left(1 - \alpha \cdot \frac{t}{T}\right)$$

This reflects the intuition that early denoising is dominated by coarse structure (amenable to early exit), while late denoising requires fine detail preservation.

### 2.3 Key Innovations

1. **Training-Free Per-Token Adaptation**: First method to achieve spatially-varying depth without learned routers, exit heads, or fine-tuning
2. **No Additional Parameters**: Uses only signals already present in the model's forward pass
3. **Content-Dependent**: Adaptation is determined by each token's actual convergence behavior, not predefined schedules
4. **Orthogonal**: Can combine with timestep reduction (distillation), caching, and quantization

### 2.4 Differentiation from Related Work

| Method | Training-Free | Spatial Adaptation | Layer Adaptation | Timestep Adaptation |
|--------|--------------|-------------------|------------------|---------------------|
| DeepCache | ✓ | ✗ | ✗ | ✓ |
| Δ-DiT | ✓ | ✗ | ✓ (fixed) | ✓ |
| DeeDiff | ✓ | ✗ | ✗ | ✓ |
| DiffRatio-MoD | ✗ (requires fine-tuning) | ✓ | ✓ | ✓ |
| **CAD-DiT (ours)** | **✓** | **✓** | **✓** | **✓** |

**Key distinction from DiffRatio-MoD**: DiffRatio-MoD achieves spatial adaptation through learned routers that require joint fine-tuning with the base model. CAD-DiT achieves similar spatial adaptation through a training-free consistency heuristic, making it applicable to any pre-trained DiT without modification.

---

## 3. Related Work

### 3.1 Training-Free Acceleration

**DeepCache** (Ma et al., CVPR 2024) caches high-level features across timesteps but uses full model depth per timestep. CAD-DiT reduces per-timestep computation through adaptive depth.

**Δ-DiT** (Chen et al., 2024) uses a fixed schedule to skip blocks based on timestep, applying the same depth to all tokens and positions. CAD-DiT adapts depth per-token based on content.

**ASE** (Moon et al., 2024) applies timestep-aware early exiting globally—all tokens exit at the same layer. CAD-DiT enables spatially-heterogeneous exit depths.

### 3.2 Training-Based Adaptive Methods

**DiffRatio-MoD** (CVPR 2025) is the closest related work. It introduces token-level routing with learned routers that are jointly fine-tuned with model weights. While effective, this requires:
- Fine-tuning on large datasets
- Architectural modifications (router networks)
- Significant additional training compute

**Difference**: CAD-DiT provides similar spatial adaptation through a training-free consistency metric. This is positioned as an alternative for scenarios where fine-tuning is impractical (proprietary models, limited compute/data).

### 3.3 Token Reduction Methods

**ToCa/RAS** prune (completely skip) certain tokens at each step. CAD-DiT differs by updating all tokens—just with variable depth. This avoids information loss from pruning.

### 3.4 Early Exit in Other Domains

Token-level early stopping has been explored in diffusion language models (Jot, 2025) using confidence metrics. CAD-DiT extends this concept to image generation DiTs with prediction consistency across transformer layers.

---

## 4. Experiments

### 4.1 Experimental Setup

**Base Models**:
- DiT-XL/2 (ImageNet 256×256, class-conditional)
- PixArt-α (text-to-image, 1024×1024)

**Datasets**:
- ImageNet 256×256 validation set (10K samples for FID)
- COCO 2014 captions (30K samples)

**Evaluation Metrics**:
- **FID** (Fréchet Inception Distance): Image quality
- **IS** (Inception Score): Sample diversity
- **CLIP Score**: Text-image alignment
- **Computational Cost**: FLOPs per timestep, total FLOPs, wall-clock time
- **Exit Statistics**: Average exit layer per timestep, spatial distribution

**Baselines**:
1. Original full-model inference (50 steps DDIM)
2. DeepCache (caching baseline)
3. Δ-DiT (fixed layer skipping)
4. DeeDiff (global early exit)
5. DiffRatio-MoD (training-based spatial adaptation—cited for comparison)

### 4.2 Expected Results

**Primary Result**: CAD-DiT achieves 30-50% per-step FLOP reduction with <3 FID degradation compared to full inference.

**Ablation Studies**:
1. Effect of base consistency threshold $\tau_{\text{base}}$
2. Effect of timestep modulation parameter $\alpha$
3. Comparison with static per-token schedules (oracle baseline)
4. Combination with DeepCache

**Analysis Experiments**:
1. **Spatial patterns**: Which image regions exit early vs. late?
2. **Timestep patterns**: Does early exit concentrate in early/late timesteps?
3. **Failure cases**: When does adaptive depth hurt quality?

### 4.3 Discussion: Relationship to DiffRatio-MoD

We explicitly acknowledge DiffRatio-MoD (CVPR 2025) as concurrent/complementary work:

**DiffRatio-MoD advantages**:
- Learned routers may achieve better quality-efficiency trade-offs after fine-tuning
- Can achieve higher compression ratios with training

**CAD-DiT advantages**:
- No training required—works out-of-the-box on any pre-trained DiT
- Applicable to proprietary models where fine-tuning is impossible
- Lower barrier to adoption (no training infrastructure needed)

**Positioning**: CAD-DiT is a practical alternative when fine-tuning is infeasible, not a replacement for learned approaches.

---

## 5. Success Criteria

### 5.1 Confirmation

The hypothesis is **confirmed** if:
- CAD-DiT achieves ≥30% per-step FLOP reduction with FID increase <3 points on ImageNet 256×256
- Spatial patterns show meaningful variation (e.g., background tokens exit earlier than foreground)
- The method works across multiple DiT architectures without any retraining
- Qualitative visual quality is comparable to full model

### 5.2 Refutation

The hypothesis is **refuted** if:
- FID degradation exceeds 5 points at 30% FLOP reduction
- Prediction consistency is not correlated with actual generation quality
- The overhead of consistency computation outweighs the savings
- Spatially-varying depth provides no benefit over simple global thresholds

---

## 6. Broader Impact

### 6.1 Significance

If successful, this work would:
1. Establish that training-free spatial adaptation is possible for DiT acceleration
2. Provide a practical acceleration method deployable without training infrastructure
3. Enable efficient inference on proprietary or resource-constrained models
4. Inform the design of future training-based methods through analysis of consistency patterns

### 6.2 Limitations

1. **Heuristic-based**: Consistency metrics may not perfectly predict quality degradation
2. **Hyperparameter sensitivity**: Thresholds may require tuning for different model sizes
3. **Architecture-specific**: Designed for DiT; U-Net adaptations would require modification
4. **Not always optimal**: Learned methods (DiffRatio-MoD) may achieve better trade-offs with sufficient training

### 6.3 Future Work

1. Hybrid approaches: Use CAD-DiT to initialize/supervise learned routers
2. Cross-architecture transfer: Learn consistency thresholds on one model, apply to others
3. Extension to video: Temporal consistency for video DiTs
4. Integration with quantization and other compression methods

---

## 7. References

1. Peebles & Xie (2023). "Scalable Diffusion Models with Transformers." *ICCV*.
2. Ma et al. (2024). "DeepCache: Accelerating Diffusion Models for Free." *CVPR*.
3. Chen et al. (2024). "Δ-DiT: A Training-Free Acceleration Method Tailored for Diffusion Transformers." *arXiv*.
4. Moon et al. (2024). "Adaptive Score Estimation: Accelerating Diffusion Models Through Optimized Network Skipping." *arXiv*.
5. Tang et al. (2023). "DeeDiff: Dynamic Uncertainty-Aware Early Exiting for Accelerating Diffusion Model Generation." *ICCV*.
6. Liu et al. (2025). "Layer- and Timestep-Adaptive Differentiable Token Compression Ratios for Efficient Diffusion Transformers (DiffRatio-MoD)." *CVPR*.
7. Zou et al. (2024). "ToCa: Token-wise Caching for Efficient Video Generation." *arXiv*.
8. Liu et al. (2025). "TeaCache: Timestep Embedding Tells It's Time to Cache for Video Diffusion Models." *arXiv*.
9. Shykula et al. (2025). "Just on Time: Token-Level Early Stopping for Diffusion Language Models." *arXiv*.
10. Song et al. (2023). "Consistency Models." *ICML*.

---

## 8. Preliminary Implementation Plan

### 8.1 Week 1: Infrastructure
- Set up DiT-XL/2 inference pipeline
- Implement PCM metric computation
- Establish baseline performance numbers

### 8.2 Week 2: Core Method
- Implement token-conditional forward pass
- Add timestep-aware thresholding
- Debug and validate correctness

### 8.3 Week 3: Experiments
- Run main experiments on ImageNet
- Ablation studies for thresholds
- Baseline comparisons

### 8.4 Week 4: Analysis and Writing
- Analyze spatial/temporal patterns
- Finalize plots and tables
- Draft paper sections

---

*Proposal Version 2.0 - March 2026*
*This revision addresses feedback on the initial proposal by ensuring the method is truly training-free and clearly differentiating from DiffRatio-MoD (CVPR 2025).*
