# Research Proposal: Internal Consistency as Guide (ICG): Test-Time Self-Correction for Diffusion Models via Cross-Noise-Level Verification

## 1. Introduction

### 1.1 Background and Motivation

Diffusion models have emerged as the dominant paradigm in generative modeling, achieving state-of-the-art results in image synthesis, video generation, and beyond. These models learn to reverse a forward noising process by training a neural network to predict either the noise, the clean data, or the score function at various noise levels. Sampling involves iteratively denoising from pure Gaussian noise to a clean sample through a trajectory of T steps.

Despite their success, diffusion models face a fundamental limitation: **generation quality varies significantly across samples and regions**, with some samples requiring more refinement than others. Current approaches treat all samples uniformly, using a fixed number of denoising steps regardless of sample difficulty. While test-time scaling (TTS) methods have emerged to address this—using search, evolutionary algorithms, or reward models—these approaches require substantial additional compute, external reward models, or complex optimization.

### 1.2 Key Insight and Hypothesis

Our key insight is that **a well-generated sample should exhibit internal consistency**: predictions of the clean image from different noise levels along the same trajectory should agree. Conversely, disagreement between predictions at different timesteps signals uncertainty or poor generation quality. This internal disagreement can serve as a **training-free, self-contained quality signal** that guides test-time refinement.

**Hypothesis**: By measuring cross-noise-level prediction consistency during the denoising process, we can (1) identify samples/regions that need additional refinement, and (2) trigger targeted test-time correction without requiring external reward models or expensive search procedures.

### 1.3 Problem Statement

Given a pretrained diffusion model and a sampling trajectory, how can we:
1. Quantify internal prediction consistency across noise levels?
2. Use inconsistency signals to identify low-quality regions?
3. Allocate additional compute adaptively for self-correction?

## 2. Proposed Approach

### 2.1 Core Concept: Cross-Noise-Level Consistency

During diffusion sampling, the model makes predictions at various timesteps. For a standard noise-prediction model $\epsilon_\theta(x_t, t)$, we can derive an estimate of the clean image at any timestep:

$$\hat{x}_0^{(t)} = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

For a well-behaved generation, $\hat{x}_0^{(t)}$ should be consistent across different $t$ values. We define the **Internal Consistency Score (ICS)** as:

$$\text{ICS}(t_1, t_2) = \text{sim}(\hat{x}_0^{(t_1)}, \hat{x}_0^{(t_2)})$$

where $\text{sim}(\cdot, \cdot)$ measures similarity (e.g., LPIPS, SSIM, or MSE in latent space).

### 2.2 Test-Time Self-Correction Algorithm

Our method, **ICG (Internal Consistency as Guide)**, operates in three phases:

**Phase 1: Standard Sampling with Consistency Tracking**
- Run standard DDIM/DDPM sampling for T steps
- At each step $t$, compute $\hat{x}_0^{(t)}$ and track ICS between adjacent steps
- Identify timesteps with low ICS (high inconsistency)

**Phase 2: Inconsistency-Guided Local Refinement**
- For regions/timesteps with ICS below threshold $\tau$:
  - Add controlled noise to "rewind" to higher noise level
  - Perform additional denoising steps with attention to inconsistent regions
  - Optionally use spatial attention masks based on per-pixel disagreement

**Phase 3: Verification and Iteration**
- Re-compute ICS after refinement
- If overall consistency improves, accept; otherwise retry with different noise realizations
- Iterate until consistency threshold met or budget exhausted

### 2.3 Adaptive Compute Allocation

Different from uniform refinement, ICG allocates compute based on sample difficulty:

$$N_{\text{steps}}^{(i)} = N_{\text{base}} + \lambda \cdot (1 - \text{ICS}^{(i)})$$

where $\text{ICS}^{(i)}$ is the average internal consistency for sample $i$.

### 2.4 Spatially-Adaptive Refinement

For image generation, inconsistency may vary spatially. We compute per-pixel consistency maps:

$$M_{\text{inconsistent}} = \text{Threshold}(\text{Var}_t(\hat{x}_0^{(t)}))$$

This mask guides where to apply additional denoising iterations.

## 3. Related Work and Novelty

### 3.1 Consistency Models (Song et al., 2023)
Consistency models train the network to satisfy $f(x_t, t) = f(x_{t'}, t')$ for points on the same trajectory. While related, our approach differs fundamentally:
- **Consistency models** enforce consistency through training
- **ICG** uses inconsistency as a diagnostic signal at test time without retraining
- ICG can be applied to any pretrained diffusion model

### 3.2 Test-Time Scaling for Diffusion (Ma et al., 2025; He et al., 2025)
Recent TTS methods (EvoSearch, TTSnap) scale inference compute via search or evolutionary algorithms. ICG differs by:
- Not requiring external reward models
- Not needing population-based search
- Using internal model signals instead of external evaluation

### 3.3 Temporal Dynamics in dLLMs (2025)
Recent work on diffusion language models exploits "temporal oscillation" where correct answers appear mid-generation. Our work extends this insight to continuous image diffusion:
- We focus on prediction consistency rather than answer emergence
- We develop spatially-adaptive refinement for images
- We provide a principled framework for consistency-guided correction

### 3.4 Adaptive Sampling Methods
AdaDiff (Tang et al., 2024) uses uncertainty for early exit. ICG differs by:
- Using cross-timestep consistency rather than single-step uncertainty
- Enabling active correction rather than just early termination
- Supporting targeted refinement of specific regions

## 4. Experiments

### 4.1 Datasets and Baselines

**Datasets**:
- CIFAR-10 (32×32): Fast prototyping and ablations
- ImageNet 64×64 and 256×256: Standard benchmarks
- LSUN Bedrooms/Churches: Complex scene generation

**Baselines**:
- Standard DDIM/DDPM with fixed steps
- Consistency distillation (CD) methods
- EvoSearch (test-time evolutionary search)
- TTSnap (noise-aware pruning)
- DPM-Solver++ (fast ODE solver)

### 4.2 Evaluation Metrics

- **FID** (Frechet Inception Distance): Overall sample quality
- **Precision/Recall**: Fidelity vs. diversity trade-off
- **ICS Distribution**: Internal consistency scores
- **NFE** (Number of Function Evaluations): Computational cost
- **Success Rate**: Fraction of samples achieving target consistency

### 4.3 Experimental Plan

**Experiment 1: Consistency-Quality Correlation**
- Verify that samples with higher ICS have better FID
- Show that ICS correlates with human judgments

**Experiment 2: ICG vs. Baselines**
- Compare ICG to standard sampling with equal NFE budget
- Compare to search-based TTS with equal compute
- Demonstrate better quality-efficiency trade-off

**Experiment 3: Ablation Studies**
- Effect of consistency threshold $\tau$
- Impact of spatially-adaptive vs. uniform refinement
- Comparison of different similarity metrics for ICS

**Experiment 4: Generalization**
- Test on different model architectures (UNet, DiT)
- Evaluate on different pre-trained models (SD 1.5, SDXL)
- Demonstrate training-free applicability

### 4.4 Expected Results

We expect ICG to:
1. Achieve 10-20% FID improvement over standard sampling with same NFE
2. Match or exceed EvoSearch quality with 50% less compute
3. Show strong correlation between ICS and sample quality
4. Demonstrate that spatially-adaptive refinement outperforms uniform approaches

## 5. Success Criteria

The project is successful if:
1. **ICS reliably predicts sample quality** (correlation > 0.7 with FID/human ratings)
2. **ICG improves over standard sampling** (statistically significant FID reduction)
3. **ICG is compute-efficient** (better FID-NFE Pareto frontier than baselines)
4. **Method is general** (works across architectures and datasets)

The hypothesis is refuted if:
1. ICS does not correlate with sample quality
2. Additional refinement guided by ICS does not improve samples
3. Compute overhead outweighs quality gains

## 6. Broader Impact and Limitations

### 6.1 Contributions
- First method to use cross-noise-level consistency for test-time self-correction in image diffusion
- Training-free approach applicable to any pretrained diffusion model
- Spatially-adaptive compute allocation for efficient refinement

### 6.2 Limitations
- Additional memory needed to store intermediate predictions
- Refinement adds latency (though less than search-based methods)
- Effectiveness depends on base model quality

### 6.3 Future Directions
- Extend to video diffusion (temporal consistency)
- Combine with consistency model training
- Explore ICS as a reward signal for RL fine-tuning

## 7. References

1. Song, Y., et al. (2023). Consistency Models. ICML 2023.
2. Ma, S., et al. (2025). Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps. arXiv:2501.09732.
3. He, H., et al. (2025). Scaling Image and Video Generation via Test-Time Evolutionary Search. ICLR 2025.
4. Tang, S., et al. (2024). AdaDiff: Accelerating Diffusion Models through Step-Wise Adaptive Computation. ECCV 2024.
5. Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020.
6. Song, J., et al. (2021). Denoising Diffusion Implicit Models. ICLR 2021.
7. Karras, T., et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS 2022.
8. Lipman, Y., et al. (2023). Flow Matching for Generative Modeling. ICLR 2023.
9. Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
10. Zhang, Q., & Chen, Y. (2022). Fast Sampling of Diffusion Models with Exponential Integrator. ICLR 2023.
