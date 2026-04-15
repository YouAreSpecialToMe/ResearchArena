# Coherent Particle Sampling: Verifier-Free Inference-Time Scaling for Diffusion Models via Prediction Self-Consistency

## Introduction

### Context

Diffusion models and flow matching models have become the dominant paradigm for high-quality image generation, powering state-of-the-art systems such as Stable Diffusion XL, Stable Diffusion 3, and FLUX. A critical frontier in this space is **inference-time scaling**: the idea that, analogous to scaling test-time compute in language models (e.g., chain-of-thought, tree search), one can improve generation quality by investing additional computation during sampling rather than during training.

Recent work has established that simply increasing the number of denoising steps yields diminishing returns beyond ~50 steps (Ma et al., CVPR 2025). More sophisticated approaches use **particle-based sampling** with Feynman-Kac (FK) steering (Singhal et al., 2025) or Sequential Monte Carlo (SMC) methods, where multiple candidate trajectories ("particles") are generated in parallel, and periodically resampled based on scores from **external reward models** (e.g., CLIP score, aesthetic predictors, or task-specific verifiers). These methods achieve significant quality improvements but introduce a critical dependency: they require an external reward model that is well-aligned with the desired generation quality.

### Problem Statement

The reliance on external reward models for inference-time scaling presents several limitations:
1. **Reward model availability**: High-quality reward models are not always available for all domains or quality criteria.
2. **Reward hacking**: Particle-based methods can exploit reward model imperfections, producing samples that score highly on the reward metric but exhibit artifacts.
3. **Computational overhead**: Running an additional neural network (the reward model) at every resampling step adds significant compute cost.
4. **Reward-quality misalignment**: Metrics like CLIP score or aesthetic score are imperfect proxies for actual generation quality, and optimizing for them does not guarantee perceptually better images.

### Key Insight

During the denoising process, the diffusion model produces an estimate of the final clean image $\hat{x}_0(t)$ at each timestep $t$. For a well-converged sampling trajectory solved with sufficiently small step sizes, these "denoised predictions" should converge smoothly as $t \to 0$. When they change drastically between consecutive steps, it indicates that the sampling trajectory is experiencing high discretization error — and such trajectories empirically produce lower-quality samples.

We propose that the **temporal coherence of denoised predictions** along a sampling trajectory serves as a powerful, intrinsic quality signal that can replace external reward models for particle-based inference-time scaling.

### Hypothesis

**Selecting diffusion sampling trajectories based on the temporal coherence of their intermediate denoised predictions — without any external reward model — achieves comparable or superior generation quality to reward-guided particle sampling, at lower computational cost.**

## Proposed Approach

### Overview

We introduce **Coherent Particle Sampling (CoPS)**, a training-free, verifier-free method for inference-time scaling of diffusion and flow matching models. CoPS generates multiple sampling trajectories (particles) in parallel and uses an intrinsic **Prediction Coherence Score (PCS)** to guide particle resampling, without requiring any external reward model.

### Method Details

#### 1. Prediction Coherence Score (PCS)

At each denoising step $t_i$, the model produces a denoised prediction $\hat{x}_0^{(k)}(t_i)$ for each particle $k$. We define the instantaneous coherence at step $i$ as:

$$c_i^{(k)} = -d\left(\hat{x}_0^{(k)}(t_i),\ \hat{x}_0^{(k)}(t_{i-1})\right)$$

where $d(\cdot, \cdot)$ is a distance function (L2, LPIPS, or cosine distance in a feature space). The cumulative Prediction Coherence Score up to step $i$ is:

$$\text{PCS}_i^{(k)} = \sum_{j=1}^{i} w_j \cdot c_j^{(k)}$$

where $w_j$ are importance weights that emphasize timestep regions where coherence is most predictive of final quality (empirically, mid-range timesteps tend to matter most).

Higher PCS indicates a more coherent, smoothly converging trajectory — which we hypothesize correlates with higher final sample quality.

#### 2. Particle Resampling with PCS

CoPS follows the Feynman-Kac particle framework but replaces the external reward potential with PCS:

1. **Initialize** $K$ particles from the same initial noise (or different noises for diversity)
2. **Denoise** all particles for $S$ steps using the standard sampler (DDIM, Euler, DPM-Solver)
3. **At resampling steps** (every $R$ denoising steps):
   - Compute $\text{PCS}$ for each particle
   - Resample particles proportional to $\exp(\alpha \cdot \text{PCS})$, where $\alpha$ is a temperature parameter
   - Add noise perturbation to resampled particles to maintain diversity (optional jittering)
4. **Continue denoising** until the final step
5. **Select** the particle with the highest final PCS as the output

#### 3. Adaptive Step Allocation (ASA)

As a complementary technique, we propose using the PCS signal to adaptively allocate step sizes within a fixed NFE (Number of Function Evaluations) budget:

1. Run a **pilot trajectory** with a coarse schedule (e.g., 10 uniform steps)
2. Compute the PCS at each step — high prediction change indicates high local curvature
3. **Redistribute steps**: allocate more steps (smaller $\Delta t$) to timestep regions with high prediction change, and fewer steps to regions with low change
4. Run the **refined trajectory** with the optimized schedule

This is complementary to particle sampling and can be combined with it.

#### 4. Theoretical Justification

For a diffusion model with score function $s_\theta(x_t, t) \approx \nabla_x \log p_t(x_t)$, the denoised prediction is:

$$\hat{x}_0(x_t, t) = \frac{x_t + (1 - \bar{\alpha}_t) s_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

For the probability flow ODE solved with an exact solver, $\hat{x}_0$ would be constant along the trajectory. The deviation $\|\hat{x}_0(t_i) - \hat{x}_0(t_{i-1})\|$ is bounded by the local truncation error of the ODE solver, which depends on:
- The step size $\Delta t = t_i - t_{i-1}$
- The curvature of the score field (higher-order derivatives of $s_\theta$)

Thus, PCS directly measures the accumulated discretization error — trajectories with lower PCS (higher coherence) have lower total discretization error and produce samples closer to the true ODE solution.

For flow matching models, the analogous quantity is the velocity prediction $v_\theta(x_t, t)$ and the denoised estimate $\hat{x}_0 = x_t - t \cdot v_\theta(x_t, t)$, with the same convergence property.

### Key Innovations

1. **First verifier-free particle sampling for image diffusion models**: Unlike FK steering (Singhal et al., 2025) and related methods that require external reward models, CoPS uses only the diffusion model's own predictions.
2. **Principled intrinsic quality signal**: PCS is grounded in ODE solver theory — it measures discretization error, which directly causes quality degradation.
3. **Universal applicability**: Works with any pretrained diffusion or flow matching model (DDPM, DDIM, flow matching, rectified flow) and any sampler (Euler, DPM-Solver, etc.), with zero additional training.
4. **Adaptive step allocation**: First method to use prediction coherence for dynamic step size scheduling within a fixed compute budget.

## Related Work

### Inference-Time Scaling for Diffusion Models

**FK Steering** (Singhal et al., 2025) provides a general framework for inference-time scaling using Feynman-Kac interacting particle systems with external reward functions as potentials. **Feynman-Kac Correctors** (Skreta et al., ICML 2025) extend this with principled PDE-based corrections. **Scaling Inference-Time Compute** (Ma et al., CVPR 2025) demonstrates that combining search with verifiers significantly outperforms simply increasing denoising steps. **All of these methods require external reward models or verifiers.** Our work eliminates this requirement by deriving an intrinsic quality signal from the model's own predictions.

**VFScale** (2026) proposes verifier-free test-time scaling but focuses on discrete reasoning tasks (Maze, Sudoku) and requires training a modified loss function. **RFG** (Chen et al., 2025) achieves reward-free guidance for diffusion language models using log-likelihood ratios between enhanced and reference models — it still requires two models and is designed for text, not images. Our method is training-free, uses a single pretrained model, and targets image generation.

### Guidance Methods

**Classifier-Free Guidance** (Ho & Salimans, 2022) is the standard approach to conditional generation but uses a fixed scale, leading to quality-diversity tradeoffs. **CFG-Zero*** (Fan et al., 2025) improves CFG for flow matching by zeroing early steps and optimizing guidance scale. **Perturbed-Attention Guidance** (Ahn et al., 2024) and **Self-Attention Guidance** (Hong et al., ICCV 2023) use internal attention maps for training-free guidance. **Model-Guidance** (Tang et al., 2025) removes CFG entirely via a modified training objective but requires retraining. **Dynamic Negative Guidance** (ICLR 2025) proposes timestep-varying guidance. Our method is orthogonal to these — CoPS operates at the particle/trajectory selection level, while guidance methods modify individual denoising steps. They can be combined.

### Sampling Efficiency

**Variance-Reduction Guidance** (Xu et al., ICME 2025) optimizes the sampling trajectory offline by reducing prediction error variance but does not perform online particle resampling. **DPM-Solver** (Lu et al., 2022) and **DDIM** (Song et al., ICLR 2021) provide efficient ODE solvers for diffusion. **Adaptive Non-Uniform Timestep Sampling** (Kim et al., CVPR 2025) optimizes training schedules, not inference schedules. Our adaptive step allocation specifically targets inference-time allocation based on prediction coherence.

### Uncertainty in Diffusion Models

**Pixel-Wise Aleatoric Uncertainty** (De Vita et al., WACV 2025) estimates uncertainty from score perturbations for guided sampling. **Generative Uncertainty** (2025) provides a Bayesian post-hoc framework for uncertainty estimation. These methods focus on uncertainty quantification rather than using it for particle-based scaling. Our PCS is not an uncertainty estimate per se — it is a direct measure of ODE solver accuracy that we repurpose as a selection criterion.

## Experiments

### Experimental Setup

**Models:**
- Stable Diffusion 1.5 (diffusion, UNet-based) — primary model for comprehensive evaluation
- Stable Diffusion XL (diffusion, UNet-based) — scaling evaluation
- Stable Diffusion 3 Medium (flow matching, DiT-based) — flow matching evaluation

**Benchmarks:**
- **MS-COCO 2017 validation set** (5K random subset): FID-5K, CLIP score
- **PartiPrompts** (1.6K prompts): compositional generation quality
- **DrawBench** (200 prompts): text-image alignment evaluation

**Metrics:**
- **FID** (Fréchet Inception Distance): distribution-level quality
- **CLIP Score**: text-image alignment
- **ImageReward**: learned human preference metric
- **Aesthetic Score**: visual quality predictor
- **Prediction Coherence Score** (proposed): to validate that PCS correlates with external quality metrics

**Baselines:**
1. Standard sampling (DDIM 50 steps, Euler 50 steps)
2. DPM-Solver++ (20 steps)
3. Random particle selection (K particles, random pick — controls for multiple sampling)
4. Best-of-K with CLIP reward (FK-style with CLIP as external reward)
5. Best-of-K with aesthetic reward
6. Self-Attention Guidance (SAG)
7. Perturbed-Attention Guidance (PAG)

### Main Experiments

**Experiment 1: CoPS vs. Baselines (Fixed NFE Budget)**
Compare all methods under the same total NFE budget (e.g., 200 NFE):
- Standard: 1 trajectory × 200 steps
- Random-K: 4 trajectories × 50 steps, random selection
- CoPS: 4 particles × 50 steps, PCS-guided resampling
- CLIP-guided: 4 particles × 50 steps, CLIP-reward resampling

This isolates the effect of the selection criterion while controlling for total compute.

**Experiment 2: Scaling Behavior**
Vary the number of particles K ∈ {1, 2, 4, 8, 16} with proportionally fewer steps per particle (constant total NFE). Measure how quality scales with K under PCS guidance vs. reward guidance vs. random selection. Hypothesis: PCS-guided shows positive scaling similar to reward-guided, while random selection plateaus.

**Experiment 3: Flow Matching Models**
Evaluate CoPS on SD3 Medium to demonstrate that prediction coherence is equally informative for flow matching (velocity-based) models as for diffusion (noise-based) models.

### Ablation Studies

**A1: Distance Metric for PCS**
Compare L2, LPIPS, cosine similarity in CLIP feature space, and MSE in VAE latent space for computing prediction coherence. Hypothesis: perceptual metrics (LPIPS) outperform pixel-space metrics (L2).

**A2: Resampling Frequency**
Vary resampling interval R ∈ {every step, every 5 steps, every 10 steps, only at midpoint}. Too frequent resampling may reduce diversity; too infrequent may miss quality signals.

**A3: Timestep Weighting**
Compare uniform weighting vs. emphasizing early/mid/late timesteps in the PCS accumulation. Test our hypothesis that mid-range timesteps carry the most signal.

**A4: Adaptive Step Allocation**
Compare fixed uniform schedule vs. PCS-guided adaptive schedule under the same NFE budget. Measure quality improvement from intelligent step redistribution.

**A5: Combination with Guidance Methods**
Apply CoPS on top of CFG, PAG, and SAG to show it is complementary to existing guidance methods.

### Expected Results

Based on the theoretical grounding (PCS ∝ discretization error) and analogy with reward-guided methods:

1. CoPS should match or approach the quality of CLIP-guided particle sampling, without requiring CLIP inference at each resampling step.
2. CoPS should significantly outperform random particle selection, demonstrating that PCS is an informative quality signal.
3. Quality should scale positively with the number of particles (given constant total NFE), showing true inference-time scaling.
4. Adaptive step allocation should provide modest but consistent improvements over fixed schedules.
5. CoPS should be compatible with and complementary to existing guidance methods.

### Computational Budget

All experiments fit within 8 hours on 1× NVIDIA RTX A6000 (48GB):
- SD 1.5 inference: ~0.5s/image (20 steps, DDIM) → ~10K images/hour
- COCO 5K evaluation per method: ~30 minutes
- 7 baselines + 3 CoPS variants + 5 ablations = ~15 configurations
- Total: ~75K images → ~7.5 hours

## Success Criteria

### Confirming the hypothesis:
- CoPS achieves FID within 5% of CLIP-guided particle sampling on COCO 5K
- CoPS achieves CLIP/ImageReward scores within 3% of reward-guided methods
- CoPS significantly outperforms random particle selection (>10% improvement on FID)
- PCS shows strong rank correlation (Spearman's ρ > 0.5) with external quality metrics across particles

### Refuting the hypothesis:
- PCS does not correlate with final sample quality (ρ < 0.2)
- CoPS performs no better than random particle selection
- Quality does not scale with the number of particles under PCS guidance

## References

1. Singhal, R., Horvitz, Z., Teehan, R., Ren, M., Yu, Z., McKeown, K., & Ranganath, R. (2025). A General Framework for Inference-time Scaling and Steering of Diffusion Models. arXiv:2501.06848.

2. Skreta, M., Akhound-Sadegh, T., Ohanesian, V., Bondesan, R., Aspuru-Guzik, A., Doucet, A., Brekelmans, R., Tong, A., & Neklyudov, K. (2025). Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts. ICML 2025. arXiv:2503.02819.

3. Ma, N., Tong, S., Jia, H., Hu, H., Su, Y.-C., Zhang, M., Yang, X., Li, Y., Jaakkola, T., Jia, X., & Xie, S. (2025). Scaling Inference Time Compute for Diffusion Models. CVPR 2025. arXiv:2501.09732.

4. Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance. NeurIPS 2022 Workshop on Deep Generative Models and Downstream Applications. arXiv:2207.12598.

5. Fan, W., Chen, Y., Chen, D., Kang, Y., Zhu, J., & Wen, J.-R. (2025). CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models. arXiv:2503.18886.

6. Ahn, D., Cho, H., Min, J., Jang, W., Kim, J., Kim, S., Park, H. H., Jin, K. H., & Kim, S. (2024). Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance. arXiv:2403.17377.

7. Hong, S., Lee, G., Jang, W., & Kim, S. (2023). Improving Sample Quality of Diffusion Models Using Self-Attention Guidance. ICCV 2023. arXiv:2210.00939.

8. Tang, Z., Bao, J., Chen, D., & Guo, B. (2025). Diffusion Models without Classifier-free Guidance. arXiv:2502.12154.

9. Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. ICLR 2021. arXiv:2010.02502.

10. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022). DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Models. NeurIPS 2022. arXiv:2206.00927.

11. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. ICLR 2023. arXiv:2210.02747.

12. Xu, S., Liu, Y., & Kong, A. W.-K. (2025). Variance-Reduction Guidance: Sampling Trajectory Optimization for Diffusion Models. ICME 2025. arXiv:2510.21792.

13. De Vita, S., et al. (2025). Diffusion Model Guided Sampling with Pixel-Wise Aleatoric Uncertainty Estimation. WACV 2025. arXiv:2412.00205.

14. Chen, T., Xu, M., Leskovec, J., & Ermon, S. (2025). RFG: Test-Time Scaling for Diffusion Large Language Model Reasoning with Reward-Free Guidance. arXiv:2509.25604.

15. VFScale (2026). VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model. arXiv:2502.01989.
