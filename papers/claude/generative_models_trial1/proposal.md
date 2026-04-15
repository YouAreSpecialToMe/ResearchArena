# Spectral Consistency Distillation: Frequency-Aware Few-Step Generation for Flow Matching Models

## Introduction

Flow matching models have emerged as a leading paradigm for generative modeling, achieving state-of-the-art sample quality across images, audio, and video (Lipman et al., 2023). However, their practical deployment is hindered by the computational cost of iterative ODE solving, typically requiring 20-100 neural function evaluations (NFEs) for high-quality generation. Consistency distillation methods (Song et al., 2023; Geng et al., 2024) address this by training student models that generate high-quality samples in 1-4 steps, but they suffer from a critical limitation: **high-frequency detail degradation**.

The core problem is well-documented but poorly addressed: when distilling multi-step flow models into few-step generators, the generated images lose fine-grained textures, sharp edges, and high-frequency details. This manifests as blurriness, loss of texture fidelity, and reduced perceptual quality — particularly visible at higher resolutions. Current distillation methods treat all spatial frequencies equally in their training objectives, despite strong evidence that different frequency components converge at vastly different rates along the ODE trajectory (Nash et al., 2021; Rissanen et al., 2023).

**Key Insight**: The denoising process in diffusion and flow models has a natural spectral hierarchy — low-frequency components (global structure, color) converge early in the trajectory, while high-frequency components (textures, edges) are resolved only in late stages. Standard distillation losses (MSE in pixel or latent space) weight all frequencies equally, wasting model capacity on already-converged low frequencies and under-investing in the harder-to-distill high frequencies.

**Hypothesis**: By decomposing the distillation loss into frequency bands and applying frequency-aware weighting — upweighting high-frequency components and using more teacher steps for high-frequency supervision — we can significantly improve the quality of few-step generation, particularly in terms of high-frequency detail preservation, without increasing inference cost.

## Proposed Approach

### Overview

We propose **Spectral Consistency Distillation (SCD)**, a frequency-aware distillation framework for flow matching models. SCD modifies the distillation training pipeline in three ways:

1. **Spectral Loss Decomposition**: Replace the standard MSE distillation loss with a frequency-decomposed loss computed in the DCT domain
2. **Frequency-Adaptive Teacher Supervision**: Use different numbers of teacher ODE steps to generate supervision targets for different frequency bands
3. **Progressive Spectral Refinement**: Train the student model with a curriculum that first focuses on low-frequency consistency, then progressively increases emphasis on high-frequency fidelity

### Method Details

#### 1. Spectral Loss Decomposition

Given a teacher flow model $v_\phi(x_t, t)$ and a student model $f_\theta$ that maps $(x_t, t) \mapsto \hat{x}_0$, the standard consistency distillation loss is:

$$\mathcal{L}_{CD} = \mathbb{E}_{x_0, t} \left[ \| f_\theta(x_t, t) - \text{sg}[\text{Teacher}(x_t, t)] \|^2 \right]$$

We decompose this into $K$ frequency bands using the 2D DCT. Let $\text{DCT}(x)$ denote the block DCT of image $x$, and let $M_k$ be a binary mask selecting the $k$-th frequency band (ordered from lowest to highest frequency). Our spectral loss is:

$$\mathcal{L}_{SCD} = \sum_{k=1}^{K} w_k \cdot \mathbb{E}_{x_0, t} \left[ \| M_k \odot \text{DCT}(f_\theta(x_t, t) - \text{sg}[\text{Target}_k(x_t, t)]) \|^2 \right]$$

where $w_k$ are learnable or analytically-derived per-band weights, and $\text{Target}_k$ is the frequency-band-specific teacher target.

#### 2. Frequency-Adaptive Teacher Supervision

The key innovation is that different frequency bands get teacher targets generated with different numbers of ODE steps:

- **Low-frequency bands** ($k = 1, ..., K/2$): Teacher uses $N_{\text{low}}$ steps (e.g., 10 steps) — these converge quickly
- **High-frequency bands** ($k = K/2+1, ..., K$): Teacher uses $N_{\text{high}}$ steps (e.g., 50-100 steps) — these require more steps for accurate targets

This ensures the student receives high-quality supervision for the components that are hardest to distill (high frequencies), while avoiding unnecessary computation for components that converge with few steps (low frequencies).

The per-band teacher targets are extracted by:
1. Running the teacher with $N_k$ steps to get $\hat{x}_0^{(k)}$
2. Extracting the $k$-th frequency band: $\text{Target}_k = M_k \odot \text{DCT}(\hat{x}_0^{(k)})$

#### 3. Progressive Spectral Refinement

Training follows a curriculum:
- **Phase 1** (warmup): Train with uniform frequency weights, standard consistency loss. This establishes basic generation capability.
- **Phase 2** (spectral refinement): Gradually increase weights $w_k$ for higher frequency bands. This shifts model capacity toward high-frequency fidelity.
- **Phase 3** (fine-tuning): Fix weights based on the spectral error profile of the student and fine-tune with the full SCD loss.

#### 4. Adaptive Frequency Weighting

The per-band weights $w_k$ can be set:
- **Analytically**: Inversely proportional to the power spectral density of natural images (emphasize under-represented high frequencies)
- **Learned**: Using a simple attention mechanism over frequency bands that observes the per-band distillation error and adjusts weights to equalize errors across bands
- **Error-driven**: Set $w_k \propto \text{Error}_k^{\alpha}$ where $\text{Error}_k$ is the running average of the per-band distillation error

### Architecture

The student model architecture remains identical to the teacher (a standard flow matching network, e.g., U-Net or DiT). The only additions are:
- DCT/IDCT operations for spectral decomposition (negligible compute)
- Per-band weight parameters ($K$ scalars)
- No additional parameters are needed at inference time — the student generates samples identically to a standard consistency model

### Inference

At inference time, the student model is used exactly like a standard consistency model — no spectral decomposition is needed. The spectral loss only affects training. This means SCD adds zero overhead at inference time.

## Related Work

### Flow Matching and Continuous Normalizing Flows
Flow matching (Lipman et al., 2023) provides a simulation-free training framework for continuous normalizing flows, learning velocity fields that transport between noise and data distributions. Rectified flow (Liu et al., 2022) learns straight-line ODE paths, enabling few-step generation. Our work builds on these as the teacher model backbone.

### Consistency Models and Distillation
Consistency models (Song et al., 2023) learn to map any point on the ODE trajectory directly to the clean data endpoint, enabling single-step generation. Improved consistency training (Song & Dhariwal, 2023) introduced Pseudo-Huber losses and lognormal noise schedules. Easy Consistency Tuning (Geng et al., 2024) showed that starting from a pretrained diffusion model drastically reduces training cost. Consistency Flow Matching (Yang et al., 2024) enforces velocity consistency for straight flows. Our method is complementary to all these — it modifies the distillation loss, not the consistency formulation itself.

### Progressive and Flow Map Distillation
Progressive distillation (Salimans & Ho, 2022) iteratively halves the number of sampling steps. Align Your Flow (Sabour et al., 2025) introduces continuous-time flow map objectives with autoguidance. "How to build a consistency model" (Boffi et al., 2025) presents a unified self-distillation framework. Self-Corrected Flow Distillation (Dao et al., 2025) combines consistency and adversarial training. Our method addresses an orthogonal dimension — the spectral decomposition of the loss — and can be combined with any of these approaches.

### Spectral Analysis of Diffusion Models
Recent work has analyzed diffusion models in the frequency domain. The forward diffusion process destroys high frequencies before low frequencies, and the reverse process generates low frequencies first (Rissanen et al., 2023). Spectral Regularization (Chandran et al., 2026) adds Fourier/wavelet losses during training (not distillation). Frequency-Decoupled Guidance (Sadat et al., 2025) applies different CFG scales to different frequency bands during inference. Adaptive Spectral Feature Forecasting (Han et al., 2026) uses Chebyshev polynomials for training-free acceleration. Our work is the first to apply spectral decomposition specifically to the consistency distillation objective.

### Diffusion Transformer Efficiency
DiT (Peebles & Xie, 2023) introduced transformer-based diffusion models. Dynamic DiT (DyDiT, ICLR 2025) adaptively allocates computation across timesteps. DiffCR (CVPR 2025) learns per-layer and per-timestep token compression ratios. Our method operates at a different level — modifying the training loss rather than the architecture.

## Experiments

### Setup

**Datasets**:
- Primary: CIFAR-10 (32×32, unconditional and class-conditional)
- Secondary: ImageNet-64 (64×64, class-conditional)

**Teacher Models**:
- CIFAR-10: Flow matching model trained with the OT-CFM objective (using torchcfm library or custom implementation). U-Net architecture similar to DDPM++.
- ImageNet-64: Pretrained DiT-S/2 or custom flow matching model.

**Student Models**: Same architecture as teacher, trained with SCD loss.

**Baselines**:
1. Standard consistency distillation (CD) — vanilla MSE loss
2. Progressive distillation (PD) — iterative step halving
3. Rectified flow with few steps — direct Euler sampling
4. Consistency Flow Matching (Consistency-FM) — velocity consistency
5. CD + Pseudo-Huber loss (improved consistency training)

**Metrics**:
- FID (Fréchet Inception Distance) — overall quality
- sFID (Spatial FID) — structure quality
- Precision and Recall — fidelity vs. diversity tradeoff
- Per-frequency error: MSE computed per DCT frequency band between generated and real data
- LPIPS — perceptual similarity

**Step budgets**: 1-step, 2-step, 4-step generation.

### Planned Experiments

**Experiment 1: Main Comparison (Table 1)**
Compare SCD against all baselines on CIFAR-10 and ImageNet-64 at 1, 2, and 4 steps. Report FID, sFID, Precision, Recall.

**Experiment 2: Spectral Error Analysis (Figure 1)**
For each method, compute the per-frequency-band MSE between generated images and a held-out test set. Show that SCD reduces high-frequency error compared to baselines while maintaining low-frequency quality.

**Experiment 3: Ablation Study (Table 2)**
Ablate each component of SCD:
- (a) SCD without frequency-adaptive teacher supervision (uniform teacher steps)
- (b) SCD without progressive spectral refinement (train from scratch with full spectral loss)
- (c) SCD with different numbers of frequency bands $K$ (2, 4, 8, 16)
- (d) SCD with different weighting strategies (analytic, learned, error-driven)

**Experiment 4: Qualitative Comparison (Figure 2)**
Visual comparison of generated samples from SCD vs. baselines, highlighting high-frequency detail preservation (zoomed-in crops of textures, edges).

**Experiment 5: Computational Overhead (Table 3)**
Training time comparison: SCD vs. baselines. The spectral decomposition adds minimal overhead (DCT is O(N log N)), and the main cost increase is from running the teacher with more steps for high-frequency bands (amortized over training).

### Expected Results

We expect SCD to:
1. Achieve 10-25% lower FID compared to standard consistency distillation at the same step count
2. Show the largest improvements in 1-step generation (where high-frequency degradation is most severe)
3. Demonstrate significantly lower high-frequency error while maintaining comparable low-frequency quality
4. Add less than 20% training overhead compared to vanilla consistency distillation (from the DCT computation and multi-resolution teacher calls)

## Success Criteria

The hypothesis is confirmed if:
1. SCD consistently outperforms standard consistency distillation in FID across both datasets and all step budgets (1, 2, 4 steps)
2. Per-frequency error analysis shows SCD specifically reduces high-frequency error (top 50% frequency bands) by at least 15% compared to the baseline
3. Visual inspection confirms better texture and edge quality in SCD-generated images

The hypothesis is refuted if:
1. SCD shows no significant improvement over standard consistency distillation (within noise margin of FID)
2. The frequency-weighted loss causes instability or mode collapse
3. The computational overhead exceeds 50% of baseline training time

## References

1. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. ICLR 2023. arXiv:2210.02747.

2. Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency Models. ICML 2023. arXiv:2303.01469.

3. Song, Y. & Dhariwal, P. (2023). Improved Techniques for Training Consistency Models. arXiv:2310.14189.

4. Liu, X., Gong, C., & Liu, Q. (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. ICLR 2023. arXiv:2209.03003.

5. Peebles, W. & Xie, S. (2023). Scalable Diffusion Models with Transformers. ICCV 2023. arXiv:2212.09748.

6. Salimans, T. & Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models. ICLR 2022. arXiv:2202.00512.

7. Geng, Z., Pokle, A., Luo, W., Lin, J., & Kolter, J.Z. (2024). Consistency Models Made Easy. ICLR 2025. arXiv:2406.14548.

8. Yang, L., Zhang, Z., Zhang, Z., Liu, X., Xu, M., Zhang, W., Meng, C., Ermon, S., & Cui, B. (2024). Consistency Flow Matching: Defining Straight Flows with Velocity Consistency. arXiv:2407.02398.

9. Sabour, A., Fidler, S., & Kreis, K. (2025). Align Your Flow: Scaling Continuous-Time Flow Map Distillation. arXiv:2506.14603.

10. Boffi, N.M., Albergo, M.S., & Vanden-Eijnden, E. (2025). How to Build a Consistency Model: Learning Flow Maps via Self-Distillation. NeurIPS 2025. arXiv:2505.18825.

11. Dao, Q., Phung, H., Dao, T., Metaxas, D., & Tran, A. (2025). Self-Corrected Flow Distillation for Consistent One-Step and Few-Step Text-to-Image Generation. AAAI 2025. arXiv:2412.16906.

12. Chandran, S., dos Santos, N.R., Wu, Y., Ver Steeg, G., & Papalexakis, E. (2026). Spectral Regularization for Diffusion Models. arXiv:2603.02447.

13. Sadat, S., Vontobel, T., Salehi, F., & Weber, R.M. (2025). Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales. arXiv:2506.19713.

14. Han, J., Shi, J., Li, P., Ye, H., Guo, Q., & Ermon, S. (2026). Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration. CVPR 2026. arXiv:2603.01623.

15. Rissanen, S., Heinonen, M., & Solin, A. (2023). Generative Modelling with Inverse Heat Dissipation. ICLR 2023. arXiv:2206.13397.
