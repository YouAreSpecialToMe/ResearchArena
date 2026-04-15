# Single-Pass Classifier-Free Guidance for Diffusion Transformers via Conditioning Interpolation

## Introduction

### Context
Classifier-Free Guidance (CFG) (Ho & Salimans, 2022) has become the de facto standard for improving sample quality in conditional generative models. By linearly combining conditional and unconditional model predictions, CFG steers the generation process toward high-quality, condition-aligned samples. However, CFG requires **two full forward passes** per denoising step -- one conditional and one unconditional -- effectively doubling the computational cost of inference.

This computational overhead is especially burdensome for Diffusion Transformers (DiT) (Peebles & Xie, 2023) and flow matching models like SiT (Ma et al., 2024), which use large transformer backbones with billions of parameters. In these architectures, each forward pass involves self-attention over all image tokens, making the 2x cost of CFG a significant bottleneck for practical deployment.

### Problem Statement
While several recent works have proposed efficiency improvements for CFG -- including feature caching across timesteps (TaylorSeer, AB-Cache), attention sharing between branches (DiTFastAttn), guidance distillation via adapters (AGD), and guidance truncation (TCFG, guidance interval) -- these methods either require additional training, introduce approximation at the feature level, or are limited to specific timestep ranges. **No existing method achieves training-free, architecture-aware, single-pass CFG that exploits the specific conditioning mechanism of Diffusion Transformers.**

### Key Insight
In DiT-family architectures, class conditioning is injected through Adaptive Layer Normalization (AdaLN), which modulates each transformer block via learned scale and shift parameters computed from the conditioning embedding. Crucially, AdaLN is a **linear operation** in its scale and shift parameters: the output of each AdaLN block is `gamma * LayerNorm(x) + beta`, which is linear in `(gamma, beta)`.

This observation suggests that instead of running two full forward passes and linearly combining their outputs (as in standard CFG), we can **linearly combine the conditioning parameters** (scale, shift, gate) at the AdaLN level and run a **single forward pass** with these interpolated parameters. If the model's dependence on conditioning is approximately linear, this single-pass approach should closely approximate standard CFG while halving the computational cost.

### Hypothesis
**The output of DiT/SiT models is approximately linear in the AdaLN conditioning parameters, enabling single-pass CFG via conditioning-space extrapolation that achieves quality comparable to standard two-pass CFG with approximately 2x inference speedup.**

## Proposed Approach

### Overview
We propose **Conditioning-Space Guidance (CSG)**, a training-free method that approximates classifier-free guidance in a single forward pass by extrapolating in the AdaLN parameter space of Diffusion Transformers.

### Method Details

#### Standard CFG (Baseline)
Given a model `v(x_t, t, c)` predicting the velocity field (in flow matching) or noise (in diffusion), standard CFG computes:

```
v_cond = v(x_t, t, c)           # conditional forward pass
v_uncond = v(x_t, t, null)      # unconditional forward pass
v_guided = v_uncond + w * (v_cond - v_uncond)
         = (1 + w) * v_cond - w * v_uncond
```

This requires 2 full forward passes through the entire transformer backbone.

#### CSG: Conditioning-Space Guidance

**Step 1: Compute AdaLN parameters for both conditions.**
For each transformer block `l` (l = 1, ..., L), the AdaLN module computes scale-shift-gate parameters from the conditioning embedding:

```
(gamma_c^l, beta_c^l, alpha_c^l) = AdaLN_MLP_l(t_emb + embed(c))    # conditional
(gamma_0^l, beta_0^l, alpha_0^l) = AdaLN_MLP_l(t_emb + embed(null))  # unconditional
```

Each AdaLN MLP is a small 2-layer network (negligible cost compared to the transformer blocks).

**Step 2: Extrapolate in AdaLN parameter space.**
Compute guided AdaLN parameters via linear extrapolation:

```
gamma_g^l = gamma_0^l + w * (gamma_c^l - gamma_0^l)
beta_g^l  = beta_0^l  + w * (beta_c^l  - beta_0^l)
alpha_g^l = alpha_0^l + w * (alpha_c^l - alpha_0^l)
```

**Step 3: Single forward pass with guided parameters.**
Run a single forward pass through the transformer using `(gamma_g^l, beta_g^l, alpha_g^l)` as the AdaLN parameters at each layer:

```
v_csg = DiT(x_t, t, guided_adaln_params)   # SINGLE forward pass
```

**Computational cost**: 2L small MLP evaluations + 1 transformer forward pass, compared to 2 full transformer forward passes for standard CFG. Since the AdaLN MLPs are ~0.1% of total compute, this achieves approximately **2x speedup**.

#### Per-Layer Guidance (CSG-PL)
A unique advantage of CSG is that it naturally enables **per-layer guidance control**. Instead of using a single guidance weight `w` for all layers, we can use layer-specific weights `w_l`:

```
gamma_g^l = gamma_0^l + w_l * (gamma_c^l - gamma_0^l)
```

This allows stronger guidance in layers that benefit from it (e.g., early layers for global structure) and weaker guidance in others (e.g., late layers for fine details), potentially improving quality beyond standard CFG.

#### Hybrid CSG (CSG-H)
For critical timesteps where approximation quality may degrade, we combine CSG with full CFG:

```
if t in critical_timesteps:
    v = standard_CFG(x_t, t, c, w)      # full 2-pass CFG
else:
    v = CSG(x_t, t, c, w)               # single-pass CSG
```

By using full CFG only at a small fraction of timesteps (e.g., 20%), we achieve most of the speedup while maintaining full-CFG quality.

### Key Innovations
1. **Training-free single-pass guidance**: No adapter training, no distillation -- just modify the conditioning injection.
2. **Architecture-aware**: Specifically designed for AdaLN-based DiT models, exploiting the linearity of the conditioning mechanism.
3. **Per-layer guidance control**: CSG naturally enables per-layer guidance weights, a capability not available with standard CFG.
4. **Compatibility**: CSG can be combined with other guidance improvements (APG, guidance interval, ERG) orthogonally.

## Related Work

### Classifier-Free Guidance
Ho & Salimans (2022) introduced CFG for diffusion models, jointly training conditional and unconditional models and combining their predictions at inference. CFG has become the standard approach for conditional generation across diffusion models, flow matching, and consistency models.

### Guidance Improvements for Flow Matching
CFG-Zero* (Fan et al., 2025) addresses inaccurate flow estimation in early timesteps by zeroing out the first few ODE steps and optimizing the guidance scale. Rectified-CFG++ (Saini et al., 2025) proposes a predictor-corrector guidance that keeps samples near the learned transport path. "On the Guidance of Flow Matching" (Feng et al., 2025) provides a general theoretical framework for guidance in flow matching. Our work is orthogonal to these -- CSG addresses the computational cost of guidance, not the guidance strategy itself, and can be combined with any of these methods.

### Efficient Guidance
**Guidance distillation**: AGD (Phan et al., 2025) trains lightweight adapters (~2% parameters) to simulate CFG in a single pass. Unlike AGD, CSG is completely training-free.
**Feature caching**: TaylorSeer and AB-Cache cache intermediate features across timesteps. These methods still require two forward passes when they do compute (they just skip some steps). CSG reduces cost at every step.
**Attention sharing**: DiTFastAttn shares attention outputs between conditional and unconditional branches when they are similar. CSG goes further by eliminating the separate unconditional pass entirely.
**Guidance interval**: Kynkaanniemi et al. (2024) showed that guidance is harmful at early and late timesteps and should only be applied in the middle. CSG is compatible with this -- we can use CSG within the guidance interval and skip guidance outside it.

### Adaptive Projected Guidance
APG (Sadat et al., 2024) decomposes the CFG signal into parallel and perpendicular components, showing the parallel component causes oversaturation. APG and CSG address different aspects: APG improves guidance direction, CSG improves guidance efficiency. They can be combined -- apply APG's projection to CSG's output.

### Diffusion Transformer Architecture
DiT (Peebles & Xie, 2023) introduced the transformer-based architecture for diffusion models with AdaLN conditioning. SiT (Ma et al., 2024) extended this to flow matching with interpolant transformers. Both use AdaLN for class conditioning, making them natural targets for CSG.

## Experiments

### Setup
- **Models**: DiT-XL/2 and SiT-XL/2 pretrained on ImageNet 256x256 (publicly available checkpoints)
- **Hardware**: 1x NVIDIA RTX A6000 (48GB VRAM)
- **Evaluation**: FID-50K, Inception Score (IS), Precision, Recall, FDDINOv2 on ImageNet 256x256

### Experiment 1: CSG vs. Standard CFG Quality Comparison
- Generate 50K images using standard CFG (2-pass) and CSG (1-pass) at guidance scales w = {1.5, 2.0, 3.0, 4.0, 5.0, 7.5}
- Use 250-step ODE solving (Euler method)
- Compare FID, IS, Precision, Recall
- **Expected result**: CSG achieves FID within 5-10% of standard CFG at moderate guidance scales (1.5-4.0), with larger gaps at high scales

### Experiment 2: Speedup Measurement
- Measure wall-clock time for 50K image generation with standard CFG vs. CSG
- Report throughput (images/second) and speedup ratio
- **Expected result**: ~1.8-1.9x speedup (not exactly 2x due to AdaLN MLP overhead and other non-transformer compute)

### Experiment 3: Linearity Analysis
- At each timestep and layer, measure the approximation error: ||v_CSG - v_CFG|| / ||v_CFG||
- Analyze how error varies across timesteps, layers, and guidance scales
- Visualize attention maps to understand where the approximation breaks down
- **Expected result**: Error is lowest at early timesteps (high noise), increases at late timesteps; lower layers show better approximation than deeper layers

### Experiment 4: Per-Layer Guidance (CSG-PL) Ablation
- Test different per-layer guidance schedules: uniform, increasing, decreasing, bell-curve
- Compare against standard CSG and standard CFG
- **Expected result**: Decreasing guidance schedule (stronger in early layers, weaker in later layers) outperforms uniform guidance

### Experiment 5: Hybrid CSG (CSG-H)
- Test CSG with full CFG at {10%, 20%, 30%, 50%} of timesteps (selected as the middle timesteps where guidance matters most)
- Compare quality-speed tradeoff
- **Expected result**: Using full CFG at 20% of timesteps recovers nearly all quality while maintaining ~1.6x speedup

### Experiment 6: Combination with Other Methods
- CSG + APG: Apply APG's parallel/perpendicular decomposition to CSG output
- CSG + Guidance Interval: Use CSG within the optimal guidance interval
- **Expected result**: Combinations yield additive benefits

### Ablations
- Model size: DiT-S/2, DiT-B/2, DiT-L/2, DiT-XL/2 (if pretrained available)
- Number of ODE steps: 25, 50, 100, 250
- Flow matching (SiT) vs. diffusion (DiT) comparison
- Embedding-space extrapolation vs. AdaLN-parameter-space extrapolation

## Success Criteria

### Confirms hypothesis:
- CSG achieves FID within 10% of standard CFG on ImageNet 256x256 at guidance scale w=4.0
- CSG achieves at least 1.7x speedup over standard CFG
- CSG-H with 20% full CFG steps matches standard CFG quality within 5% FID

### Refutes hypothesis:
- CSG FID is more than 25% worse than standard CFG even at low guidance scales
- The conditioning linearity assumption breaks down catastrophically (approximation error > 50%)
- No per-layer schedule improves upon uniform guidance

### Partial success:
- CSG works well at low guidance scales (w < 3) but degrades at high scales
- CSG works better for flow matching (SiT) than diffusion (DiT) or vice versa
- CSG-PL finds meaningful per-layer guidance patterns even if raw quality is lower

## References

1. Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance. NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.
2. Peebles, W. & Xie, S. (2023). Scalable Diffusion Models with Transformers. ICCV 2023.
3. Ma, N., Goldstein, M., Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E., & Xie, S. (2024). SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers. ECCV 2024.
4. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. ICLR 2023.
5. Sadat, S., Hilliges, O., & Weber, R.M. (2024). Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models. arXiv:2410.02416.
6. Fan, W., Zheng, A.Y., Yeh, R.A., & Liu, Z. (2025). CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models. arXiv:2503.18886.
7. Kynkaanniemi, T., Aittala, M., Karras, T., Laine, S., Aila, T., & Lehtinen, J. (2024). Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models. NeurIPS 2024.
8. Saini, S. et al. (2025). Rectified-CFG++ for Flow Based Models. NeurIPS 2025.
9. Phan, H.A. et al. (2025). Efficient Distillation of Classifier-Free Guidance using Adapters. arXiv:2503.07274.
10. Feng, R., Yu, C., Deng, W., Hu, P., & Wu, T. (2025). On the Guidance of Flow Matching. ICML 2025.
11. Karras, T. et al. (2024). Guiding a Diffusion Model with a Bad Version of Itself. arXiv:2406.02507.
12. Zheng, H., He, J., Zheng, M., & Sun, J. (2024). DiTFastAttn: Attention Compression for Diffusion Transformer Models. NeurIPS 2024.
