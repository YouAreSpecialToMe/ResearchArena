# Distance-Aware Flow Matching for LiDAR Point Cloud Generation

## Problem Statement and Motivation

LiDAR point cloud generation is essential for autonomous driving simulation, data augmentation, and 3D scene understanding. However, LiDAR scans exhibit a unique geometric bias: point density decreases quadratically with distance from the sensor, creating severe sparsity in distant regions. Standard flow matching models treat all points equally, resulting in poor generation quality for far-away objects that are critical for safe navigation.

Recent work on adaptive timestep selection for flow matching (Tariq et al., 2026) addresses inference efficiency for motion planning but does not address the fundamental spatial imbalance in LiDAR geometry. We hypothesize that incorporating radial distance awareness into the flow matching objective can significantly improve generation quality for distant structures while maintaining overall sample fidelity.

## Key Insight and Hypothesis

**Key Insight**: LiDAR point clouds exhibit distance-dependent density variation that violates the uniform spatial assumptions implicit in standard flow matching. Near-field regions (0-20m) contain 60-70% of points, while far-field regions (50-80m) contain only 5-10% of points, yet both regions contribute equally to the standard MSE loss. This causes the model to under-fit distant structures.

**Hypothesis**: A distance-weighted flow matching objective that upweights gradients from distant points will improve generation quality for sparse far-field regions without degrading near-field fidelity. Specifically, we predict that a radial distance-dependent weighting scheme will reduce Chamfer Distance by 25-35% on far-field evaluation compared to standard uniform weighting.

## Proposed Approach: DistFlow

We propose **DistFlow** (Distance-Aware Flow Matching), a modified flow matching framework that explicitly accounts for LiDAR's distance-dependent geometry.

### Core Method

**1. Radial Distance-Weighted Flow Loss**

Standard flow matching minimizes:
```
L_FM = E_{t,x_0,x_1}[||v_θ(x_t, t) - (x_1 - x_0)||²]
```

DistFlow introduces distance-aware weighting:
```
L_DistFlow = E_{t,x_0,x_1}[w(r) · ||v_θ(x_t, t) - (x_1 - x_0)||²]
```

where `r` is the radial distance from the sensor and `w(r)` is a learned or hand-designed weighting function. We explore two primary variants:

- **Inverse Distance Weighting (IDW)**: `w(r) = 1 + α·(r/R_max)^β` with β ∈ [1,2]
  - Simple, interpretable, no extra parameters during training
  - α controls the strength of distance weighting (default α=1.0)
  - β controls the curvature (linear β=1 vs quadratic β=2)

- **Learned Adaptive Weighting (LAW)**: A small MLP (~8K params) that predicts sample-wise weights
  - Input: For each point, compute local density features:
    - k-NN distance statistics (mean, std) with k=8
    - Radial distance normalized by R_max
    - Height above ground (z-coordinate)
  - Architecture: 3-layer MLP (4 → 32 → 16 → 1) with ReLU activations
  - Output: Per-point weight w_i ∈ [0.5, 2.0] via sigmoid scaling
  - Training: End-to-end with the main flow matching objective
  - The small MLP overhead (~0.2% of total parameters) adds minimal compute

**2. Architecture: Point Transformer with Distance Encoding**

Our velocity network uses a lightweight Point Transformer backbone:
- Input: Noisy point cloud x_t ∈ R^(N×3) + radial distances r ∈ R^N + timestep t
- Point embedding: 64-dim MLP
- 4 transformer layers with 4-head attention (256 hidden dim)
- Distance-conditioned timestep embedding via FiLM:
  - Compute distance statistics (mean, max) of the input point cloud
  - Concatenate with sinusoidal timestep embedding
  - Project to FiLM γ, β via 2-layer MLP
- Output: Predicted velocity v ∈ R^(N×3)

Total parameters: ~4.2M (trainable on single A6000)

**3. Multi-Scale Distance Stratification**

To handle extreme density variation, we implement a coarse-to-fine sampling strategy:
- Train on stratified point sets with fixed samples per distance bucket
- Near (0-20m): 2048 points, Mid (20-50m): 1024 points, Far (50m+): 512 points
- Fuse gradients with distance-aware weights during backpropagation

### Distinctions from Prior Work

| Method | Domain | Key Idea | How DistFlow Differs |
|--------|--------|----------|---------------------|
| Tariq et al. (2026) | Motion planning | Adaptive timestep selection based on velocity variance | We address spatial sampling imbalance, not temporal |
| RAP (Sun et al., 2025) | Point cloud registration | Flow matching for SE(3) pose estimation | We target generative modeling, not registration |
| DynamicCity (Bian et al., 2024) | Occupancy forecasting | Latent flow matching for 4D scenes | We focus on raw point generation with distance-aware training |
| LiFlow (Matteazzi & Tutsch, 2026) | LiDAR scene completion | Flow matching with nearest-neighbor correspondences for completion | **Fundamentally different problem**: LiFlow is conditional generation (incomplete scan → complete scan) using paired data and nearest-neighbor flow matching. DistFlow is unconditional generation (noise → realistic scan) that explicitly addresses distance-dependent density imbalance through loss weighting. LiFlow's NFM loss establishes point-wise correspondences between paired scans; DistFlow weights gradients by radial distance to handle spatial imbalance. The techniques are orthogonal—LiFlow could benefit from DistFlow's distance weighting. |
| Decomposable FM (Haji-Ali et al., 2025) | Image/video | Multi-scale spectral decomposition via Laplacian pyramid | We use radial distance (sensor geometry), not frequency decomposition, for weighting |

## Related Work

**Flow Matching for 3D Data**. Flow matching has been applied to molecular generation (Klein et al., 2024), point cloud registration (Sun et al., 2025), and occupancy forecasting (Bian et al., 2024). However, these methods treat spatial dimensions uniformly without accounting for LiDAR's characteristic distance-dependent sparsity.

**Point Cloud Generation**. Early methods used GANs (Achlioptas et al., 2018) and VAEs (Luo & Hu, 2021). Diffusion-based approaches (Luo & Hu, 2021; Zeng et al., 2022) achieve better quality but require many sampling steps. Flow matching offers faster sampling but has not been adapted for LiDAR's unique geometry.

**LiDAR Scene Completion**. LiFlow (Matteazzi & Tutsch, 2026) applies flow matching to scene completion, using nearest-neighbor correspondences between incomplete and complete scans. While both works address LiDAR data, we tackle a different problem: unconditional generation for simulation and augmentation, not conditional completion. Our distance-aware weighting is a training-time modification that could complement LiFlow's architecture.

**Adaptive Sampling for Generative Models**. Recent work explores timestep curriculum (Sun, 2026) and variance-aware sampling (Tariq et al., 2026). Our approach is complementary—we adapt spatial sampling based on radial distance rather than temporal trajectory characteristics.

## Experimental Plan

### Datasets

**KITTI-360** (Liao et al., 2022): 320K LiDAR scans, 360° field of view
- Training: sequences 00-07 (270K scans)
- Validation: sequences 09-10 (50K scans)
- Preprocessing: Ground removal, downsample to 8192 points per scan
- Data preparation time: ~2 hours (download, extraction, preprocessing)

**nuScenes** (Caesar et al., 2020): 1.4M scans with annotations
- Used for cross-dataset generalization evaluation

### Evaluation Metrics

**Generation Quality**:
- Chamfer Distance (CD): Minimum point-to-point distances
- Earth Mover's Distance (EMD): Optimal transport cost
- 1-NN Accuracy: Classifier-based realism score

**Distance-Stratified Metrics**:
- CD-near (0-20m), CD-mid (20-50m), CD-far (50m+)
- Our primary metric: improvement ratio CD-far / CD-overall

### Baselines

1. **Standard Flow Matching** (Lipman et al., 2023): Uniform weighting
2. **Density-Weighted Flow**: Weight by local point density (not radial distance)
3. **Class-Balanced Flow**: Weight by semantic class frequency
4. **Diffusion (PointFlow)**: Luo & Hu (2021) - diffusion-based baseline

### Expected Results

We predict DistFlow will achieve:
- **25-35% reduction** in CD-far compared to standard FM
- **10-15% improvement** in overall CD
- **Comparable or better** CD-near (no degradation in near-field quality)
- **Faster convergence**: 30% fewer training epochs to reach target CD

### Ablation Studies

1. Weighting function comparison (IDW vs LAW)
2. Effect of β parameter in inverse distance weighting
3. Impact of multi-scale stratification
4. Generalization to unseen datasets (nuScenes)

## Success Criteria

**Confirming the hypothesis**:
- Statistically significant improvement (p < 0.05, paired t-test) in CD-far over standard FM
- No significant degradation in CD-near
- Qualitative visualization shows improved structure in distant regions (buildings, vehicles at 50m+)

**Refuting the hypothesis** (would also be valuable):
- If uniform weighting is optimal, this validates standard practice
- If distance weighting degrades near-field quality, reveals fundamental trade-off
- If LAW underperforms IDW, suggests simple geometric weighting suffices

## Computational Feasibility

**Training**: 
- Model size: 4.2M parameters
- Batch size: 32 on A6000 48GB
- Training time: ~6 hours for 200 epochs on KITTI-360

**Inference**:
- 50-step Euler sampling: ~0.3s per 8192-point scan
- Comparable to standard flow matching overhead

**Total budget**: Well within 8-hour limit with room for ablations

## Broader Impact

Successful distance-aware flow matching could:
1. Enable higher-fidelity simulation of rare but critical scenarios (distant pedestrians, emergency vehicles)
2. Reduce data collection costs by generating realistic training data for far-field perception
3. Provide a principled framework for handling spatial bias in other sensor modalities (RADAR, depth cameras)

## References

1. Lipman, Y., et al. (2023). Flow matching for generative modeling. ICLR.
2. Tariq, F. M., et al. (2026). Adaptive time step flow matching for autonomous driving motion planning. arXiv:2602.10285.
3. Sun, Y., et al. (2025). RAP: Registration via adaptive point flow. NeurIPS.
4. Bian, G., et al. (2024). DynamicCity: Large-scale 4D occupancy generation. arXiv:2410.18084.
5. Liao, Y., et al. (2022). KITTI-360: A novel dataset and benchmarks. PAMI.
6. Caesar, H., et al. (2020). nuScenes: A multimodal dataset. CVPR.
7. Achlioptas, P., et al. (2018). Learning representations and generative models for 3D point clouds. ICML.
8. Luo, S., & Hu, W. (2021). Diffusion probabilistic models for 3D point cloud generation. CVPR.
9. Klein, L., et al. (2024). Equivariant flow matching. NeurIPS.
10. Sun, P. (2026). Curriculum sampling: A two-phase curriculum for efficient training of flow matching. arXiv:2603.12517.
11. Matteazzi, A., & Tutsch, D. (2026). LiFlow: Flow matching for 3D LiDAR scene completion. arXiv:2602.02232.
12. Haji-Ali, M., et al. (2025). Improving progressive generation with decomposable flow matching. NeurIPS.
