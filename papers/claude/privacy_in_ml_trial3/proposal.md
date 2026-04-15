# The Compounding Cost: How Differential Privacy and Model Compression Jointly Amplify Fairness Degradation

## Introduction

### Context

Machine learning models deployed in production must simultaneously satisfy multiple desiderata: they must protect user privacy (e.g., for GDPR/CCPA compliance), be efficient enough for deployment on resource-constrained devices (e.g., mobile phones, edge servers), and treat all demographic groups fairly. In practice, this means models are often (1) trained with differentially private stochastic gradient descent (DP-SGD) to provide formal privacy guarantees, and then (2) compressed via pruning or quantization to reduce inference cost. Both steps are well-studied individually, but their *interaction* with respect to fairness remains poorly understood.

### The Problem

Two independent lines of research have established that:

1. **Differential privacy amplifies bias.** DP-SGD's gradient clipping and noise addition disproportionately degrade accuracy on underrepresented subgroups (Bagdasaryan & Shmatikov, 2019). The root cause is gradient misalignment: clipping distorts gradients for minority groups more than majority groups because minority-group gradients tend to have larger norms and different directions (Esipova et al., 2023).

2. **Model compression amplifies bias.** Pruning and quantization disproportionately harm performance on "Compression Identified Exemplars" (CIE) — atypical examples that often belong to underrepresented groups (Hooker et al., 2020). Additionally, compression exacerbates privacy leakage, making training data more vulnerable to membership inference attacks (Li et al., 2025).

**The critical gap:** No prior work has systematically studied whether these two effects *compound* when applied together. In real-world deployment pipelines, a model undergoes both DP training and compression. If the fairness costs are super-additive — i.e., the combined degradation exceeds the sum of individual degradations — then current separate analyses dramatically underestimate the true cost to minority groups.

### Key Insight

We hypothesize that DP-SGD and pruning share a common mechanism for harming minority groups: both operations preferentially remove or distort features with low signal-to-noise ratio, which are precisely the features that encode minority-group information. In DP-SGD, gradient clipping truncates large-norm gradients (which are more common for minority groups), reducing the learned signal for these groups. In pruning, magnitude-based importance criteria remove small-weight features first — and in DP-trained models, minority-relevant weights are systematically smaller due to the reduced gradient signal. This creates a **feedback loop**: DP training produces models where minority features are encoded with small weights, and pruning then preferentially removes exactly those features.

### Hypothesis

**Primary hypothesis:** The fairness degradation from combining DP-SGD training with magnitude-based pruning is *super-additive* — significantly worse than the sum of the degradation from DP-SGD alone and pruning alone, particularly for underrepresented subgroups.

**Secondary hypothesis:** A fairness-aware pruning criterion that accounts for per-subgroup feature importance can break this feedback loop and achieve comparable compression ratios with substantially less fairness degradation.

## Proposed Approach

### Overview

We propose a three-part contribution:

1. **Empirical characterization** of the compounding fairness cost when DP-SGD and model compression are applied together, measured across multiple datasets, architectures, privacy budgets, and compression rates.

2. **Mechanistic analysis** explaining *why* the costs compound, through examination of weight magnitude distributions, gradient norms, and feature importance per subgroup across the DP + compression pipeline.

3. **FairPrune-DP** — a simple but effective fairness-aware pruning algorithm designed specifically for DP-trained models that uses worst-group feature importance to select which weights to prune.

### Method Details

#### Part 1: Measuring the Compounding Effect

For each dataset and architecture, we train four model variants:
- **Baseline (B):** Standard training, no compression
- **DP-only (D):** DP-SGD training, no compression
- **Comp-only (C):** Standard training + pruning
- **DP+Comp (DC):** DP-SGD training + pruning

For each variant, we measure:
- Overall accuracy
- Per-subgroup accuracy (for each protected attribute)
- Worst-group accuracy
- Accuracy gap: max-group accuracy − min-group accuracy
- Membership inference attack (MIA) success rate, stratified by subgroup

The **compounding ratio** is defined as:

$$\text{CR} = \frac{\Delta_{DC}}{\Delta_D + \Delta_C}$$

where $\Delta_X$ denotes the worst-group accuracy drop of variant $X$ relative to the baseline $B$. A compounding ratio CR > 1 indicates super-additive degradation.

#### Part 2: Mechanistic Analysis

We analyze:
- **Weight magnitude distributions per subgroup:** For each weight, compute its contribution to subgroup-specific Fisher information. Show that DP-trained models have systematically smaller weights for minority-relevant features.
- **Pruning vulnerability maps:** Identify which weights are pruned first under magnitude-based pruning and show these overlap with minority-relevant features more in DP-trained models than in standard models.
- **Gradient norm analysis:** Track per-subgroup gradient norms during DP-SGD training to quantify the differential clipping effect.

#### Part 3: FairPrune-DP

Standard magnitude-based pruning assigns importance score $s_i = |w_i|$ to each weight and removes weights with the smallest scores. This is subgroup-agnostic.

**FairPrune-DP** replaces this with a worst-group importance criterion:

1. Given a small held-out calibration set with subgroup labels, compute per-subgroup importance for each weight:
   $$s_i^g = \mathbb{E}_{(x,y) \sim \mathcal{D}_g} \left[ \left(\frac{\partial \mathcal{L}(x, y)}{\partial w_i}\right)^2 \right]$$
   where $\mathcal{D}_g$ is the data distribution for subgroup $g$.

2. Compute the fairness-aware importance as:
   $$s_i^{\text{fair}} = \min_g \; s_i^g$$
   This ensures a weight is only considered "unimportant" if it is unimportant for *all* subgroups.

3. Prune weights with the smallest $s_i^{\text{fair}}$ values.

4. Optionally fine-tune for a few epochs (with DP-SGD if privacy budget allows).

**Why this works:** Standard pruning removes weights that are globally least important, which in DP-trained models disproportionately correspond to minority-relevant features. FairPrune-DP prevents this by ensuring that features critical to any subgroup are retained, even if their global importance is low.

**Computational overhead:** Computing per-subgroup Fisher information requires one additional forward-backward pass per subgroup over the calibration set. For $G$ subgroups and a calibration set of size $n$, this adds $O(G \cdot n)$ gradient computations — negligible compared to training cost.

### Baselines

We compare against:
- **Magnitude pruning (MP):** Standard weight-magnitude pruning (Han et al., 2015)
- **Fisher pruning (FP):** Pruning based on global Fisher information
- **FairDP + MP:** Train with FairDP (Xu et al., 2020), then apply standard magnitude pruning
- **FairPrune (no DP):** Our method applied to non-DP models (ablation)
- **Bi-level Fair Pruning (BFP):** Dai et al. (2023) — joint fairness-pruning optimization
- **No pruning (DP-only):** Upper bound on fairness for each DP budget

## Related Work

### Differential Privacy and Fairness

Bagdasaryan & Shmatikov (2019) first demonstrated that DP-SGD has disparate impact on model accuracy, with underrepresented classes suffering larger accuracy drops. Esipova et al. (2023) identified gradient misalignment from inequitable clipping as the root cause and proposed methods to mitigate it (ICLR 2023 Spotlight). Uniyal et al. (2021) compared DP-SGD and PATE, finding PATE has less disparate impact. Demelius et al. (2025) showed that hyperparameter tuning alone cannot reliably mitigate DP-SGD's disparate impact. Tran et al. (2023) proposed FairDP with certified fairness bounds under DP. Our work differs by studying how compression *compounds* these existing fairness costs.

### Model Compression and Fairness

Hooker et al. (2020) introduced Compression Identified Exemplars (CIE) and showed pruning amplifies bias against underrepresented features. Iofinova et al. (2023) provided in-depth analysis and countermeasures for bias in pruned vision models (CVPR 2023). Dai et al. (2023) proposed bi-level optimization for joint fairness and pruning. Our work is the first to study compression fairness specifically in the context of DP-trained models.

### Privacy and Model Compression

Li et al. (2025, CompLeak) showed that compression exacerbates privacy leakage through membership inference attacks. Shang et al. (2025, WeMem, NDSS) showed iterative pruning increases memorization and proposed defenses. Adamczewski et al. (2023) proposed pre-pruning to help DP-SGD scale. Gao et al. (2025, DPQuant) introduced dynamic quantization scheduling for DP-SGD. These works study privacy-compression or fairness-compression in isolation; we study the three-way interaction.

### Disparate Privacy Vulnerability

Kulynych et al. (2019) showed membership inference attacks have disparate vulnerability across subgroups. Chang & Shokri (2022) studied disparate effects of MIA and defenses. Our work connects these findings to the compression pipeline: we measure whether compression amplifies disparate privacy vulnerability, especially in DP-trained models.

## Experiments

### Datasets

1. **CelebA** (Liu et al., 2015): 200K celebrity face images with 40 binary attributes. We use "Smiling" as the target and analyze fairness across gender (Male/Female) and age (Young/Old). This is the standard benchmark for studying fairness in computer vision.

2. **UTKFace** (Zhang et al., 2017): 23K face images with age, gender, and ethnicity annotations. We use age-group classification (binned into 5 groups) and analyze fairness across ethnicity.

3. **CIFAR-10 with synthetic imbalance**: We create imbalanced versions with minority classes subsampled to 10% and 1% of their original size, simulating real-world class imbalance.

### Models

- **ResNet-18**: Standard architecture for CelebA and CIFAR-10 experiments
- **MobileNetV2**: Lightweight architecture commonly used in deployment (already compressed by design)

### Training Configuration

- **DP-SGD** via Opacus (PyTorch) with privacy budgets ε ∈ {1, 2, 4, 8, ∞}
- Batch size: 256, with Poisson sampling for DP
- Clipping norm: tuned per dataset via grid search over {0.1, 0.5, 1.0, 2.0}
- Training: 30 epochs for CelebA, 50 epochs for CIFAR-10/UTKFace
- δ = 1/N (where N = dataset size)

### Compression Configuration

- **Unstructured pruning** at sparsity levels {30%, 50%, 70%, 90%}
- **Structured pruning** (filter-level) at sparsity levels {30%, 50%, 70%}
- **Post-training quantization** to INT8 and INT4
- **Fine-tuning after pruning**: 5 epochs (with DP-SGD if applicable)

### Evaluation Metrics

1. **Accuracy metrics**: Overall accuracy, per-subgroup accuracy, worst-group accuracy
2. **Fairness metrics**: Accuracy gap (max − min subgroup), equalized odds difference, demographic parity difference
3. **Privacy metrics**: MIA success rate (balanced accuracy) per subgroup, using LiRA (Carlini et al., 2022)
4. **Compounding ratio (CR)**: As defined above, measured for worst-group accuracy and MIA vulnerability
5. **Compression-fairness Pareto frontier**: Plotting compression ratio vs. fairness gap for each method

### Experimental Plan

| Experiment | Purpose | Estimated Time |
|---|---|---|
| Exp 1: Baselines (no DP, no compression) | Establish baseline accuracy/fairness | 1 hour |
| Exp 2: DP-SGD at varying ε | Measure DP-only fairness cost | 2 hours |
| Exp 3: Compression of standard models | Measure compression-only fairness cost | 0.5 hours |
| Exp 4: Compression of DP models | Measure combined fairness cost + CR | 0.5 hours |
| Exp 5: FairPrune-DP vs baselines | Evaluate proposed method | 1 hour |
| Exp 6: Mechanistic analysis | Weight distributions, gradient norms | 0.5 hours |
| Exp 7: MIA per subgroup | Privacy leakage disparities | 1.5 hours |
| **Total** | | **~7 hours** |

### Expected Results

1. **Compounding ratio > 1** for worst-group accuracy across all datasets and privacy budgets, especially at tight privacy (ε ≤ 4) and high compression (≥ 50% sparsity).

2. **Mechanistic evidence**: In DP-trained models, minority-relevant weights have systematically lower magnitude than in standard models, causing magnitude-based pruning to disproportionately remove them.

3. **FairPrune-DP** achieves comparable overall accuracy and compression to standard pruning while reducing the fairness gap by 30-50%.

4. **MIA disparities amplified**: Compression increases MIA success rate more for minority subgroups in DP-trained models than in standard models.

## Success Criteria

### What would confirm the hypothesis:
- Compounding ratio CR > 1.2 consistently across datasets and settings
- FairPrune-DP reduces worst-group accuracy gap by ≥ 20% relative to standard pruning of DP models, at comparable compression ratios
- Clear mechanistic evidence (weight magnitude distributions, pruning overlap analysis)

### What would refute the hypothesis:
- CR ≈ 1.0 consistently (effects are merely additive, not compounding)
- No significant difference in which weights are pruned between DP and non-DP models
- FairPrune-DP provides no benefit over standard pruning

### Partial confirmation scenarios:
- CR > 1 only at certain privacy budgets or compression rates → still publishable with narrower claims
- FairPrune-DP helps but less than expected → ablation-focused paper

## References

1. Bagdasaryan, E., & Shmatikov, V. (2019). Differential Privacy Has Disparate Impact on Model Accuracy. *NeurIPS 2019*. arXiv:1905.12101.

2. Hooker, S., Moorosi, N., Clark, G., Bengio, S., & Denton, E. (2020). Characterising Bias in Compressed Models. *arXiv:2010.03058*.

3. Esipova, M.S., Ghomi, A.A., Luo, Y., & Cresswell, J.C. (2023). Disparate Impact in Differential Privacy from Gradient Misalignment. *ICLR 2023 (Spotlight)*. arXiv:2206.07737.

4. Li, N., Gao, Y., Hu, H., Kuang, B., & Fu, A. (2025). CompLeak: Deep Learning Model Compression Exacerbates Privacy Leakage. *arXiv:2507.16872*.

5. Shang, J., Wang, J., Wang, K., Liu, J., Jiang, N., Armanuzzaman, M., & Zhao, Z. (2025). Defending Against Membership Inference Attacks on Iteratively Pruned Deep Neural Networks. *NDSS 2025*.

6. Kulynych, B., Yaghini, M., Cherubin, G., Veale, M., & Troncoso, C. (2019). Disparate Vulnerability to Membership Inference Attacks. *arXiv:1906.00389*.

7. Adamczewski, K., He, Y., & Park, M. (2023). Pre-Pruning and Gradient-Dropping Improve Differentially Private Image Classification. *arXiv:2306.11754*.

8. Gao, Y., Tu, R., Pekhimenko, G., & Vijaykumar, N. (2025). DPQuant: Efficient and Differentially-Private Model Training via Dynamic Quantization Scheduling. *arXiv:2509.03472*.

9. Dai, Y., Li, G., Luo, F., Ma, X., & Wu, Y. (2023). Integrating Fairness and Model Pruning Through Bi-level Optimization. *arXiv:2312.10181*.

10. Uniyal, A., Naidu, R., Kotti, S., Singh, S., Kenfack, P.J., Mireshghallah, F., & Trask, A. (2021). DP-SGD vs PATE: Which Has Less Disparate Impact on Model Accuracy? *arXiv:2106.12576*.

11. Demelius, L., Kowald, D., Kopeinik, S., Kern, R., & Trügler, A. (2025). Private and Fair Machine Learning: Revisiting the Disparate Impact of Differentially Private SGD. *Transactions on Machine Learning Research*.

12. Tran, K., & Phan, L.N. (2023). FairDP: Achieving Fairness Certification with Differential Privacy. *arXiv:2305.16474*.

13. Iofinova, E., Peste, A., & Alistarh, D. (2023). Bias in Pruned Vision Models: In-Depth Analysis and Countermeasures. *CVPR 2023*.

14. Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022). Membership Inference Attacks From First Principles. *IEEE S&P 2022*.

15. Abadi, M., Chu, A., Goodfellow, I., McMahan, H.B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. *CCS 2016*.

16. Han, S., Pool, J., Tung, J., & Dally, W.J. (2015). Learning both Weights and Connections for Efficient Neural Networks. *NeurIPS 2015*.

17. Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. *ICCV 2015*.

18. Chang, H., & Shokri, R. (2022). Understanding Disparate Effects of Membership Inference Attacks and their Countermeasures. *ACM ASIACCS 2022*.
