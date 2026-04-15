# Calibrated Neural Collapse: Controlling Within-Class Representation Geometry for Reliable Supervised Learning

## Introduction

### Context

Deep neural networks trained with supervised objectives exhibit a remarkable geometric phenomenon in their final-layer representations known as **neural collapse** (NC) (Papyan et al., 2020). As training progresses beyond zero training error, four interrelated properties emerge: (NC1) within-class feature variability collapses — features converge to their class means; (NC2) class means converge to the vertices of a simplex equiangular tight frame (ETF); (NC3) the last-layer classifier converges to a scaled version of the class means (self-duality); and (NC4) classification simplifies to nearest-class-center decision-making. Neural collapse has been shown to be globally optimal in deep regularized networks (2025) and has become a central lens for understanding supervised representation learning.

Separately, **model calibration** — the alignment between predicted confidence and actual correctness probability — is critical for reliable deployment of deep learning systems (Guo et al., 2017). Overconfident predictions can lead to catastrophic failures in safety-critical applications such as medical diagnosis, autonomous driving, and financial decision-making. Modern neural networks are notoriously miscalibrated, typically exhibiting severe overconfidence (Guo et al., 2017).

### Problem Statement

We identify a fundamental tension between neural collapse and calibration that has not been formally characterized. As NC1 drives within-class features to collapse onto their class means, all samples — regardless of their proximity to decision boundaries — produce nearly identical, maximally confident predictions. A sample that lies close to the decision boundary between two classes produces the same high-confidence prediction as a sample that sits squarely in the center of its class cluster. This collapse of within-class variability destroys precisely the geometric information needed for calibrated uncertainty estimation: the distance from a sample's representation to the nearest decision boundary.

### Key Insight

We observe that within-class feature variance serves as a natural proxy for prediction uncertainty. In a well-structured representation space, samples near decision boundaries should have features that reflect this ambiguity — they should be farther from their class center or closer to neighboring class centers. Full neural collapse eliminates this signal. However, preventing collapse entirely would sacrifice the discriminative structure (NC2, NC3) that makes representations effective for classification. The key is to find the **calibration-optimal degree of partial collapse**: enough NC2 structure for accurate class separation, but enough preserved within-class variance to encode boundary-proximity information for calibration.

### Hypothesis

We hypothesize that there exists a calibration-optimal degree of neural collapse — specifically, a non-zero within-class covariance structure — that simultaneously maintains high classification accuracy and significantly improves calibration compared to both fully collapsed (standard training) and uncollapsed (under-trained) representations. We propose a training-time regularizer, **Calibrated Collapse Regularizer (CCR)**, that targets this optimum by maintaining a minimum within-class feature spread while preserving inter-class separation.

## Proposed Approach

### Overview

We propose **Calibrated Collapse Regularizer (CCR)**, a simple yet principled training-time regularizer that prevents excessive neural collapse (NC1) while preserving the beneficial geometric structure (NC2) of supervised representations. CCR adds a single penalty term to the standard training loss that maintains a minimum level of within-class feature variance, ensuring that the learned representations retain information about sample-to-boundary proximity that is essential for calibrated predictions.

### Method Details

#### Loss Formulation

The total training objective is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{CCR}}$$

where $\mathcal{L}_{\text{task}}$ is the standard cross-entropy loss and $\mathcal{L}_{\text{CCR}}$ is the calibrated collapse regularizer.

#### Calibrated Collapse Regularizer (CCR)

For a mini-batch $\mathcal{B}$ with features $\{z_i\}_{i \in \mathcal{B}}$ extracted from the penultimate layer, and class assignments $\{y_i\}_{i \in \mathcal{B}}$, we define:

1. **Running class prototypes** (exponential moving average):
   $$\mu_c^{(t)} = \alpha \cdot \mu_c^{(t-1)} + (1-\alpha) \cdot \bar{z}_c^{(t)}$$
   where $\bar{z}_c^{(t)}$ is the mean of class-$c$ features in the current batch, and $\alpha$ is the momentum coefficient.

2. **Within-class spread** for class $c$:
   $$S_c = \frac{1}{|\mathcal{B}_c|} \sum_{i \in \mathcal{B}_c} \|z_i - \mu_c\|^2$$
   where $\mathcal{B}_c = \{i \in \mathcal{B} : y_i = c\}$.

3. **CCR loss** (hinge-style variance floor):
   $$\mathcal{L}_{\text{CCR}} = \frac{1}{|\mathcal{C}_\mathcal{B}|} \sum_{c \in \mathcal{C}_\mathcal{B}} \max\left(0, \; \tau - S_c\right)$$
   where $\mathcal{C}_\mathcal{B}$ is the set of classes present in the batch and $\tau > 0$ is the target minimum within-class spread.

The hinge formulation is deliberate: it only activates when within-class spread drops below $\tau$, allowing the model to naturally form well-separated clusters while preventing excessive collapse. When $S_c \geq \tau$ for all classes, the regularizer contributes zero gradient, and training proceeds as standard supervised learning.

#### Adaptive Threshold via Inter-Class Distance

Rather than using a fixed $\tau$, we propose an **adaptive threshold** that scales with the inter-class separation:

$$\tau_{\text{adaptive}} = \gamma \cdot \min_{c \neq c'} \|\mu_c - \mu_{c'}\|^2$$

where $\gamma \in (0, 1)$ controls what fraction of the minimum inter-class distance should be preserved as within-class spread. This ensures the regularizer adapts to the evolving geometry of the representation space during training:
- Early in training (large inter-class distances are not yet established), the threshold is permissive.
- Late in training (when collapse accelerates), the threshold scales with the geometric structure.

#### Spectral Extension (CCR-S)

For a richer characterization, we also propose a **spectral variant** that controls the effective dimensionality of within-class features, not just their total variance:

$$\mathcal{L}_{\text{CCR-S}} = -\frac{1}{|\mathcal{C}_\mathcal{B}|} \sum_{c \in \mathcal{C}_\mathcal{B}} \text{EffRank}(\hat{\Sigma}_c)$$

where $\text{EffRank}(\hat{\Sigma}_c) = \exp\left(-\sum_k p_k \log p_k\right)$ with $p_k = \lambda_k / \sum_j \lambda_j$ and $\lambda_k$ are the eigenvalues of the within-class covariance $\hat{\Sigma}_c$ (maintained as a running estimate). This prevents both variance collapse (NC1) and dimensional collapse (all variance concentrating in a single direction).

### Key Innovations

1. **Calibration-motivated partial collapse**: Unlike prior work that controls NC for OOD detection (Harun et al., 2025) or analyzes NC-calibration connections post-hoc (Guo et al., 2025), we explicitly design a training objective targeting the calibration-optimal NC degree.

2. **Adaptive threshold**: The threshold scales with inter-class geometry, requiring no dataset-specific tuning of the absolute spread level.

3. **Minimal overhead**: CCR requires only maintaining running class prototypes and computing within-class distances — negligible cost compared to the forward/backward pass.

4. **Training-time intervention**: Unlike post-hoc methods (temperature scaling, Feature Clipping), CCR shapes the representation during learning, producing inherently better-calibrated features that benefit any downstream calibration method applied on top.

### Theoretical Analysis

Under simplifying assumptions (Gaussian within-class features, simplex ETF class means, shared covariance), we derive a relationship between the within-class covariance magnitude and expected calibration error:

**Proposition (informal):** Let features follow $z | y=c \sim \mathcal{N}(\mu_c, \sigma^2 I)$ with class means at simplex ETF vertices. The expected calibration error of the Bayes-optimal classifier satisfies:
- As $\sigma \to 0$ (full NC1 collapse): $\text{ECE} \to |1 - \text{Accuracy}|$ (maximally miscalibrated when accuracy < 1)
- As $\sigma \to \infty$ (no collapse): $\text{ECE} \to 0$ but accuracy $\to 1/K$ (random chance)
- There exists $\sigma^* > 0$ minimizing ECE subject to accuracy $\geq$ threshold

This establishes that the calibration-optimal within-class variance is neither zero nor infinite, motivating our regularization approach. We will formalize this with explicit ECE bounds as a function of $\sigma$ and the number of classes $K$.

## Related Work

### Neural Collapse

Neural collapse was first identified by Papyan, Han, and Donoho (2020) as a terminal phase of training in deep classifiers. Subsequent work established NC as globally optimal under various settings: Zhu et al. (2021) proved it for the unconstrained features model; recent work (2025) extended proofs to deep regularized ResNets and Transformers. Xu and Liu (2023) proposed the Variability Collapse Index (VCI) to quantify the degree of NC1. Our work builds on this understanding but argues that full NC1 is not always desirable — calibration benefits from partial collapse.

### Controlling Neural Collapse

Most closely related to our work, **Harun, Gallardo, and Kanan (2025)** proposed controlling the degree of neural collapse for OOD detection and transfer learning. They use entropy regularization in the encoder to mitigate NC while enforcing NC via fixed ETF projectors in the classifier. Their key finding — that stronger NC improves OOD detection but degrades generalization — motivates our investigation of how NC degree affects calibration. Our work differs in three ways: (1) we target calibration as the primary objective, (2) we use a variance-floor regularizer rather than entropy regularization, providing more direct geometric control, and (3) we provide theoretical analysis linking NC1 degree to ECE.

### Calibration Methods

**Post-hoc methods:** Temperature scaling (Guo et al., 2017) applies a single learned temperature to logits. Platt scaling (Platt, 1999) and histogram binning (Zadrozny & Elkan, 2001) are classical approaches. Feature Clipping (Tao et al., 2024) clips feature magnitudes post-hoc to reduce overconfidence — notably, they observe that high-ECE samples have larger feature variance, which supports our hypothesis from a different angle.

**Training-time methods:** Label smoothing (Szegedy et al., 2016; Müller et al., 2019) and Mixup (Zhang et al., 2018) improve calibration as a side effect of regularization. Guo et al. (2025, TMLR) analyzed these through the NC lens, showing label smoothing accelerates NC and can cause overconfidence through excessive NC1. BalCAL (Ni et al., 2025) balances a learnable classifier with a fixed ETF classifier for calibration — but operates at the classifier level, not the representation level.

**Representation-aware calibration:** Park et al. (2024) analyzed how soft-label regularization affects calibration through the representation space, finding that regularization reduces feature magnitudes and increases cosine similarity to class centers. Ghosal et al. (2025) proposed RelCal, which guides feature selection per class. Our approach is complementary — we directly regularize the within-class geometric structure of representations during training.

### Supervised Contrastive Learning

Supervised contrastive learning (SupCon; Khosla et al., 2020) learns representations by pulling same-class features together and pushing different-class features apart. VarCon (Wang et al., 2025) reformulates SupCon as variational inference with adaptive temperature. ProCo (Du et al., 2024) uses von Mises-Fisher distributions. I-Con (Alshammari et al., 2025) unifies many representation learning losses under a single information-theoretic framework. Our CCR regularizer can be combined with any of these supervised representation learning objectives.

## Experiments

### Datasets

1. **CIFAR-10** (10 classes, 50K train / 10K test): Standard benchmark for calibration studies.
2. **CIFAR-100** (100 classes, 50K train / 10K test): More challenging with fine-grained classes — tests scaling of CCR to many classes.
3. **TinyImageNet** (200 classes, 100K train / 10K val): Larger-scale validation with more classes and higher resolution (64×64).

### Architectures

- **ResNet-18** and **ResNet-50**: Standard architectures used in neural collapse and calibration literature, enabling direct comparison with prior work.

### Training Protocol

- SGD with momentum 0.9, weight decay 5e-4
- Cosine annealing learning rate schedule, initial LR 0.1
- 200 epochs for CIFAR-10/100, 100 epochs for TinyImageNet
- Batch size 256
- Standard data augmentation (random crop, horizontal flip, normalization)

### Baselines

1. **Standard CE**: Cross-entropy training (full neural collapse baseline)
2. **Label Smoothing** (LS): CE with soft targets (ε=0.1)
3. **Mixup**: Input mixing augmentation (α=0.2)
4. **Temperature Scaling** (TS): Post-hoc temperature on standard CE model
5. **SupCon + Linear**: Supervised contrastive pretraining + linear classifier
6. **Entropy Regularization**: Adapted from Harun et al. (2025) for calibration
7. **Feature Clipping**: Post-hoc method from Tao et al. (2024)
8. **CCR (ours)**: With fixed threshold τ
9. **CCR-adaptive (ours)**: With adaptive threshold
10. **CCR-S (ours)**: Spectral variant

### Metrics

**Classification performance:**
- Top-1 accuracy
- Top-5 accuracy (for CIFAR-100 and TinyImageNet)

**Calibration:**
- Expected Calibration Error (ECE, 15 bins)
- Maximum Calibration Error (MCE)
- Adaptive ECE (AdaECE)
- Reliability diagrams
- Negative log-likelihood (NLL)

**Neural collapse metrics:**
- NC1: $\text{tr}(\Sigma_W \Sigma_B^{-1}) / K$ (within-class variability relative to between-class)
- NC2: Std of pairwise cosine similarities of centered class means (ETF convergence)
- NC3: $\|\hat{W} - \hat{M}\|_F$ (self-duality gap)
- NC4: Nearest-class-center classification accuracy vs. network accuracy

**OOD detection** (to compare with Harun et al.):
- AUROC using maximum softmax probability
- AUROC using Mahalanobis distance in feature space
- OOD datasets: SVHN (for CIFAR models), LSUN, iSUN, Textures

**Transfer learning** (optional, time permitting):
- Linear probe accuracy on downstream tasks after representation learning

### Key Experiments

**Experiment 1: CCR effectiveness.** Train ResNet-18/50 on CIFAR-10/100 with and without CCR. Compare accuracy, ECE, and NC metrics against all baselines. Main result table.

**Experiment 2: NC-calibration Pareto frontier.** Sweep λ (CCR strength) and γ (adaptive threshold fraction) to map the full accuracy-vs-ECE trade-off curve. Compare this frontier against the frontier achievable with label smoothing, Mixup, and entropy regularization.

**Experiment 3: NC metrics tracking.** Plot NC1, NC2, NC3, NC4 metrics throughout training for standard CE vs. CCR. Show that CCR selectively controls NC1 while preserving NC2/NC3.

**Experiment 4: Layer-wise analysis.** Measure within-class variance and calibration contribution at different network layers (after each residual block). Investigate whether CCR applied only to the penultimate layer is sufficient or whether deeper regularization helps.

**Experiment 5: Combination with post-hoc calibration.** Apply temperature scaling on top of CCR-trained models. Show that CCR + TS achieves better calibration than TS alone, demonstrating that better representations benefit post-hoc methods.

**Experiment 6: Ablation study.**
- Fixed vs. adaptive threshold
- CCR vs. CCR-S (spectral variant)
- Sensitivity to λ, γ, momentum α
- Effect of batch size on CCR stability

**Experiment 7: OOD detection.** Compare CCR against Harun et al.'s entropy regularization on OOD detection benchmarks. Investigate whether the calibration-optimal NC degree differs from the OOD-detection-optimal NC degree.

### Expected Results

1. CCR achieves 30-50% reduction in ECE compared to standard CE training, with less than 1% accuracy loss.
2. The accuracy-ECE Pareto frontier for CCR dominates the frontiers of label smoothing and Mixup.
3. NC1 metrics show controlled partial collapse under CCR while NC2/NC3 remain close to their fully collapsed values.
4. CCR + temperature scaling achieves the best overall calibration, outperforming TS applied to standard models.
5. The calibration-optimal NC1 value differs from the OOD-detection-optimal value, confirming that different downstream tasks require different collapse degrees.

### Computational Budget

| Experiment | Runs | Time/run | Total |
|---|---|---|---|
| Exp 1: Main results (3 datasets × 2 archs × 4 methods) | 24 | ~15 min | ~6 hrs |
| Exp 2: Pareto sweep (10 λ values × 1 dataset × 1 arch) | 10 | ~12 min | ~2 hrs |
| Exp 3-7: Analysis (reuse Exp 1/2 checkpoints) | — | — | ~0 hrs |
| **Total** | | | **~8 hrs** |

Note: Experiments 3-7 are analysis on saved checkpoints from Experiments 1-2, requiring minimal additional compute. Some Exp 1 runs can be parallelized or shortened by reducing epochs for ablations.

## Success Criteria

### Primary (must achieve):
- CCR reduces ECE by ≥20% relative to standard CE training on at least 2/3 datasets, with accuracy loss ≤1.5%.
- CCR demonstrates selective NC1 control: NC1 metric is measurably higher (less collapsed) under CCR while NC2 remains within 10% of the fully collapsed value.

### Secondary (strong paper):
- CCR's Pareto frontier dominates label smoothing and Mixup on at least 2/3 datasets.
- CCR + temperature scaling outperforms temperature scaling alone.
- Theoretical ECE-NC1 relationship is empirically validated (rank correlation between NC1 and ECE across sweep).

### Tertiary (bonus):
- Calibration-optimal NC1 differs measurably from OOD-optimal NC1 (confirming task-dependent optimal collapse).
- CCR-S (spectral variant) outperforms CCR on datasets with many classes (CIFAR-100, TinyImageNet).

### What would refute the hypothesis:
- If controlling NC1 does not significantly affect ECE (NC1 and calibration are independent).
- If preventing NC1 collapse requires sacrificing >3% accuracy (the trade-off is too steep).
- If post-hoc temperature scaling on standard models achieves equal or better calibration than CCR without any accuracy cost.

## References

1. Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training. *Proceedings of the National Academy of Sciences*, 117(40), 24652-24663.

2. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised contrastive learning. *Advances in Neural Information Processing Systems*, 33, 18661-18673.

3. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the International Conference on Machine Learning (ICML)*, 1321-1330.

4. Harun, M. Y., Gallardo, J., & Kanan, C. (2025). Controlling neural collapse enhances out-of-distribution detection and transfer learning. *Proceedings of the International Conference on Machine Learning (ICML)*. arXiv:2502.10691.

5. Guo, P., Chen, Y., & Weinberger, K. Q. (2025). Cross entropy versus label smoothing: A neural collapse perspective. *Transactions on Machine Learning Research (TMLR)*. arXiv:2402.03979.

6. Ni, H., Li, Z., Zhao, H., & Zhang, L. (2025). Balancing two classifiers via a simplex ETF structure for model calibration. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. arXiv:2504.10007.

7. Tao, L., Du, X., Zhu, T., & Li, L. (2024). Feature clipping for uncertainty calibration. *Proceedings of the AAAI Conference on Artificial Intelligence*. arXiv:2410.19796.

8. Xu, Y., & Liu, C. (2023). Quantifying the variability collapse of neural networks. *arXiv preprint* arXiv:2306.03440.

9. Park, J., Kim, S., Lee, J., & Hwang, S. J. (2024). Impact of regularization on calibration and robustness: From the representation space perspective. *arXiv preprint* arXiv:2410.03999.

10. Wang, Z., Li, B., & Ermon, S. (2025). Variational supervised contrastive learning. *arXiv preprint* arXiv:2506.07413.

11. Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. (2022). Fine-tuning can distort pretrained features and underperform out-of-distribution. *Proceedings of the International Conference on Learning Representations (ICLR)*. arXiv:2202.10054.

12. Zhu, Z., Ding, T., Zhou, J., Li, X., You, C., Sulam, J., & Qu, Q. (2021). A geometric analysis of neural collapse with unconstrained features. *Advances in Neural Information Processing Systems*, 34, 29820-29834.

13. Alshammari, S., Hershey, J., Feldmann, T., Freeman, W. T., & Hamilton, M. (2025). I-Con: A unifying framework for representation learning. *Proceedings of the International Conference on Learning Representations (ICLR)*. arXiv:2504.16929.

14. Du, H., Wang, L., Song, M., & Huang, G. (2024). Probabilistic contrastive learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. arXiv:2403.06726.

15. Müller, R., Kornblith, S., & Hinton, G. (2019). When does label smoothing help? *Advances in Neural Information Processing Systems*, 32, 4694-4703.

16. Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). Mixup: Beyond empirical risk minimization. *Proceedings of the International Conference on Learning Representations (ICLR)*.

17. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2818-2826.

18. Aghajanyan, A., Shrivastava, A., Gupta, A., Goyal, N., Zettlemoyer, L., & Gupta, S. (2020). Better fine-tuning by reducing representational collapse. *arXiv preprint* arXiv:2008.03156.

19. Rafailov, R., Shao, T., Srinivasan, R., & Finn, C. (2025). Seq-VCR: Preventing collapse in intermediate transformer representations for enhanced reasoning. *Proceedings of the International Conference on Learning Representations (ICLR)*. arXiv:2411.02344.

20. Ghosal, A., et al. (2025). Better features, better calibration. *Proceedings of ECML PKDD*.

21. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 10(3), 61-74.

22. Cui, J., Zhang, Y., Liu, T., & Wang, C. (2025). An inclusive theoretical framework of robust supervised contrastive loss against label noise. *arXiv preprint* arXiv:2501.01130.
