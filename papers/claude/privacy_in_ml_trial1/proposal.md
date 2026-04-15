# MemPrune: Privacy-Enhancing Neural Network Pruning via Memorization-Aware Gradient Dispersion Criterion

## Introduction

### Context and Motivation

Model compression through neural network pruning is essential for deploying deep learning models on resource-constrained devices. However, recent work has revealed a troubling paradox: standard pruning techniques, particularly iterative magnitude pruning, significantly *increase* models' vulnerability to membership inference attacks (MIAs) (Yuan et al., 2022; Shang et al., 2025). This occurs because pruning forces the remaining weights to compensate for removed capacity, leading to tighter memorization of individual training samples.

This creates a critical tension for practitioners who need both efficient models (via pruning) and privacy protection. Current approaches address this tension through post-hoc defenses---applying regularization or differential privacy during the retraining phase after pruning (Shang et al., 2025). However, these approaches treat the symptom (increased memorization after pruning) rather than the cause (the pruning criterion itself is agnostic to memorization).

### Problem Statement

Standard pruning criteria---weight magnitude (Han et al., 2015), gradient sensitivity (Molchanov et al., 2019), or SNIP (Lee et al., 2019)---select weights for removal based on their contribution to the model's *overall* performance. They do not distinguish between weights that encode *generalizable knowledge* (useful for both training and unseen data) and weights that encode *memorized information* (useful only for specific training samples). As a result, standard pruning may preferentially retain memorization-heavy weights while removing generalization-important weights, amplifying privacy risks.

### Key Insight

Agarwal et al. (2022) demonstrated that the Variance of Gradients (VoG), computed per *sample* across training epochs, effectively identifies difficult and memorized examples. We extend this insight from the sample domain to the weight domain: we observe that weights contributing primarily to memorization exhibit a distinctive gradient signature---**high gradient dispersion across individual training samples at a fixed checkpoint**. A weight that has memorized specific training examples will respond very differently (high per-sample gradient variance) to different inputs, because it has learned sample-specific rather than general features. In contrast, weights encoding general features will exhibit relatively consistent (low-variance) gradients across samples.

Critically, our per-weight Gradient Dispersion Score (GDS) differs from VoG in both granularity and purpose: VoG operates per-sample across time to rank example difficulty, whereas GDS operates per-weight across samples at a single checkpoint to rank weight memorization contribution for pruning.

### Theoretical Grounding

Our hypothesis that high per-weight gradient variance signals memorization is grounded in two complementary theoretical perspectives:

**1. Coherent Gradients Hypothesis (Chatterjee, 2020; Zielinski et al., 2020)**: The overall gradient in SGD is the sum of per-example gradients. It is strongest in *coherent* directions---those that reduce the loss on many examples simultaneously. Weights encoding generalizable features are primarily updated by these coherent gradient directions, leading to low per-sample gradient variance. Conversely, weights that memorize individual samples are updated by *incoherent* gradient signals that vary across examples, producing high per-sample gradient variance. By pruning high-GDS weights, we selectively remove the capacity used for incoherent (memorization-specific) updates.

**2. Information-theoretic perspective (Harutyunyan et al., ICML 2020)**: The mutual information between network weights and training labels, $I(W; Y|X)$, upper-bounds generalization error. Weights with high gradient dispersion across samples effectively encode more information about individual training labels (high mutual information), because their gradient updates are driven by sample-specific rather than shared signals. Pruning these weights reduces $I(W; Y|X)$, simultaneously improving generalization bounds and reducing memorization that enables privacy attacks.

These frameworks predict a testable consequence: if we inject known label noise (creating "canary" memorized samples), the weights with highest GDS should disproportionately contribute to the gradients of these canary samples. We design an explicit validation experiment around this prediction (Experiment 4).

### Hypothesis

By using per-weight gradient dispersion as a pruning criterion---removing high-dispersion (memorization-heavy) weights first---we can simultaneously achieve model compression and privacy enhancement. This approach should yield pruned models that are:
1. Less vulnerable to membership inference attacks than magnitude-pruned models at the same sparsity level
2. Comparable or better in test accuracy, since removing memorization does not harm generalization
3. Complementary with differential privacy (DP-SGD) for stronger privacy guarantees

**Important caveat**: Following Feldman (2020), we acknowledge that some memorization may be necessary for generalization on rare subpopulations. Therefore, we include per-class and subgroup accuracy analysis to ensure MemPrune does not disproportionately harm underrepresented classes. Our goal is to remove *excessive* memorization (privacy-leaking) rather than all memorization.

## Proposed Approach

### Overview

We propose **MemPrune**, a pruning method that uses a novel **Gradient Dispersion Score (GDS)** to identify and remove weights that contribute to memorization rather than generalization. The method consists of three phases:

1. **Gradient Dispersion Profiling**: Compute per-weight GDS by measuring the variance of per-sample gradients across training data
2. **Memorization-Aware Pruning**: Remove weights with the highest GDS (most memorization-contributing) up to the target sparsity
3. **Fine-Tuning**: Retrain the pruned model, optionally with DP-SGD for formal privacy guarantees

### Method Details

#### Phase 1: Gradient Dispersion Score (GDS) Computation

Given a trained model with parameters $\theta = \{w_1, w_2, \ldots, w_d\}$ and training dataset $D_{train} = \{(x_i, y_i)\}_{i=1}^{N}$, we compute the GDS for each weight $w_j$:

$$\text{GDS}(w_j) = \text{Var}_{(x,y) \sim D_{train}} \left[ \frac{\partial \mathcal{L}(f_\theta(x), y)}{\partial w_j} \right]$$

where $\mathcal{L}$ is the loss function and $f_\theta$ is the model. In practice, we estimate this using a random subset $S \subset D_{train}$ of size $|S| = M$ (e.g., $M = 5000$):

$$\widehat{\text{GDS}}(w_j) = \frac{1}{M-1} \sum_{i=1}^{M} \left( g_j^{(i)} - \bar{g}_j \right)^2$$

where $g_j^{(i)} = \frac{\partial \mathcal{L}(f_\theta(x_i), y_i)}{\partial w_j}$ and $\bar{g}_j = \frac{1}{M} \sum_i g_j^{(i)}$.

**Normalization**: To account for scale differences across layers, we normalize GDS within each layer:

$$\text{GDS}_{\text{norm}}(w_j) = \frac{\text{GDS}(w_j) - \mu_l}{\sigma_l + \epsilon}$$

where $\mu_l$ and $\sigma_l$ are the mean and standard deviation of GDS values within layer $l$.

**Memory-efficient computation**: For large models (e.g., VGG-16 with 14.7M parameters), computing per-sample gradients via vmap can be memory-intensive. We use chunked computation: process micro-batches of $B_{\text{micro}} = 8$--$16$ samples at a time, accumulating gradient statistics (running mean and variance via Welford's online algorithm) across chunks. For VGG-16, each micro-batch of 16 samples requires approximately $14.7\text{M} \times 4\text{B} \times 16 \approx 940\text{MB}$ for the per-sample gradient tensor, well within the 48GB A6000 budget even with model parameters, activations, and overhead.

#### Phase 2: Memorization-Aware Pruning

Given target sparsity $s \in (0, 1)$, we create a binary mask $m \in \{0, 1\}^d$:

$$m_j = \begin{cases} 0 & \text{if } \text{GDS}_{\text{norm}}(w_j) \text{ is in the top } s \text{ fraction} \\ 1 & \text{otherwise} \end{cases}$$

The pruned model uses parameters $\theta' = m \odot \theta$. This removes the highest-GDS weights first, targeting memorization-specific parameters.

**Hybrid criterion (ablation)**: We also explore combining GDS with magnitude:

$$\text{Score}(w_j) = \alpha \cdot \text{GDS}_{\text{norm}}(w_j) + (1 - \alpha) \cdot (-|w_j|_{\text{norm}})$$

where higher score means higher pruning priority (high GDS, low magnitude). This is explored as a single ablation point ($\alpha = 0.5$) rather than a primary contribution.

#### Phase 3: Fine-Tuning

After pruning, we fine-tune the model for $T_{ft}$ epochs with standard SGD. For one experiment (Experiment 3), we also evaluate fine-tuning with DP-SGD (Abadi et al., 2016) to test whether memorization-aware pruning is complementary with formal privacy mechanisms.

### Key Innovations

1. **From sample-level to weight-level gradient variance for pruning**: While Agarwal et al. (2022) use per-sample gradient variance across epochs to identify hard examples, we adapt this signal to the per-weight domain at a single checkpoint, producing a pruning criterion that directly targets memorization-contributing weights.
2. **Privacy-motivated pruning criterion**: Unlike Huang et al. (2020), who prove that standard pruning is theoretically equivalent to adding DP noise, and Adamczewski & Park (2023), who use standard pruning to reduce DP-SGD dimensionality, we design a pruning criterion *specifically optimized* to remove memorization.
3. **Theoretically grounded**: Our criterion is motivated by the Coherent Gradients Hypothesis and information-theoretic memorization bounds, and we validate it with a controlled canary-based experiment.

## Related Work

### Membership Inference Attacks (MIA)

MIAs aim to determine whether a specific data point was used to train a model. Shokri et al. (2017) introduced shadow model-based MIAs. Yeom et al. (2018) showed the connection between overfitting and MIA vulnerability. Carlini et al. (2022) proposed LiRA (Likelihood Ratio Attack), currently the strongest MIA method, using the likelihood ratio test between "in" and "out" model distributions. Zarifzadeh et al. (2024) proposed RMIA, an efficient reference-based MIA.

**How we differ**: We do not propose a new attack but rather a defense that operates at the model compression stage.

### Pruning and Privacy

Yuan et al. (2022, USENIX Security) systematically studied how pruning affects MIA vulnerability, finding that both one-shot and iterative pruning increase privacy risks compared to unpruned models. The key mechanism is that pruning forces the remaining weights to overfit (memorize) training data more tightly.

Shang et al. (2025, NDSS) proposed WeMem, which defends against MIAs during iterative pruning by (1) reducing data reuse during retraining and (2) addressing inherent memorability of samples. Their approach modifies the *retraining process* but not the *pruning criterion*.

Wang et al. (2021, IJCAI) showed that aggressive one-shot pruning can reduce MIA vulnerability, but at significant accuracy cost. Their pruning criterion remains standard (magnitude-based); the privacy benefit comes from aggressive capacity reduction.

Huang et al. (2020) proved that pruning a neural network layer is theoretically equivalent to adding a certain amount of differentially private noise to its hidden-layer activations. This establishes that pruning *can* serve as a privacy mechanism, but their work does not propose a memorization-aware pruning criterion. **Our contribution**: We design a pruning criterion that *maximizes* the privacy benefit by specifically targeting memorization-encoding weights.

Adamczewski & Park (2023) explored using neural network pruning to improve the scalability of DP-SGD by reducing the dimensionality of the parameter space. They use standard pruning criteria. **Our contribution**: Rather than using pruning to make DP-SGD more efficient, we design a pruning criterion that directly removes memorization.

**How we differ from all prior pruning-privacy work**: All existing approaches either (a) analyze how standard pruning affects privacy (Yuan et al., Huang et al.), (b) modify the retraining process after standard pruning (Shang et al., Wang et al.), or (c) use standard pruning to improve DP-SGD scalability (Adamczewski & Park). MemPrune is the first to design a *pruning criterion explicitly optimized for memorization removal*, targeting the root cause of privacy leakage in pruned models.

### Gradient Variance and Memorization

Agarwal et al. (2022, CVPR) proposed Variance of Gradients (VoG), computing per-sample gradient variance *across training epochs* to estimate example difficulty, demonstrating that high-VoG samples are disproportionately memorized.

**How we differ**: VoG operates per-sample across time; our GDS operates per-weight across samples at a single checkpoint. The conceptual leap is translating the gradient-variance-as-memorization-signal from the sample domain to the weight domain.

Chatterjee (2020) proposed the Coherent Gradients Hypothesis: SGD gradients are strongest in directions that reduce loss on many examples, explaining why neural networks generalize despite overparameterization. Zielinski et al. (2020) validated this at scale on ImageNet with ResNet and VGG models, showing that suppressing incoherent (weak) gradient directions reduces overfitting and memorization. **Our connection**: GDS directly measures gradient incoherence per weight; pruning high-GDS weights removes the capacity for incoherent updates.

Harutyunyan et al. (ICML 2020) showed that controlling mutual information $I(W; Y|X)$ between weights and training labels reduces memorization and improves generalization bounds. **Our connection**: High-GDS weights encode more sample-specific label information (high mutual information), so pruning them reduces $I(W; Y|X)$.

### Variance-Based Pruning for Compression

Berisha et al. (2025, ICCV) proposed variance-based pruning that selects neurons for structured pruning based on lowest *activation variance*.

**How we differ**: (1) Different variance signal: activation variance vs. per-sample gradient variance; (2) Different direction: they prune low-variance, we prune high-variance; (3) Different granularity: structured vs. unstructured; (4) Different objective: compression vs. privacy.

### Differential Privacy in Deep Learning

Abadi et al. (2016) introduced DP-SGD, which clips per-sample gradients and adds calibrated Gaussian noise. While providing formal privacy guarantees, DP-SGD significantly degrades model utility.

**How we differ**: MemPrune provides empirical privacy protection through pruning. When combined with DP-SGD, it can potentially achieve the same formal privacy level with less noise.

### Memorization in Deep Learning

Feldman (2020) showed that some memorization is necessary for optimal generalization, particularly for rare subpopulations. Feldman and Zhang (2020) proposed methods to estimate per-sample memorization via influence estimation.

**How we differ**: We use gradient dispersion as a computationally efficient proxy for memorization, applicable per-weight for pruning. We explicitly address Feldman's long-tail concern with per-class accuracy analysis.

## Experiments

### Experimental Setup

**Datasets**:
- CIFAR-10 (10 classes, 50K train / 10K test)
- CIFAR-100 (100 classes, 50K train / 10K test)
- ImageNet-100 (100-class subset of ImageNet, ~130K train / 5K test) --- for demonstrating practical relevance at larger scale

**Models**:
- ResNet-18 (11.2M parameters) --- primary model for all datasets
- VGG-16 (14.7M parameters) --- CIFAR-10/100 only, with chunked per-sample gradient computation ($B_{\text{micro}}=16$)

**Sparsity levels**: 50%, 70%, 90%, 95%

**Privacy evaluation**:
- Loss-based MIA (Yeom et al., 2018)
- LiRA (Carlini et al., 2022) with 16 shadow models for key configurations (ResNet-18/CIFAR-10 at 70% and 90% sparsity for all methods)
- Metric-based MIA using prediction entropy, modified entropy, and confidence

**Baselines**:
- No pruning (dense model)
- Random pruning
- Magnitude pruning (Han et al., 2015)
- Gradient sensitivity pruning (Molchanov et al., 2019)
- WeMem (Shang et al., 2025) --- magnitude pruning with privacy-aware retraining
- Magnitude pruning + DP-SGD fine-tuning

### Experiment 1: Privacy-Utility Tradeoff Across Pruning Criteria

Compare all pruning methods across sparsity levels on CIFAR-10, CIFAR-100, and ImageNet-100. For each configuration, measure:
- Test accuracy (overall)
- MIA success rate (true positive rate at fixed low false positive rate)
- MIA AUC-ROC

Expected result: MemPrune achieves lower MIA vulnerability than magnitude pruning at matched accuracy, particularly at high sparsity (90-95%).

### Experiment 2: Per-Class and Subgroup Fairness Analysis

Evaluate whether MemPrune disproportionately harms rare or underrepresented classes:
- Report per-class accuracy for all 100 classes on CIFAR-100
- Compare the accuracy distribution (mean, min, std across classes) between MemPrune and magnitude pruning
- Compute per-class MIA AUC to check whether privacy improvements are uniform
- Specifically analyze the bottom-10 (lowest accuracy) classes

This directly addresses the Feldman (2020) concern.

### Experiment 3: Combination with DP-SGD

Compare MemPrune + DP-SGD fine-tuning with magnitude pruning + DP-SGD fine-tuning at privacy levels $\varepsilon \in \{1, 4, 8\}$. Expected result: MemPrune + DP-SGD achieves higher accuracy at the same $\varepsilon$, because memorization-aware pruning removes the memorization burden before DP-SGD is applied.

### Experiment 4: GDS Validation via Label-Noise Canaries (Critical)

Validate that GDS captures memorization using ground-truth memorized samples:

1. **Canary injection**: Train a model on CIFAR-10 with 10% of labels randomly flipped (canary samples with known ground-truth memorization status).
2. **GDS computation**: Compute per-weight GDS on this model.
3. **Per-weight contribution to canaries**: For each weight $w_j$, compute its average gradient magnitude on canary samples vs. clean samples. Define the *canary contribution ratio* $\text{CCR}(w_j) = \frac{\mathbb{E}_{\text{canary}}[|g_j|]}{\mathbb{E}_{\text{clean}}[|g_j|]}$.
4. **Correlation analysis**: Measure Spearman correlation between GDS and CCR. If GDS captures memorization, high-GDS weights should have high CCR (they disproportionately serve canary/memorized samples).
5. **Pruning validation**: Compare MIA vulnerability when pruning high-GDS weights vs. random weights vs. magnitude-based weights on this noisy-label model. MemPrune should show the largest reduction in MIA vulnerability on canary samples.

This provides a controlled test with ground-truth memorization, avoiding the circularity of using proxy-based validation.

### Experiment 5: Comparison with Variance-Based Pruning

Directly compare MemPrune (prune high per-sample gradient variance weights) with:
- Activation variance pruning (inspired by Berisha et al., 2025): prune low activation-variance neurons
- Reverse-GDS pruning: prune low-GDS weights (ablation showing direction matters)

This demonstrates that the privacy benefit stems specifically from our memorization-motivated gradient dispersion criterion.

### Experiment 6: Ablation Study

- Effect of GDS estimation subset size $M$: 1000, 2000, 5000, 10000
- Layer-wise vs. global GDS normalization
- Hybrid criterion (GDS + magnitude) with $\alpha = 0.5$ as a single ablation point
- Effect of fine-tuning epochs $T_{ft}$

### Experiment 7: Robustness and Scale Analysis

- Test against multiple MIA methods (not just the one we optimize for)
- Report results with error bars over 3 random seeds
- ImageNet-100 results with ResNet-18 at 70% and 90% sparsity to demonstrate practical relevance beyond CIFAR
- LiRA evaluation: 16 shadow models for key configurations

### Compute Budget Estimate

| Component | Time (est.) |
|---|---|
| Train base models (ResNet-18 ×3 datasets, VGG-16 ×2 datasets) | ~1.5h |
| GDS computation (all models, chunked for VGG-16) | ~0.5h |
| Pruning + fine-tuning (all methods, sparsity levels, seeds) | ~3h |
| LiRA shadow models (16 models, CIFAR-10/ResNet-18 only) | ~1.5h |
| Ablations and analysis experiments | ~1h |
| **Total** | **~7.5h** |

VGG-16 per-sample gradient memory: with $B_{\text{micro}}=16$, peak GPU memory is ~4GB for the gradient tensor plus ~6GB for model + activations ≈ 10GB, well within 48GB. We use Welford's online algorithm for streaming variance computation, avoiding storage of all per-sample gradients.

## Success Criteria

### Primary (must achieve):
1. MemPrune reduces MIA AUC by at least 3-5 percentage points compared to magnitude pruning at 90% sparsity while maintaining within 1% test accuracy
2. MemPrune + DP-SGD outperforms magnitude pruning + DP-SGD at the same epsilon in terms of test accuracy
3. GDS-CCR Spearman correlation > 0.3 in the label-noise canary experiment (Experiment 4)
4. Per-class accuracy on CIFAR-100 shows MemPrune does not cause >2% additional degradation on bottom-10 classes vs. magnitude pruning

### Secondary (nice to have):
5. MemPrune at high sparsity achieves comparable MIA resistance to moderate-epsilon DP-SGD without formal DP mechanism
6. Consistent privacy improvement on ImageNet-100, demonstrating practical relevance beyond CIFAR

### Refutation criteria:
- If GDS shows no correlation with canary contribution ratio, the fundamental hypothesis is wrong
- If magnitude pruning already achieves low MIA rates at high sparsity, there's no room for improvement
- If MemPrune causes significant accuracy degradation (>3% compared to magnitude pruning), the approach is impractical
- If MemPrune disproportionately harms rare classes (>5% accuracy drop on bottom-10 CIFAR-100 classes vs. magnitude pruning), the Feldman concern is validated

## References

1. Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. *ACM CCS 2016*.

2. Adamczewski, K. & Park, M. (2023). Differential Privacy Meets Neural Network Pruning. *arXiv:2303.04612*.

3. Agarwal, C., D'souza, D., & Hooker, S. (2022). Estimating Example Difficulty Using Variance of Gradients. *CVPR 2022*.

4. Berisha, U., Mehnert, J., & Condurache, A. P. (2025). Variance-Based Pruning for Accelerating and Compressing Trained Networks. *ICCV 2025*.

5. Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022). Membership Inference Attacks From First Principles. *IEEE S&P 2022*.

6. Chatterjee, S. (2020). Coherent Gradients: An Approach to Understanding Generalization in Gradient Descent-Based Optimization. *ICLR 2020*.

7. Feldman, V. (2020). Does Learning Require Memorization? A Short Tale about a Long Tail. *STOC 2020*.

8. Feldman, V. & Zhang, C. (2020). What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation. *NeurIPS 2020*.

9. Han, S., Pool, J., Tung, J., & Dally, W. J. (2015). Learning both Weights and Connections for Efficient Neural Networks. *NeurIPS 2015*.

10. Harutyunyan, H., Reing, K., Ver Steeg, G., & Galstyan, A. (2020). Improving Generalization by Controlling Label-Noise Information in Neural Network Weights. *ICML 2020*.

11. Huang, Y., Su, Y., Ravi, S., Song, Z., Arora, S., & Li, K. (2020). Privacy-preserving Learning via Deep Net Pruning. *arXiv:2003.01876*.

12. Lee, N., Ajanthan, T., & Torr, P. H. S. (2019). SNIP: Single-shot Network Pruning based on Connection Sensitivity. *ICLR 2019*.

13. Molchanov, P., Mallya, A., Tyree, S., Frosio, I., & Kautz, J. (2019). Importance Estimation for Neural Network Pruning. *CVPR 2019*.

14. Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership Inference Attacks Against Machine Learning Models. *IEEE S&P 2017*.

15. Wang, Y., Wang, C., Wang, Z., Zhou, S., Liu, H., Bi, J., Ding, C., & Rajasekaran, S. (2021). Against Membership Inference Attack: Pruning is All You Need. *IJCAI 2021*.

16. Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. *IEEE CSF 2018*.

17. Yuan, X., & Zhang, L. (2022). Membership Inference Attacks and Defenses in Neural Network Pruning. *USENIX Security 2022*.

18. Zarifzadeh, S., Liu, P., & Shokri, R. (2024). Low-Cost High-Power Membership Inference Attacks. *ICML 2024*.

19. Shang, J., Wang, J., Wang, K., Liu, J., Jiang, N., Armanuzzaman, M., & Zhao, Z. (2025). Defending Against Membership Inference Attacks on Iteratively Pruned Deep Neural Networks. *NDSS 2025*.

20. Zielinski, P., Krishnan, S., & Chatterjee, S. (2020). Weak and Strong Gradient Directions: Explaining Memorization, Generalization, and Hardness of Examples at Scale. *arXiv:2003.07422*.
