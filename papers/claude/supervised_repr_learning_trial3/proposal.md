# Confusion-Calibrated Supervised Contrastive Learning: Adaptive Class-Pair Reweighting from Training Dynamics

## 1. Introduction

### Context

Supervised contrastive learning (SupCon) has emerged as a powerful alternative to cross-entropy for learning visual representations (Khosla et al., 2020). By pulling together representations of same-class samples and pushing apart different-class samples on a normalized hypersphere, SupCon produces representations that transfer better, are more robust to corruptions, and achieve competitive or superior classification accuracy compared to cross-entropy training.

A key theoretical result about SupCon—and about supervised representation learning more broadly—is **neural collapse** (Papyan et al., 2020): during the terminal phase of training, last-layer features converge to class means that form a **simplex equiangular tight frame (ETF)**, where all class pairs are equidistant. Graf et al. (2021) proved that the SupCon loss also attains its minimum at this simplex ETF geometry.

### Problem Statement

While the simplex ETF is the optimal geometry for the training classification task under idealized conditions, it has a fundamental limitation: **it treats all class pairs identically**. In practice, some class pairs are inherently harder to distinguish (e.g., "leopard" vs. "jaguar") while others are trivially separable (e.g., "airplane" vs. "mushroom"). Standard SupCon allocates equal representational effort to all negative pairs, which means:

1. **Wasted capacity**: The model devotes equal effort to pushing apart already-well-separated classes.
2. **Insufficient separation of confusable classes**: Hard-to-distinguish classes receive no additional repulsion, leaving them inadequately separated.
3. **Suboptimal transfer**: The resulting equidistant geometry discards inter-class structure that could benefit downstream tasks, few-shot learning, and robustness.

### Key Insight

During training, a model's own confusion patterns—which classes it struggles to separate in the current representation space—provide a rich, dynamic signal about where representational capacity is most needed. We propose to **use this signal to adaptively calibrate the supervised contrastive loss**, increasing repulsion for class pairs that the model currently confuses and relaxing repulsion for pairs that are already well-separated.

### Hypothesis

**Adaptively reweighting negative pairs in supervised contrastive learning based on class-pair confusability scores derived from the model's evolving representation geometry will produce more discriminative representations (higher linear probe accuracy) and more structured representations (better transfer and few-shot performance) compared to standard SupCon, without requiring external knowledge such as label hierarchies.**

## 2. Proposed Approach

### 2.1 Overview

We introduce **Confusion-Calibrated Supervised Contrastive Learning (CC-SupCon)**, which augments the standard SupCon loss with a class-pair confusability matrix that dynamically modulates the repulsion strength between different classes. The confusability matrix is computed from the model's own representation space and updated throughout training with exponential moving averages, creating an implicit curriculum that focuses learning on the hardest class distinctions.

### 2.2 Background: Supervised Contrastive Loss

The standard SupCon loss for an anchor sample $i$ with label $y_i$ in a batch of $N$ augmented samples is:

$$\mathcal{L}_i^{\text{SupCon}} = -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

where $P(i)$ is the set of positives (same class as $i$), $A(i)$ is the set of all samples except $i$, $\mathbf{z}$ are L2-normalized embeddings, and $\tau$ is the temperature.

In this formulation, all negative pairs contribute equally to the denominator, regardless of which class they belong to.

### 2.3 Class-Pair Confusability Matrix

We maintain a running set of **class centroids** $\{\boldsymbol{\mu}_c\}_{c=1}^{C}$ computed as exponential moving averages of the L2-normalized embeddings for each class:

$$\boldsymbol{\mu}_c^{(t)} = \alpha \cdot \boldsymbol{\mu}_c^{(t-1)} + (1-\alpha) \cdot \frac{1}{|B_c|}\sum_{i \in B_c} \mathbf{z}_i$$

where $B_c$ is the set of class-$c$ samples in the current mini-batch and $\alpha$ is the momentum coefficient.

The **confusability score** between classes $i$ and $j$ is defined as:

$$\text{conf}(i, j) = \frac{\exp(\boldsymbol{\mu}_i \cdot \boldsymbol{\mu}_j / \tau_c)}{\sum_{k \neq i} \exp(\boldsymbol{\mu}_i \cdot \boldsymbol{\mu}_k / \tau_c)}$$

This softmax-based score measures how much class $j$ "competes" with other classes for proximity to class $i$ in the current representation space. Higher confusability means classes $i$ and $j$ are closer together (more confusable). The temperature $\tau_c$ controls the sharpness of the confusability distribution.

We symmetrize this as: $\hat{\text{conf}}(i,j) = \frac{1}{2}[\text{conf}(i,j) + \text{conf}(j,i)]$

### 2.4 Confusion-Calibrated SupCon Loss

We modify the SupCon loss to weight negative pairs by confusability:

$$\mathcal{L}_i^{\text{CC-SupCon}} = -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} w(y_i, y_a) \cdot \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

where for negative pairs ($y_a \neq y_i$):
$$w(y_i, y_a) = 1 + \beta \cdot \hat{\text{conf}}(y_i, y_a)$$

and for positive pairs ($y_a = y_i$): $w(y_i, y_a) = 1$.

Here $\beta \geq 0$ is a scalar hyperparameter controlling the strength of confusion calibration. When $\beta = 0$, CC-SupCon reduces to standard SupCon.

**Intuition**: When two classes are highly confusable (high $\hat{\text{conf}}$), their negative pair weights increase, amplifying the repulsion and encouraging the model to invest more representational capacity in separating them. As training progresses and classes become better separated, their confusability decreases, and the weights naturally relax.

### 2.5 Key Design Choices

1. **Class-pair level, not instance-level**: Unlike hard negative mining (Robinson et al., 2021), which operates on individual samples, our weighting operates at the class-pair level. This is more stable (aggregated over many samples) and directly targets the inter-class structure. We provide a formal variance reduction argument in Section 2.6.

2. **Self-discovered structure**: Unlike methods that require external label hierarchies (Zhang et al., 2022; Zeng et al., 2024) or semantic embeddings, our confusability matrix is derived entirely from the model's own training dynamics.

3. **Implicit curriculum**: Early in training when representations are random, all class pairs have similar confusability, so CC-SupCon behaves like standard SupCon. As training progresses, easy pairs become well-separated (low confusability → weight ≈ 1) while hard pairs retain high weights. This creates a natural coarse-to-fine curriculum.

4. **Minimal overhead**: Computing class centroids and the confusability matrix requires only $O(C^2 \cdot d)$ operations per batch (where $C$ is the number of classes and $d$ is the embedding dimension), which is negligible compared to the forward/backward pass.

### 2.6 Why Class-Pair Weighting Outperforms Instance-Level Hard Negative Mining: A Variance Reduction Argument

Instance-level hard negative mining (Robinson et al., 2021; Jiang et al., 2022) selects or upweights individual negative samples that are close to the anchor in embedding space. While effective, this approach suffers from high variance: within a single batch, the hardest negative for a given anchor may be an outlier sample that is unrepresentative of its class (e.g., an atypical "dog" image that happens to lie near "cat" due to background similarity rather than semantic confusability).

Formally, let $g_{ij}$ denote the gradient contribution from negative sample $j$ to anchor $i$. Instance-level mining weights individual $g_{ij}$ by their hardness, leading to gradient variance proportional to $\text{Var}[g_{ij} | y_i, y_j]$, the within-class-pair variance. CC-SupCon instead weights by $\hat{\text{conf}}(y_i, y_j)$, which depends only on class identities. The effective gradient for class pair $(c, c')$ is:

$$\bar{g}_{cc'} = \hat{\text{conf}}(c, c') \cdot \frac{1}{|S_{cc'}|} \sum_{(i,j) \in S_{cc'}} g_{ij}$$

where $S_{cc'}$ is the set of negative pairs with classes $c$ and $c'$ in the batch. By the law of large numbers, the averaged gradient $\frac{1}{|S_{cc'}|}\sum g_{ij}$ has variance reduced by a factor of $1/|S_{cc'}|$ compared to any individual $g_{ij}$. The confusability weight $\hat{\text{conf}}(c, c')$, being a momentum-smoothed statistic over many batches, adds negligible additional variance.

**The key insight**: CC-SupCon trades fine-grained per-sample targeting for lower-variance, class-level targeting. When class pairs contain many samples (as in standard classification), this averaging reduces noise substantially while still directing representational effort toward the hardest class distinctions. This is analogous to how class-balanced sampling reduces variance compared to individual importance weighting in long-tailed recognition.

We will empirically validate this argument by comparing gradient norms and their variance between CC-SupCon and an instance-level hard negative baseline (Section 4.6).

## 3. Related Work

### Supervised Contrastive Learning
Khosla et al. (2020) introduced SupCon, extending self-supervised contrastive learning (Chen et al., 2020) to the supervised setting. SupCon uses label information to define positive pairs (same class) and achieves 81.4% top-1 on ImageNet with ResNet-200, outperforming cross-entropy. However, it treats all negative pairs equally and is sensitive to batch size (Fan et al., 2024).

### Neural Collapse
Papyan et al. (2020) discovered that during the terminal phase of training, features collapse to class means forming a simplex ETF. Graf et al. (2021) proved SupCon also converges to this geometry. While the simplex ETF is optimal for balanced classification, it discards inter-class structure. Our work explicitly breaks this equidistant symmetry to create more informative representations.

### Confusion-Guided Contrastive Learning

**Lovász Theta Contrastive Learning** (Smyrnis et al., 2022) is the closest prior work to ours. They establish a connection between the Lovász theta function and the InfoNCE loss, and use a learned weighted similarity graph to modulate contrastive learning. In the supervised setting, their approach **softens (decreases) the repulsion** for partially similar classes—classes that are semantically similar contribute less to the negative term, with the rationale that similar classes should not be pushed as far apart. CC-SupCon takes the **opposite direction**: we **increase repulsion** for confusable class pairs. The key conceptual distinction is the difference between *semantic similarity* (which Lovász Theta preserves) and *confusion* (which CC-SupCon resolves). Two classes may be semantically similar yet perfectly separable in representation space (e.g., "cat" and "dog" share visual features but can be reliably distinguished), while confusability specifically measures the model's current failure to separate them. Moreover, Lovász Theta uses a fixed or learned similarity graph, while CC-SupCon's confusability matrix evolves dynamically with training, creating an adaptive curriculum. Lovász Theta was presented as a workshop paper and reported ~1% supervised improvement on CIFAR-10/100; we target larger gains through dynamic adaptation.

**CCDC** (Chen et al., 2024) proposes a dynamic contrastive learning framework guided by class confidence and confusion degree for medical image segmentation. Their hard class mining strategy selects the most confusing classes as negative examples for pixel-level contrastive learning. While CCDC shares the intuition of using inter-class confusion to guide contrastive learning, it differs from CC-SupCon in several ways: (i) CCDC is designed for dense prediction (segmentation) with pixel-level pairs, while CC-SupCon targets image-level representation learning; (ii) CCDC uses confusion to *select* hard classes (binary mining), while CC-SupCon uses continuous *soft weights* that modulate repulsion strength; (iii) CCDC does not maintain a dynamic momentum-updated confusability matrix—it recomputes confusion per batch. Our continuous weighting approach provides smoother gradients and a more stable curriculum.

### Hard Negative Mining
Robinson et al. (2021) proposed sampling hard negatives in self-supervised contrastive learning. Jiang et al. (2022) extended this to the supervised setting with H-SCL. These methods operate at the **instance level** within a batch. Our approach operates at the **class-pair level** across training, providing more stable and semantically meaningful weighting (see Section 2.6 for a variance reduction argument).

### Class-Aware Contrastive Learning
Zhu et al. (2022) proposed Balanced Contrastive Learning (BCL) to address class imbalance in long-tailed distributions. Our work is orthogonal: CC-SupCon addresses **class confusability** (which classes are hard to separate), not class imbalance (which classes have fewer samples). The two approaches are complementary and could be combined.

### Adaptive Loss Mechanisms
VarCon (Wang et al., 2025) reformulates SupCon as variational inference with a posterior-weighted ELBO, providing a confidence-adaptive temperature that tightens pull strength for hard samples and relaxes it for confident ones. Our approach is complementary: VarCon adapts to per-sample difficulty, while CC-SupCon adapts to per-class-pair difficulty. Temperature scheduling (Kukleva et al., 2023) adjusts a global temperature during training but does not differentiate between class pairs.

### Label Hierarchy in Contrastive Learning
Several works incorporate explicit label hierarchies into contrastive learning (Zhang et al., 2022; Zeng et al., 2024). These require external knowledge (taxonomy trees, WordNet distances). CC-SupCon discovers inter-class structure from the training dynamics themselves, making it applicable to any classification dataset without additional annotations.

### Sample Similarity Graphs in Contrastive Learning
The X-Sample Contrastive Loss (Sobal et al., 2025) uses sample similarity graphs to encode richer relational structure in contrastive learning. While this work modifies how positive similarities are encoded, it does not specifically address the problem of confusion-guided negative reweighting that CC-SupCon targets.

### Representation Quality Metrics
Wang and Isola (2020) identified alignment and uniformity as key properties of contrastive representations. Our work adds a third consideration: **calibrated inter-class separation**, where the separation between classes is proportional to their difficulty of distinction.

## 4. Experiments

### 4.1 Datasets

- **CIFAR-100** (Krizhevsky, 2009): 100 classes, 500 training images per class, 32×32 resolution. Provides a rich class structure with 20 superclasses containing 5 fine-grained classes each—ideal for studying confusability.
- **ImageNet-100**: 100-class subset of ImageNet (Deng et al., 2009), selected following the protocol of Tian et al. (2020). 224×224 resolution, ~1300 images per class.

### 4.2 Architecture and Training

- **Encoder**: ResNet-50 (He et al., 2016) with a 2-layer MLP projection head (2048 → 2048 → 128)
- **Optimizer**: SGD with momentum 0.9, weight decay 1e-4
- **Learning rate**: 0.5 with cosine annealing
- **Batch size**: 512 (256 images × 2 augmentations)
- **Epochs**: 500 for CIFAR-100, 200 for ImageNet-100
- **Temperature**: τ = 0.1 (standard for SupCon)
- **Augmentations**: Standard SimCLR augmentation pipeline (random crop, color jitter, grayscale, horizontal flip, Gaussian blur)

### CC-SupCon Hyperparameters
- **Centroid momentum** α: 0.99
- **Confusability temperature** τ_c: {0.05, 0.1, 0.2} (ablated)
- **Calibration strength** β: {0.5, 1.0, 2.0, 5.0} (ablated)

### 4.3 Evaluation Protocol

**Linear Probe Accuracy**: Freeze the encoder, train a linear classifier on top. This is the standard evaluation for representation quality.

**k-NN Accuracy**: k-nearest neighbors classification (k=200) on frozen features. Measures representation structure without any additional training.

**Transfer Learning**: Train the encoder on CIFAR-100, evaluate linear probe on STL-10 and transfer to CIFAR-10.

**Few-Shot Classification**: 5-way 1-shot and 5-way 5-shot classification on held-out classes (following the meta-learning evaluation protocol).

### 4.4 Baselines

1. **Cross-Entropy (CE)**: Standard cross-entropy training with the same architecture
2. **SupCon** (Khosla et al., 2020): Standard supervised contrastive learning
3. **SupCon + Hard Negatives (H-SCL)**: SupCon with instance-level hard negative sampling (Jiang et al., 2022)
4. **Balanced Contrastive Learning (BCL)** (Zhu et al., 2022): Class-averaging variant of SupCon

### 4.5 Statistical Design

All main comparisons (SupCon vs. CC-SupCon) use **5 seeds** with a **paired design**: the same 5 random seeds are used for both methods, and significance is assessed with a paired t-test. This paired design controls for seed-specific variance (e.g., data augmentation randomness, initialization) and provides adequate statistical power to detect a 1-2% improvement. For ablation studies, we use 3 seeds to stay within the compute budget.

### 4.6 Ablation Studies

1. **Calibration strength (β)**: Sweep β ∈ {0, 0.5, 1.0, 2.0, 5.0} to understand the effect of confusion calibration intensity. β=0 recovers standard SupCon.
2. **Confusability temperature (τ_c)**: Study the sharpness of the confusability distribution.
3. **Centroid momentum (α)**: Compare α ∈ {0.9, 0.95, 0.99, 0.999} to study the tradeoff between responsiveness and stability.
4. **Warm-up period**: Test whether starting with standard SupCon for W epochs before enabling calibration helps or hurts.
5. **Batch size sensitivity**: Evaluate SupCon and CC-SupCon at batch sizes {128, 256, 512, 1024} with 3 seeds each. We hypothesize that CC-SupCon is less sensitive to batch size because the class-pair weights provide a curriculum that partially compensates for fewer negatives per batch at smaller sizes.

### 4.7 Analysis

1. **Confusability matrix visualization**: Visualize the learned confusability matrix at different training stages. Compare to CIFAR-100's superclass structure (e.g., do aquatic mammals group together as confusable?).
2. **Representation geometry**: Measure inter-class distances and compare to the simplex ETF baseline. Show that CC-SupCon produces non-uniform distances that correlate with semantic similarity.
3. **Neural collapse metrics**: Track NC1 (within-class variability collapse) and NC2 (simplex ETF convergence) metrics throughout training for CC-SupCon vs. SupCon.
4. **Per-class accuracy breakdown**: Identify which classes benefit most from confusion calibration.
5. **Gradient analysis**: Measure and compare gradient norms for confusable vs. well-separated class pairs under CC-SupCon and standard SupCon. We expect CC-SupCon to produce systematically larger gradient magnitudes for confusable pairs, providing direct evidence for the mechanistic claim that the method focuses representational effort where it is needed. Additionally, compare gradient variance between CC-SupCon and instance-level H-SCL to empirically validate the variance reduction argument from Section 2.6.

### 4.8 Expected Results

We expect CC-SupCon to:
- Improve linear probe accuracy by 1-2% over SupCon on CIFAR-100 and ImageNet-100
- Show larger gains on k-NN evaluation (since k-NN benefits more from well-structured representations)
- Improve transfer learning performance (representations preserve meaningful inter-class structure)
- Discover confusability structure that aligns with semantic class relationships (e.g., within-superclass confusability should be higher than between-superclass)
- Be less sensitive to batch size than standard SupCon (class-pair weights provide a curriculum that partially compensates for fewer negatives per batch)
- Exhibit lower gradient variance than instance-level hard negative mining while maintaining comparable or larger average gradient magnitude for confusable pairs

## 5. Success Criteria

### Primary (must achieve)
- CC-SupCon achieves higher linear probe accuracy than SupCon on at least one of {CIFAR-100, ImageNet-100} by a statistically significant margin (paired t-test, p < 0.05, 5 seeds)
- The learned confusability matrix shows meaningful structure (higher confusability within semantic groups than between them)

### Secondary (should achieve)
- Improvements on k-NN evaluation and transfer learning tasks
- Ablation studies confirm the contribution of each component
- CC-SupCon is less sensitive to batch size than standard SupCon (batch size sensitivity experiment)
- Gradient analysis confirms larger gradients for confusable pairs under CC-SupCon
- Analysis reveals interpretable confusability dynamics during training

### Failure indicators
- CC-SupCon performs worse than or identical to SupCon on all metrics → confusability weighting may be harmful or irrelevant
- Confusability matrix shows no meaningful structure → the signal may be too noisy to be useful
- Results are highly sensitive to β with no robust operating range → the method is impractical

## 6. Computational Budget

| Experiment | Dataset | Epochs | Est. Time (A6000) |
|---|---|---|---|
| SupCon baseline (×5 seeds) | CIFAR-100 | 500 | ~2.5h |
| CC-SupCon (×5 seeds, main) | CIFAR-100 | 500 | ~2.5h |
| CC-SupCon ablations (4 runs × 3 seeds) | CIFAR-100 | 200 | ~0.6h |
| Batch size sensitivity (4 sizes × 2 methods × 1 seed) | CIFAR-100 | 200 | ~0.8h |
| Linear probes + k-NN eval | CIFAR-100 | — | ~0.3h |
| CE baseline (×1 seed) | CIFAR-100 | 200 | ~0.2h |
| Gradient analysis | CIFAR-100 | — | ~0.1h |
| Analysis & visualization | — | — | ~0.2h |
| **Total** | | | **~7.2h** |

This fits within the 8-hour compute budget on a single NVIDIA RTX A6000. We drop ImageNet-100 experiments to accommodate the increased seed count (5 instead of 3) and the batch size sensitivity study, focusing on CIFAR-100 where the rich 20-superclass structure provides the most informative testbed for confusability-guided learning.

## 7. References

1. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., and Krishnan, D. (2020). Supervised Contrastive Learning. *NeurIPS 2020*.
2. Papyan, V., Han, X.Y., and Donoho, D.L. (2020). Prevalence of Neural Collapse during the terminal phase of deep learning training. *PNAS*, 117(40), 24652-24663.
3. Graf, F., Hofer, C., Niethammer, M., and Kwitt, R. (2021). Dissecting Supervised Contrastive Learning. *ICML 2021*.
4. Robinson, J., Chuang, C.Y., Sra, S., and Jegelka, S. (2021). Contrastive Learning with Hard Negative Samples. *ICLR 2021*.
5. Wang, T. and Isola, P. (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. *ICML 2020*.
6. Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML 2020*.
7. Zhu, J., Wang, Z., Chen, J., Chen, Y.P.P., and Jiang, Y.G. (2022). Balanced Contrastive Learning for Long-Tailed Visual Recognition. *CVPR 2022*.
8. Wang, Z., Fan, J., Nguyen, T., Ji, H., and Liu, G. (2025). Variational Supervised Contrastive Learning. *NeurIPS 2025*. arXiv:2506.07413.
9. Kukleva, A., Kuehne, H., Schiele, B., and Sener, F. (2023). Temperature Schedules for Self-Supervised Contrastive Methods on Long-Tail Data. *ICLR 2023*.
10. Zhang, Y., Jiang, H., Miura, Y., Manning, C.D., and Langlotz, C.P. (2022). Use All the Labels: A Hierarchical Multi-Label Contrastive Learning Framework. *CVPR 2022*.
11. He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
12. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. *Technical Report*.
13. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., and Fei-Fei, L. (2009). ImageNet: A Large-Scale Hierarchical Image Database. *CVPR 2009*.
14. Tian, Y., Krishnan, D., and Isola, P. (2020). Contrastive Multiview Coding. *ECCV 2020*.
15. Jiang, R., Nguyen, T., Ishwar, P., and Aeron, S. (2022). Supervised Contrastive Learning with Hard Negative Samples. *arXiv:2209.00078*.
16. Smyrnis, G., Jordan, M., Uppal, A., Daras, G., and Dimakis, A. (2022). Lovász Theta Contrastive Learning. *NeurIPS 2022 Workshop on Self-Supervised Learning*.
17. Chen, J., Chen, C., Huang, W., Zhang, J., Debattista, K., and Han, J. (2024). Dynamic Contrastive Learning Guided by Class Confidence and Confusion Degree for Medical Image Segmentation. *Pattern Recognition*, 145, 109881.
18. Fan, W., et al. (2024). CLCE: An Approach to Refining Cross-Entropy and Contrastive Learning for Optimized Learning Fusion. *ECAI 2024*.
19. Sobal, V., Ibrahim, M., Balestriero, R., Cabannes, V., Bouchacourt, D., Astolfi, P., Cho, K., and LeCun, Y. (2025). X-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs. *ICLR 2025*.
20. Zeng, Z., et al. (2024). Hierarchical Contrastive Learning. *NeurIPS 2024*.
