# Curriculum Contrastive Learning for Enzyme Function Prediction

## Introduction

### Context and Problem Statement

Accurate prediction of enzyme function, typically encoded as Enzyme Commission (EC) numbers, is fundamental to genomics, synthetic biology, and biocatalysis. With millions of protein sequences deposited in databases like UniProt, the gap between known sequences and experimentally characterized functions continues to widen. EC numbers follow a hierarchical four-level classification (e.g., 2.7.10.1), encoding progressively finer functional distinctions from general reaction type to specific substrate.

Recent deep learning methods have made substantial progress. CLEAN (Yu et al., 2023) introduced supervised contrastive learning for EC prediction, outperforming BLASTp on standard benchmarks. ProtDETR (Yang et al., 2025) reframed the problem as residue-level detection using learnable functional queries. HIT-EC (Dumontet et al., 2026) proposed a four-level hierarchical transformer. CLEAN-Contact (Yang et al., 2024) augmented CLEAN with structural contact information. MAPred (Rong et al., 2025) introduced autoregressive prediction leveraging 3D structural tokens. EnzHier (Duan et al., 2026) combines multi-scale feature integration with hierarchical contrastive learning using a U-Net backbone and triplet loss. Yim et al. (2024) proposed hierarchical contrastive learning that gradually incorporates EC hierarchy by adjusting hierarchical weights within a modified Hierarchical Multi-level Contrastive (HMC) loss.

### Key Limitation

Despite these advances, existing methods that exploit the EC hierarchy do so within a **single training phase** -- either through hierarchical margin adjustments (EnzHier), hierarchical weight scheduling within a unified loss (Yim et al., 2024), multi-level architectures (HIT-EC), or autoregressive digit prediction (MAPred). None employ a **multi-phase curriculum learning** strategy that trains sequentially from coarse to fine EC levels, with each phase producing a dedicated checkpoint that initializes the next. This distinction matters because:

1. **Class imbalance**: Fine-grained EC classes (level 4) exhibit extreme long-tail distributions, with some classes having thousands of examples and others fewer than five. Starting with coarse levels provides stronger initial supervision.
2. **Optimization landscape**: Simultaneously optimizing across ~6000 fine-grained classes creates a complex loss landscape. Sequential phase-wise training provides a smoother optimization trajectory, as each phase inherits a well-structured representation from the previous level.
3. **Representation structure**: Single-phase hierarchical methods must balance coarse and fine objectives simultaneously, creating tension between them. Multi-phase curriculum training builds coarse structure first, then refines it -- explicitly encoding the EC hierarchy into the representation space through the training schedule itself.

### How CurrEC Differs from Yim et al. (2024)

Yim et al. (2024) proposed a hierarchical contrastive learning approach that modifies the loss function within a **single training phase** by gradually increasing hierarchical weights applied to different EC levels. Their HMC loss incorporates all hierarchy levels simultaneously, with weight hyperparameters controlling the relative importance of each level. In contrast, CurrEC adopts a fundamentally different strategy: **sequential multi-phase training** where each phase exclusively defines positive pairs at a single EC level, initializes from the previous phase's checkpoint, and applies consistency regularization to preserve previously learned structure. The key differences are:

- **Training regime**: Yim et al. use one continuous training phase with loss-level weight adjustments; CurrEC trains four separate phases with full checkpoint inheritance.
- **Positive pair definition**: Yim et al. define positive pairs across all levels simultaneously with hierarchical weights; CurrEC redefines positive pairs at each phase to match the current EC level.
- **Anti-forgetting mechanism**: Yim et al. rely on hierarchical weights to balance levels; CurrEC uses explicit consistency regularization losses from all previously trained levels.
- **Curriculum principle**: CurrEC applies curriculum learning (Bengio et al., 2009) -- the idea that ordering training from easy to hard improves generalization -- to the natural difficulty hierarchy of EC levels, which Yim et al. do not.

### Key Insight and Hypothesis

We propose **CurrEC** (Curriculum Enzyme Classification), a curriculum contrastive learning framework that trains enzyme representations progressively from coarse to fine EC classification levels. Inspired by the seminal work of Bengio et al. (2009) showing that curriculum learning improves generalization through better optimization trajectories, we leverage the natural hierarchy of EC numbers as a built-in curriculum: level-1 classes (7 broad reaction types) are "easy" concepts, while level-4 classes (~6000 specific enzymes) are "hard" concepts.

**Hypothesis**: Progressive coarse-to-fine contrastive training produces more structured and discriminative enzyme representations than both flat training and single-phase hierarchical training, leading to improved EC number prediction, especially for rare and underrepresented enzyme classes.

## Proposed Approach

### Overview

CurrEC uses a frozen protein language model (ESM-2) to extract sequence embeddings, followed by a lightweight projection network trained with a phased curriculum contrastive strategy. The key innovation is the training protocol, not the architecture.

### Method Details

#### Step 1: Embedding Extraction
For each enzyme sequence, extract a fixed-dimensional representation by mean-pooling per-residue embeddings from a frozen ESM-2 model (650M parameters). This produces a 1280-dimensional vector per enzyme.

#### Step 2: Projection Network
A two-layer MLP with batch normalization and ReLU activation projects the ESM-2 embedding to a 512-dimensional normalized representation space. This is the only trainable component.

#### Step 3: Curriculum Training Protocol

The training proceeds in four phases, corresponding to the four levels of the EC hierarchy:

**Phase 1 - Coarse (EC Level 1):** Train with Supervised Contrastive Loss (SupCon; Khosla et al., 2020) where positive pairs are enzymes sharing the same level-1 EC class (7 classes: oxidoreductases, transferases, hydrolases, lyases, isomerases, ligases, translocases).

**Phase 2 - Sub-class (EC Level 2):** Fine-tune from Phase 1 checkpoint. Positive pairs now share the same level-2 EC class (~70 classes). A consistency regularization term preserves level-1 structure:

$$\mathcal{L}_2 = \mathcal{L}_{SupCon}^{(2)} + \lambda \cdot \mathcal{L}_{SupCon}^{(1)}$$

**Phase 3 - Sub-sub-class (EC Level 3):** Fine-tune from Phase 2 checkpoint. Positive pairs share the same level-3 EC class (~300 classes). Consistency regularization includes both level-1 and level-2 terms:

$$\mathcal{L}_3 = \mathcal{L}_{SupCon}^{(3)} + \lambda \cdot \left(\mathcal{L}_{SupCon}^{(1)} + \mathcal{L}_{SupCon}^{(2)}\right)$$

**Phase 4 - Specific (EC Level 4):** Fine-tune from Phase 3 checkpoint. Positive pairs share the full EC number (~6000 classes):

$$\mathcal{L}_4 = \mathcal{L}_{SupCon}^{(4)} + \lambda \cdot \left(\mathcal{L}_{SupCon}^{(1)} + \mathcal{L}_{SupCon}^{(2)} + \mathcal{L}_{SupCon}^{(3)}\right)$$

**General formulation:** During Phase $k$ ($k = 1, \dots, 4$), the total loss is:

$$\mathcal{L}_k = \mathcal{L}_{SupCon}^{(k)} + \lambda \sum_{l=1}^{k-1} \mathcal{L}_{SupCon}^{(l)}$$

where $\mathcal{L}_{SupCon}^{(l)}$ is the standard Supervised Contrastive Loss (Khosla et al., 2020) computed using EC level-$l$ labels as the grouping criterion. The regularization coefficient $\lambda$ controls the strength of anti-forgetting constraints. This formulation is unambiguous: each consistency term is simply the SupCon loss evaluated with coarser-level labels, ensuring that the learned representations maintain cluster structure at all previously trained EC levels.

#### Step 4: Dynamic Margin Scheduling

Across phases, the contrastive temperature decreases progressively. In Phase 1, a higher temperature suffices to separate 7 coarse classes. By Phase 4, a lower temperature is needed to produce sharper distinctions among thousands of fine-grained classes. Specifically:

$$\tau_k = \tau_{base} \cdot \gamma^{(k-1)}$$

where $k$ is the phase index (1-4), $\tau_{base}$ is the base temperature, and $\gamma < 1$ is a decay factor (e.g., $\gamma = 0.85$). This progressive sharpening is motivated by the observation that coarse-level discrimination requires less representational precision than fine-grained discrimination.

#### Step 5: Inference

At inference, we follow the CLEAN protocol: compute the centroid of each EC class in the learned embedding space, then assign EC numbers based on cosine distance to centroids. We support both max-separation (assign the EC with largest separation from other candidates) and sum-distance (assign based on summed distances) selection strategies.

### Key Innovations

1. **Multi-phase curriculum for EC prediction**: First application of sequential coarse-to-fine training phases to enzyme function prediction, with checkpoint inheritance between phases. Distinguished from single-phase hierarchical weight adjustment (Yim et al., 2024) and joint multi-level training (EnzHier) by the curriculum principle.
2. **Explicit consistency regularization**: Formalized as SupCon losses computed at all previously trained EC levels (Equation above), providing a mathematically precise anti-forgetting mechanism that maintains hierarchical representation structure.
3. **Progressive temperature scheduling**: Adapts contrastive temperature across curriculum phases to match the discrimination difficulty at each EC level.
4. **Improved rare-class performance**: Coarse-level pre-training provides stronger supervision signal for underrepresented fine-grained classes through shared representations.

## Related Work

### Enzyme Function Prediction

**CLEAN** (Yu et al., 2023) introduced supervised contrastive learning for EC annotation using ESM-1b embeddings with average pooling. It uses a flat SupCon loss where all enzymes with the same EC number are positives. CLEAN achieves strong performance but does not exploit the hierarchical structure of EC numbers. On the New-392 benchmark, CLEAN achieves Precision/Recall/F1 of 0.575/0.491/0.502 and on Price-149, 0.538/0.408/0.438 (as reported by Rong et al., 2025).

**ProtDETR** (Yang et al., 2025) reformulates EC prediction as residue-level detection with learnable functional queries. Published at ICLR 2025, it achieves Precision/Recall of 0.594/0.608 on New-392 and recall of 0.507 on Price-149, substantially improving recall over CLEAN. It provides interpretability through cross-attention maps but uses a classification loss rather than contrastive learning and does not model the EC hierarchy.

**MAPred** (Rong et al., 2025) uses autoregressive prediction of EC digits, integrating sequence and 3D structural tokens. It currently holds the strongest reported numbers: Precision/Recall/F1 of 0.651/0.632/0.610 on New-392 and 0.554/0.487/0.493 on Price-149. It exploits the hierarchical nature of EC numbers through its autoregressive structure but requires 3D structural information.

**HIT-EC** (Dumontet et al., 2026) uses a four-level hierarchical transformer architecture aligned with EC levels. It trains all levels jointly with a masked loss for incomplete annotations. Unlike CurrEC, it uses separate transformer encoders per level rather than a curriculum strategy.

**CLEAN-Contact** (Yang et al., 2024) extends CLEAN by incorporating protein structural contact maps through graph neural networks. It improves upon CLEAN but still uses flat contrastive training without curriculum.

**EnzHier** (Duan et al., 2026) combines multi-scale feature integration via a U-Net backbone with hierarchical triplet loss, using margin adjustment based on EC hierarchy. Published at ISBRA 2025 (LNCS vol. 15757, Springer). It reports a 23% higher F1-score on benchmark cross-validation versus prior methods. However, EnzHier adjusts margins within a single training phase rather than using a multi-phase curriculum. It does not employ checkpoint inheritance or explicit consistency regularization across levels.

**Yim et al. (2024)** proposed "Hierarchical Contrastive Learning for Enzyme Function Prediction" at the ICML 2024 Workshop on ML for Life and Material Science. Their approach gradually incorporates the EC hierarchical structure into a modified HMC (Hierarchical Multi-level Contrastive) loss by adjusting hierarchical weights across EC levels within a single training phase. They show that gradually increasing hierarchical weight improves consensus across levels. CurrEC differs fundamentally by using **sequential multi-phase training** with checkpoint inheritance and explicit SupCon-based consistency regularization, rather than loss-level weight adjustment within a single phase (see Introduction for detailed comparison).

### How CurrEC Differs

CurrEC is distinguished from all prior work by its **progressive multi-phase curriculum training** strategy with explicit consistency regularization. The closest related works are:

| Method | Hierarchy-aware? | Training regime | Key difference from CurrEC |
|--------|-----------------|-----------------|---------------------------|
| CLEAN | No | Single-phase flat | No hierarchy exploitation |
| Yim et al. | Yes (weight scheduling) | Single-phase | Loss weight adjustment, not curriculum |
| EnzHier | Yes (triplet margins) | Single-phase | Margin adjustment, no curriculum |
| HIT-EC | Yes (architecture) | Joint multi-level | Separate encoders, no curriculum |
| MAPred | Yes (autoregressive) | Single-phase | Different paradigm (classification) |
| **CurrEC** | **Yes (curriculum)** | **Multi-phase sequential** | **Curriculum + consistency reg.** |

### Foundational Methods

**Supervised Contrastive Learning** (Khosla et al., 2020) extends self-supervised contrastive learning to the supervised setting, using label information to define positive pairs. CurrEC builds on SupCon but adapts the definition of "positive pairs" across curriculum phases and reuses the SupCon loss at coarser levels for consistency regularization.

**Curriculum Learning** (Bengio et al., 2009) formalized the idea that training examples ordered from easy to hard improve generalization, acting as a continuation method for non-convex optimization. CurrEC applies this principle to the natural difficulty hierarchy of EC levels.

**ESM-2** (Lin et al., 2023) provides the protein language model backbone, offering state-of-the-art protein sequence representations that capture evolutionary and structural information.

## Experiments

### Planned Setup

**Backbone**: ESM-2 (650M parameters, frozen). Embeddings extracted once and cached.

**Training Data**: Swiss-Prot enzyme sequences with EC annotations, following the CLEAN data processing pipeline. Approximately 200K-250K enzyme sequences.

**Architecture**: Two-layer MLP projection head (1280 -> 1024 -> 512) with batch normalization and ReLU.

**Training**:
- Phase 1: 30 epochs, learning rate 5e-4
- Phase 2: 20 epochs, learning rate 1e-4
- Phase 3: 20 epochs, learning rate 5e-5
- Phase 4: 20 epochs, learning rate 1e-5
- Optimizer: AdamW with cosine annealing within each phase
- Batch size: 512
- Consistency regularization weight: $\lambda = 0.5$
- Temperature schedule: $\tau_{base} = 0.1$, $\gamma = 0.85$

### Benchmarks

Following established protocol from CLEAN (Yu et al., 2023):

1. **New-392**: 392 enzymes with experimentally verified EC numbers not in the training set. Tests generalization to novel enzymes.
2. **Price-149**: 149 enzymes from Price et al. with experimentally characterized functions. Tests annotation of understudied enzymes.
3. **Cross-validation**: 10-fold CV on the training set with sequence identity filtering (<40% between train and test).

### Metrics

- **F1-score (macro and micro)**: Primary metric for EC prediction accuracy
- **Precision and Recall**: Per-class analysis
- **F1 on rare classes**: Performance on EC classes with fewer than 10 training examples
- **Embedding visualization**: t-SNE/UMAP of learned representations colored by EC level

### Baselines

1. **CLEAN** (Yu et al., 2023): Flat contrastive learning with ESM-1b. Published numbers: New-392 F1=0.502, Price-149 F1=0.438.
2. **BLASTp**: Sequence homology-based baseline.
3. **Flat SupCon**: Same architecture and ESM-2 embeddings as CurrEC, but trained in a single flat phase at level-4 only (ablation control).
4. **Joint Hierarchical SupCon**: Same architecture, single-phase training with all 4 EC levels simultaneously using a weighted combination of SupCon losses at all levels (i.e., $\mathcal{L} = \sum_{l=1}^{4} w_l \cdot \mathcal{L}_{SupCon}^{(l)}$). This isolates the curriculum effect from the hierarchy-awareness effect.
5. **Reverse Curriculum**: Train fine-to-coarse (anti-curriculum ablation).

**Published reference numbers** (not re-implemented, for context):
- ProtDETR (Yang et al., 2025): New-392 P/R=0.594/0.608; Price-149 R=0.507
- MAPred (Rong et al., 2025): New-392 P/R/F1=0.651/0.632/0.610; Price-149 P/R/F1=0.554/0.487/0.493

Note: ProtDETR and MAPred use different backbones (fine-tuned ESM-2, 3D structural tokens) and training paradigms (detection, autoregressive classification), so direct comparison is informative but not apples-to-apples. CurrEC's primary comparison is against flat and joint hierarchical baselines using the same backbone and inference protocol.

### Ablation Studies

1. **Curriculum order**: Coarse-to-fine vs. fine-to-coarse vs. random-order vs. simultaneous (joint hierarchical)
2. **Number of phases**: 2-phase (level 1+2 then 3+4) vs. 4-phase (one per level)
3. **Consistency regularization**: With vs. without $\mathcal{L}_{consistency}$ ($\lambda = 0$ vs. $\lambda = 0.5$)
4. **Consistency weight $\lambda$**: Sweep over {0, 0.1, 0.25, 0.5, 1.0}
5. **Temperature scheduling**: With vs. without progressive temperature decay
6. **ESM model size**: ESM-2 (8M) vs. ESM-2 (150M) vs. ESM-2 (650M)

### Expected Results

We expect CurrEC to:
1. Outperform flat SupCon (same backbone, no curriculum) by 3-5% F1, demonstrating the value of curriculum training
2. Outperform joint hierarchical SupCon by 1-3% F1, isolating the benefit of the curriculum schedule over single-phase hierarchy awareness
3. Match or exceed CLEAN performance while using the same inference protocol
4. Show the largest improvements on **rare EC classes** (fewer than 10 examples), where coarse-level pre-training provides the most benefit
5. Produce embedding spaces with clear hierarchical structure visible in t-SNE/UMAP visualizations
6. Show that coarse-to-fine curriculum outperforms fine-to-coarse and random-order ablations

## Success Criteria

### Primary (confirms hypothesis)
- CurrEC achieves higher macro-F1 than flat SupCon on New-392 and Price-149 benchmarks
- CurrEC outperforms joint hierarchical SupCon (same hierarchy awareness, no curriculum), isolating the curriculum effect
- CurrEC improves F1 on rare EC classes (n < 10) by at least 5% relative to flat training
- Coarse-to-fine curriculum order outperforms reverse and random curriculum orders

### Secondary (strengthens contribution)
- CurrEC matches or exceeds CLEAN on standard benchmarks
- Embedding visualizations show clear hierarchical clustering
- Consistency regularization demonstrably prevents forgetting of coarse-level structure
- Ablation over $\lambda$ shows optimal performance at intermediate values (not 0 or very large)

### Negative results that would refute hypothesis
- No significant difference between curriculum and flat training
- Joint hierarchical SupCon matches or exceeds curriculum training (curriculum adds nothing beyond hierarchy awareness)
- Curriculum order has no impact on performance
- Curriculum training hurts performance on common (well-represented) EC classes without offsetting gains on rare classes

## References

1. Yu, T., Cui, H., Li, J.C., Luo, Y., Jiang, G., & Zhao, H. (2023). Enzyme function prediction using contrastive learning. *Science*, 379(6639), 1358-1363.

2. Yang, Z., Su, B., Chen, J., & Wen, J.R. (2025). Interpretable Enzyme Function Prediction via Residue-Level Detection. *ICLR 2025*. arXiv:2501.05644.

3. Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., ..., & Rives, A. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130.

4. Dumontet, L., Han, S.R., Oh, T.J., & Kang, M. (2026). Trustworthy prediction of enzyme commission numbers using a hierarchical interpretable transformer. *Nature Communications*.

5. Yang, Y., Jerger, A., Feng, S., Wang, Z., Brasfield, C., Cheung, M.S., Zucker, J., & Guan, Q. (2024). Improved enzyme functional annotation prediction using contrastive learning with structural inference. *Communications Biology*, 7(1), 1690.

6. Rong, D., Zhong, B., Zheng, W., Hong, L., & Liu, N. (2025). Autoregressive enzyme function prediction with multi-scale multi-modality fusion. *Briefings in Bioinformatics*, 26(5), bbaf476.

7. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., ..., & Krishnan, D. (2020). Supervised Contrastive Learning. *NeurIPS 2020*, 33, 18661-18673.

8. Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum Learning. *ICML 2009*, 41-48.

9. Duan, H., Zhu, J., Liu, Q., & Li, G. (2026). EnzHier: Accurate Enzyme Function Prediction Through Multi-scale Feature Integration and Hierarchical Contrastive Learning. In *Bioinformatics Research and Applications (ISBRA 2025)*, LNCS vol. 15757, Springer, Singapore.

10. Yim, S., Hwang, D., Kim, K., & Han, S. (2024). Hierarchical Contrastive Learning for Enzyme Function Prediction. *ICML 2024 Workshop on ML for Life and Material Science*.
