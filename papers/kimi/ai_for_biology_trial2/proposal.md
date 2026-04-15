# Research Proposal: CROSS-GRN: Directionality-Aware Cross-Modal Attention for Signed Gene Regulatory Network Inference from Single-Cell Multi-Omics

## 1. Introduction

### 1.1 Background and Motivation

Gene regulatory networks (GRNs) are fundamental to understanding cellular function, development, and disease. The advent of single-cell multi-omics technologies, particularly paired single-cell RNA sequencing (scRNA-seq) and single-cell ATAC sequencing (scATAC-seq), has provided unprecedented opportunities to decipher gene regulation by simultaneously measuring gene expression and chromatin accessibility in the same cells.

Recent advances have shown remarkable progress in multi-omics integration:

1. **Cross-Modal Generation**: Cisformer (Genome Biology 2025) [1] demonstrated that decoder-only transformers with cross-attention can accurately translate between gene expression and chromatin accessibility modalities, achieving state-of-the-art performance in cross-modality generation.

2. **Cell Representation Learning**: Attune (GigaScience 2024) [2] uses asymmetric teacher-student networks with cross-modal contrastive learning to integrate scRNA-seq and scATAC-seq data, enabling improved cell clustering and peak-gene interaction inference.

3. **Cross-Attention for Directionality**: XATGRN (BMC Bioinformatics 2025) [3] recently proposed cross-attention for directionality-aware GRN inference from bulk RNA data, predicting activation/repression relationships.

However, a critical gap remains: **while cross-attention for GRN directionality has been explored with bulk data, no existing method combines asymmetric cross-attention with single-cell multi-omics integration and explicit cell-type conditioning to learn context-specific, signed regulatory networks.**

### 1.2 Key Insight

Gene regulation is inherently:
1. **Directional**: Transcription factors (TFs) regulate target genes, not vice versa
2. **Signed**: TFs can activate (positive) or repress (negative) their targets
3. **Context-specific**: Regulatory relationships vary across cell types and states

Current methods fall short in distinct ways:
- Cisformer and Attune use symmetric cross-attention that treats gene-peak relationships as undirected
- XATGRN uses cross-attention for directionality but only on bulk RNA data (no multi-omics) and uses GNN architectures, not transformers
- GRNFormer [4] and AttentionGRN [5] address GRN inference but neither combines cross-modal attention with cell-type conditioning for signed edge prediction

We hypothesize that by introducing **directionality-aware asymmetric cross-attention** specifically designed for single-cell multi-omics with **cell-type conditioning**, we can learn signed, context-specific regulatory networks that are both more accurate and more biologically interpretable than existing approaches.

### 1.3 Problem Statement

Current approaches suffer from three key limitations:

1. **Symmetric Attention in Multi-Omics**: Cisformer and Attune use standard cross-attention that treats TF→target and target→TF as equivalent, missing the inherent directionality of regulation.

2. **Unsigned or Context-Agnostic Networks**: While XATGRN predicts signed edges, it operates on bulk RNA without multi-omics integration or cell-type conditioning. AttentionGRN models directed structure but does not predict activation vs. repression or leverage multi-modal data.

3. **Static Networks**: Existing single-cell methods infer cell-type-specific networks through separate analyses rather than learning a unified model with explicit cell-type conditioning.

### 1.4 Relationship to Recent Work

**Explicit Differentiation from XATGRN**: XATGRN (BMC Bioinformatics 2025, arXiv:2412.16220) is the most closely related work conceptually. Both use cross-attention for directionality-aware GRN inference with signed edge prediction. However, CROSS-GRN differs in three critical dimensions:

| Aspect | XATGRN | CROSS-GRN |
|--------|--------|-----------|
| **Data modality** | Bulk RNA-seq only | Single-cell multi-omics (RNA + ATAC) |
| **Architecture** | GNN-based with dual complex graph embedding | Transformer-based with asymmetric cross-attention |
| **Cell-type conditioning** | None (bulk data) | Explicit cell-type modulation of attention weights |
| **Cross-modal attention** | N/A | Expression ↔ Accessibility cross-attention |

The key innovation of CROSS-GRN is the **combination** of (1) asymmetric cross-attention, (2) single-cell multi-omics integration, and (3) cell-type conditioning for context-specific signed GRNs—a combination not explored by XATGRN or any other existing method.

## 2. Proposed Approach

### 2.1 Overview

We propose **CROSS-GRN (Cross-modal Regulatory cOherence with Signed Supervision for GRN inference)**, a transformer-based architecture that:

1. Processes gene expression and chromatin accessibility through modality-specific encoders
2. Uses **dual asymmetric cross-attention** to explicitly model TF→target and target→TF relationships separately
3. Incorporates **cell-type-conditioned attention modulation** for context-specific regulatory networks
4. Jointly predicts **signed regulatory edges** (activation/repression) alongside expression prediction
5. Extracts interpretable, directional regulatory networks from asymmetric attention weights

### 2.2 Architecture Details

**Input Representation**:
- Gene expression: Log-normalized count matrix X ∈ ℝ^(N×G) for N cells, G genes
- Chromatin accessibility: Binary/score matrix A ∈ ℝ^(N×P) for P peaks, mapped to genes within ±500kb
- Cell type labels: Discrete labels or soft cluster assignments for each cell

**Core Architecture**:
```
Expression Encoder: 8-layer transformer with gene tokens
  ↓ Dual Asymmetric Cross-Modal Attention
     - Forward: TFs → target genes (Q=TFs, K/V=targets)
     - Reverse: targets → TFs (Q=targets, K/V=TFs)
  ↓ Cell-Type-Conditioned Modulation
Accessibility Encoder: 8-layer transformer with peak tokens
  ↓ Joint Training Heads
Expression Prediction + Signed GRN Edge Prediction
```

**Key Innovation 1 - Dual Asymmetric Cross-Attention**:
Unlike Cisformer's symmetric cross-attention and XATGRN's gene-gene attention on bulk data, we maintain separate query projections for regulatory sources (TFs) and targets in a multi-omic context:

```
# Forward direction (TFs regulating targets)
Q_forward = W_Q^TF · H_TF_tokens
K_forward = W_K^target · H_gene_tokens  
V_forward = W_V^target · H_gene_tokens
Attention_forward = softmax(Q_forward K_forward^T / √d) V_forward

# Reverse direction (for comparison/regularization)
Q_reverse = W_Q^target · H_gene_tokens
K_reverse = W_K^TF · H_TF_tokens
V_reverse = W_V^TF · H_TF_tokens
```

The asymmetry in query/key projections explicitly models the directional nature of regulation across expression and accessibility modalities.

**Key Innovation 2 - Cell-Type-Conditioned Attention**:
We modulate cross-attention weights based on cell type embeddings:

```
Cell_embedding = Embedding(cell_type)
Modulation = MLP(Cell_embedding)  # Learned per-cell-type modulation
Modulated_attention = Attention_forward ⊙ Modulation
```

This enables the same TF-target pair to have different regulatory strengths across cell types—a capability absent in XATGRN (bulk data) and not explored in AttentionGRN.

**Key Innovation 3 - Signed Edge Prediction**:
We extend the GRN prediction head to predict both edge existence and regulatory sign:

```
Edge_existence = sigmoid(MLP([h_TF, h_target, h_peak]))
Edge_sign = tanh(MLP([h_TF, h_target, accessibility_correlation]))
# Final signed edge weight = Edge_existence × Edge_sign
```

### 2.3 Training Objectives

**Primary Objectives (Self-Supervised)**:
- Masked gene expression prediction: Predict expression of masked genes from remaining genes + chromatin
- Masked peak prediction: Predict accessibility of masked peaks

**Secondary Objectives (Supervised)**:
- GRN edge existence: Binary classification of TF-target edges
- GRN edge sign: Regression to {-1, +1} for repression/activation

**Tertiary Objective (Regularization)**:
- Directionality consistency: Forward attention should be more informative than reverse attention for GRN prediction
- Cell-type diversity: Encourage different attention patterns across cell types via contrastive loss

**Loss Function**:
```
L = L_expr_pred + λ₁ L_atac_pred + λ₂ L_edge_existence + λ₃ L_edge_sign + λ₄ L_directionality + λ₅ L_cell_type_div
```

### 2.4 GRN Extraction

After training, we extract cell-type-specific GRNs by:
1. Computing asymmetric forward attention weights between TF tokens and target gene tokens
2. Modulating by cell-type-specific learned embeddings
3. Thresholding and sign extraction based on the signed edge prediction head
4. Aggregating across cells of the same type to produce population-level signed GRNs

## 3. Related Work

### 3.1 Cross-Modal Generation Methods

**Cisformer** [1] uses a decoder-only transformer with symmetric cross-attention for RNA↔ATAC generation. Key differences:
- Cisformer uses **symmetric** cross-attention for generation; CROSS-GRN uses **asymmetric** attention for directional GRN inference
- Cisformer extracts CRE-gene links from attention as secondary analysis; CROSS-GRN explicitly trains for **signed TF-target edges**
- Cisformer infers one network; CROSS-GRN learns **cell-type-conditioned** networks through modulation

**Attune** [2] uses cross-modal contrastive learning with a transformer decoder for peak-gene interaction. Key differences:
- Attune focuses on **cell representation learning** for clustering; CROSS-GRN focuses on **regulatory network inference**
- Attune does not model regulatory **directionality or sign**
- CROSS-GRN introduces explicit **cell-type conditioning** for context-specific networks

### 3.2 Cross-Attention GRN Methods

**XATGRN** [3] is the most conceptually similar work. It uses cross-attention for directionality-aware GRN inference from bulk RNA data with activation/repression prediction. Key differences:
- **Data scope**: XATGRN uses bulk RNA-seq only; CROSS-GRN uses single-cell **multi-omics** (RNA + ATAC)
- **Architecture**: XATGRN uses GNNs with dual complex graph embedding; CROSS-GRN uses **transformer-based asymmetric cross-attention**
- **Cell-type conditioning**: XATGRN has no cell-type component (bulk data); CROSS-GRN explicitly **conditions on cell type** for context-specific networks

**AttentionGRN** [5] uses a graph transformer with directed structure encoding for GRN inference from scRNA-seq. Key differences:
- AttentionGRN uses single-modality (scRNA-seq) with prior network structure; CROSS-GRN uses **paired multi-omics**
- AttentionGRN does not predict **activation vs. repression** (sign); CROSS-GRN explicitly predicts signed edges
- AttentionGRN does not have **cell-type conditioning**; CROSS-GRN learns context-specific networks

**GRNFormer** [4] integrates multi-scale GRNs into RNA foundation models using adaptive cross-attention. Key differences:
- GRNFormer focuses on **enhancing foundation models** with GRN knowledge; CROSS-GRN focuses on **inferring GRNs** from multi-omics
- GRNFormer uses GRNs as prior knowledge for downstream tasks; CROSS-GRN **learns GRNs de novo** from multi-omics with signed edges
- GRNFormer does not predict regulatory sign or condition on cell type for GRN variation

### 3.3 Multi-Omics GRN Methods

**scMultiomeGRN** [6] uses graph convolutional networks with cross-modal attention for GRN inference. Unlike our approach, it:
- Uses GNNs rather than transformers as the primary architecture
- Does not model **activation vs. repression**
- Does not explicitly condition on cell type for network modulation

**SCENIC+** [7] uses motif-based regulatory inference with multi-omics integration. Unlike our approach:
- Relies heavily on prior motif knowledge
- Does not learn representations end-to-end with expression prediction
- Cannot generalize to novel regulatory mechanisms not in motif databases

**SCRIPro** [8] infers cell-type-specific GRNs using chromatin reconstruction and in silico deletion. Unlike our approach:
- Uses separate analyses for different cell types rather than learning a unified, conditionable model
- Does not leverage transformer-based cross-modal attention

### 3.4 Summary of Novelty

CROSS-GRN is the first method to combine: 
1. **Asymmetric cross-modal attention** between expression and accessibility
2. **Cell-type conditioning** for context-specific network learning
3. **Signed edge prediction** (activation/repression)
4. **Single-cell multi-omics** integration

While XATGRN uses cross-attention for directionality and sign prediction, it operates on bulk RNA with GNNs and lacks cell-type conditioning. While GRNFormer and AttentionGRN use transformers for GRNs, neither combines multi-omics, signed prediction, and cell-type conditioning. CROSS-GRN fills this unique gap at the intersection of these approaches.

## 4. Experiments

### 4.1 Datasets

We will evaluate on publicly available paired scRNA-seq/scATAC-seq datasets:

1. **10x Genomics Multiome PBMC** (~10k cells, 10 cell types) - benchmark dataset with known immune cell regulons
2. **SHARE-seq mouse skin** [9] (~15k cells) - developmental system with dynamic GRN changes
3. **SEA-AD human brain** [10] (~100k cells, subset for training efficiency) - disease context with microglia activation
4. **Norman et al. K562 perturb-seq** [11] - with CRISPR ground truth for signed regulation validation

### 4.2 Baselines

**Expression Prediction**:
- **Cisformer** [1]: Cross-modal generation (most direct comparison)
- **GET** [12]: Expression prediction from chromatin
- **BABEL** [13]: Cross-modal translation baseline

**GRN Inference**:
- **scMultiomeGRN** [6]: GNN-based multi-omics GRN
- **SCENIC+** [7]: Motif-based multi-omics GRN
- **AttentionGRN** [5]: Graph transformer for GRN from scRNA-seq

**Signed/Directional GRN**:
- **XATGRN** [3]: Cross-attention for signed GRN from bulk RNA (adapted for comparison)
- **TRaCE+** [14]: Signed GRN from KO data (K562 only)
- **GENIE3 + correlation sign**: Baseline signed inference

### 4.3 Evaluation Metrics

**Expression Prediction**:
- Pearson correlation between predicted and observed expression
- R² (coefficient of determination)
- MSE on differentially expressed genes

**GRN Inference (Existence)**:
- AUROC and AUPRC against ChIP-seq/ground truth
- Early Precision Rate (EPR) at top-k predictions
- Recovery of known TF-target relationships from TRRUST [15]

**Signed GRN Inference**:
- Signed AUROC (accounts for correct sign prediction)
- Accuracy of sign prediction on edges with known directionality
- Recovery of activating vs. repressive regulons

**Cell-Type Specificity**:
- Jaccard similarity between cell-type-specific GRNs (should be moderate, not 0 or 1)
- Enrichment of cell-type-specific marker TFs in respective networks
- Consistency with perturbation experiments across cell types

### 4.4 Ablation Studies

We will conduct rigorous ablation studies to isolate the contribution of each component:

1. **Symmetric vs. Asymmetric attention**: Replace asymmetric with standard symmetric cross-attention
2. **Without cell-type conditioning**: Remove cell-type embeddings and modulation
3. **Without sign prediction**: Train only for edge existence, not sign
4. **Single-modality**: Train with RNA-only vs. multi-omics
5. **XATGRN architecture comparison**: Implement XATGRN's GNN approach on our single-cell multi-omics data

### 4.5 Expected Results

We expect CROSS-GRN to:
1. Match or exceed Cisformer's cross-modal generation performance (Pearson r > 0.85)
2. Outperform scMultiomeGRN and AttentionGRN by 10-15% in standard GRN AUROC
3. Achieve >70% accuracy in predicting regulatory sign (activation vs. repression)
4. Show meaningful variation in TF-target edge weights across cell types (Jaccard 0.3-0.7 between related types)
5. Better predict perturbation outcomes than unsigned methods (lower MSE on expression changes)
6. Demonstrate that asymmetric attention significantly outperforms symmetric attention (p < 0.05)

## 5. Success Criteria

### 5.1 Confirming Hypothesis

Success would be demonstrated by:
1. **Quantitative**: Statistically significant improvement in GRN inference AUROC compared to scMultiomeGRN and AttentionGRN (paired t-test, p < 0.05)
2. **Directionality**: Forward (TF→target) attention significantly more predictive than reverse attention (t-test, p < 0.01)
3. **Signed Edges**: Sign prediction accuracy significantly above random (binomial test, p < 0.001) on validation set
4. **Cell-Type Conditioning**: Model with cell-type modulation outperforms static model (ablation study, p < 0.05)
5. **Comparison to XATGRN**: CROSS-GRN outperforms XATGRN when both are applied to single-cell multi-omics data
6. **Biological**: Recovered GRNs show cell-type-specific enrichment of known transcription factors and correctly distinguish activating from repressive regulons

### 5.2 Refuting Hypothesis

If our approach fails, we would observe:
1. Asymmetric attention provides no benefit over symmetric attention (Cisformer-style)
2. Sign prediction is no better than random or correlation-based baselines
3. Cell-type conditioning does not improve over cell-agnostic training

In this case, we would investigate whether:
- Directionality cannot be learned from observational data without perturbations
- The sign of regulation is inherently ambiguous from static multi-omics data
- Cell-type differences are better captured by separate models rather than conditioning

## 6. Computational Requirements and Feasibility

### 6.1 Resource Requirements

**Model Size**: ~35M parameters (smaller than Cisformer due to asymmetric weight sharing)
- 8 layers × 2 encoders × 384 hidden dim
- Dual cross-attention modules with separate projections: ~8M parameters
- Cell-type conditioning network: ~2M parameters

**Training**:
- Single NVIDIA RTX A6000 (48GB) is sufficient
- Expected training time: 3-4 hours per dataset
- Total experiments (including ablations): ~6-8 hours

**Data**:
- Preprocessed datasets: <50GB storage
- All datasets are publicly available

### 6.2 Implementation Plan

1. **Week 1**: Data preprocessing and baseline evaluation
2. **Week 2**: Core model implementation (asymmetric attention + cell-type conditioning)
3. **Week 3**: Training, hyperparameter tuning, and ablation studies
4. **Week 4**: Evaluation, analysis, and paper writing

## 7. Significance and Impact

### 7.1 Scientific Contribution

1. **Technical**: First method to combine asymmetric cross-attention, cell-type conditioning, and signed edge prediction for single-cell multi-omics GRN inference—distinct from XATGRN which uses bulk data and GNNs
2. **Biological**: Enabling inference of context-specific, signed regulatory networks from observational single-cell multi-omics data
3. **Methodological**: Demonstrating that directionality and sign can be learned from static multi-omics with appropriate architectural inductive biases

### 7.2 Practical Applications

1. **Disease Mechanism Discovery**: Identifying cell-type-specific dysregulation in disease
2. **Therapeutic Targeting**: Distinguishing activators from repressors for drug intervention
3. **Cell Engineering**: Predicting outcomes of genetic perturbations in specific cell types

## 8. References

[1] Ji, L., et al. (2025). Cisformer: a scalable cross-modality generation framework for decoding transcriptional regulation at single-cell resolution. *Genome Biology*, 26, 340.

[2] Yang, Y., Xie, C., He, Q., et al. (2024). Cross-modal contrastive learning decodes developmental regulatory features through chromatin potential analysis. *GigaScience*, 13, giaf053.

[3] Xiong, J., Yin, N., Liang, S., et al. (2025). Cross-attention graph neural networks for inferring gene regulatory networks with skewed degree distribution. *BMC Bioinformatics*, 26, 40.

[4] Qiu, M., Hu, X., Zhan, F., et al. (2025). GRNFormer: A biologically-guided framework for integrating gene regulatory networks into RNA foundation models. *arXiv preprint arXiv:2503.01682*.

[5] Gao, Z., Su, Y., Tang, J., et al. (2025). AttentionGRN: a functional and directed graph transformer for gene regulatory network reconstruction from scRNA-seq data. *Briefings in Bioinformatics*, 26(2), bbaf118.

[6] Wang, Y., et al. (2024). Deep learning-based cell-specific gene regulatory networks inferred from single-cell multiome data. *Nucleic Acids Research*, 52(6), 2787-2803.

[7] Bravo González-Blas, C., et al. (2023). SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks. *Nature Methods*, 20, 1355-1367.

[8] Chang, Z., et al. (2024). SCRIPro: Single-cell and spatial multiomic inference of gene regulatory networks. *Bioinformatics*, 40(8), btae477.

[9] Ma, S., et al. (2020). Chromatin potential identified by shared single-cell profiling of RNA and chromatin. *Cell*, 183(4), 1103-1116.

[10] SEA-AD Consortium (2024). Seattle Alzheimer's Disease Brain Cell Atlas. *bioRxiv*.

[11] Norman, T.M., et al. (2019). Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. *Science*, 365(6455), 786-793.

[12] Fu, X., et al. (2025). A foundation model of transcription across human cell types. *Nature*, 637, 965-973.

[13] Wu, K.E., et al. (2021). BABEL enables cross-modality translation between multiomic profiles at single-cell resolution. *Proceedings of the National Academy of Sciences*, 118(15), e2023070118.

[14] Ud-Dean, S.M.M., et al. (2016). TRaCE+: Ensemble inference of gene regulatory networks from transcriptional expression profiles of gene knock-out experiments. *BMC Bioinformatics*, 17, 252.

[15] Han, H., et al. (2018). TRRUST v2: an expanded reference database of human and mouse transcriptional regulatory interactions. *Nucleic Acids Research*, 46(D1), D380-D386.

[16] Cao, Z.J. & Gao, G. (2022). Multi-omics single-cell data integration and regulatory inference with graph-linked embedding. *Nature Biotechnology*, 40, 1458-1466.

[17] Zhang, S., et al. (2023). Inference of cell type-specific gene regulatory networks on cell lineages from single cell omic datasets. *Nature Communications*, 14, 3064.
