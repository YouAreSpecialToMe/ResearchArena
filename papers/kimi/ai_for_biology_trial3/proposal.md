# Structure-Aware Conditional VAE for Multi-Objective Therapeutic Peptide Design

## Introduction

### Background and Motivation

Therapeutic peptides represent a promising drug class that bridges the gap between small molecules and biologics, offering high specificity and potency with lower toxicity. However, their clinical translation is severely limited by poor pharmacokinetic (PK) properties—particularly membrane permeability, proteolytic stability, and solubility. Over 90% of peptides in clinical development fail due to these ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) challenges.

Recent advances in AI-driven peptide design have made remarkable progress in two separate directions:
1. **Structure-based design** (RFdiffusion, ProteinMPNN, RFpeptides): Generate peptides that fold into desired backbone structures but do not explicitly optimize for PK properties.
2. **Property-based prediction** (GraphCPP, LightCPPgen, PEGASUS): Predict membrane permeability, stability, or other properties but do not generate novel sequences.

**The Critical Gap**: While recent multimodal methods (MMCD, Multi-Peptide) combine sequence and structure information, they either use simple late fusion strategies or focus only on prediction tasks. No existing method jointly optimizes both structural validity AND pharmacokinetic properties within a unified disentangled latent space that enables precise property control.

### Key Insight

The conformational flexibility of cyclic peptides—their "chameleonic" ability to expose polar groups in aqueous environments while burying them in membrane-like environments—is the key determinant of membrane permeability. This structural mechanism cannot be captured by sequence-only models but requires explicit 3D structure awareness during generation. Cross-attention fusion between sequence and structure representations enables early interaction between modalities, capturing dependencies that late fusion approaches miss.

### Proposed Solution

We propose **StruCVAE-Pep** (Structure-Conditioned Variational Autoencoder for Peptide Design), a deep generative model that:
1. Uses a **multimodal encoder** with cross-attention fusion to process both peptide sequences and 3D structures
2. Learns a **disentangled latent space** with explicit structure, property, and sequence factors
3. Enables **conditional generation** where target permeability/stability values guide sequence generation
4. Employs **stabilized training techniques** to prevent posterior collapse common in disentangled VAEs

---

## Proposed Approach

### Architecture Overview

**StruCVAE-Pep** consists of four key components:

#### 1. Multimodal Structure-Sequence Encoder with Cross-Attention Fusion
- **Sequence Encoder**: Pre-trained protein language model (ESM-2 650M) fine-tuned on cyclic peptides
- **Structure Encoder**: Graph neural network (DMPNN) operating on 3D conformations with bond/angle information
- **Fusion Module**: Cross-attention mechanism where structure features attend to sequence features and vice versa, enabling early interaction between modalities

**Advantage over late fusion**: Unlike MMCD (which aligns independent sequence and structure encoders via contrastive learning) and Multi-Peptide (which concatenates separately-trained embeddings), cross-attention allows each modality to dynamically select relevant information from the other during encoding.

#### 2. Disentangled Latent Space
- **Structure latent (z_s)**: Captures backbone conformation and fold information
- **Property latent (z_p)**: Encodes permeability, stability, and solubility factors
- **Sequence latent (z_seq)**: Captures residue-level sequence patterns

Each latent factor has its own encoder head and prior distribution, enabling independent control during generation.

#### 3. Conditional Decoder
- Autoregressive decoder generating sequences conditioned on concatenated latents (z_s, z_p, z_seq)
- Property prediction heads providing auxiliary supervision during training

#### 4. Structure Validation Module
- Self-consistency loss encouraging generated sequences to have consistent predicted structures
- Optional AlphaFold2 confidence filtering for final designs

### Training Objectives

The model is trained with a composite loss function:

```
L_total = L_reconstruction + β(t)·L_KL + γ·L_property + δ·L_structure + ε·L_adversarial
```

Where:
- **L_reconstruction**: Cross-entropy loss for sequence reconstruction
- **L_KL**: KL divergence regularizing the latent space (with cyclical annealing schedule β(t))
- **L_property**: MSE loss for permeability/stability prediction (multi-task)
- **L_structure**: Contrastive loss ensuring structure latent captures conformational similarity
- **L_adversarial**: Gradient reversal loss to enforce disentanglement (following PLUM)

### Addressing Training Instability in Disentangled VAEs

Disentangled VAEs are known to suffer from **posterior collapse**, where the posterior distribution collapses to the prior, losing all information about the input. We employ multiple stabilization techniques:

1. **Cyclical KL Annealing** (Fu et al., 2019): Gradually increases β from 0 to 1 over multiple cycles, allowing the model to learn meaningful representations before full regularization kicks in.

2. **Free Bits** (Kingma et al., 2016): Sets a minimum threshold for KL divergence per latent dimension, preventing any single dimension from collapsing.

3. **Aggressive Encoder Training** (He et al., 2019): Performs multiple encoder updates per decoder update when the encoder is "lagging" behind.

4. **Decoder Weakening via Word Dropout**: Randomly drops conditioning tokens in the autoregressive decoder to force reliance on latent variables.

5. **β-VAE Regularization**: Uses β > 1 for the property latent to encourage disentanglement while maintaining β ≈ 0.5-1 for other latents to preserve reconstruction quality.

### Property Conditioning

During generation, users specify target property values (y_target) which are encoded and combined with sampled latents:

```
z_p = μ_p(y_target) + σ_p(y_target) · ε
```

This enables property-targeted generation: sampling peptides predicted to have specific permeability/stability values.

---

## Related Work

### Structure-Based Peptide Design

**RFdiffusion** (Watson et al., Nature 2023) revolutionized protein design by training a diffusion model on protein backbones. **RFpeptides** extends this to cyclic peptides but focuses solely on binding interfaces without PK optimization. **ProteinMPNN** (Dauparas et al., Science 2022) designs sequences for fixed backbones but requires post-hoc property filtering.

Our key distinction: We integrate structure and property prediction into a single generative model with a disentangled latent space, enabling gradient-based multi-objective optimization with explicit control over individual factors.

### Property Prediction for Peptides

**GraphCPP** (Imre et al., 2025) uses graph neural networks to predict cell-penetrating peptides, achieving state-of-the-art classification accuracy. **PEGASUS** (Merck, 2024/2025) combines multiple ML models for cyclic peptide permeability prediction with experimental validation. **CycPeptMPDB** (Li et al., JCIM 2023) established the first comprehensive database of ~7,000 cyclic peptides with measured permeability.

Our contribution: Rather than just predicting properties, we use property prediction as a differentiable guide for sequence generation with explicit 3D structure constraints.

### Multi-Objective Peptide Generation

**PepTune** (Tang et al., ICML 2025) uses masked diffusion with Monte Carlo Tree Guidance for multi-objective optimization. **HMAMP** (2024) employs hypervolume-driven multi-objective RL for antimicrobial peptides. **PepMLM** fine-tunes ESM-2 for target-conditioned peptide generation.

Our novelty: We introduce structure-aware disentangled representation learning with cross-modal fusion, separating structural and property factors to enable controllable generation with explicit 3D structure constraints.

### Transformer-Based CVAE for Peptides (PepMorph)

**PepMorph** (Costa & Zavadlav, arXiv 2025) proposes a Transformer-based Conditional VAE with a masking mechanism for morphology-aware peptide generation. Key features:
- Uses ESM-2 alphabet for tokenization
- Masking mechanism enables partial conditioning on arbitrary descriptor subsets
- Trained on aggregation propensity and geometric descriptors
- 83% success rate in CG-MD validation for morphology targeting

**Our difference from PepMorph**: While PepMorph uses a sequence-only CVAE with descriptor conditioning, StruCVAE-Pep integrates explicit 3D structure encodings via cross-attention fusion and learns a disentangled latent space separating structure, property, and sequence factors. Our approach targets pharmacokinetic properties (permeability, stability, solubility) rather than aggregation morphology.

### Disentangled VAE for Antimicrobial Peptides (PLUM)

**PLUM** (Banerjee et al., bioRxiv 2026) is a structured conditional VAE for antimicrobial peptide design:
- Three disentangled latent subspaces: Zseq (sequence), Zfunc (function), Zlength (length)
- Uses adversarial predictors with gradient reversal to enforce disentanglement
- LSTM-based encoder and autoregressive decoder
- Achieves 88.5% AMP yield vs 82.7% for HydrAMP baseline

**Our difference from PLUM**: While PLUM disentangles sequence, function, and length for antimicrobial peptides using sequence-only inputs, StruCVAE-Pep adds an explicit 3D structure encoder (DMPNN) and uses cross-attention fusion between modalities. We target therapeutic peptides with pharmacokinetic optimization (membrane permeability, proteolytic stability, solubility) rather than antimicrobial activity.

### Multimodal Peptide Prediction (Multi-Peptide)

**Multi-Peptide** (JCIM 2024) combines PeptideBERT (transformer) with GNN for peptide property prediction:
- Uses AlphaFold-generated structures
- Contrastive learning aligns sequence and structure embeddings
- Late fusion via embedding concatenation

**Our difference**: Multi-Peptide focuses on prediction tasks and uses late fusion. StruCVAE-Pep is a generative model with early fusion via cross-attention, enabling structure-aware sequence generation rather than just property prediction.

### Multi-Modal Contrastive Diffusion (MMCD)

**MMCD** (Wang et al., arXiv 2023/2024) fuses sequence and structure modalities in a diffusion framework:
- Co-generates peptide sequences and 3D structures
- Inter-contrastive learning aligns sequence and structure embeddings
- Intra-contrastive learning differentiates therapeutic vs non-therapeutic peptides
- Uses late fusion with separate encoders for each modality

**Our difference**: MMCD uses diffusion with late fusion (contrastive alignment of independent encoders). StruCVAE-Pep uses a VAE framework with early cross-attention fusion and explicit disentanglement of structure vs property factors. Our approach enables direct conditioning on continuous property values (e.g., target permeability) rather than just binary therapeutic/non-therapeutic classification.

### Structure-Aware Protein Language Models

**SaProt** (Su et al., ICLR 2024) introduced structure-aware vocabulary combining residue tokens with Foldseek 3Di structure tokens. Published in **ICLR 2024** (not Nature Communications). This demonstrated that integrating structure information improves downstream task performance.

Our extension: We apply structure-aware representations to the generative setting, conditioning peptide generation on both sequence and explicit 3D structure information via cross-attention fusion.

---

## Experiments

### Datasets

1. **CycPeptMPDB** (Li et al., JCIM 2023): 7,334 cyclic peptides with PAMPA permeability measurements
2. **PROSO-II**: ~4,000 peptide solubility measurements
3. **PepTherDev**: Curated stability measurements from literature
4. **PDB cyclic peptides**: ~2,000 high-resolution structures for structure pre-training

### Evaluation Metrics

**Generation Quality**:
- Validity: % of generated sequences chemically valid (correct cyclization)
- Novelty: % not in training set (via sequence similarity)
- Diversity: Average pairwise edit distance among generated peptides

**Property Prediction**:
- Permeability: R² and RMSE vs. experimental PAMPA values
- Stability: Classification accuracy for protease resistance
- Multi-task: Performance across all PK properties

**Structure Quality**:
- Designability: pLDDT > 70 for AlphaFold2 predictions
- Diversity: Structural clustering of generated backbones

### Experimental Setup

**Baselines**:
1. **VAE-only**: Sequence-only variational autoencoder
2. **PepMorph**: Transformer-based CVAE with masking (reimplemented for PK properties)
3. **PLUM**: Disentangled VAE adapted for PK properties
4. **MMCD-lite**: Diffusion-based multimodal method (reimplemented for comparison)
5. **Structure→Property pipeline**: RFpeptides + PEGASUS filtering
6. **DMPNN**: Direct property prediction from Liu et al. 2025 benchmark (R² ≈ 0.65 on random split)

**Proposed Variants**:
1. **StruCVAE-Pep (full)**: Complete model with all components
2. **StruCVAE-Pep (-structure)**: Ablated without structure encoder
3. **StruCVAE-Pep (-disentangled)**: Single latent space (no factorization)
4. **StruCVAE-Pep (-cross-attn)**: Late fusion instead of cross-attention

### Expected Results

We hypothesize that:

1. **Structure awareness improves property prediction**: Models incorporating 3D structure will achieve R² > 0.70 on permeability prediction (vs. R² ≈ 0.65 for DMPNN sequence-only baseline from Liu et al. 2025), as conformational features (e.g., polar surface burial) directly impact membrane permeation.

2. **Cross-attention outperforms late fusion**: Early fusion via cross-attention will provide 5-10% better property prediction than late fusion approaches (as in MMCD/Multi-Peptide) by capturing modality interactions during encoding.

3. **Disentangled latents enable controllable generation**: The model will generate peptides with target permeability values within 0.5 log unit of specified targets, demonstrating precise property control.

4. **Joint training improves over pipelines**: End-to-end optimization will produce 2-3x more "drug-like" peptides (simultaneously permeable, stable, soluble) compared to sequential generate-then-filter approaches.

5. **Training stabilization works**: Using cyclical annealing + free bits + aggressive encoder training, we will achieve <20% posterior collapse rate (measured by inactive latent dimensions) vs >50% for vanilla disentangled VAE.

6. **Novelty without sacrificing quality**: Generated peptides will achieve >90% novelty while maintaining comparable predicted properties to known therapeutic cyclic peptides.

---

## Success Criteria

### Confirming Evidence

The hypothesis would be strongly supported if:
- Permeability prediction R² > 0.70 on held-out test set (vs. R² ≈ 0.65 for DMPNN baseline)
- Generated peptides with specified permeability achieve mean absolute error < 0.5 log Papp units
- >30% of generated peptides pass all three PK filters (permeable, stable, soluble) vs. <10% for baselines
- Structural validation shows >80% of generated sequences fold to compact, designable structures
- Cross-attention fusion shows >5% improvement over late fusion ablation
- <20% of latent dimensions collapse during training (measured by KL < 0.1)

### Refuting Evidence

The hypothesis would be challenged if:
- Structure information does not improve property prediction (structure vs. sequence ablation shows <3% difference)
- Disentangled latents collapse (>50% inactive dimensions) or fail to provide independent control over properties
- Generated sequences fail structural validation (<50% designable) indicating mode collapse
- Property conditioning is ineffective (property variance of conditioned generation similar to unconditional)
- Cross-attention provides no benefit over late fusion (<2% difference)

### Ablation Studies

We will conduct systematic ablations to validate each component:
1. **Without structure encoder**: Measure property prediction degradation
2. **Without disentanglement**: Assess controllability loss
3. **Without cross-attention fusion**: Compare late vs early fusion
4. **Without training stabilization**: Measure posterior collapse rate
5. **Varying β (KL weight)**: Explore reconstruction-quality vs. latent regularization tradeoff

---

## Impact and Significance

### Scientific Contribution

1. **Cross-attention fusion for multimodal peptide generation**: Demonstrating that early interaction between sequence and structure modalities via cross-attention improves over late fusion approaches used in prior work (MMCD, Multi-Peptide).

2. **Stabilized disentangled VAE for peptides**: Systematically addressing posterior collapse in disentangled VAEs through combined cyclical annealing, free bits, and aggressive encoder optimization—providing a recipe for training stable disentangled models on peptide data.

3. **Structure-property joint optimization**: Enabling gradient-based multi-objective optimization across both structure and PK objectives within a unified framework.

4. **Explicit disentangled control**: Separating structure, property, and sequence factors enables interpretable manipulation of specific ADMET profiles during generation.

### Practical Impact

1. **Accelerated peptide drug discovery**: Reducing experimental screening burden by 10-100x through in silico pre-optimization.

2. **Oral peptide therapeutics**: By explicitly optimizing for membrane permeability, the model addresses the key barrier to oral peptide drugs.

3. **De novo immunogenicity reduction**: Controllable generation can avoid known immunogenic motifs while retaining therapeutic function.

---

## References

### Key Papers

1. **Li et al. (2023)**. "CycPeptMPDB: A Comprehensive Database of Membrane Permeability of Cyclic Peptides." *Journal of Chemical Information and Modeling*. [Dataset foundation]

2. **Dauparas et al. (2022)**. "Robust deep learning-based protein sequence design using ProteinMPNN." *Science*. [Structure-based sequence design baseline]

3. **Watson et al. (2023)**. "De novo design of protein structure and function with RFdiffusion." *Nature*. [Diffusion-based backbone generation]

4. **Liu et al. (2025)**. "Systematic benchmarking of 13 AI methods for predicting cyclic peptide membrane permeability." *Journal of Cheminformatics*. [Property prediction benchmark showing DMPNN achieves R² ≈ 0.65]

5. **Tang et al. (2025)**. "PepTune: de novo generation of therapeutic peptides with multi-objective-guided discrete diffusion." *ICML 2025*. [Multi-objective diffusion baseline]

6. **Su et al. (2024)**. "SaProt: Protein Language Modeling with Structure-aware Vocabulary." *ICLR 2024*. [Structure-aware representation learning - CORRECTED VENUE]

7. **Imre et al. (2025)**. "GraphCPP: The new state-of-the-art method for cell-penetrating peptide prediction via graph neural networks." *British Journal of Pharmacology*. [Graph-based property prediction]

8. **Costa & Zavadlav (2025)**. "Morphology-Specific Peptide Discovery via Masked Conditional Generative Modeling." *arXiv:2509.02060*. [PepMorph - Transformer CVAE with masking]

9. **Banerjee et al. (2026)**. "A PLUM Job: Peptide modeLs for Understanding and engineering antiMicrobial therapeutics." *bioRxiv*. [PLUM - Disentangled VAE with Zseq/Zfunc/Zlength]

10. **Wang et al. (2023)**. "A Multi-Modal Contrastive Diffusion Model for Therapeutic Peptide Generation." *arXiv:2312.15665*. [MMCD - Late fusion via contrastive learning]

11. **Multi-Peptide (2024)**. "Multi-Peptide: Multimodality Leveraged Language-Graph Learning of Peptide Properties." *Journal of Chemical Information and Modeling*. [Late fusion for prediction]

12. **He et al. (2019)**. "Lagging Inference Networks and Posterior Collapse in Variational Autoencoders." *ICLR 2019*. [Aggressive encoder training for VAE stability]

13. **Fu et al. (2019)**. "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing." *NAACL 2019*. [Cyclical KL annealing]

14. **Kingma et al. (2016)**. "Improved Variational Inference with Inverse Autoregressive Flow." *NeurIPS 2016*. [Free bits technique]

15. **Rives et al. (2021)**. "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences." *PNAS* (ESM-1b/ESM-2). [Foundation model]

16. **Chen et al. (2024)**. "HMAMP: Hypervolume-Driven Multi-Objective Antimicrobial Peptides Design." *arXiv*. [Multi-objective RL baseline]

---

## Resource Requirements

### Computational Resources (Available)
- GPU: 1x NVIDIA RTX A6000 (48GB VRAM) - Sufficient for training
- RAM: 60GB - Adequate for data loading and processing
- Training time: ~6-8 hours for full model on CycPeptMPDB (~7k samples)

### Software Dependencies
- PyTorch 2.0+ with CUDA support
- ESM-2 (via fair-esm or transformers)
- RDKit (molecule processing)
- OpenFold/AlphaFold2 (structure validation)
- DGL/PyTorch Geometric (GNN components)

### Data Requirements
- CycPeptMPDB (~7k peptides with permeability) ✓ Publicly available
- PROSO-II solubility dataset ✓ Publicly available
- PDB cyclic peptide structures ✓ Publicly available

---

## Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Data Preparation | 1 hour | Download, clean, and preprocess datasets |
| Model Development | 3 hours | Implement StruCVAE-Pep architecture |
| Training | 2 hours | Train model with stabilization techniques |
| Evaluation | 1 hour | Benchmark against baselines |
| Analysis | 1 hour | Ablation studies and result analysis |

Total: ~8 hours, fitting within the allocated compute budget.
