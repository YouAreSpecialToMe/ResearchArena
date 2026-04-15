# Residual Epistasis Networks: Structure-Conditioned Graph Neural Networks for Correcting Additive Protein Fitness Predictions

## Introduction

### Context

Predicting the functional effects of protein mutations is a central problem in computational biology, with applications spanning protein engineering, drug resistance prediction, and understanding genetic disease. Protein language models (PLMs) such as ESM-2 (Lin et al., 2023) have emerged as powerful zero-shot predictors of mutation effects, achieving strong correlations with experimentally measured fitness across hundreds of deep mutational scanning (DMS) assays in the ProteinGym benchmark (Notin et al., 2024).

However, PLMs have a critical blind spot: **epistasis** — the phenomenon where the combined effect of multiple mutations differs from the sum of their individual effects. When predicting the fitness of multi-mutant variants, PLMs default to an additive assumption, summing the log-likelihood ratios of individual mutations. This approximation breaks down precisely when it matters most: in protein engineering, where combinations of mutations are selected through directed evolution, and in understanding genetic disease, where pathogenic variants often involve complex epistatic interactions.

### Problem Statement

Despite recent progress in benchmarking PLMs on epistasis (Nambiar et al., 2025) and using dynamics-based graph neural networks for epistasis prediction (Ozkan et al., 2025), no method explicitly learns the **non-additive residual** between a PLM's additive prediction and the true multi-mutant fitness using structural contact information. Current approaches either combine sequence and structure end-to-end (Zhang et al., 2024) without separating additive from epistatic contributions, or require expensive molecular dynamics simulations to derive structural features (Ozkan et al., 2025).

### Key Insight

We observe that epistatic interactions are strongly mediated by structural proximity — mutations at residues that are distant in sequence but close in 3D structure are far more likely to exhibit epistasis than those that are structurally distant (Olson et al., 2014; Tang et al., 2026). This suggests that a graph neural network operating over the protein's structural contact map can learn to predict the epistatic correction needed on top of a PLM's additive baseline. By explicitly decomposing fitness into an additive PLM component and a structural epistatic correction, we obtain both improved predictions and interpretable insights into the structural determinants of epistasis.

### Hypothesis

A lightweight graph attention network (GAT) trained on AlphaFold-predicted structural contact maps, using ESM-2 per-residue embeddings as node features, can learn to predict the non-additive (epistatic) component of multi-mutant protein fitness. When combined with ESM-2's additive baseline, this **Residual Epistasis Network (REN)** will significantly outperform additive PLM predictions on multi-mutant fitness prediction benchmarks, particularly for variants involving structurally proximal mutations.

## Proposed Approach

### Overview

We propose **Residual Epistasis Networks (REN)**, a two-stage framework that decomposes multi-mutant fitness prediction into:

1. **Additive stage:** ESM-2 zero-shot log-likelihood ratio scores, summed across mutation sites (no training required)
2. **Epistatic correction stage:** A graph attention network (GAT) over the AlphaFold-predicted structural contact map that predicts the residual between the additive prediction and the true fitness

The final prediction is: **fitness(v) = PLM_additive(v) + GAT_epistatic(v)**

### Method Details

#### Stage 1: Additive PLM Baseline

For each protein in the dataset:
1. Compute ESM-2 (650M parameter) per-residue embeddings for the wildtype sequence
2. For each mutation site, compute the log-likelihood ratio: LLR(m) = log P(mutant_aa | context) - log P(wildtype_aa | context)
3. For multi-mutant variants, the additive prediction is: PLM_additive(v) = Σ_i LLR(m_i)

This is the standard zero-shot PLM fitness prediction, which serves as our strong additive baseline.

#### Stage 2: Structural Epistasis Correction via GAT

**Graph construction:**
- **Nodes:** One node per residue in the protein
- **Node features:** ESM-2 per-residue embeddings (1280-dimensional for ESM-2 650M), concatenated with:
  - One-hot encoding of mutation identity at that position (20 dims for amino acid + 1 dim for "no mutation")
  - Position-specific scoring matrix (PSSM) conservation scores from MSA
- **Edges:** Residue pairs with Cβ-Cβ distance < 10Å in the AlphaFold-predicted structure (using structures from the AlphaFold Protein Structure Database or predicted on-the-fly for proteins not in AFDB)
- **Edge features:** Cβ-Cβ distance, relative sequence separation, AlphaFold pLDDT confidence at both residues

**Architecture:**
- 3-layer Graph Attention Network (GAT) with 8 attention heads per layer
- Hidden dimension: 256
- For each multi-mutant variant, aggregate node embeddings at mutation sites via attention-weighted pooling
- Feed aggregated representation through a 2-layer MLP to predict the epistatic correction: ε(v) = f_GAT(G, mutation_sites)
- Final prediction: fitness(v) = PLM_additive(v) + ε(v)

**Training objective:**
- Minimize MSE between predicted and true epistatic residuals
- The training target is: ε_true(v) = fitness_observed(v) - PLM_additive(v)
- This explicitly trains the GNN to learn ONLY what the PLM misses — the non-additive component

**Training protocol:**
- Cross-protein generalization: train on a subset of ProteinGym proteins, evaluate on held-out proteins
- Within-protein generalization: for complete combinatorial landscapes (GB1, TrpB), train on lower-order mutants (singles + doubles), predict higher-order (triples, quadruples)
- Use ProteinGym's pre-built cross-validation folds for multi-mutant evaluation

#### Stage 3: Analysis and Interpretation

- Visualize learned GAT attention weights on protein structures to identify structural features associated with epistasis
- Correlate predicted epistasis magnitudes with known biophysical properties (distance, secondary structure contacts, allosteric pathways)
- Ablation: compare edge definitions (contact map vs. sequence-based, different distance thresholds)

### Key Innovations

1. **Explicit additive-epistatic decomposition:** Unlike end-to-end methods (S3F), we separate the PLM's contribution (additive) from the structural correction (epistatic), making each component interpretable and independently evaluable.

2. **Residual learning on structural graphs:** Training the GNN only on the PLM's residuals prevents it from re-learning what the PLM already knows and focuses its capacity on the genuinely non-additive signal.

3. **Scalable structural features:** Unlike dynamics-based approaches (Ozkan et al., 2025) requiring MD simulations, we use static AlphaFold structures available for most proteins, making the method applicable proteome-wide.

4. **Order-agnostic epistasis prediction:** The GAT architecture naturally handles arbitrary numbers of simultaneous mutations through attention-weighted aggregation, without needing separate models for doubles, triples, etc.

## Related Work

### Protein Language Models for Fitness Prediction

ESM-2 (Lin et al., 2023), Tranception (Notin et al., 2022), and EVE (Frazer et al., 2021) achieve state-of-the-art zero-shot fitness prediction on ProteinGym's single-mutant benchmarks. However, their multi-mutant predictions rely on additive assumptions (summing individual mutation effects), which systematically fails for epistatic combinations. Notin et al. (2025) showed that while zero-shot models perform well on non-epistatic combinations, they fail on strongly epistatic multi-mutants. Our work directly addresses this failure mode.

### Structure-Aware Protein Language Models

SaProt (Su et al., 2024) incorporates 3D structure through Foldseek's 3Di structural tokens, achieving top ranks on ProteinGym. ProtSSN (Tan et al., 2023) uses structure-aware pre-training. S3F (Zhang et al., NeurIPS 2024) combines PLM embeddings with GVP networks encoding backbone and surface topology, showing improved epistasis capture on GB1. **Key difference:** These methods merge structure and sequence end-to-end and do not explicitly decompose additive vs. epistatic contributions. REN's residual formulation is orthogonal — it could use any of these as the base model.

### Epistasis Prediction Methods

Nambiar et al. (2025) systematically benchmarked PLMs on epistasis, showing that raw PLM scores align with structural epistasis in zero-shot settings. Ozkan et al. (2025) built a dynamics-based GNN using asymmetric dynamic coupling indices, achieving strong epistasis prediction without training on experimental data but requiring costly MD simulations. Epistatic Net (Aghazadeh et al., 2021) uses sparse spectral regularization for epistasis but does not leverage PLM embeddings or structural information. EpiNNet (Lipsh-Sokolik & Fleishman, 2024) addresses epistasis in designed proteins but not natural fitness landscapes. **Key difference:** REN uniquely combines PLM additive baselines with static structural contact GNNs, trained specifically on the non-additive residual.

### Multi-Mutant Fitness Prediction

MULTI-evolve (Tran et al., 2026) combines PLMs with epistatic modeling for directed evolution, achieving up to 10-fold improvements but is a practical engineering tool rather than a general prediction framework. The interpretable neural network of Otwinowski et al. (2025, eLife) decomposes fitness into additive and higher-order epistatic components using transformers. **Key difference:** REN provides an explicit structural mechanism for epistasis through the contact graph, rather than learning it from sequence alone.

## Experiments

### Datasets

**Primary benchmark:** ProteinGym v1.3 multi-mutant substitution datasets (69 assays, ~1.77M multi-mutant variants), using the provided cross-validation folds for multi-mutant evaluation.

**Complete combinatorial landscapes** (for systematic epistasis analysis):
- **GB1 Wu 2016:** 149,361 variants across 4 positions (order ≤ 4), IgG-Fc binding
- **GB1 Olson 2014:** 536,962 variants (all pairwise doubles), IgG-Fc binding
- **avGFP Sarkisyan 2016:** 51,714 variants (order ≤ 15), fluorescence
- **His3 Pokusaeva 2019:** 496,137 variants, growth fitness
- **CreiLOV Chen 2023:** 167,529 variants across 15 sites, fluorescence

### Baselines

1. **Zero-shot PLM (additive):** ESM-2 650M, ESM-1v, Tranception (with retrieval)
2. **Structure-aware PLMs:** SaProt, ProtSSN
3. **End-to-end PLM+structure:** S3F (Zhang et al., 2024)
4. **Epistasis-specific:** Global epistasis model (additive + nonlinear sigmoid), Epistatic Net
5. **Simple baselines:** Ridge regression on one-hot encoding, additive + pairwise interaction terms

### Metrics

- **Spearman correlation (ρ):** Primary metric, correlating predicted with observed fitness across all multi-mutant variants
- **Epistasis-specific Spearman correlation:** Correlation computed only on the non-additive component (observed fitness minus additive prediction from single-mutant data)
- **NDCG@k:** Normalized discounted cumulative gain for top-k fitness variant ranking (relevant for protein engineering applications)
- **Order-stratified performance:** Evaluate separately on doubles, triples, quadruples, etc. to assess generalization to higher-order mutations
- **Per-protein Spearman:** Following ProteinGym protocol, compute per-assay Spearman and report the average

### Experimental Setup

1. **Cross-protein evaluation:** Leave-one-protein-out cross-validation on ProteinGym multi-mutant assays
2. **Within-protein generalization:** Train on singles + doubles from GB1 Olson 2014, predict triples + quadruples from GB1 Wu 2016
3. **Ablation studies:**
   - Contact threshold (8Å, 10Å, 12Å, 15Å)
   - Node features: ESM-2 embeddings only vs. ESM-2 + PSSM vs. ESM-2 + one-hot mutation identity
   - GNN depth (1, 2, 3, 4 layers)
   - Edge features: with/without distance, with/without pLDDT
   - Base model: ESM-2 650M vs. ESM-2 3B vs. SaProt vs. Tranception

### Expected Results

1. REN should improve Spearman correlation over additive ESM-2 by 0.05-0.15 on multi-mutant datasets, with larger improvements on proteins exhibiting strong epistasis
2. The epistasis-specific Spearman correlation should show the largest gains, as this directly measures what the correction module learns
3. Performance gains should increase with mutation order (triples > doubles), as epistatic effects compound
4. GAT attention weights should correlate with known contact-mediated epistatic interactions, especially in GB1 where the structural basis of epistasis is well-characterized
5. The method should be particularly strong on structurally compact proteins with dense contact networks

### Computational Budget

- ESM-2 650M forward passes for all ProteinGym proteins: ~1 hour on 1x A6000
- AlphaFold structure download/prediction: ~30 minutes
- GNN training (per cross-validation fold): ~15 minutes
- Full cross-validation + ablations: ~4-5 hours
- Analysis and visualization: ~1 hour
- **Total: ~7-8 hours** (within budget)

## Success Criteria

### Confirmatory (hypothesis supported)

1. REN achieves statistically significant improvement (p < 0.05, paired Wilcoxon test) over additive ESM-2 on the ProteinGym multi-mutant benchmark average Spearman correlation
2. The improvement is larger for high-epistasis proteins (top quartile by epistasis magnitude) than low-epistasis proteins
3. GAT attention weights on structurally proximal mutation pairs correlate with measured epistatic effects (Spearman ρ > 0.3 on GB1)

### Refutatory (hypothesis rejected)

1. No significant improvement over additive ESM-2 across the benchmark
2. The learned epistatic corrections are uncorrelated with structural proximity
3. End-to-end methods (S3F) match or exceed REN without explicit decomposition, suggesting the decomposition adds no value

### Stretch goals

1. REN generalizes from low-order to high-order mutations: trained on doubles, predicts triples/quadruples better than baselines
2. Attention analysis reveals known allosteric pathways without supervision
3. REN achieves SOTA on the full ProteinGym multi-mutant leaderboard

## References

1. Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., ... & Rives, A. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130.

2. Notin, P., Kollasch, A., Ritter, D., van Niekerk, L., Paul, S., Spinner, H., ... & Marks, D. S. (2024). ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design. *Advances in Neural Information Processing Systems*, 36.

3. Notin, P., Dias, M., Frazer, J., Marchena-Hurtado, J., Gomez, A. N., Marks, D., & Gal, Y. (2022). Tranception: Protein Fitness Prediction with Autoregressive Transformers and Inference-time Retrieval. *International Conference on Machine Learning (ICML)*.

4. Frazer, J., Notin, P., Dias, M., Gomez, A., Min, J. K., Busia, K., ... & Marks, D. S. (2021). Disease variant prediction with deep generative models of evolutionary data. *Nature*, 599(7883), 91-95.

5. Nambiar, A., Littlefield, S. B., Cuellar, C., Khorana, R., & Maslov, S. (2025). Protein Language Models Capture Structural and Functional Epistasis in a Zero-Shot Setting. *bioRxiv*, 2025.09.14.676130.

6. Ozkan, B., et al. (2025). A Protein Dynamics-Based Deep Learning Model Enhances Predictions of Fitness and Epistasis. *PNAS*, 122(42).

7. Zhang, Z., Notin, P., Huang, Y., Lozano, A., Chenthamarakshan, V., Marks, D., Das, P., & Tang, J. (2024). Multi-Scale Representation Learning for Protein Fitness Prediction. *Advances in Neural Information Processing Systems (NeurIPS)*.

8. Su, J., Han, C., Zhou, Y., Shan, J., Zhou, X., & Yuan, F. (2024). SaProt: Protein Language Modeling with Structure-aware Vocabulary. *International Conference on Learning Representations (ICLR)*.

9. Tan, Y., Zhou, B., Zheng, L., Fan, G., & Hong, L. (2023). Semantical and Topological Protein Encoding Toward Enhanced Bioactivity and Thermostability. *bioRxiv*.

10. Tran, V. Q., Nemeth, M., Bartie, L. J., Chandrasekaran, S. S., Fanton, A., Moon, H. C., Hie, B. L., Konermann, S., & Hsu, P. D. (2026). Rapid Directed Evolution Guided by Protein Language Models and Epistatic Interactions. *Science*.

11. Tang, M., Cromie, G. A., Kabir, A., Timour, M. S., et al. (2026). Predicting Epistasis Across Proteins by Structural Logic. *PNAS*.

12. Lipsh-Sokolik, R., & Fleishman, S. J. (2024). Addressing Epistasis in the Design of Protein Function. *PNAS*, 121(34).

13. Aghazadeh, A., Nisonoff, H., Ocal, O., et al. (2021). Epistatic Net allows the sparse spectral regularization of deep neural networks for inferring fitness functions. *Nature Communications*, 12, 5225.

14. Olson, C. A., Wu, N. C., & Sun, R. (2014). A comprehensive biophysical description of pairwise epistasis throughout an entire protein domain. *Current Biology*, 24(22), 2643-2651.

15. Wu, N. C., Dai, L., Olson, C. A., Lloyd-Smith, J. O., & Sun, R. (2016). Adaptation in protein fitness landscapes is facilitated by indirect paths. *eLife*, 5, e16965.

16. Sarkisyan, K. S., Bolotin, D. A., Meer, M. V., Usmanova, D. R., Mishin, A. S., Sharonov, G. V., ... & Kondrashov, F. A. (2016). Local fitness landscape of the green fluorescent protein. *Nature*, 533(7603), 397-401.

17. Pokusaeva, V. O., Usmanova, D. R., Putintseva, E. V., Espinar, L., Carey, L. B., & Kondrashov, A. S. (2019). An experimental assay of the interactions of amino acids from orthologous sequences shaping a complex fitness landscape. *PLoS Genetics*, 15(3), e1008079.

18. Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., ... & Hassabis, D. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596(7873), 583-589.

19. Otwinowski, J. (2025). An Interpretable Neural Network Unveils Higher-Order Epistasis in Large Protein Sequence-Function Relationships. *eLife* (reviewed preprint).

20. Johnston, K. E., et al. (2024). A combinatorially complete epistatic fitness landscape in an enzyme active site. *PNAS*, 121(32), e2400439121.
