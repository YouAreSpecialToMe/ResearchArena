# EpiGNN: Epistasis-Aware Multi-Mutation Protein Fitness Prediction via Message Passing on Language Model-Derived Residue Coupling Graphs

## Introduction

### Context

Protein engineering requires predicting how mutations affect protein function — a task known as *fitness prediction*. Deep mutational scanning (DMS) experiments have generated millions of single-mutation measurements, and protein language models (PLMs) like ESM-2 have achieved impressive zero-shot accuracy on these benchmarks by computing log-likelihood ratios at mutated positions. However, real-world protein engineering typically involves introducing *multiple simultaneous mutations*, where the combined effect is not simply the sum of individual effects. This phenomenon, known as *epistasis*, remains a major challenge for computational prediction.

### Problem Statement

Current PLM-based fitness predictors handle multi-mutation variants by assuming additivity: the predicted fitness of a multi-mutant is the sum of predicted single-mutation effects. This assumption systematically fails when epistatic interactions are present. Recent work by Dieckhaus & Kuhlman (2025) demonstrated that even state-of-the-art stability prediction models fail to capture epistatic interactions of double point mutations, with additive models performing comparably to more complex non-additive predictors. Meanwhile, Nambiar et al. (2025) showed that PLMs *do* encode pairwise epistatic information in their internal representations — but this information is not exploited by standard scoring approaches.

### Key Insight

We hypothesize that the residue-residue coupling information encoded in PLM attention maps and hidden state correlations can serve as a structural prior for modeling epistatic interactions between co-occurring mutations. Rather than treating mutations independently, we propose to model them as interacting nodes in a *mutation-centric graph*, where edges represent PLM-derived residue couplings. Message passing on this graph allows mutation effects to propagate through the coupling network, producing non-additive fitness predictions.

### Hypothesis

A graph neural network operating on PLM-derived residue coupling graphs will significantly improve multi-mutation fitness prediction over additive PLM baselines, and the improvement will correlate with the magnitude of epistatic interactions in the target protein.

## Proposed Approach

### Overview

We propose **EpiGNN**, a lightweight graph neural network architecture that predicts epistatic corrections to additive PLM fitness scores. The key components are:

1. **PLM Feature Extraction**: Extract per-residue embeddings and attention-based coupling scores from ESM-2 for each protein.
2. **Mutation Graph Construction**: For each multi-mutant, construct a small subgraph where nodes are mutated positions and edges are weighted by PLM-derived residue couplings.
3. **Message Passing**: Apply graph attention layers to propagate mutation effects through the coupling network, producing a predicted epistatic correction.
4. **Fitness Prediction**: Final fitness = additive PLM score + learned epistatic correction.

### Method Details

#### Step 1: PLM Feature Extraction

For each protein in the dataset:
- Run ESM-2 (650M parameters) on the wild-type sequence to obtain:
  - Per-residue embeddings **h_i** ∈ ℝ^d from the final hidden layer
  - Attention maps **A** ∈ ℝ^{L×L×H} across all heads and layers
- For each mutation at position *i* (wild-type residue *w* → mutant residue *m*):
  - Compute mutation feature: **δ_i** = ESM-2(**seq_mut**)_i − ESM-2(**seq_wt**)_i (change in residue embedding)
  - Compute scalar mutation score: s_i = log P(m|context) − log P(w|context) (standard masked marginal scoring)

#### Step 2: Residue Coupling Graph

Extract a residue coupling matrix **C** ∈ ℝ^{L×L} from ESM-2:
- **Attention-based coupling**: Average attention weights across selected heads/layers known to encode structural contacts (following Rao et al., 2021). Apply APC (Average Product Correction) to remove phylogenetic background signal.
- **Optional structural augmentation**: If predicted structures are available (via ESMFold), add distance-based edges for residue pairs within 8Å at the Cα level.

For a multi-mutant with mutations at positions {p_1, p_2, ..., p_k}, construct a *mutation subgraph* G = (V, E):
- Nodes V = {p_1, ..., p_k}, with features **δ_{p_i}** (mutation embeddings)
- Edges E: fully connected among mutated positions, with edge features:
  - Coupling score C_{p_i, p_j}
  - Sequence separation |p_i − p_j|
  - (Optional) Euclidean distance from predicted structure

#### Step 3: Graph Neural Network

Apply 2–3 layers of graph attention (GATv2) on the mutation subgraph:

```
h_i^{(l+1)} = h_i^{(l)} + Σ_j α_{ij} W^{(l)} h_j^{(l)}
```

where α_{ij} is an attention coefficient that incorporates edge features (coupling, distance). The final node representations are aggregated via sum pooling and passed through an MLP to predict the epistatic correction term ε:

```
ε = MLP(Σ_i h_i^{(final)})
```

#### Step 4: Final Prediction

```
Fitness(multi-mutant) = Σ_i s_i + ε
```

where s_i are the additive single-mutation PLM scores and ε is the learned epistatic correction from EpiGNN.

#### Training Objective

Minimize MSE between predicted and observed fitness:
```
L = Σ ||Fitness_pred − Fitness_obs||²
```

We also experiment with training on the *epistasis residual* directly:
```
L_epi = Σ ||ε_pred − (Fitness_obs − Σ_i s_i)||²
```

### Key Innovations

1. **Mutation-centric graph**: Unlike prior work that builds full protein-level graphs (e.g., GraphESMStable, dynamics GNN), we construct small subgraphs over only the mutated positions. This makes the model lightweight, focused on inter-mutation interactions, and scalable to higher-order mutants.

2. **PLM-derived coupling as structural prior**: Instead of requiring expensive molecular dynamics simulations (Zheng et al., 2025) or hand-crafted structural features, we extract residue coupling information directly from the PLM's attention maps — a cheap and universal proxy for structural/functional proximity.

3. **Architecture-agnostic**: EpiGNN is a plug-in module that works on top of *any* PLM scoring method (ESM-2, Tranception, EVE, etc.), correcting their additive assumptions.

4. **Explicit epistasis modeling**: By predicting a correction term rather than the full fitness, the model directly targets the non-additive component, making it easier to learn and interpret.

## Related Work

### Protein Language Models for Fitness Prediction

PLMs such as ESM-2 (Lin et al., 2023), ESM-1v (Meier et al., 2021), and Tranception (Notin et al., 2022) achieve strong zero-shot fitness prediction by leveraging evolutionary statistics learned from large protein sequence databases. However, these models predict multi-mutation effects additively by summing independent single-mutation scores, which fails to capture epistatic interactions. Our work builds on PLM representations but introduces an explicit non-additive correction.

### Epistasis in Protein Language Models

Nambiar et al. (2025) demonstrated that PLMs capture pairwise structural and functional epistasis in a zero-shot setting, showing that attention maps align with residue-residue contacts and that nonlinear transformations of PLM outputs correlate with functional couplings. However, their work is analytical rather than predictive — they study what PLMs encode but do not build a model to exploit this for fitness prediction. Tsui & Aghazadeh (2024) studied higher-order interaction recovery from PLMs theoretically. Our work operationalizes these findings into a practical prediction method.

### Structure-Based Epistasis Prediction

Zheng et al. (2025) introduced a dynamics-based GNN that uses molecular dynamics simulations to compute residue coupling (DCIasym), achieving strong epistasis prediction. However, MD simulations are computationally expensive (hours to days per protein), limiting scalability. We replace MD-derived dynamics with PLM attention-derived coupling, which requires only a single forward pass through the PLM.

### Multi-Mutation Stability Prediction

SPURS (Li & Luo, 2025) uses rewired protein generative models (ESM-2 + ProteinMPNN) for stability prediction, including higher-order mutations. GraphESMStable (2026) combines ESM embeddings with 3D structural features and an epistasis decoder for stability (ΔΔG) prediction. Both focus on thermodynamic stability rather than general fitness, and use full protein-level graphs rather than mutation-centric subgraphs. Dieckhaus & Kuhlman (2025) showed that current stability models, including these approaches, struggle with epistatic interactions, motivating our focused approach.

### Fitness Landscape Modeling

The GB1 combinatorial fitness landscape (Wu et al., 2016) provided early evidence that epistasis is pervasive and structured. ProteinGym (Notin et al., 2023) established large-scale benchmarks for fitness prediction including multi-mutation assays. Our work uses these datasets for training and evaluation.

## Experiments

### Datasets

1. **ProteinGym Multi-Mutation Substitution Benchmark**: We will use all DMS assays from ProteinGym v1.3 that contain multi-mutation variants (double, triple, and higher-order substitutions). Key proteins include:
   - GB1 (IgG-binding domain, 4-position combinatorial landscape)
   - avGFP (green fluorescent protein)
   - Additional proteins with multi-mutation data across diverse families

2. **SKEMPI-derived epistasis data**: For proteins where binding affinity (ΔΔG) data is available for both single and double mutations, we can compute experimental epistasis scores.

### Baselines

- **Additive ESM-2**: Sum of single-mutation masked marginal scores (standard approach)
- **Additive Tranception**: Sum of autoregressive single-mutation scores
- **EVE (Evolutionary model of Variant Effect)**: Deep generative model, additive for multi-mutants
- **ESM-2 + MLP (no graph)**: Same mutation features but without graph structure (ablation)
- **Random graph EpiGNN**: Same architecture but with random edge weights (ablation)
- **Distance-only graph EpiGNN**: Edges based on 3D distance only, no PLM coupling (ablation)

### Metrics

- **Spearman's ρ** between predicted and observed fitness (primary metric)
- **RMSE** of fitness predictions
- **Epistasis correlation**: Spearman's ρ between predicted and observed epistasis (non-additive component)
- **Per-protein analysis**: Correlation between model improvement (over additive baseline) and magnitude of epistasis in each protein

### Experimental Protocol

1. **Within-protein evaluation**: For each protein with sufficient multi-mutation data, 5-fold cross-validation on the multi-mutation variants.
2. **Cross-protein evaluation**: Leave-one-protein-out evaluation to assess generalization.
3. **Ablation studies**:
   - Remove graph structure (MLP-only)
   - Remove PLM coupling edges (random edges)
   - Remove structural distance edges
   - Vary number of GNN layers (1, 2, 3, 4)
   - Compare different PLM backbones (ESM-2 650M, ESM-2 150M, ProtT5)
4. **Analysis**:
   - Visualize learned attention patterns on the mutation graph
   - Correlate edge importance with known structural contacts
   - Case studies on proteins with well-characterized epistatic interactions (e.g., GB1)

### Computational Requirements

- **PLM feature extraction**: ESM-2 650M inference on ~50-100 proteins, ~2-3 hours on 1× A6000
- **Graph construction**: Attention extraction + coupling computation, <30 minutes
- **GNN training**: Small model (~100K parameters), trains in <1 hour per fold
- **Total**: Well within 8-hour budget on 1× NVIDIA RTX A6000 (48GB VRAM)

### Expected Results

1. EpiGNN should achieve 5-15% improvement in Spearman ρ over additive PLM baselines on multi-mutation fitness prediction.
2. Improvement should be largest for proteins with strong epistatic interactions and smallest for nearly additive landscapes.
3. PLM-derived coupling edges should significantly outperform random edges, validating the structural prior.
4. The mutation-centric graph should outperform full protein graphs (more focused, less noise).

## Success Criteria

### Confirming the hypothesis
- EpiGNN achieves statistically significant improvement (p < 0.05, paired t-test) in Spearman correlation over additive ESM-2 baseline on ≥70% of multi-mutation DMS assays.
- The epistasis correlation metric shows the model captures non-additive effects beyond what the additive baseline predicts.
- Ablation removing graph structure (MLP-only) shows significantly worse performance than the full EpiGNN, confirming the value of the coupling graph.

### Refuting the hypothesis
- If EpiGNN shows no significant improvement over additive baselines, this would suggest either:
  (a) PLM attention does not encode sufficiently useful coupling information for epistasis prediction, or
  (b) The available multi-mutation DMS data is insufficient to learn epistatic corrections.
- If the MLP ablation performs comparably to EpiGNN, the graph structure is not contributing meaningfully.

## References

1. Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., & Rives, A. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123–1130.

2. Meier, J., Rao, R., Verkuil, R., Liu, J., Sercu, T., & Rives, A. (2021). Language models enable zero-shot prediction of the effects of mutations on protein function. *Advances in Neural Information Processing Systems*, 34.

3. Notin, P., Kollasch, A.W., Ritter, D., van Niekerk, L., Paul, S., Spinner, H., Rollins, N., Shaw, A., Weitzman, R., Frazer, J., Dias, M., Franceschi, D., Orber, R., Gal, Y., & Marks, D.S. (2023). ProteinGym: Large-scale benchmarks for protein fitness prediction and design. *Advances in Neural Information Processing Systems*, 36.

4. Notin, P., Dias, M., Frazer, J., Marchena-Hurtado, J., Gomez, A.N., Marks, D.S., & Gal, Y. (2022). Tranception: Protein fitness prediction with autoregressive transformers and inference-time retrieval. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162.

5. Nambiar, A., Littlefield, S.B., Cuellar, C., Khorana, R., & Maslov, S. (2025). Protein Language Models Capture Structural and Functional Epistasis in a Zero-Shot Setting. *bioRxiv*, 2025.09.14.676130.

6. Zheng, L., et al. (2025). A protein dynamics–based deep learning model enhances predictions of fitness and epistasis. *Proceedings of the National Academy of Sciences*, 122(42).

7. Dieckhaus, H., & Kuhlman, B. (2025). Protein stability models fail to capture epistatic interactions of double point mutations. *Protein Science*, 34(1), e70003.

8. Li, Z., & Luo, Y. (2025). Generalizable and scalable protein stability prediction with rewired protein generative models. *Nature Communications*, 16.

9. Rao, R., Liu, J., Verkuil, R., Meier, J., Canny, J., Abbeel, P., Sercu, T., & Rives, A. (2021). MSA Transformer. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139.

10. Wu, N.C., Dai, L., Olson, C.A., Lloyd-Smith, J.O., & Sun, R. (2016). Adaptation in protein fitness landscapes is facilitated by indirect paths. *eLife*, 5, e16965.

11. Tsui, D., & Aghazadeh, A. (2024). On Recovering Higher-order Interactions from Protein Language Models. *arXiv*, 2405.06645.

12. GraphESMStable. (2026). GraphESMStable: A Deep Learning Framework for Protein Stability Prediction Integrating Sequence and Structural Features with Epistasis Modeling. *bioRxiv*, 2026.01.08.698524.
