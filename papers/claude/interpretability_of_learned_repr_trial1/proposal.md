# The Convergent Core: Connecting Seed Stability, Cross-Model Universality, and Causal Importance of Sparse Autoencoder Features

## Introduction

Sparse autoencoders (SAEs) have become the dominant tool for extracting interpretable features from the internal representations of large language models (LLMs). By learning overcomplete dictionaries that decompose dense, polysemantic activations into sparse, monosemantic features, SAEs promise to reveal the fundamental units of computation within neural networks (Cunningham et al., 2023; Bricken et al., 2023). This approach has yielded impressive results: SAE features have been used for circuit discovery (Marks et al., 2025), model steering (Templeton et al., 2024), and mechanistic understanding of model behaviors (Anthropic, 2025).

However, a critical and underexplored tension has emerged in the field. On one hand, Paulo & Belrose (2025) demonstrated that SAEs trained on the same model and data with different random seeds learn substantially different features—only ~30% of features are shared across seeds for a 131K-latent SAE on Llama 3 8B. On the other hand, Lan et al. (2024) and Thasarathan et al. (2025) have shown that SAE feature *spaces* exhibit remarkable universality across entirely different model architectures. This presents a paradox: **how can individual features be unreliable across random seeds, yet the feature spaces they span be universal across models?**

We hypothesize that SAE feature dictionaries contain two distinct populations: a **convergent core** of features that are reliably discovered regardless of initialization (seed-stable) and correspond to genuine computational primitives shared across models (universal), and a **variable periphery** of features that represent arbitrary decompositions of a complementary subspace—individually unreliable, but collectively meaningful. We further hypothesize that convergent core features are more causally important for model behavior than peripheral features.

This paper presents the first systematic study connecting three dimensions of SAE feature quality—**seed stability**, **cross-model universality**, and **causal importance**—on the same set of features. Our findings provide a principled, computation-free method for identifying which SAE features are "real" and which are artifacts of the optimization landscape, with immediate practical implications for mechanistic interpretability, model steering, and circuit discovery.

## Proposed Approach

### Overview

We propose a three-stage experimental framework:

1. **Multi-seed SAE training**: Train SAEs with multiple random seeds on several small-to-medium LLMs to identify seed-stable vs. seed-variable features.
2. **Cross-model alignment**: Align SAE feature dictionaries across models to measure which features are universal vs. model-specific.
3. **Causal importance assessment**: Measure the functional impact of ablating seed-stable vs. seed-variable features on model behavior.

### Method Details

#### Stage 1: Seed Stability Analysis

For each model $M$ in our suite, we train $K=5$ SAEs with different random seeds on the same layer's residual stream activations. We use TopK SAEs (Gao et al., 2024) with dictionary size $D=16{,}384$, as TopK has been shown to exhibit greater seed sensitivity than ReLU SAEs (Paulo & Belrose, 2025), making it a more informative test case.

To measure seed stability, we compute pairwise feature alignment across all $\binom{K}{2}=10$ seed pairs using:
- **Decoder cosine similarity**: For each feature $f_i$ in SAE$_a$, find its best match in SAE$_b$ via $\max_j \cos(d_i^a, d_j^b)$ where $d$ denotes decoder vectors.
- **Activation correlation**: For matched features, compute Pearson correlation of activation patterns over a held-out dataset.
- **Hungarian matching**: Use the Hungarian algorithm for optimal 1-1 matching to avoid double-counting.

A feature is classified as **core** if its best-match cosine similarity exceeds threshold $\tau$ in at least $M_{\min}$ of $K$ seeds. We sweep $\tau \in \{0.7, 0.8, 0.9\}$ and $M_{\min} \in \{3, 4, 5\}$ to produce a continuous "convergence score" for each feature.

#### Stage 2: Cross-Model Universality Measurement

We train SAEs on corresponding layers of different models (e.g., middle layer of GPT-2 Small, Pythia-160M, Pythia-410M). To align features across models with different embedding dimensions, we:

1. Use the activation correlation method from Lan et al. (2024): run both models on the same dataset, collect SAE activations, and compute correlation between features across models.
2. Compute a cross-model alignment score for each feature using the mean of its top-1 match scores across model pairs.

We then test the central hypothesis: **seed-stable features have higher cross-model alignment scores than seed-variable features.** This is evaluated via rank correlation (Spearman's $\rho$) between each feature's convergence score and its cross-model alignment score.

#### Stage 3: Causal Importance Assessment

For each feature, we measure causal importance via:

1. **Zero-ablation**: Set the feature's activation to zero during a forward pass and measure the KL divergence between the original and perturbed output distributions, averaged over a diverse evaluation set.
2. **Downstream task impact**: Measure accuracy changes on targeted tasks (e.g., indirect object identification, greater-than comparisons, subject-verb agreement) when ablating individual features.

We compare the causal importance distributions of core vs. peripheral features and test whether convergence score predicts causal importance.

#### Stage 4: Subspace Analysis

To understand the variable periphery, we analyze whether:
- The *union* of peripheral features across seeds spans a consistent subspace (measured via principal angle analysis between peripheral subspaces from different seeds).
- This peripheral subspace carries information not captured by the convergent core (measured via reconstruction quality using only core features vs. core + peripheral features).

### Key Innovations

1. **Joint analysis across three axes**: No prior work has simultaneously measured seed stability, cross-model universality, and causal importance on the same features.
2. **Convergence score as a quality metric**: We propose that a feature's seed convergence score serves as a cheap, model-free proxy for feature quality—no need for expensive automated interpretability scoring or causal interventions.
3. **Periphery analysis**: Rather than dismissing unstable features as noise, we characterize the complementary subspace they collectively span.

## Related Work

### Sparse Autoencoders for Interpretability

SAEs were introduced for LLM interpretability by Cunningham et al. (2023) and Bricken et al. (2023), who demonstrated that sparse dictionary learning can decompose polysemantic neuron activations into monosemantic features. Gao et al. (2024) scaled SAEs to GPT-4 with 16M latents and introduced TopK SAEs with clean scaling laws. Templeton et al. (2024) demonstrated that SAE features can be used for model steering in Claude 3 Sonnet. Our work builds directly on this foundation but asks *which* of these features are reliable.

### SAE Reproducibility

Paulo & Belrose (2025) first demonstrated that SAEs trained with different seeds learn different features, with only ~30% overlap. Gorton & Crook (2026) showed that L2 weight regularization improves seed consistency and doubles steering success rates. Marks et al. (2024) proposed Mutual Feature Regularization (MFR), training multiple SAEs jointly to encourage shared features. Our work differs fundamentally: rather than modifying SAE training to improve stability (an engineering contribution), we **exploit the natural variation across seeds as a diagnostic signal** to identify which features correspond to genuine model representations (a scientific contribution).

### Feature Universality

Lan et al. (2024) showed that SAE feature *spaces* are similar across LLMs under rotation-invariant transformations, and that semantic subspaces (e.g., calendar tokens) show especially high cross-model similarity. Thasarathan et al. (2025) introduced Universal SAEs (USAEs) that jointly learn a shared concept space across multiple vision models. Our work connects universality to seed stability—testing whether the features that are universal across models are the same ones that are stable across seeds.

### Feature Quality and Evaluation

SAEBench (Karvonen et al., 2025) provides comprehensive benchmarks across interpretability, reconstruction, and disentanglement. Chanin et al. (2025) identified feature splitting and absorption as fundamental failure modes. Bussmann et al. (2025) introduced Matryoshka SAEs for hierarchical feature organization. Paulo et al. (2024) developed automated interpretability scoring including intervention-based metrics. Our convergence score complements these evaluation approaches by providing a training-free quality signal.

### Nonlinear and Multi-Dimensional Features

Engels et al. (2025) demonstrated that some language model features are irreducibly multi-dimensional (e.g., circular representations of days/months), challenging the linear feature assumption underlying standard SAEs. This suggests that some seed-variable features may represent different 1D projections of the same multi-dimensional feature—a hypothesis our subspace analysis can test.

## Experiments

### Models and Data

- **Models**: GPT-2 Small (85M params, 12 layers), Pythia-160M (12 layers), Pythia-410M (24 layers)
- **Layers**: Middle residual stream (layer 6 for GPT-2/Pythia-160M, layer 12 for Pythia-410M)
- **Training data**: OpenWebText (for GPT-2 SAEs), The Pile (for Pythia SAEs), 100M tokens each
- **Evaluation data**: 10M tokens held out from training distribution

### SAE Configuration

- **Architecture**: TopK SAEs (Gao et al., 2024)
- **Dictionary size**: 16,384 (expansion factor 16x for GPT-2 Small with d_model=768)
- **Sparsity**: k=64 (following standard practice)
- **Training**: Adam optimizer, batch size 4096, ~50K steps per SAE
- **Seeds**: 5 seeds per model-layer combination (15 SAEs total)

### Evaluation Metrics

1. **Seed stability**: Pairwise decoder cosine similarity, activation correlation, convergence score
2. **Cross-model alignment**: Feature-level correlation scores across model pairs
3. **Causal importance**: Zero-ablation KL divergence, downstream task accuracy change
4. **Interpretability** (supplementary): Automated interpretability scores using the Paulo et al. (2024) pipeline
5. **Subspace analysis**: Principal angles between peripheral subspaces, reconstruction loss comparison

### Benchmarks and Tasks

For causal importance evaluation:
- **Indirect Object Identification (IOI)**: "When Mary and John went to the store, John gave a drink to ___"
- **Greater-Than**: "The war lasted from 1723 to 17___"
- **Subject-Verb Agreement**: "The keys to the cabinet ___"
- **General language modeling**: Perplexity on held-out text

### Baselines

1. **Random partition**: Randomly split features into "core" and "peripheral" sets of the same sizes
2. **Activation frequency**: Sort features by how often they activate (high-frequency = "core")
3. **Decoder norm**: Sort features by L2 norm of decoder vectors
4. **Automated interpretability score**: Use autointerp scores as a feature quality baseline

### Expected Results

1. **Core features are universal**: Seed convergence score correlates positively with cross-model alignment score ($\rho > 0.3$).
2. **Core features are more important**: The top-k core features have higher mean causal importance (KL divergence) than top-k peripheral features.
3. **Convergence score predicts quality**: Convergence score correlates with automated interpretability scores and causal importance better than simpler baselines (activation frequency, decoder norm).
4. **The periphery spans a consistent subspace**: Principal angles between peripheral subspaces from different seeds are smaller than expected by chance, indicating that the periphery represents a genuine complementary subspace—even though individual features within it are unreliable.
5. **Core alone is insufficient**: Using only core features for reconstruction produces measurably worse language modeling performance than core + peripheral, confirming the periphery carries real information.

### Computational Budget

| Component | Estimated Time | GPU Memory |
|-----------|---------------|------------|
| SAE training (15 SAEs × ~20 min each) | ~5 hours | ~8 GB per SAE |
| Feature matching & alignment analysis | ~30 min | ~16 GB |
| Causal ablation studies | ~1.5 hours | ~12 GB |
| Autointerp scoring (subset of features) | ~45 min | ~8 GB |
| **Total** | **~8 hours** | **< 48 GB peak** |

## Success Criteria

### Primary (must achieve for paper to succeed)

1. **Statistically significant correlation** between seed stability and cross-model universality (Spearman's $\rho > 0.2$, $p < 0.01$).
2. **Core features are more causally important**: Mean KL divergence of core features > mean KL divergence of peripheral features, with effect size (Cohen's $d$) > 0.3.
3. **Convergence score outperforms random baseline**: As a feature quality predictor, convergence score achieves higher AUC-ROC than random partition on at least 2 of 3 models.

### Secondary (strengthens the paper)

4. Convergence score correlates with automated interpretability scores.
5. Peripheral subspaces show above-chance consistency across seeds.
6. Results replicate across TopK and ReLU SAE architectures.

### Negative results that would still be publishable

- If seed stability does **not** predict universality, this would challenge the assumption that "real features" should be stable across both seeds and models, suggesting the two notions of feature quality are orthogonal—itself an important finding.
- If core features are not more causally important, this would suggest that SAE seed variability is a benign phenomenon (different but equally valid decompositions), rather than a signal-vs-noise distinction.

## References

1. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. *arXiv:2309.08600*.

2. Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., ... & Olah, C. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Transformer Circuits Thread*.

3. Paulo, G. & Belrose, N. (2025). Sparse Autoencoders Trained on the Same Data Learn Different Features. *arXiv:2501.16615*.

4. Gorton, P. & Crook, O.M. (2026). Stable and Steerable Sparse Autoencoders with Weight Regularization. *arXiv:2603.04198*.

5. Marks, L., Paren, A., Krueger, D., & Barez, F. (2024). Enhancing Neural Network Interpretability with Feature-Aligned Sparse Autoencoders. *arXiv:2411.01220*.

6. Lan, M., Torr, P., Meek, A., Khakzar, A., Krueger, D., & Barez, F. (2024). Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models. *arXiv:2410.06981*.

7. Thasarathan, H., Forsyth, J., Fel, T., Kowal, M., & Derpanis, K.G. (2025). Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment. *arXiv:2502.03714*.

8. Gao, L., Dupré la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). Scaling and Evaluating Sparse Autoencoders. *arXiv:2406.04093*.

9. Karvonen, A., Rager, C., Lin, J., Tigges, C., Bloom, J., Chanin, D., ... & Nanda, N. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. *ICML 2025*.

10. Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., & Bloom, J. (2025). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. *NeurIPS 2025 (Oral)*.

11. Bussmann, B., Nabeshima, N., Karvonen, A., & Nanda, N. (2025). Learning Multi-Level Features with Matryoshka Sparse Autoencoders. *arXiv:2503.17547*.

12. Paulo, G., Mallen, A., Juang, C., & Belrose, N. (2024). Automatically Interpreting Millions of Features in Large Language Models. *arXiv:2410.13928*.

13. Engels, J., Michaud, E.J., Liao, I., Gurnee, W., & Tegmark, M. (2025). Not All Language Model Features Are One-Dimensionally Linear. *ICLR 2025*.

14. Biderman, S., Schoelkopf, H., Anthony, Q., et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. *ICML 2023*.

15. Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., ... & Olah, C. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread*.

16. Chanin, D. & Garriga-Alonso, A. (2026). SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data. *arXiv:2602.14687*.

17. Marks, S., Rager, C., Michaud, E.J., Belinkov, Y., Bau, D., & Mueller, A. (2025). Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models. *ICLR 2025*.
