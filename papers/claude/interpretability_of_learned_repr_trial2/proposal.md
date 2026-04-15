# The Functional Anatomy of Sparse Features in Language Models

## Introduction

Sparse Autoencoders (SAEs) have emerged as the primary tool for decomposing neural network activations into interpretable features (Cunningham et al., 2024; Bricken et al., 2023). By training an overcomplete dictionary with a sparsity constraint, SAEs extract features that are often monosemantic—activating for a single, human-interpretable concept. This line of work has yielded remarkable insights, from identifying features for specific entities and concepts in Claude 3 Sonnet (Templeton et al., 2024) to enabling circuit tracing in production language models (Lindsey et al., 2025).

However, a critical question remains unanswered: **how do individual SAE features collectively organize to produce model capabilities?** Current evaluation frameworks assess SAE quality through global metrics—reconstruction loss, sparsity, and aggregate interpretability scores (Gao et al., 2024; Karvonen et al., 2025). Individual features are studied in isolation through automated interpretability pipelines (Bills et al., 2023; Eleuther AI, 2024). Neither approach reveals the *functional organization* of the feature space: which features support factual recall, which enable syntactic processing, and which underpin reasoning.

This gap matters for both scientific understanding and practical application. If a model capability is supported by a small number of highly specialized features (localized), it is amenable to feature-level interpretation and targeted intervention. If it requires the coordinated activation of hundreds of features (distributed), single-feature analysis will miss the forest for the trees, and the capability may live partly in the SAE's "dark matter"—the reconstruction residual that standard SAEs fail to capture (Sharkey & Beren, 2024).

**Key Insight:** We hypothesize that model capabilities exist on a spectrum from *localized* (supported by few, specialized features) to *distributed* (requiring many, broadly-shared features), and that this localization spectrum predicts when SAE-based interpretability succeeds or fails.

**Hypothesis:** For capabilities that are functionally localized in SAE feature space, individual features will be more interpretable, more causally faithful, and more stable across SAE training runs. For distributed capabilities, SAE features will be less interpretable, the reconstruction residual will contain more task-relevant information, and feature-level interventions will be less effective.

## Proposed Approach

### Overview

We propose a systematic framework for mapping sparse autoencoder features to model capabilities, creating what we call the **Functional Feature Atlas**—a structured representation of how the model's feature space supports different behavioral capabilities.

### Method Details

**Step 1: Capability Taxonomy and Evaluation Suites**

We define six core capability categories for language models, each with targeted evaluation datasets:

| Capability | Evaluation Task | Dataset/Source |
|---|---|---|
| Factual Knowledge | Cloze-style fact completion | LAMA (Petroni et al., 2019) |
| Syntactic Processing | Subject-verb agreement | BLiMP (Warstadt et al., 2020) |
| Semantic Understanding | Word sense disambiguation | WiC (Pilehvar & Camacho-Collados, 2019) |
| Sentiment Analysis | Sentiment classification | SST-2 (Socher et al., 2013) |
| Named Entity Recognition | Entity type prediction | Custom prompts from CoNLL |
| Simple Reasoning | Next-step inference | Custom logical completion tasks |

For each category, we curate 500-1000 evaluation examples where GPT-2 Small achieves non-trivial accuracy, ensuring the model actually uses the relevant capability.

**Step 2: Gradient-Based Feature Attribution**

For each capability and each SAE layer, we compute feature importance using gradient-weighted activation:

$$\text{importance}(f_i, x) = |a_i(x) \cdot \nabla_{a_i} \mathcal{L}(x)|$$

where $a_i(x)$ is the activation of SAE feature $i$ on input $x$, and $\nabla_{a_i} \mathcal{L}(x)$ is the gradient of the task-relevant loss with respect to that activation. This gives a per-example, per-feature importance score that can be aggregated across examples for each capability.

This approach is computationally efficient (one forward + backward pass per example) while providing a principled measure of feature importance that accounts for both activation magnitude and downstream sensitivity.

**Step 3: Targeted Causal Validation**

For the top-K features identified by gradient attribution for each capability, we perform targeted activation patching:
- **Zero ablation**: Set the feature activation to zero and measure capability-specific performance change
- **Mean ablation**: Set to the mean activation across a reference distribution
- **Activation amplification**: Scale the feature activation by 2x and measure the effect

This validates that gradient-based rankings reflect genuine causal relationships, not mere correlations.

**Step 4: Localization Analysis**

For each capability $c$ and layer $l$, we define the **Functional Localization Index (FLI)**:

$$\text{FLI}(c, l) = 1 - \frac{H(\mathbf{p}_c)}{H(\mathbf{u})}$$

where $\mathbf{p}_c$ is the normalized importance distribution over features for capability $c$, $H(\cdot)$ is Shannon entropy, and $\mathbf{u}$ is the uniform distribution. FLI = 1 means the capability is concentrated in a single feature; FLI = 0 means it is uniformly distributed across all features.

We analyze how FLI varies across capabilities and layers, testing whether some capabilities are inherently more localized than others.

**Step 5: Feature Sharing and Capability Overlap**

We construct a **capability overlap matrix** $O$ where:

$$O_{ij} = \frac{|\text{TopK}(c_i) \cap \text{TopK}(c_j)|}{K}$$

measuring the fraction of top-K features shared between capabilities $i$ and $j$. This reveals the modular vs. shared organization of the model's feature space.

**Step 6: Dark Matter Analysis**

For capabilities where SAE features explain limited variance, we analyze the reconstruction residual:
- Train linear probes on the SAE residual $r = x - \hat{x}$ for each capability
- Compare probe accuracy on the residual vs. on the SAE features
- Quantify how much task-relevant information is captured by features vs. lost in the residual

### Key Innovations

1. **Functional Localization Index (FLI)**: A new metric quantifying how concentrated a capability is in feature space, enabling principled comparison across capabilities, layers, and models.

2. **Capability overlap matrix**: A structured representation of feature sharing that reveals modular vs. shared functional organization.

3. **Residual capability analysis**: Direct measurement of how much capability-relevant information is lost in SAE reconstruction, connecting the "dark matter" literature to specific model behaviors.

4. **Feature Atlas visualization**: A comprehensive visualization mapping the feature space to capabilities, providing an intuitive representation of functional organization.

## Related Work

### Sparse Autoencoders for Interpretability

SAEs have become the dominant approach for extracting interpretable features from neural networks. Cunningham et al. (2024) demonstrated that SAE features are more interpretable than individual neurons, while Anthropic's Scaling Monosemanticity project (Templeton et al., 2024) showed SAEs can extract millions of meaningful features from production models. Several architectural variants have been proposed: TopK SAEs (Gao et al., 2024), JumpReLU SAEs (Rajamanoharan et al., 2024), and Switch SAEs using mixture-of-experts routing.

**How our work differs**: Prior SAE work evaluates features individually or through global metrics. We provide the first systematic mapping from features to capabilities, revealing the functional organization that individual feature analysis misses.

### SAE Evaluation and Limitations

SAEBench (Karvonen et al., 2025) provides the most comprehensive SAE evaluation to date, with eight diverse metrics across 200+ SAEs. A critical finding is that proxy metrics (reconstruction loss, sparsity) do not reliably predict downstream utility. The non-canonicality result (Leask et al., 2025) shows SAEs with different seeds learn different features, questioning whether any single SAE captures the "true" feature decomposition. Feature absorption (Chanin et al., 2025) demonstrates that hierarchical concepts break SAE sparsity assumptions.

**How our work differs**: We evaluate SAE faithfulness at the *capability level*, not globally. This reveals which capabilities are well-served by SAEs and which are not—information that global metrics cannot provide.

### SAE Dark Matter

Sharkey & Beren (2024) decomposed SAE reconstruction error into unlearned linear features, dense features, and nonlinear errors, showing that much "dark matter" is linearly predictable. Domain-specific SAEs (2025) reduce dark matter by specializing to narrower activation distributions.

**How our work differs**: We connect dark matter to specific capabilities, showing not just *how much* is missed but *what kinds of information* are lost, enabling targeted improvements.

### Mechanistic Interpretability and Circuits

Anthropic's circuit tracing (Lindsey et al., 2025) traces computation through cross-layer transcoders for individual inputs. The Open Problems paper (2025) identifies functional organization of features as an open question. The linear representation hypothesis (Park et al., 2024) characterizes the geometry of learned concepts.

**How our work differs**: Circuit tracing operates per-input; we aggregate across examples to characterize capability-level organization. This provides a complementary, higher-level view of model function.

### Feature Attribution Methods

Gradient-based attribution has a long history in interpretability (Sundararajan et al., 2017; Shrikumar et al., 2017). Activation patching (Vig et al., 2020; Meng et al., 2022) provides causal validation. Our contribution is applying these established tools systematically to SAE features across a comprehensive capability taxonomy, yielding novel findings about functional organization.

## Experiments

### Setup

- **Model**: GPT-2 Small (124M parameters, 12 layers, 768-dimensional residual stream)
- **SAEs**: Pre-trained SAEs from SAE Lens (Bloom et al., 2024) at each residual stream layer, with dictionary sizes of 16K and 32K
- **Hardware**: 1× NVIDIA RTX A6000 (48GB VRAM)
- **Evaluation**: 500-1000 examples per capability, 6 capabilities
- **Feature attribution**: Gradient × activation with targeted ablation validation

### Experiment 1: Feature-Capability Mapping

For each of the 6 capabilities across all 12 layers, compute gradient-based feature importance and identify top-100 features. Create layer × capability heatmaps showing where each capability is most concentrated.

**Expected result**: Different capabilities peak at different layers (syntactic features in early layers, semantic in middle, factual knowledge in later layers), consistent with prior probing work but now grounded in specific SAE features.

### Experiment 2: Functional Localization Index

Compute FLI for each capability at each layer. Compare across capabilities to determine which are localized vs. distributed.

**Expected result**: Factual knowledge and named entities are highly localized (few specific features), while syntactic processing and reasoning are more distributed.

### Experiment 3: Causal Validation

For top-50 features per capability (at the peak layer), perform zero-ablation and measure capability-specific performance drop. Compare to ablating random features of equal number.

**Expected result**: Ablating top-50 features causes >50% of the total capability degradation for localized capabilities, but <20% for distributed capabilities.

### Experiment 4: Capability Overlap Matrix

Compute pairwise overlap of top-100 features for all capability pairs. Visualize as a heatmap and perform clustering.

**Expected result**: Semantically related capabilities (factual knowledge ↔ NER, syntax ↔ reasoning) share more features than unrelated ones, revealing a meaningful modular structure.

### Experiment 5: Dark Matter per Capability

Train linear probes on SAE features and on SAE residuals for each capability. Compare probe accuracy to measure how much capability-relevant information is in the features vs. the residual.

**Expected result**: For localized capabilities, >80% of probe accuracy comes from features; for distributed capabilities, >30% comes from the residual, confirming that distribution correlates with dark matter dependence.

### Experiment 6: Comparison Across SAE Architectures

Repeat the analysis with two SAE architectures (ReLU with L1, TopK) and two dictionary sizes (16K, 32K). Measure whether FLI scores are stable across these choices.

**Expected result**: Localization patterns are largely consistent across architectures, confirming they reflect model properties rather than SAE artifacts. Wider SAEs may improve coverage of distributed capabilities.

### Experiment 7: Localization Predicts Interpretability

For features supporting localized vs. distributed capabilities, compare their automated interpretability scores (using EleutherAI's autointerp pipeline or LLM-based description accuracy).

**Expected result**: Features supporting localized capabilities have higher interpretability scores, providing a mechanistic explanation for why some SAE features are more interpretable than others.

## Success Criteria

### Confirming the hypothesis

The hypothesis is confirmed if:
1. FLI varies significantly across capabilities (at least 2× range between most and least localized)
2. Localization predicts causal faithfulness: ablating top features for localized capabilities causes >3× more performance degradation than for distributed capabilities
3. Localization predicts dark matter dependence: residual probes are >2× more accurate for distributed capabilities
4. These patterns are consistent across at least 2 SAE architectures

### Refuting the hypothesis

The hypothesis is refuted if:
1. All capabilities have similar FLI scores (no localized/distributed distinction)
2. There is no correlation between FLI and causal faithfulness of top features
3. Residual information content is independent of capability type

### Minimum viable contribution

Even if the localization hypothesis is partially refuted, the feature-capability mapping itself is a novel contribution: the Functional Feature Atlas, the capability overlap matrix, and the per-capability SAE evaluation framework are all new tools for the interpretability community.

## References

1. Bills, S., et al. (2023). Language models can explain neurons in language models. OpenAI Blog.
2. Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread.
3. Chanin, D., Wilken-Smith, T., et al. (2025). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. NeurIPS 2025.
4. Cunningham, H., et al. (2024). Sparse Autoencoders Find Highly Interpretable Features in Language Models. ICLR 2024.
5. Gao, L., Dupré la Tour, T., et al. (2024). Scaling and Evaluating Sparse Autoencoders. OpenAI Technical Report.
6. Karvonen, A., et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. ICML 2025.
7. Leask, T., Bussmann, B., et al. (2025). Sparse Autoencoders Do Not Find Canonical Units of Analysis. ICLR 2025.
8. Lindsey, J., et al. (2025). Circuit Tracing: Revealing Computational Graphs in Language Models. Anthropic.
9. Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022.
10. Mueller, M., et al. (2025). MIB: A Mechanistic Interpretability Benchmark. ICML 2025.
11. Open Problems in Mechanistic Interpretability. (2025). arXiv:2501.16496.
12. Park, K., Choe, Y. J., Veitch, V. (2024). The Linear Representation Hypothesis and the Geometry of Large Language Models. ICML 2024.
13. Petroni, F., et al. (2019). Language Models as Knowledge Bases? EMNLP 2019.
14. Rajamanoharan, S., et al. (2024). Improving Dictionary Learning with Gated Sparse Autoencoders. arXiv:2404.16014.
15. Sharkey, L. & Beren, M. (2024). Decomposing The Dark Matter of Sparse Autoencoders. arXiv:2410.14670.
16. Sundararajan, M., et al. (2017). Axiomatic Attribution for Deep Networks. ICML 2017.
17. Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Anthropic.
18. Warstadt, A., et al. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English. TACL.
