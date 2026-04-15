# Faithful by Consensus: Identifying Causally Important Features Through Multi-Seed Sparse Autoencoder Agreement

## Introduction

Sparse autoencoders (SAEs) have emerged as the dominant tool for decomposing neural network representations into interpretable features (Cunningham et al., 2023; Bricken et al., 2023). By mapping dense, polysemantic activations into high-dimensional sparse codes, SAEs promise to reveal the fundamental units of computation within language models. However, a critical challenge undermines this promise: SAEs trained on the same model and data with different random seeds discover substantially different feature sets, with only ~30% of features shared across runs (Paulo & Belrose, 2025). This instability raises a fundamental question: *which SAE features, if any, reflect the model's true computational structure?*

We propose that **multi-seed agreement provides a powerful, free signal for identifying causally faithful features**. Our core hypothesis is that features consistently recovered across independently trained SAEs — which we term *consensus features* — are not merely artifacts of stable optimization landscapes, but are causally important for model behavior. Conversely, features found by only a single SAE run (*singleton features*) are more likely to be artifacts of the decomposition process itself, such as arbitrary tilings of feature manifolds (Lecomte et al., 2025).

**Key Insight:** While Paulo & Belrose (2025) established that SAE features are unstable and showed that shared features tend to be more interpretable, they did not investigate whether this stability signal predicts *causal importance* — the degree to which a feature actually matters for model predictions. We bridge this gap by systematically measuring the causal importance of consensus vs. singleton features through ablation experiments, sparse probing, and feature steering, establishing multi-seed consensus as both a theoretical indicator and practical tool for SAE feature curation.

**Hypothesis:** Consensus features (recovered in ≥80% of independent SAE training runs) are significantly more causally important for model predictions than singleton features (recovered in ≤1 run), as measured by KL divergence under feature ablation, sparse probing accuracy, and steering effectiveness. Furthermore, a curated "consensus dictionary" containing only consensus features provides better per-feature utility than the full dictionary from any single SAE run.

## Proposed Approach

### Overview

We train multiple SAEs (N=10 per layer) with different random seeds on the same model and data, match features across runs, assign each feature a *consensus score* (the fraction of runs recovering it), and then evaluate features at different consensus levels on multiple faithfulness metrics.

### Method Details

**Step 1: Multi-Seed SAE Training.** We train N=10 SAEs independently at each of K=3 layers (early, middle, late) of Pythia-160M using the SAELens library (Bloom, 2024). Each SAE uses the same architecture (TopK with k=50), dictionary size (d=16,384), and training data (100M tokens from The Pile), varying only the random seed. We also replicate key experiments with ReLU+L1 SAEs to assess architecture sensitivity, following the observation that ReLU SAEs show greater seed stability (Paulo & Belrose, 2025).

**Step 2: Cross-Seed Feature Matching.** For each pair of SAE runs (i, j) on the same layer, we compute cosine similarity between all decoder weight vectors. We use the Hungarian algorithm to find the optimal bijective matching that maximizes total similarity, following Paulo & Belrose (2025). A feature pair is considered "matched" if cosine similarity exceeds 0.7 in the decoder space. We then build a bipartite matching graph across all 10 runs and identify connected components — each component represents a "feature identity" that may appear in 1 to 10 runs.

**Step 3: Consensus Scoring.** Each feature identity receives a consensus score c ∈ {0.1, 0.2, ..., 1.0}, equal to the fraction of SAE runs in which it was recovered. We partition features into three tiers:
- **Consensus features** (c ≥ 0.8): found in ≥8 of 10 runs
- **Partial features** (0.3 ≤ c < 0.8): found in some but not most runs
- **Singleton features** (c ≤ 0.2): found in ≤2 runs

**Step 4: Causal Importance Evaluation.** For each consensus tier, we measure:

1. **Feature Ablation (Primary metric):** For each feature f, we zero its activation on a held-out evaluation set and measure the KL divergence between the original and ablated model output distributions. Higher KL divergence indicates greater causal importance. We use a diverse evaluation set of 10K sequences across multiple domains.

2. **Sparse Probing:** Following SAEBench (Karvonen et al., 2025), we train sparse linear probes on subsets of features from each consensus tier to predict known linguistic properties (part-of-speech, named entity type, sentiment). We measure whether consensus features achieve higher probing accuracy per feature.

3. **Feature Steering:** We test whether consensus features produce more coherent behavioral changes when amplified or suppressed. We measure steering effectiveness by the magnitude and consistency of output distribution shifts on targeted evaluation prompts.

4. **Auto-Interpretability:** We use an LLM-based automated interpretability pipeline (following Bills et al., 2023) to score features on interpretability. We compare scores across consensus tiers and test whether consensus score predicts interpretability beyond what firing frequency alone explains.

**Step 5: Consensus Dictionary Construction.** We construct a "consensus dictionary" by selecting, for each consensus feature identity, the representative feature vector closest to the centroid of its matched instances across runs. We evaluate this consensus dictionary against individual SAE dictionaries on SAEBench metrics (sparse probing, feature absorption, reconstruction quality) to test whether curation via consensus improves practical utility.

**Step 6: Connecting Instability to Feature Manifolds.** We analyze what makes singleton features different from consensus features. We hypothesize that singletons correspond to manifold tilings — arbitrary discretizations of continuous feature manifolds (Lecomte et al., 2025). We test this by measuring: (a) the local density of singleton vs. consensus features in decoder weight space, (b) whether singletons form clusters that tile continuous regions, and (c) whether singletons have higher activation correlation with nearby singletons (indicating they subdivide a manifold).

### Key Innovations

1. **Consensus score as feature quality metric:** A simple, training-free signal that requires only running the matching algorithm across existing SAE runs.
2. **Causal validation of stability:** Moving beyond the observation that stable features are more interpretable to showing they are more causally important.
3. **Consensus dictionaries:** A practical method for curating higher-quality feature dictionaries by leveraging multi-seed agreement.
4. **Instability-manifold connection:** Providing empirical evidence that feature instability arises from manifold tiling, connecting two important research threads.

## Related Work

### Sparse Autoencoders for Interpretability
SAEs were introduced for language model interpretability by Cunningham et al. (2023) and scaled by Bricken et al. (2023). Gao et al. (2024) introduced k-sparse autoencoders with clean scaling laws. Recent architectural variants include Matryoshka SAEs (Bussmann et al., 2025) for hierarchical features, crosscoders for cross-layer features (Lieberum et al., 2024), and transcoders (Dunefsky et al., 2025). Our work is orthogonal to architecture improvements — consensus scoring can be applied to any SAE variant.

### SAE Feature Stability
Paulo & Belrose (2025) demonstrated that only ~30% of SAE features are shared across random seeds, with TopK SAEs being more seed-dependent than ReLU SAEs. They showed shared features fire more frequently and have higher auto-interpretability scores. Our work extends this by (a) connecting stability to causal importance, (b) proposing consensus scoring as a practical metric, and (c) analyzing the source of instability through the manifold lens.

### SAE Evaluation
SAEBench (Karvonen et al., 2025) provides comprehensive SAE evaluation across 8 metrics. Chanin et al. (2024) identified feature absorption as a failure mode. SynthSAEBench (2025) uses synthetic ground truth. Our consensus metric complements these by providing a feature-level quality signal rather than an SAE-level benchmark.

### Feature Geometry and Manifolds
Li et al. (2025) revealed multi-scale geometric structure in SAE features (crystals, lobes, power-law eigenvalues). Lecomte et al. (2025) showed SAEs tile feature manifolds with increasing dictionary size. Our work connects manifold tiling to instability: we hypothesize that manifold tiles are precisely the features that fail to reproduce across seeds.

### Causal Methods in Interpretability
Activation patching (Zhang & Nanda, 2024) and sparse feature circuits (Marks et al., 2025) provide causal evaluation of model components. We adapt feature ablation techniques to compare causal importance across consensus tiers, providing a new lens on feature quality.

### Feature Universality
Lan et al. (2025) showed SAE features exhibit universality across different models. Universal SAEs (Moayeri et al., 2026) learn shared concept spaces. Our work studies a complementary question: universality *within* a single model across SAE training runs, connecting to the question of what the model's "true" features are.

## Experiments

### Setup
- **Model:** Pythia-160M (EleutherAI), chosen for fast SAE training while being large enough for meaningful features
- **Layers:** 3 representative layers — Layer 2 (early), Layer 6 (middle), Layer 10 (late) — out of 12 total
- **SAE Architecture:** TopK (k=50) and ReLU+L1, dictionary size 16,384 (expansion factor 16×)
- **Training Data:** 100M tokens from The Pile (deduplicated)
- **Seeds:** 10 independent training runs per layer per architecture
- **Evaluation Data:** 10K held-out sequences from The Pile
- **Hardware:** 1× NVIDIA RTX A6000 (48GB VRAM)
- **Estimated Time:** ~7 hours total (3h SAE training, 1.5h matching, 2.5h evaluation)

### Experiments and Expected Results

**Experiment 1: Consensus Score Distribution.**
Characterize the distribution of consensus scores across layers and architectures. Expected: bimodal distribution with peaks near 0.1 (singletons) and 0.9-1.0 (consensus), consistent with Paulo & Belrose (2025).

**Experiment 2: Consensus Predicts Causal Importance.**
Primary experiment. For each feature, measure KL divergence under ablation. Plot mean KL divergence vs. consensus score. Expected: strong positive correlation (Spearman ρ > 0.4), with consensus features having 2-5× higher mean causal importance than singletons.

**Experiment 3: Consensus Predicts Interpretability.**
Replicate Paulo & Belrose's interpretability finding and extend with controls. Expected: consensus features score higher on auto-interpretability even after controlling for firing frequency.

**Experiment 4: Sparse Probing with Consensus Features.**
Train k-sparse probes (k=5,10,20) using features from each consensus tier. Expected: probes using consensus features achieve higher accuracy per feature than probes using singletons or random features.

**Experiment 5: Feature Steering Effectiveness.**
For 50 consensus and 50 singleton features with comparable firing rates, amplify activation by 3× and measure output distribution shift. Expected: consensus features produce larger, more consistent behavioral changes.

**Experiment 6: Consensus Dictionary Evaluation.**
Build a consensus dictionary and evaluate on SAEBench metrics against individual SAE runs. Expected: consensus dictionary achieves comparable or better reconstruction quality with fewer features, and substantially better sparse probing and feature absorption scores.

**Experiment 7: Singleton Analysis — Manifold Tiling.**
Analyze whether singleton features cluster in decoder weight space, form continuous manifold tiles, and have high mutual activation correlation. Expected: singletons form denser clusters than consensus features, supporting the manifold tiling hypothesis.

**Ablation Studies:**
- Vary number of seeds (N=3,5,8,10) to assess minimum seeds needed
- Vary dictionary size (8K, 16K, 32K) to test robustness
- Compare TopK vs ReLU+L1 stability patterns
- Vary consensus threshold (0.5, 0.7, 0.8, 0.9)

### Metrics
- KL divergence under feature ablation (causal importance)
- Sparse probing accuracy (downstream utility)
- Steering effect size (behavioral relevance)
- Auto-interpretability score (human-alignment)
- Feature matching rate (stability)
- Reconstruction loss (SAE quality)

## Success Criteria

**Primary (must show):**
1. Consensus score significantly predicts causal importance (p < 0.01, effect size Cohen's d > 0.5)
2. Consensus features achieve higher per-feature sparse probing accuracy than singletons

**Secondary (would strengthen):**
3. Consensus dictionaries match or exceed single-run SAE quality on SAEBench metrics
4. Singleton features show manifold tiling signatures (clustering, high mutual correlation)
5. Results hold across both TopK and ReLU+L1 architectures

**Would refute hypothesis:**
- No significant correlation between consensus score and causal importance
- Singleton features are equally or more causally important than consensus features
- Consensus dictionaries perform worse than individual SAE dictionaries

## References

1. Paulo, G. & Belrose, N. (2025). Sparse Autoencoders Trained on the Same Data Learn Different Features. arXiv:2501.16615.

2. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600. ICLR 2024.

3. Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J.E., Hume, T., Carter, S., Henighan, T., & Olah, C. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread.

4. Gao, L., Dupré la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). Scaling and Evaluating Sparse Autoencoders. arXiv:2406.04093. ICLR 2025.

5. Marks, S., Rager, C., Michaud, E.J., Belinkov, Y., Bau, D., & Mueller, A. (2024). Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models. arXiv:2403.19647. ICLR 2025.

6. Karvonen, A., Rager, C., Lin, J., Tigges, C., Bloom, J., Chanin, D., Lau, Y.-T., Farrell, E., McDougall, C., Ayonrinde, K., Till, D., Wearden, M., Conmy, A., Marks, S., & Nanda, N. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. ICML 2025.

7. Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., & Bloom, J. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507. NeurIPS 2025.

8. Li, Y., Michaud, E.J., Baek, D.D., Engels, J., Sun, X., & Tegmark, M. (2024). The Geometry of Concepts: Sparse Autoencoder Feature Structure. Entropy, 27(4), 344, 2025.

9. Zhang, F. & Nanda, N. (2023). Towards Best Practices of Activation Patching in Language Models: Metrics and Methods. arXiv:2309.16042. ICLR 2024.

10. Lecomte, V., et al. (2025). Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds. arXiv:2509.02565.

11. Cho, S., Oh, H., Lee, D., Vieira, L.R., Bermingham, A., & El Sayed, Z. (2025). FaithfulSAE: Towards Capturing Faithful Features with Sparse Autoencoders without External Dataset Dependencies. ACL 2025 SRW.

12. Bloom, J. (2024). SAELens: Training Sparse Autoencoders on Language Models. GitHub.

13. Lieberum, T., et al. (2024). Sparse Crosscoders for Cross-Layer Features and Model Diffing. Transformer Circuits Thread.

14. Lan, M., et al. (2025). Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models. arXiv:2410.06981.

15. Dunefsky, J., et al. (2025). Transcoders Beat Sparse Autoencoders for Interpretability. arXiv:2501.18823.

16. Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., Goh, G., Sutskever, I., Leike, J., Wu, J., & Saunders, W. (2023). Language Models Can Explain Neurons in Language Models. OpenAI Blog.

17. Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M.A., Purohit, S., Prashanth, U.S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML 2023.
