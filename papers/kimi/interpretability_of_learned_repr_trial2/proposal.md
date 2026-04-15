# Research Proposal: Measuring and Mitigating the Causal-Semantic Disconnect in Sparse Autoencoders

## 1. Introduction

### 1.1 Background and Motivation

Sparse Autoencoders (SAEs) have emerged as the dominant approach for interpreting neural network representations, promising to decompose polysemantic activations into interpretable, monosemantic features. By learning an overcomplete dictionary with sparsity constraints, SAEs aim to recover the "true" features that neural networks use internally, providing a window into the black box of deep learning models.

However, a growing body of evidence suggests that current SAE evaluation metrics—primarily reconstruction loss and L0 sparsity—fail to capture what we actually care about: whether the discovered features are **causally important** for model behavior. Recent work has identified a troubling "causal-semantic disconnect":

1. **High reconstruction does not imply feature recovery**: He et al. (2025) show that SAEs can achieve 71% explained variance while recovering only 9% of ground-truth features in synthetic settings.

2. **Interpretable features are not necessarily causal**: Multiple studies (Chaudhary & Geiger, 2024; Minegishi et al., 2025) find that many SAE features, even those that appear semantically meaningful, have minimal causal effect on model outputs when intervened upon.

3. **Steering with SAE features causes unintended side effects**: Raedler et al. (2025), Anthropic (2024), and Chalnev et al. (2024) demonstrate that steering with SAE features often achieves target behaviors through unintended mechanisms, degrading model capabilities in unpredictable ways.

This disconnect undermines the core promise of mechanistic interpretability: if we cannot distinguish between features that merely correlate with concepts and those that actually drive behavior, our ability to understand and safely control AI systems is severely limited.

### 1.2 Key Insight and Hypothesis

Our key insight is that while recent work has made significant progress in identifying causally-relevant features for steering (Arad et al., 2025), existing metrics capture only partial aspects of causal importance. Arad et al.'s "output score" measures how much a feature affects the output distribution when intervened upon—similar to a sufficiency criterion. However, a comprehensive measure of causal influence should also consider **necessity** (whether the behavior fails without the feature) and **consistency** (whether the effect is stable across contexts).

We propose the **Intervention Fidelity Score (IFS)**, which combines these three components to better quantify how reliably a feature influences model behavior. Features with high intervention fidelity are those that truly drive model behavior, while low-fidelity features are merely correlational or epiphenomenal.

**Primary Hypothesis**: Ranking SAE features by IFS, which extends prior work by incorporating necessity and consistency alongside sufficiency, will enable more precise behavioral steering with significantly fewer unintended side effects compared to both activation-based selection and output-score-based selection alone.

**Secondary Hypothesis**: The correlation between traditional SAE quality metrics (reconstruction, sparsity) and causal importance remains weak even when using more comprehensive causal metrics, suggesting fundamental misalignment in current SAE training objectives.

### 1.3 Research Questions

1. How does IFS differ from existing causal feature selection metrics like Arad et al.'s output scores in identifying steering-relevant features?

2. How strongly do existing SAE quality metrics correlate with IFS compared to simpler causal metrics?

3. Does incorporating necessity and consistency components improve steering precision and reduce side effects beyond sufficiency-only selection?

4. What properties distinguish high-IFS from low-IFS features, and can these insights inform better SAE training objectives?

---

## 2. Proposed Approach

### 2.1 Overview

We propose a systematic framework for measuring and mitigating the causal-semantic disconnect in SAEs. Our approach consists of three main components:

1. **Intervention Fidelity Scoring**: A method to efficiently estimate how reliably each SAE feature influences model behavior when activated, extending prior work by incorporating necessity and consistency alongside sufficiency.

2. **Correlation Analysis**: A comprehensive study examining the relationship between IFS and existing SAE metrics, including comparisons to output-score-based selection.

3. **Fidelity-Guided Steering**: A steering methodology that selects features based on IFS rather than activation magnitude or output scores alone.

### 2.2 Method Details

#### 2.2.1 Intervention Fidelity Measurement

For each SAE feature $f$, we measure its intervention fidelity through a combination of three components:

**Necessity Score**: The drop in target behavior probability when feature $f$ is ablated:
$$\text{Nec}(f) = \mathbb{E}_{x \sim \mathcal{D}}[P(y_{\text{target}}|x) - P(y_{\text{target}}|x, f=0)]$$

**Sufficiency Score**: The increase in target behavior probability when feature $f$ is injected into neutral contexts (similar to Arad et al.'s output score):
$$\text{Suf}(f) = \mathbb{E}_{x \sim \mathcal{D}_{\text{neutral}}}[P(y_{\text{target}}|x, f=\alpha) - P(y_{\text{target}}|x)]$$

**Consistency Score**: The variance in causal effect across different input contexts:
$$\text{Cons}(f) = 1 - \frac{\text{Var}_{x \sim \mathcal{D}}[\Delta P(y_{\text{target}}|x, f)]}{\max_{f'} \text{Var}[\Delta P(y_{\text{target}}|x, f')]}$$

The **Intervention Fidelity Score (IFS)** combines these:
$$\text{IFS}(f) = \sqrt{\text{Nec}(f) \cdot \text{Suf}(f)} \cdot \text{Cons}(f)$$

**Key Differentiation from Arad et al. (2025)**: While Arad et al.'s output score corresponds primarily to our sufficiency component, IFS adds necessity (measuring whether the feature is required for the behavior) and consistency (measuring stability across contexts). This three-component formulation addresses cases where a feature may have high output effects in some contexts but not others, or where multiple redundant features each show sufficiency without individual necessity.

To make this computationally tractable, we use attribution patching (Syed et al., 2024) to approximate these scores. However, we acknowledge important limitations of this approach (see Section 6.2).

#### 2.2.2 Correlation Analysis Study

We systematically compare IFS against existing SAE metrics:

- **Structural metrics**: Reconstruction MSE, L0 sparsity, dead feature fraction
- **Functional metrics**: Sparse probing accuracy, auto-interpretability scores (Paulo & Belrose, 2025)
- **Causal metrics**: Feature ablation effect, activation patching attribution, Arad et al.'s output scores

For each metric pair, we compute correlation coefficients and analyze whether high scores on traditional metrics reliably predict high intervention fidelity. We specifically compare:
1. Traditional metrics vs. IFS
2. Output scores (sufficiency-only) vs. IFS (necessity + sufficiency + consistency)
3. Single-component metrics vs. the full IFS composite

#### 2.2.3 Fidelity-Guided Steering

Standard SAE steering selects features with highest activation on target prompts and upweights them by a fixed coefficient (Templeton et al., 2024). Arad et al. (2025) improved this by filtering for high output scores. We propose **Fidelity-Weighted Steering (FWS)**:

1. Identify candidate features that activate on target prompts
2. Compute full IFS (not just output score) for each candidate
3. Rank by IFS rather than raw activation or output score alone
4. Select top-k features by IFS
5. Apply steering weighted by $\text{IFS}(f)^\beta$ (with $\beta$ as a tunable hyperparameter)

We hypothesize that FWS will achieve comparable or better steering performance than both activation-based and output-score-based selection, with fewer features and reduced side effects.

### 2.3 Innovations and Contributions

1. **Extended causal metric (IFS)**: Building on Arad et al.'s output score, we introduce a comprehensive three-component metric (necessity, sufficiency, consistency) for identifying causal SAE features.

2. **Systematic comparison**: We provide a comprehensive analysis of how different metrics correlate with causal importance, comparing IFS against both traditional metrics and prior causal metrics.

3. **Practical steering improvement**: Fidelity-Weighted Steering offers a practical improvement over current methods by incorporating all three causal components.

4. **Analysis of metric complementarity**: We characterize when and why necessity and consistency components provide additional value over sufficiency alone.

---

## 3. Related Work

### 3.1 Sparse Autoencoders for Interpretability

SAEs were introduced to mechanistic interpretability by Bricken et al. (2023) and Cunningham et al. (2023), building on earlier sparse coding work (Olshausen & Field, 1997). The core architecture learns to reconstruct activations $x$ as $\hat{x} = W_{\text{dec}} \cdot f(x) + b_{\text{dec}}$ where $f(x) = \text{sparse}(W_{\text{enc}} \cdot x + b_{\text{enc}})$.

Recent architectural improvements include TopK SAEs (Gao et al., 2024), JumpReLU SAEs (Rajamanoharan et al., 2024), Gated SAEs (Rajamanoharan et al., 2024), and End-to-End SAEs (Braun et al., 2024) which optimize for KL divergence rather than reconstruction.

### 3.2 Causal Interpretability Methods

Activation patching (Vig et al., 2020; Meng et al., 2022) allows causal interventions by swapping activations between clean and corrupted runs. Attribution patching (Syed et al., 2024) approximates this efficiently using gradients.

Marks et al. (2024) use SAE features to construct sparse causal circuits, demonstrating that some SAE features are causally important. However, they do not systematically study what fraction of features are causal or how to identify them efficiently.

**Attribution Patching Limitations**: We note important limitations of attribution patching as a causal approximation. Attribution patching uses gradients to provide a first-order linear approximation of activation patching effects (Syed et al., 2024). This approximation can produce false negatives in certain scenarios, particularly in deep models or when non-linear interactions are significant (Kramár et al., 2024). Recent work has shown that gradient-based approximations may struggle with saturation effects, discontinuous activations, and path-dependent causal pathways (Ferrando et al., 2024; Wu et al., 2024). We address these limitations by validating our IFS rankings against direct activation patching on a subset of features.

### 3.3 Critiques of Current SAE Evaluation

Several concurrent works identify limitations in SAE evaluation:

- **He et al. (2025)** in "Sanity Checks for SAEs" show that SAEs perform similarly to random baselines on downstream tasks, questioning whether they learn meaningful features.
- **Minegishi et al. (2025)** demonstrate that SAEs optimized for MSE-L0 Pareto frontiers may actually harm interpretability.
- **Chaudhary & Geiger (2024)** and **Karvonen et al. (2025)** find that many SAE features lack causal effects.

Our work complements these by providing a unified framework to measure the causal-semantic disconnect and a practical method to mitigate it.

### 3.4 Steering and Its Limitations

Activation steering (Zou et al., 2023; Turner et al., 2023) modifies model behavior by adding steering vectors to activations. SAE-based steering (Templeton et al., 2024; O'Brien et al., 2024) uses SAE features as steering targets.

Recent work has identified steering limitations:
- **Raedler et al. (2025)** show that steering in fine-tuned models often achieves desired behavior through unintended mechanisms.
- **Tan et al. (2025)** find that steering vectors can be brittle and dataset-dependent.
- **Anthropic (2024)** document "off-target effects" where steering one bias inadvertently affects others.

**SAE-Targeted Steering (SAE-TS)**: Chalnev et al. (2024) address steering side effects through SAE-TS, which constructs steering vectors to target specific SAE features while minimizing unintended side effects. SAE-TS uses linear effect approximators to predict how steering vectors affect SAE features, enabling optimization for targeted behavior. While SAE-TS focuses on constructing better steering vectors, our work complements it by focusing on selecting better features to steer. The two approaches could be combined: using IFS to select high-fidelity features, then applying SAE-TS to construct optimal steering vectors for those features.

**Output Scores for Feature Selection**: Arad et al. (2025) propose "input scores" and "output scores" to distinguish features that capture input patterns from those that affect model outputs. Their output score measures the alignment between a feature's effect on the output distribution and its expected tokens, which corresponds to our sufficiency component. They demonstrate that filtering for high output scores yields 2-3x improvements in steering quality. Our IFS extends this by adding necessity (measuring whether the feature is required for the behavior) and consistency (measuring stability across contexts). This addresses limitations of sufficiency-only selection: features may be sufficient in some contexts but not others, or multiple redundant features may each show sufficiency without individual necessity.

### 3.5 How Our Work Differs

While prior work has made significant progress on identifying causal features for steering, we extend and complement it in several ways:

1. **Extended metric**: We build on Arad et al.'s output score by adding necessity and consistency components, creating a more comprehensive measure of causal influence.

2. **Systematic analysis**: We provide a detailed comparison of how different metric combinations perform for feature selection, quantifying the marginal value of each IFS component.

3. **Complementary to SAE-TS**: While SAE-TS improves steering through better vector construction, we improve it through better feature selection—these approaches are compatible and potentially synergistic.

4. **Characterization of feature properties**: We analyze what distinguishes high-IFS from low-IFS features, providing insights for future SAE training.

---

## 4. Experiments

### 4.1 Models and Datasets

**Models**: We use publicly available language models that are tractable for causal analysis:
- GPT-2 Small (124M) - for initial validation and synthetic experiments
- Pythia-160M/410M (Biderman et al., 2023) - for scaling analysis
- Gemma-2-2B (Team et al., 2024) - for instruction-tuned model analysis

**SAEs**: We use pretrained SAEs from SAELens (Bloom & Chanin, 2024) when available, training our own only when necessary (ensuring reproducibility).

**Datasets**:
- **Synthetic**: Formal language tasks with known ground-truth features (following Chaudhary & Geiger, 2024)
- **Behavioral**: TruthfulQA (Lin et al., 2022), BBQ (Parrish et al., 2022), StereoSet (Nadeem et al., 2021) for evaluation
- **General capability**: MMLU (Hendrycks et al., 2021), HellaSwag (Zellers et al., 2019) for side effect measurement

### 4.2 Experimental Design

#### Experiment 1: Quantifying the Causal-Semantic Disconnect

**Goal**: Measure the correlation between traditional SAE metrics, output scores, and full IFS.

**Procedure**:
1. Select 5-10 SAE checkpoints with varying MSE/sparsity tradeoffs
2. For each feature, compute:
   - Traditional metrics: reconstruction contribution, L0 norm, auto-interpretability score
   - Arad et al.'s output score
   - IFS components (necessity, sufficiency, consistency) via attribution patching on 500 contrastive prompt pairs
3. Compute correlations between all metric pairs
4. Identify features that score high on traditional metrics but low on IFS (the "disconnect")
5. Compare predictive power of output scores vs. full IFS for identifying effective steering features

**Expected Outcome**: Significant disconnect between reconstruction-based metrics and both output scores and IFS; IFS shows stronger correlation with actual steering effectiveness than output scores alone due to necessity and consistency components capturing edge cases.

#### Experiment 2: Comparing Feature Selection Methods

**Goal**: Compare IFS-guided steering against activation-based selection, output-score filtering (Arad et al.), and random baselines.

**Procedure**:
1. Select target behaviors: truthfulness, refusal, sentiment, bias
2. For each behavior:
   - Compute steering vectors using top-k features by activation (baseline)
   - Compute steering vectors using top-k features by output score (Arad et al.)
   - Compute steering vectors using top-k features by IFS (our method)
   - Evaluate on: target behavior change, general capability preservation, unintended behavior shifts
3. Measure side effects through:
   - Performance on unrelated tasks (MMLU, HellaSwag)
   - Activation of unrelated SAE features during steering

**Expected Outcome**: IFS-guided steering achieves comparable or better target behavior modification than both activation-based and output-score-based selection, with fewer features and reduced side effects. The improvement over output scores demonstrates the value of necessity and consistency components.

#### Experiment 3: Component Ablation Analysis

**Goal**: Quantify the marginal contribution of each IFS component (necessity, sufficiency, consistency).

**Procedure**:
1. Compare steering effectiveness using:
   - Sufficiency only (equivalent to output score)
   - Necessity only
   - Consistency only
   - Sufficiency + Necessity
   - Sufficiency + Consistency
   - Full IFS (all three)
2. Identify which components matter most for different types of behaviors

**Expected Outcome**: Sufficiency (output score) provides the strongest signal for most behaviors, but necessity and consistency provide meaningful improvements for behaviors with redundant features or context-dependent effects.

#### Experiment 4: Understanding High-Fidelity Features

**Goal**: Identify properties that distinguish high-IFS from low-IFS features.

**Procedure**:
1. Stratify features by IFS quartiles
2. Analyze differences across strata in:
   - Feature geometry (cosine similarity between decoder directions)
   - Activation patterns (sparsity, burstiness, co-activation with other features)
   - Semantic properties (via automated interpretation)
   - Layer-wise distribution
3. Test whether high-IFS features are more likely to be "functional" (affecting logits) vs. "representational" (encoding information without direct output influence)

**Expected Outcome**: High-IFS features are more sparsely activated, less correlated with other features, and concentrated in middle layers where computation occurs. Features with high output scores but low IFS may show context-dependent sufficiency without necessity.

#### Experiment 5: Scaling Analysis

**Goal**: Understand how the causal-semantic disconnect and the value of IFS components scale with model size.

**Procedure**:
1. Repeat Experiments 1-2 across Pythia models (160M, 410M, 1B if feasible)
2. Measure: fraction of high-IFS features, correlation between metrics, relative contribution of each IFS component

**Expected Outcome**: The disconnect increases with model size (more polysemanticity), and the value of necessity/consistency components grows as feature redundancy increases.

### 4.3 Evaluation Metrics

**For measuring the disconnect**:
- Spearman correlation between metric pairs
- Precision@k: fraction of top-k features by metric A that are in top-k by metric B
- Feature overlap analysis
- Steering success rate when using top-k features by each metric

**For steering evaluation**:
- Target behavior accuracy: % change in desired behavior
- Side effect score: $\frac{1}{N} \sum_{i=1}^N |\Delta \text{capability}_i|$ across N held-out tasks
- Steering efficiency: target behavior change per feature used
- Generation quality: perplexity on held-out text

**For computational efficiency**:
- All experiments designed to complete within 8 hours on 1x A6000 (48GB)
- Attribution patching reduces causal evaluation from O(n) forward passes to O(1) with gradients
- Validation subset (10% of features) evaluated with full activation patching to verify attribution patching accuracy

### 4.4 Ablations and Controls

- **Random baseline**: Steering with random SAE features (following He et al., 2025)
- **Activation-only baseline**: Standard steering without causal filtering
- **Output-score baseline**: Arad et al.'s method for comparison
- **Feature count ablation**: Vary k (number of features) to study efficiency vs. effectiveness tradeoff
- **Attribution patching validation**: Compare attribution patching estimates to direct activation patching on subset

---

## 5. Success Criteria

### 5.1 Confirming the Hypothesis

Our primary hypothesis is confirmed if:
- **SC1**: IFS shows stronger correlation with actual steering effectiveness than output scores alone (Spearman ρ improvement ≥0.1)
- **SC2**: Fidelity-weighted steering achieves ≥90% of target behavior change with ≤50% as many features as activation-based steering
- **SC3**: IFS-guided steering has ≤50% the side effect score of activation-based steering and ≤70% of output-score-based steering
- **SC4**: Component ablation shows that necessity and/or consistency components provide measurable improvements over sufficiency alone for at least some behaviors

### 5.2 Refuting the Hypothesis

Our hypothesis is refuted if:
- **RF1**: IFS correlates similarly to output scores with steering effectiveness (difference in Spearman ρ < 0.05)
- **RF2**: Fidelity-weighted steering performs similarly or worse than output-score-based selection
- **RF3**: Component ablation shows necessity and consistency provide no meaningful improvement over sufficiency alone

### 5.3 Qualitative Success Indicators

Even if quantitative thresholds are not met, we consider the project successful if we:
- Characterize when and why necessity/consistency components matter for feature selection
- Demonstrate that different causal metrics capture different aspects of feature importance
- Offer actionable insights for combining feature selection approaches
- Release code and analysis tools that enable future research

---

## 6. Broader Impact and Limitations

### 6.1 Broader Impact

**Positive**:
- Improves AI interpretability and safety by distinguishing causal from correlational features
- Provides practical tools for more precise model control with fewer side effects
- Contributes to better evaluation standards for mechanistic interpretability
- Complements existing steering methods (SAE-TS) through better feature selection

**Potential Negative**:
- More precise steering could potentially be misused to bypass safety measures
- Understanding which features are causal could aid in adversarial manipulation

We will release our findings responsibly, focusing on defensive applications and discussing risks in our paper.

### 6.2 Limitations

1. **Computational constraints**: We focus on smaller models (≤2B parameters) due to resource limits; findings may not directly transfer to frontier models.

2. **Attribution patching approximation**: We use gradient-based approximations for causal effects. Attribution patching provides only a first-order linear approximation and can produce:
   - **False negatives**: Features with non-linear or threshold effects may be underestimated (Kramár et al., 2024)
   - **Saturation errors**: Gradient estimates may be inaccurate when features are already at extreme activation values
   - **Path-dependence misses**: Approximations may not capture indirect causal pathways through multiple layers
   
   We mitigate these by validating on a subset of features using full activation patching.

3. **Behavior selection**: Our analysis is limited to behaviors we can define and measure; there may be important model behaviors outside our test set.

4. **SAE architecture**: Findings may be specific to the SAE variants we test (primarily TopK and standard ReLU SAEs).

5. **Sufficiency dominance**: Output scores (sufficiency) may dominate IFS for most practical cases, limiting the marginal value of necessity and consistency components.

### 6.3 Future Work

- Extend analysis to larger models (GPT-2 XL, LLaMA-3-8B) with compressed SAEs
- Incorporate IFS into SAE training objectives (causally-aware SAEs)
- Study how the disconnect evolves during training
- Combine IFS-based feature selection with SAE-TS for optimal steering vectors
- Investigate alternative causal metrics beyond necessity/sufficiency/consistency

---

## 7. References

1. **Bricken et al. (2023)**. "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning." *Transformer Circuits Thread*.

2. **Cunningham et al. (2023)**. "Sparse Autoencoders Find Highly Interpretable Features in Language Models." *ICLR 2024*.

3. **Gao et al. (2024)**. "Scaling and Evaluating Sparse Autoencoders." *arXiv:2406.04093*.

4. **Rajamanoharan et al. (2024)**. "Improving Dictionary Learning with Gated Sparse Autoencoders." *ICML 2024*.

5. **He et al. (2025)**. "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?" *arXiv:2502.14111*.

6. **Minegishi et al. (2025)**. "Rethinking Evaluation of Sparse Autoencoders Through the Representation of Polysemous Words." *ICLR 2025*.

7. **Marks et al. (2024)**. "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models." *ICLR 2025*.

8. **Zou et al. (2023)**. "Representation Engineering: A Top-Down Approach to AI Transparency." *NeurIPS 2023*.

9. **Turner et al. (2023)**. "Activation Addition: Steering Language Models Without Optimization." *arXiv:2308.10248*.

10. **Templeton et al. (2024)**. "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." *Anthropic Research*.

11. **Raedler et al. (2025)**. "Unintended Side Effects When Steering LLMs." *ICML 2025 Workshop*.

12. **Syed et al. (2024)**. "Attribution Patching Outperforms Automated Circuit Discovery." *BlackboxNLP 2024*.

13. **Meng et al. (2022)**. "Locating and Editing Factual Associations in GPT." *NeurIPS 2022*.

14. **Braun et al. (2024)**. "Identifying Functional Units in Language Models." *arXiv:2405.14717*.

15. **Karvonen et al. (2025)**. "SAE-Bench: A Comprehensive Benchmark for Sparse Autoencoders." *arXiv:2501.17609*.

16. **Arad et al. (2025)**. "SAEs Are Good for Steering – If You Select the Right Features." *EMNLP 2025*. https://aclanthology.org/2025.emnlp-main.519/

17. **Chalnev et al. (2024)**. "Improving Steering Vectors by Targeting Sparse Autoencoder Features." *arXiv:2411.02193*.

18. **Paulo & Belrose (2025)**. "Evaluating SAE Interpretability Without Explanations." *arXiv:2507.08473*.

19. **Kramár et al. (2024)**. "Relevance Patching: Faithful and Efficient Circuit Discovery via Activation Patching." *arXiv:2508.21258*.

20. **Ferrando et al. (2024)**. "Attribution Patching: Activation Patching At Industrial Scale." *arXiv:2410.12891*.

21. **Wu et al. (2024)**. "Open Problems in Mechanistic Interpretability." *arXiv:2501.16496*.

---

## 8. Timeline and Compute Budget

**Total Time**: ~8 hours on 1x NVIDIA RTX A6000 (48GB VRAM)

**Breakdown**:
- Experiment 1 (Correlation analysis): 2 hours
- Experiment 2 (Feature selection comparison): 3 hours
- Experiment 3 (Component ablation): 1.5 hours
- Experiment 4 (Feature characterization): 1 hour
- Experiment 5 (Scaling analysis): 0.5 hour

All experiments use pre-computed SAEs and focus on smaller models to ensure completion within the time limit. Attribution patching validation is performed on 10% of features to ensure approximation quality without excessive computation.
