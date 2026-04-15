# Research Proposal: SAE-GUIDE: Sparse Autoencoder-Guided Uncertainty-aware Information Detection and Enhancement

## 1. Introduction

### 1.1 Problem Context

Multi-hop question answering requires LLMs to synthesize information across multiple documents, yet a critical challenge remains: **determining precisely when and what information is missing during the reasoning process**. Current approaches to retrieval guidance fall into three categories, each with significant limitations:

1. **Explicit reasoning methods** (IRCOT, Self-RAG) generate intermediate reasoning steps but incur high latency from autoregressive text generation
2. **Latent alignment methods** (LaSER, March 2026) distill explicit reasoning into latent space but require costly training and don't interpret the model's internal information needs
3. **Active retrieval methods** (Probing-RAG, UAR) use hidden states to decide WHETHER to retrieve, but cannot identify WHAT specific information is missing

Recent work has revealed that LLM hidden states encode rich signals about knowledge gaps:
- **Probing-RAG** (Oct 2024) trains classifiers on hidden states to predict retrieval necessity
- **RAGLens** (Feb 2026) uses Sparse Autoencoders (SAEs) to detect hallucination-related features post-hoc
- **Sparse Latents Steer RAG** (Xin et al., ACL 2025) demonstrates that SAE features can steer RAG behavior (context vs. memory prioritization)

However, a key gap remains: **no existing method uses SAE-interpretable features to identify specific information needs in real-time and trigger targeted retrieval during multi-hop reasoning**. While Xin et al. (ACL 2025) show that SAE features can *steer* model behavior (intervention-based control), they do not address *detecting* information gaps to trigger retrieval automatically. SAEs can decompose hidden states into sparse, interpretable features that correspond to concrete concepts—providing a principled way to detect not just that information is missing, but what type of information the model is seeking.

### 1.2 Key Insight and Hypothesis

**Key Insight:** During multi-hop reasoning, specific sparse features in LLM hidden states activate when the model recognizes it lacks information needed for the next reasoning step. By training SAEs to decompose these hidden states and identifying features that correlate with information gaps, we can detect latent information needs in real-time and trigger targeted retrieval before the model commits to an incorrect reasoning path.

**Core Hypothesis:** If we (1) train SAEs on intermediate-layer hidden states during multi-hop reasoning, (2) identify sparse features that activate when bridge entities or key facts are missing, and (3) use feature activation patterns to trigger targeted retrieval, then multi-hop QA accuracy will improve significantly compared to both periodic retrieval (IRCOT) and binary active retrieval (Probing-RAG), because retrieval becomes conditioned on the model's specific internal information needs rather than generic uncertainty signals.

### 1.3 Proposed Method: SAE-GUIDE

SAE-GUIDE (Sparse Autoencoder-Guided Uncertainty-aware Information Detection and Enhancement) introduces three technical contributions:

1. **SAE-based Information-Need Detection:** Unlike Probing-RAG's binary classifiers, SAE-GUIDE trains Sparse Autoencoders on intermediate hidden states to extract interpretable features. We identify feature groups that activate when specific types of information (entity attributes, temporal facts, relational knowledge) are needed but missing.

2. **Feature-to-Query Mapping:** Instead of using hidden states directly for retrieval, we map activated SAE features to targeted retrieval queries. Each feature (or feature combination) corresponds to a specific information need, enabling precise retrieval targeting rather than generic similarity search.

3. **Uncertainty-Gated Retrieval:** We combine feature activation magnitude with uncertainty estimation to determine both WHEN to retrieve (based on cumulative feature activation) and WHAT to retrieve (based on feature semantics).

**How SAE-GUIDE Differs from Xin et al. (ACL 2025) "Sparse Latents Steer RAG":**

| Aspect | Xin et al. (ACL 2025) | SAE-GUIDE |
|--------|----------------------|-----------|
| Core mechanism | Intervention on SAE features to steer behavior | Detection of SAE features to trigger retrieval |
| SAE usage | Steering (manual intervention on latents) | Automated detection (feature-based retrieval triggering) |
| Timing | Post-hoc analysis and manual control | Real-time automatic triggering during reasoning |
| Goal | Control RAG behavior (context vs. memory) | Detect information gaps for targeted retrieval |
| Key question | "Can we control behavior by intervening on latents?" | "Can features automatically signal when to retrieve what?" |

The critical distinction: Xin et al. demonstrate that SAE features *can be used* to steer behavior through manual intervention. SAE-GUIDE asks whether these features *automatically signal* information needs that can trigger retrieval without human intervention. This is the difference between steering (control) and detecting (monitoring).

**How SAE-GUIDE Differs from LaSER (March 2026):**

| Aspect | LaSER | SAE-GUIDE |
|--------|-------|-----------|
| Core mechanism | Self-distillation from explicit CoT to latent tokens | SAE decomposition of hidden states for interpretability |
| Training requirement | Full model training with dual-view alignment | Lightweight SAE training + probe training (frozen LLM) |
| Interpretability | Black-box latent tokens | Sparse, human-interpretable features |
| Retrieval trigger | Static latent reasoning | Dynamic feature-based information-need detection |
| Differentiation | Speeds up explicit CoT | Interprets and acts on internal information gaps |

**How SAE-GUIDE Differs from Probing-RAG (Oct 2024):**

| Aspect | Probing-RAG | SAE-GUIDE |
|--------|-------------|-----------|
| Hidden state usage | Binary classifier on last hidden state | SAE decomposition of intermediate states |
| Granularity | Binary (retrieve/don't retrieve) | Continuous feature activation + semantic interpretation |
| Information targeting | Generic query based on full context | Feature-specific targeting of missing information |
| Interpretability | Black-box probe | Interpretable SAE features with semantic meaning |

**How SAE-GUIDE Differs from RAGLens (Feb 2026):**

| Aspect | RAGLens | SAE-GUIDE |
|--------|---------|-----------|
| SAE application | Post-hoc hallucination detection | Real-time information-need detection |
| Timing | After generation | During reasoning (before errors occur) |
| Feature usage | Detect factual errors | Trigger proactive retrieval |
| Intervention | None (detection only) | Active retrieval guidance |

**How SAE-GUIDE Differs from SAE-Probes (Engels et al.):**

| Aspect | SAE-Probes | SAE-GUIDE |
|--------|------------|-----------|
| Application | Sparse probing for classification tasks | Information-need detection for retrieval |
| Goal | Evaluate whether SAEs help probing | Use SAE features for targeted retrieval |
| Methodology | Linear probes on SAE features | Cumulative activation + feature-to-query mapping |
| Domain | General classification benchmarks | Multi-hop reasoning and RAG |

## 2. Related Work

### 2.1 Latent Reasoning and Retrieval

**LaSER** (Jin et al., arXiv:2603.01425, March 2026) proposes a self-distillation framework that internalizes explicit Chain-of-Thought reasoning into latent space. Their dual-view training aligns latent tokens with explicit reasoning trajectories. While LaSER focuses on making retrieval faster by eliminating text generation, SAE-GUIDE focuses on making retrieval more precise by interpreting the model's internal information needs. Critically, LaSER requires full model training, while SAE-GUIDE trains lightweight SAEs on a frozen LLM.

### 2.2 SAE-Based RAG Control and Interpretability

**Sparse Latents Steer Retrieval-Augmented Generation** (Xin et al., ACL 2025) represents the most closely related concurrent work. Using LLaMA Scope, Xin et al. identify SAE latents associated with two fundamental RAG decisions: (1) context versus memory prioritization, and (2) response generation versus query rejection. Through intervention experiments, they demonstrate that manipulating these latents can precisely control RAG behavior. Their mechanistic analysis reveals that interventions modify attention patterns in retrieval heads.

While Xin et al. establish that SAE features *can* steer RAG behavior through manual intervention, **SAE-GUIDE addresses a fundamentally different question**: Can SAE features *automatically detect* information needs to trigger retrieval without human intervention? Xin et al.'s steering approach requires knowing which features to intervene on and manually adjusting activations. SAE-GUIDE learns to recognize feature activation patterns that correlate with information gaps and automatically triggers retrieval based on these patterns. This represents a shift from manual control (steering) to automated monitoring (detection).

**RAGLens** (arXiv:2512.08892v2, Feb 2026) uses SAEs to detect hallucinations in RAG outputs by identifying interpretable features predictive of unfaithful generation. While RAGLens demonstrates SAEs can extract meaningful RAG-related features, it operates post-hoc (after generation) for detection only. SAE-GUIDE operates during reasoning to proactively trigger retrieval.

**LLaMA Scope** (He et al., 2024) provides a framework for analyzing LLaMA models with SAEs. SAE-GUIDE builds on these foundations but applies them specifically to information-need detection.

### 2.3 Active and Adaptive Retrieval

**Probing-RAG** (arXiv:2410.13339, Oct 2024) trains lightweight probes on LLM hidden states to decide whether retrieval is necessary. While effective for binary decisions, Probing-RAG cannot identify what specific information is missing. SAE-GUIDE advances this by using SAEs to decompose hidden states into interpretable features that reveal the type of information needed.

**UAR** (Unified Active Retrieval, Cheng et al., EMNLP 2024 Findings) uses multiple binary classifiers on hidden states for different retrieval criteria (intent-aware, knowledge-aware, time-aware, self-aware). SAE-GUIDE differs by using continuous feature activations rather than binary decisions, enabling more nuanced information-need detection.

**DRAGIN** (Su et al., ACL 2024) triggers retrieval based on token-level attention weights and generation probabilities. **FLARE** (Jiang et al., EMNLP 2023) uses token probability thresholds. These methods rely on surface-level generation signals rather than internal hidden state interpretation.

### 2.4 Sparse Autoencoders for Interpretability

**SAE-Probes** (Engels et al., 2024) investigate whether SAEs provide benefits for sparse probing tasks. They find that SAE features do not consistently improve probing performance over raw hidden states, highlighting challenges in applying SAEs for downstream tasks. SAE-GUIDE addresses a different problem—information-need detection for retrieval—and uses cumulative activation patterns rather than simple linear probes.

**Semantic Entropy Probes** (Kossen et al., arXiv:2406.15927, 2024) approximate semantic entropy from hidden states to detect hallucinations. SAE-GUIDE differs by targeting information needs before errors occur, rather than detecting errors after generation.

### 2.5 Multi-Hop Retrieval

**HopRAG** (Liu et al., ACL 2025 Findings) uses graph-structured knowledge exploration with retrieve-reason-prune mechanisms. **HippoRAG** (Gutiérrez et al., 2024) emulates brain-like memory with personalized PageRank. These methods focus on graph traversal rather than interpreting the LLM's internal reasoning state.

**MetaRAG** (Zhou et al., WWW 2024) uses metacognitive monitoring with discrete state classification. SAE-GUIDE provides continuous, interpretable feature-based detection rather than discrete categories.

## 3. Proposed Approach

### 3.1 Overview

SAE-GUIDE operates in four phases:

**Phase 1: SAE Training on Reasoning Hidden States**
We extract hidden states from intermediate layers (layers 12-20 of a 32-layer model) during multi-hop reasoning on training examples. We train Sparse Autoencoders to decompose these states:
```
z = SAE.encode(h)  # sparse feature activation
h_hat = SAE.decode(z)  # reconstruction
```
The SAE uses Top-K sparsity with dictionary learning to ensure interpretable features.

**Phase 2: Feature Annotation and Information-Need Mapping**
We analyze SAE feature activations to identify features correlated with specific information gaps:
- Features activating when entity attributes are needed
- Features activating when temporal information is missing  
- Features activating when relational facts are required

For each feature f_j, we learn a mapping to retrieval query augmentation:
```
query_augment_j = MLP(f_j)  # maps feature to query expansion
```

**Phase 3: Information-Need Probe Training**
We train lightweight probes that:
1. Accumulate feature activations across reasoning steps
2. Predict cumulative information uncertainty: `U_t = σ(W · cumsum(z_1:t) + b)`
3. Trigger retrieval when `U_t > θ_retrieval`

**Phase 4: Feature-Guided Retrieval**
At inference:
1. During reasoning, extract hidden states and compute SAE features
2. When cumulative activation exceeds threshold, identify top-k activated features
3. Generate targeted query using feature-to-query mappings
4. Retrieve documents matching the identified information need
5. Continue reasoning with retrieved context

### 3.2 Feature-to-Query Mapping: Concrete Examples

A key innovation of SAE-GUIDE is mapping activated features to targeted retrieval queries. Here are concrete examples of how this works:

**Example 1: Entity Bridge Detection**
```
Question: "What award did the director of Titanic win?"

Step 1: Model generates "The director of Titanic is James Cameron."
Hidden state analysis: Features f_12 ("person_entity"), f_45 ("film_director") activate
Cumulative activation: Low (known entities)
→ No retrieval triggered

Step 2: Model generates "James Cameron won the"
Hidden state analysis: Features f_12 ("person_entity"), f_203 ("award_recognition") activate strongly
Cumulative activation: High
→ Retrieve query: "James Cameron awards Academy Awards"
```

**Example 2: Temporal Fact Need**
```
Question: "When did the president who signed the ACA leave office?"

Step 1: Model identifies "Barack Obama signed the ACA"
Hidden state analysis: Features f_67 ("US_president"), f_89 ("legislation") activate

Step 2: Model needs to find when Obama left office
Hidden state analysis: Features f_156 ("temporal_inquiry"), f_67 ("US_president") activate
Cumulative activation pattern: temporal + entity combination
→ Retrieve query: "Barack Obama presidency end date"
```

**Implementation Details:**

The feature-to-query mapping is implemented as:

```python
def feature_to_query(top_features, original_question, context):
    """
    Maps activated SAE features to retrieval query augmentation.
    
    Args:
        top_features: List of (feature_id, activation_value) tuples
        original_question: The user's question
        context: Current reasoning context
    
    Returns:
        augmented_query: Query string for retrieval
    """
    # Feature type classification (learned during analysis)
    entity_features = [f for f in top_features if feature_type[f[0]] == 'entity']
    temporal_features = [f for f in top_features if feature_type[f[0]] == 'temporal']
    relation_features = [f for f in top_features if feature_type[f[0]] == 'relation']
    
    # Construct augmentation based on feature types
    augmentations = []
    
    if entity_features and temporal_features:
        # Looking for temporal info about an entity
        entity = extract_entity_from_context(context)
        augmentations.append(f"{entity} date time when")
    
    elif entity_features and relation_features:
        # Looking for relationship between entities
        entities = extract_entities_from_context(context)
        augmentations.append(f"{' '.join(entities)} relationship")
    
    elif temporal_features:
        # General temporal inquiry
        augmentations.append("year date timeline")
    
    # Combine with original question
    augmented_query = original_question + " " + " ".join(augmentations)
    return augmented_query
```

**Feature Type Learning:**

During the annotation phase, we automatically classify features by:
1. Collecting top-activating examples for each feature
2. Using an LLM to analyze patterns in activating examples
3. Assigning type labels: entity, temporal, relation, location, numeric

### 3.3 Technical Details

**SAE Architecture:**
Following RAGLens and OpenAI's SAE work, we use:
- Expansion factor: 8x (hidden_dim → 8 × hidden_dim features)
- Top-K activation: K=32 active features per input
- Loss: Reconstruction + sparsity penalty

**Hidden State Extraction:**
- Layers: 12-20 (mid-layers where reasoning representations are strongest per Yang et al., ACL 2024)
- Token positions: Last token of each reasoning step
- Aggregation: Pool across layers with learned weights

**Feature Interpretability:**
We use automated feature interpretation:
1. Collect top-activating examples for each feature
2. Use LLM to generate feature descriptions from patterns
3. Manually verify a subset for ground-truth alignment

**Retrieval Triggering:**
Unlike binary probes (Probing-RAG), we use cumulative feature activation:
```python
# At each reasoning step t
z_t = SAE.encode(h_t)  # sparse feature vector
cumulative_activation[t] = α · cumulative_activation[t-1] + z_t
uncertainty = Classifier(cumulative_activation[t])
if uncertainty > threshold:
    top_k_features = argsort(cumulative_activation[t])[-K:]
    retrieval_query = original_query + sum([feature_to_query[f] for f in top_k_features])
```

### 3.4 Architecture

**Base LLM:** Qwen2.5-7B-Instruct (frozen during SAE/probe training)

**SAE:**
- Input: 3584-dim (Qwen hidden size)
- Hidden: 28672-dim (8x expansion)
- Output: 3584-dim
- Parameters: ~200M (frozen LLM) + ~100M (SAE, trainable)

**Information-Need Probe:**
- 2-layer MLP: 28672 → 1024 → 1
- Parameters: ~30M

**Total trainable parameters:** ~130M (lightweight compared to full model fine-tuning)

### 3.5 Inference Algorithm

```
Input: Question Q, LLM M, SAE E, Probe P, Feature-to-Query Map F
Output: Answer A

1. Initialize: context C = ∅, cumulative_features = 0
2. For t = 1, 2, ... until answer generated:
   a. Generate next reasoning tokens with M using C
   b. Extract hidden state h_t from layer 16
   c. z_t = E.encode(h_t)  # sparse features
   d. cumulative_features = 0.9 · cumulative_features + z_t
   e. uncertainty = P(cumulative_features)
   f. If uncertainty > θ:
      i. top_k_features = argsort(cumulative_features)[-K:]
      ii. query_expansion = concat([F[f] for f in top_k_features])
      iii. D = retrieve(query=Q + query_expansion, k=3)
      iv. C = C ∪ D
3. Return generated answer A
```

## 4. Failure Analysis and Limitations

### 4.1 Potential Failure Modes

We explicitly acknowledge and plan to analyze the following failure modes:

**F1: SAE Features Are Not Interpretable**
- *Risk:* SAE features may not correspond to meaningful semantic concepts
- *Detection:* Human evaluation of feature interpretability (target: >50% meaningful)
- *Mitigation:* Use more training data for SAE; adjust sparsity hyperparameters
- *Impact on project:* If <30% features are interpretable, the feature-to-query mapping will fail

**F2: Features Don't Correlate with Information Needs**
- *Risk:* Activations may not reliably signal specific information gaps
- *Detection:* Correlation analysis between feature activations and ground-truth intermediate answers
- *Mitigation:* Try different SAE layers; use supervised feature selection
- *Impact on project:* If correlation r < 0.3, the core hypothesis is refuted

**F3: Cumulative Activation Provides No Benefit**
- *Risk:* Per-step activation may be as effective as cumulative tracking
- *Detection:* Ablation comparing cumulative vs. per-step activation
- *Mitigation:* Simplify to per-step; adjust decay factor α
- *Impact on project:* Would reduce claim to "SAE features help" rather than "cumulative activation helps"

**F4: Feature-to-Query Mapping Fails**
- *Risk:* Learned mappings may not improve retrieval quality
- *Detection:* Compare retrieval precision with/without feature-guided augmentation
- *Mitigation:* Use simpler heuristic mappings; manual feature annotation
- *Impact on project:* Would still demonstrate SAE-based detection but with less targeted retrieval

**F5: Overfitting to Training Distribution**
- *Risk:* Method works on HotpotQA but fails on other multi-hop datasets
- *Detection:* Cross-dataset evaluation on 2WikiMultiHopQA
- *Mitigation:* More diverse training data; stronger regularization
- *Impact on project:* Limits generalizability claims

### 4.2 Dataset Size Limitation

We acknowledge that 800 total questions (500 HotpotQA + 300 2WikiMultiHopQA) is relatively small:

- *Statistical power:* May not detect small effect sizes (<3% improvement)
- *Generalizability:* Results may not generalize to full benchmarks
- *SAE training:* Smaller dataset may yield less interpretable features

*Mitigation strategies:*
1. Use established data splits (dev sets) to maximize validity
2. Report confidence intervals and effect sizes, not just point estimates
3. Acknowledge as limitation in paper; suggest future work on full benchmarks
4. Focus on demonstrating proof-of-concept rather than state-of-the-art results

### 4.3 Computational Constraints

The 8-hour time budget imposes additional limitations:
- Simplified SAE architecture (vs. state-of-the-art)
- Limited hyperparameter search
- Automated (not extensive human) feature evaluation
- Cannot train full LaSER baseline for comparison

## 5. Experiments

### 5.1 Research Questions

1. Does SAE-GUIDE improve multi-hop QA accuracy compared to Probing-RAG and periodic retrieval (IRCOT)?
2. Do SAE features provide more interpretable information-need signals than binary probes?
3. Does feature-guided retrieval reduce unnecessary retrieval calls compared to uncertainty-only methods?
4. Can we interpret what specific information each feature corresponds to?
5. **(New)** What is the failure mode distribution? Which failure modes are most common?

### 5.2 Baselines

**Implementable within 8-hour budget:**
- **Standard RAG:** Single retrieval with question only
- **IRCOT** (Trivedi et al., ACL 2023): Periodic retrieval every N tokens (we use N=50)
- **Probing-RAG** (Oct 2024): Binary probe on hidden states for retrieval decisions
- **LaSER-inspired baseline:** Given 8-hour constraint, we compare against a simplified version using the public Contriever with query expansion as a proxy for latent reasoning

**Note:** Full LaSER training requires significant compute beyond our budget; we acknowledge this limitation and focus on demonstrating SAE-based detection vs. simpler baselines.

### 5.3 Datasets

**HotpotQA** (Yang et al., 2018): 500 questions from dev set (distractor setting) - 2-hop reasoning
**2WikiMultiHopQA** (Ho et al., 2020): 300 questions from dev set - complex multi-hop

Both datasets have ground-truth supporting documents and intermediate answers, enabling evaluation of information-need detection accuracy.

### 5.4 Evaluation Metrics

**Primary metrics:**
- Answer Accuracy (Exact Match, F1)
- Retrieval Precision@k (% of retrieved docs that are relevant)
- Retrieval Efficiency (number of retrieval calls per question)

**Secondary metrics (validating our approach):**
- Feature Interpretability Score (human evaluation of feature descriptions)
- Information-Need Detection Accuracy (% of times retrieval triggered at correct reasoning step)
- Feature Activation Correlation with ground-truth intermediate answer needs

**Failure analysis metrics:**
- Distribution of failure modes (F1-F5 above)
- Per-feature activation stability
- Cross-dataset feature consistency

### 5.5 Expected Results

**Hypothesis 1:** SAE-GUIDE achieves 5-10% higher F1 than Probing-RAG on HotpotQA by providing finer-grained information-need detection.

**Hypothesis 2:** SAE-GUIDE reduces unnecessary retrievals by 20-30% compared to IRCOT while maintaining comparable accuracy.

**Hypothesis 3:** Top-activated SAE features correlate (r > 0.6) with ground-truth intermediate information needs.

**Hypothesis 4:** Human evaluation rates SAE feature interpretations as meaningful for 60%+ of top-activating features.

**Hypothesis 5 (Failure Analysis):** The most common failure mode is F2 (features don't correlate with needs) if SAE training is insufficient; F1 (non-interpretable features) if sparsity is too high.

### 5.6 Ablations

- **Binary probe baseline:** Standard MLP on hidden states (like Probing-RAG)
- **No cumulative activation:** Use per-step feature activation only
- **No feature-to-query mapping:** Retrieve using original query only
- **Different layer selection:** Early (layers 4-8) vs. mid (12-20) vs. late (24-30) layers
- **SAE sparsity levels:** Compare K=16, 32, 64 active features
- **Feature type ablation:** Use only entity features vs. only temporal features vs. all

### 5.7 Analysis: Feature Interpretability

We will conduct automatic feature interpretation:
1. For each of top-100 most active features, collect 50 examples with highest activation
2. Use GPT-4 to generate descriptions of common patterns
3. Human annotators rate descriptions as: "clearly meaningful", "somewhat meaningful", "unclear"
4. Compare with baseline: random feature descriptions

## 6. Success Criteria

### 6.1 Confirmation Criteria

- **C1:** SAE-GUIDE achieves statistically significant improvement (p < 0.05) over Probing-RAG on HotpotQA.
- **C2:** Retrieval efficiency (calls per question) is at least 15% better than IRCOT with comparable accuracy.
- **C3:** Feature interpretability study shows >50% of top features are rated as "meaningful" or better.
- **C4:** Ablation shows SAE features outperform binary probe baseline (same architecture without SAE decomposition).

### 6.2 Refutation Criteria

- **R1:** SAE-GUIDE shows no improvement over simple binary probe (suggesting SAE decomposition adds no value for retrieval).
- **R2:** Feature interpretability is poor (<30% meaningful), suggesting SAE features don't capture semantic information needs.
- **R3:** Cumulative feature activation provides no benefit over per-step activation.
- **R4:** Feature-to-query mapping does not improve retrieval precision compared to original query.

## 7. Significance and Impact

### 7.1 Scientific Contribution

1. **Novel Application of SAEs:** First work to use SAEs for automated real-time information-need detection and retrieval triggering, extending Xin et al.'s (ACL 2025) intervention-based steering to automated detection-based triggering.

2. **Interpretable Retrieval Guidance:** Unlike black-box probes or latent tokens, SAE-GUIDE provides human-interpretable signals about what information the model needs.

3. **Lightweight Alternative:** Demonstrates that frozen-LLM SAE training can achieve competitive results versus full model training (LaSER), making the approach accessible for resource-constrained settings.

4. **Failure Mode Analysis:** First work to systematically analyze failure modes of SAE-based retrieval detection, providing guidance for future research.

### 7.2 Practical Impact

1. **Reduced Retrieval Costs:** By targeting retrieval based on specific information needs rather than generic uncertainty, we reduce unnecessary API calls.

2. **Debugging and Transparency:** Interpretable features enable practitioners to understand why retrieval was triggered and what information the model was seeking.

3. **Generalization:** The approach is model-agnostic (any LLM with extractable hidden states) and doesn't require task-specific fine-tuning.

### 7.3 Publication Target

This work targets top-tier NLP venues such as **ACL**, **EMNLP**, or **NAACL**. The novel combination of SAE interpretability with active retrieval, positioned against recent work (Xin et al. ACL 2025, LaSER, RAGLens, Probing-RAG), provides a timely and differentiated contribution.

## 8. Timeline and Resource Requirements (Realistic 8-Hour Budget)

### 8.1 Hour-by-Hour Plan

**Hour 1-2: Setup and Data Preparation**
- Set up environment, install dependencies (transformers, sae-lens)
- Download HotpotQA and 2WikiMultiHopQA dev sets (500 + 300 = 800 questions)
- Preprocess: extract reasoning chains using GPT-4 or use existing annotated data
- Set up BM25/Contriever retrieval index

**Hour 3-4: SAE Training and Feature Extraction**
- Generate hidden state dataset: run Qwen2.5-7B on training questions, extract layer 16 hidden states
- Train SAE on hidden states (fast convergence with small dataset ~2K examples)
- Extract SAE features for all training examples
- Implement cumulative feature tracking

**Hour 5: Probe Training and Baseline Implementation**
- Train information-need probe on SAE features
- Implement Probing-RAG baseline (binary MLP probe)
- Implement IRCOT baseline (periodic retrieval)
- Implement Standard RAG baseline

**Hour 6-7: Evaluation**
- Run all methods on test sets (500 HotpotQA + 300 2WikiMultiHopQA)
- Compute all metrics (accuracy, retrieval precision, efficiency)
- Run ablation studies
- Analyze feature activations

**Hour 8: Analysis, Failure Modes, and Documentation**
- Feature interpretability analysis (automated description generation)
- Failure mode analysis (classify errors into F1-F5 categories)
- Statistical significance tests
- Generate comparison tables and figures

### 8.2 Compute Budget

**Total trainable parameters:** ~130M (SAE + probe)
**Training time:** 
- Hidden state extraction: ~30 min (800 examples, cached)
- SAE training: ~60 min (2K examples, ~100M parameters)
- Probe training: ~15 min (small MLP)

**Inference time:**
- 800 questions × (reasoning + retrieval) ≈ 5 GPU hours with caching

**Total:** ~8 hours on single A6000 (48GB)

### 8.3 Justification for Scope

We acknowledge limitations given the 8-hour constraint:
- Using subsets of datasets (500+300 questions) rather than full benchmarks
- Simplified LaSER comparison (cannot fully train LaSER in 8 hours)
- Automated rather than extensive human evaluation of feature interpretability
- Limited hyperparameter search

However, the core contribution—demonstrating SAE-based information-need detection outperforms binary probes—is achievable within this scope.

## 9. References

1. Xin, C., Zhou, S., Zhu, H., Wang, W., Chen, X., Guan, X., Lu, Y., Lin, H., Han, X., & Sun, L. (2025). Sparse Latents Steer Retrieval-Augmented Generation. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 4547–4562, Vienna, Austria. Association for Computational Linguistics.

2. Jin, J., Zhang, Y., Li, M., Long, D., Xie, P., Zhu, Y., & Dou, Z. (2026). LaSER: Internalizing Explicit Reasoning into Latent Space for Dense Retrieval. arXiv:2603.01425.

3. Liu, H., Wang, Z., Chen, X., Li, Z., Xiong, F., Yu, Q., & Zhang, W. (2025). HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval-Augmented Generation. ACL 2025 Findings.

4. RAGLens Authors. (2026). Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders. arXiv:2512.08892v2.

5. Probing-RAG Authors. (2024). Probing-RAG: Self-Probing to Guide Language Models in Selective Document Retrieval. arXiv:2410.13339.

6. Engels, J., et al. (2024). Are Sparse Autoencoders Useful? A Case Study in Sparse Probing. GitHub: https://github.com/JoshEngels/SAE-Probes

7. Cheng, Z., et al. (2024). Unified Active Retrieval for Retrieval Augmented Generation. EMNLP 2024 Findings.

8. Kossen, J., Han, J., Razzak, M., Schut, L., Malik, S., & Gal, Y. (2024). Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. arXiv:2406.15927.

9. Su, H., et al. (2024). DRAGIN: Dynamic Retrieval Augmented Generation based on the Information Needs of LLMs. ACL 2024.

10. Jiang, Z., et al. (2023). Active Retrieval Augmented Generation. EMNLP 2023.

11. Trivedi, H., et al. (2023). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. ACL 2023.

12. Zhou, Y., Liu, Z., Jin, J., Nie, J.-Y., & Dou, Z. (2024). MetaRAG: Metacognitive Retrieval-Augmented Large Language Models. WWW 2024.

13. Yang, S., et al. (2024). Do Large Language Models Latently Perform Multi-Hop Reasoning? ACL 2024.

14. He, J., et al. (2024). LLaMA Scope: An Interpretability Lens for Large Language Models. (Technical Report).

15. Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread.

16. Yang, Z., et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP 2018.

17. Ho, X., et al. (2020). Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. COLING 2020.

18. Gutiérrez, B., et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. ACL 2024.
