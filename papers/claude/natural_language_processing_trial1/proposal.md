# Context-Contrastive Uncertainty Decomposition for Reliable Retrieval-Augmented Generation

## Introduction

### Context

Retrieval-Augmented Generation (RAG) has become the dominant paradigm for grounding large language model (LLM) outputs in external knowledge, enabling more factual and up-to-date responses. However, a critical open problem remains: **how to reliably estimate whether a RAG system's output is trustworthy**. When RAG systems fail, the failure can stem from multiple distinct sources -- the retriever may return irrelevant documents, the generator may hallucinate despite having good context, or the model may override retrieved evidence with incorrect parametric knowledge.

### Problem Statement

Recent axiomatic analysis (Soudani et al., 2025) has formally demonstrated that existing uncertainty estimation (UE) methods -- including token probability, semantic entropy, and verbalized confidence -- **systematically fail in RAG settings**. These methods were designed for standard LLM generation and do not account for the interaction between retrieved context and model behavior. Specifically, no existing UE method satisfies all five axioms required for reliable uncertainty estimation when external documents are incorporated into prompts.

The core issue is that current methods produce a **single, monolithic uncertainty score** that conflates fundamentally different failure modes:
- **Retrieval failure**: The retrieved documents are irrelevant or insufficient.
- **Grounding failure**: The model has relevant documents but ignores or misinterprets them.
- **Parametric override**: The model's prior knowledge conflicts with and overrides the retrieved evidence.

While concurrent work has begun to address context-awareness in uncertainty estimation -- notably CRUX (Yuan et al., 2026), which compares model behavior with and without context via entropy reduction and consistency metrics, and FRANQ (Fadeeva et al., 2025), which conditions uncertainty quantification on faithfulness to retrieved context -- these approaches use only **two conditions** (with vs. without context) and cannot distinguish between a model that genuinely grounds in relevant evidence versus one that is indiscriminately sensitive to any context (including irrelevant noise).

### Key Insight

Our key insight is that by comparing an LLM's behavior under **three** controlled conditions -- **with retrieved context**, **without context**, and **with irrelevant context** -- we can disentangle failure modes that two-condition approaches conflate. This is analogous to a controlled experiment: the "with context" setting is the treatment, "without context" is the control, and "irrelevant context" is the **placebo**. The pattern of divergence across these three conditions reveals not just *whether* the model uses context, but whether it uses *the right* context. Specifically, the third condition enables our **Context Discrimination** metric, which detects indiscriminate context sensitivity -- a failure mode invisible to two-condition frameworks like CRUX.

### Hypothesis

We hypothesize that decomposing RAG uncertainty into retrieval sensitivity, context discrimination, and parametric agreement components -- enabled by the three-condition design -- will produce significantly better selective prediction (the ability to abstain on likely-incorrect answers) compared to both holistic uncertainty measures and two-condition contrastive methods, while also enabling targeted interventions based on the identified failure mode.

## Proposed Approach

### Overview

We propose **C2UD (Context-Contrastive Uncertainty Decomposition)**, a method that decomposes RAG uncertainty into three interpretable components by analyzing how an LLM's behavior changes across controlled context conditions. C2UD requires three forward passes per query plus a lightweight logistic regression calibration step fitted on a small labeled set. The core decomposition is training-free and model-agnostic; only the final scoring aggregation requires calibration.

### Method Details

Given a query $q$ and a set of retrieved documents $D = \{d_1, \ldots, d_k\}$:

#### Step 1: Multi-Condition Generation

Generate responses under three conditions:
1. **RAG condition**: Present the query with retrieved documents. Obtain response $a_D$ and token-level log-probabilities $\mathbf{p}_D = (p_D^{(1)}, p_D^{(2)}, \ldots, p_D^{(T)})$ at each generated token position $t$.
2. **Parametric condition**: Present the query alone (no documents). Obtain response $a_\emptyset$ and log-probabilities $\mathbf{p}_\emptyset$.
3. **Control condition**: Present the query with irrelevant documents $D'$ sampled from a generic corpus (unrelated Wikipedia passages from different topic categories). Obtain response $a_{D'}$ and log-probabilities $\mathbf{p}_{D'}$.

#### Step 2: Uncertainty Component Extraction

We compute three uncertainty components using **sequence-level JSD** rather than first-token approximation.

**Sequence-Level JSD Computation**: For two distributions $\mathbf{p}$ and $\mathbf{q}$ over generated token positions $t = 1, \ldots, T$, we compute:
$$\text{JSD}_{\text{seq}}(\mathbf{p} \| \mathbf{q}) = \frac{1}{T} \sum_{t=1}^{T} \text{JSD}(p^{(t)} \| q^{(t)})$$
where $p^{(t)}$ and $q^{(t)}$ are the next-token probability distributions at position $t$. When generated sequences differ in length, we use the shared prefix length or teacher-force the shorter sequence's tokens through the other model to obtain aligned distributions. This averaging over all token positions captures distributional differences throughout the full response, not just at the first token.

**Retrieval Sensitivity (RS)**: Measures how much the model's output distribution shifts when retrieved context is provided versus absent.
$$RS(q, D) = \text{JSD}_{\text{seq}}(\mathbf{p}_D \| \mathbf{p}_\emptyset)$$
- High RS: The model uses the retrieved context (may be appropriate or may indicate susceptibility).
- Low RS: The model ignores the context (answers from parametric memory).

**Context Discrimination (CD)**: Measures whether the model differentially responds to relevant versus irrelevant context, normalized to ensure non-negativity:
$$CD(q, D) = \frac{\text{JSD}_{\text{seq}}(\mathbf{p}_D \| \mathbf{p}_{D'})}{\text{JSD}_{\text{seq}}(\mathbf{p}_D \| \mathbf{p}_{D'}) + \text{JSD}_{\text{seq}}(\mathbf{p}_\emptyset \| \mathbf{p}_{D'})}$$

This ratio formulation ensures $CD \in [0, 1]$, avoiding the interpretability issues of the subtraction form ($\text{JSD}(\mathbf{p}_D \| \mathbf{p}_{D'}) - \text{JSD}(\mathbf{p}_\emptyset \| \mathbf{p}_{D'})$) which could yield negative values. When $CD > 0.5$, the model's response to relevant documents diverges from irrelevant documents more than the parametric-only response does, indicating genuine grounding. When $CD \leq 0.5$, the model is no more discriminating of relevant context than the baseline, suggesting unreliable context use.

**Parametric Agreement (PA)**: Measures whether the RAG answer and the parametric answer agree semantically.
$$PA(q, D) = \text{NLI}(a_D, a_\emptyset)$$
where NLI is a lightweight natural language inference model (DeBERTa-v3-large fine-tuned on MNLI) that classifies whether $a_D$ entails, contradicts, or is neutral to $a_\emptyset$. We use the entailment probability as the agreement score.
- Agreement + high RS: Strong confidence (both sources agree and context is used).
- Disagreement + high CD: Trust the RAG answer (model appropriately grounded in context).
- Disagreement + low CD: Low confidence (conflict with no clear evidence of grounding).

#### Step 3: Calibrated Uncertainty Scoring

We feed the three components (RS, CD, PA) plus pairwise interactions (RS$\times$CD, RS$\times$PA, CD$\times$PA) as features to a **logistic regression classifier** for selective prediction. This classifier is fitted on a small labeled calibration set (~200 examples with binary correctness labels), separate from the test set. We use 5-fold cross-validation within the calibration set to select regularization strength.

We explicitly acknowledge this is a lightweight supervised calibration step, not a training-free method. The **decomposition** itself (computing RS, CD, PA) is training-free and model-agnostic; only the aggregation into a single score requires calibration. This is analogous to how Platt scaling or temperature scaling calibrates any confidence method.

#### Step 4: Targeted Intervention

Based on the decomposed uncertainty, C2UD enables specific remediation strategies:
- **Low CD (poor grounding)**: Trigger re-retrieval with a reformulated query or more diverse retrieval.
- **Low PA with high CD (knowledge conflict)**: The model is grounded but parametric knowledge conflicts -- flag for human review or prioritize the context-grounded answer.
- **Low RS (context ignored)**: The retrieval may be irrelevant -- try a different retrieval strategy or more passages.

### Key Innovations and Positioning

1. **Three-condition contrastive design**: Unlike CRUX (Yuan et al., 2026) and other two-condition methods that compare only with/without context, C2UD introduces an irrelevant context control condition. This enables the Context Discrimination metric, which detects indiscriminate context sensitivity -- a failure mode invisible to two-condition frameworks.

2. **Normalized Context Discrimination metric**: The ratio-based CD formulation provides a bounded, interpretable measure of genuine grounding that avoids the negativity issues of raw JSD subtraction.

3. **Sequence-level distributional comparison**: Unlike first-token approximations, our sequence-level JSD averages over all generated token positions, capturing distributional differences throughout the full response.

4. **Actionable uncertainty decomposition**: While FRANQ (Fadeeva et al., 2025) decomposes uncertainty based on faithfulness (a post-hoc assessment), C2UD decomposes at the generative process level through controlled conditions, enabling proactive interventions before the final answer is committed.

## Related Work

### Uncertainty Estimation for LLMs

**Semantic Entropy** (Kuhn et al., 2023; Farquhar et al., 2024, Nature) clusters sampled responses by semantic equivalence and computes entropy over these clusters, separating linguistic from semantic uncertainty. **Semantic Entropy Probes** (Kossen et al., 2024, NeurIPS) approximate this from hidden states in a single pass but suffer from out-of-distribution degradation. **Kernel Language Entropy** (Nikitin et al., 2024, NeurIPS) generalizes semantic entropy using pairwise semantic similarities. **SPUQ** (Gao et al., 2024, EACL) perturbs input prompts to estimate epistemic uncertainty. All these methods operate on the generation level and do not account for the RAG-specific interaction between context and generation.

### Uncertainty in RAG

**Soudani et al. (2025, ACL Findings)** present an axiomatic analysis showing that existing UE methods fail in RAG settings and propose a calibration function. However, their calibration function is a post-hoc fix that does not decompose uncertainty by source. **Counterfactual prompting** (Chen et al., 2024) uses counterfactual variations to assess RAG risk but focuses on risk control rather than uncertainty decomposition. **Noise-Aware Verbal Confidence** (Liu et al., 2026) addresses noise-aware verbal confidence for RAG but relies on verbalized confidence which is known to be poorly calibrated. **QuCo-RAG** (Huang et al., 2025) uses pre-training corpus statistics to decide when to retrieve, but does not decompose uncertainty once retrieval has occurred. **R2C** (Soudani et al., 2025, arXiv:2510.11483) addresses uncertainty in retrieval-augmented reasoning via perturbation-based consistency but targets multi-step reasoning chains rather than single-hop QA.

### Context-Aware Confidence and Contrastive Methods

**CRUX** (Yuan et al., 2026, arXiv:2508.00600) is the most closely related work. CRUX proposes a context-aware dual-metric framework using (1) contextual entropy reduction -- measuring information gain from context via Shannon entropy difference $\Delta H = H(K^{(q)}) - H(K^{(c,q)})$ -- and (2) unified consistency examination -- measuring global consistency of answers with and without context. CRUX evaluates on reading comprehension benchmarks (CoQA, SQuAD, QuAC, BioASQ, EduQG) and achieves strong AUROC results.

**C2UD differs from CRUX in three key ways**: (1) CRUX uses **two conditions** (with/without context), while C2UD uses **three conditions** (with/without/irrelevant context), enabling the Context Discrimination metric that detects models blindly sensitive to any context rather than genuinely grounding in relevant evidence; (2) CRUX uses Shannon entropy over semantic clusters requiring multiple samples, while C2UD uses sequence-level JSD over token distributions from single forward passes, making it more computationally efficient; (3) C2UD explicitly decomposes uncertainty into three named components with distinct diagnostic interpretations, enabling targeted interventions.

**FRANQ** (Fadeeva et al., 2025, arXiv:2505.21072) decomposes RAG uncertainty conditioned on faithfulness to retrieved context. FRANQ estimates $P(\text{true}) = P(\text{faithful}) \cdot UQ_{\text{faith}} + P(\text{unfaithful}) \cdot UQ_{\text{unfaith}}$, applying different UQ techniques depending on whether a claim is faithful to the retrieval. While FRANQ shares the goal of decomposing RAG-specific uncertainty, it differs fundamentally: FRANQ is a **post-hoc claim-level** assessment that first generates an answer and then classifies claims by faithfulness, whereas C2UD operates at the **generation-process level** by controlling the input conditions before generation. Additionally, FRANQ requires separate UQ pipelines for faithful vs. unfaithful conditions, while C2UD uses a unified three-condition framework.

**Context-Aware Decoding (CAD)** (Shi et al., 2024, NAACL) contrasts output distributions with and without context to improve faithfulness. C2UD builds on this contrastive principle but extends it from a decoding strategy to an uncertainty estimation framework with a third control condition and formal uncertainty decomposition.

### Comparison Table

| Method | Conditions | Decomposition | RAG-specific | Detects indiscriminate sensitivity | Targeted intervention |
|--------|-----------|---------------|-------------|-----------------------------------|----------------------|
| Semantic Entropy | 1 (sampling) | No | No | No | No |
| SPUQ | 1 (perturbation) | No | No | No | No |
| CRUX | 2 (with/without) | Partial (2 metrics) | Yes | No | No |
| FRANQ | 1 (post-hoc) | Yes (faith/unfaith) | Yes | No | No |
| Soudani calibration | 1 | No | Yes | No | No |
| **C2UD (Ours)** | **3 (with/without/irrelevant)** | **Yes (3 components)** | **Yes** | **Yes** | **Yes** |

## Experiments

### Datasets

We evaluate on three standard open-domain question answering benchmarks commonly used in RAG evaluation:

1. **Natural Questions (NQ)** (Kwiatkowski et al., 2019): Questions derived from real Google search queries with Wikipedia-based answers. We use the open-domain subset (~1000 test examples).
2. **TriviaQA** (Joshi et al., 2017): Trivia questions with evidence documents, representing broad knowledge (~1000 test examples).
3. **PopQA** (Mallen et al., 2023): Entity-centric questions focusing on long-tail knowledge, where RAG is particularly important because parametric knowledge is less reliable (~1000 test examples).

For retrieval, we use Contriever (Izacard et al., 2022) with a Wikipedia corpus, retrieving top-5 passages per query. Irrelevant control documents are sampled from unrelated Wikipedia topic categories (verified to have no keyword overlap with the query).

### Models

- **Llama 3.1 8B Instruct**: Open-source, widely used, fits on a single A6000.
- **Mistral 7B Instruct v0.3**: Different model family for generalization testing.
- **Phi-3 Mini 3.8B Instruct**: Smaller model to test scalability across model sizes.

### Baselines

1. **Token Probability**: Average log-probability of generated tokens.
2. **Verbalized Confidence**: Prompt the model to state its confidence (0-100%).
3. **Semantic Entropy** (Kuhn et al., 2023): Sample 5 responses, cluster by semantic equivalence, compute entropy. (Adapted for RAG by including retrieved context.)
4. **SPUQ** (Gao et al., 2024): Perturbation-based uncertainty via prompt paraphrasing.
5. **Self-Consistency** (Wang et al., 2023): Majority vote across 5 sampled reasoning paths.
6. **Axiomatic Calibration** (Soudani et al., 2025): The calibration function proposed by the axiomatic analysis paper.
7. **CRUX** (Yuan et al., 2026): Two-condition context-aware confidence estimation using entropy reduction and consistency. We implement CRUX's two metrics (contextual entropy reduction + unified consistency) and combine them via the same logistic regression calibration used for C2UD, for fair comparison.

### Evaluation Metrics

**Selective Prediction Quality**:
- **AUROC**: Area under ROC curve for predicting answer correctness.
- **AUPRC**: Area under precision-recall curve (more informative when classes are imbalanced).
- **Coverage@90**: Maximum fraction of questions answered while maintaining 90% accuracy.

**Calibration Quality**:
- **ECE**: Expected Calibration Error (with 10 bins).
- **Brier Score**: Combined measure of calibration and discrimination.

### Experimental Design

**Experiment 1: Selective Prediction Performance**
Compare C2UD against all baselines (including CRUX) on AUROC, AUPRC, and Coverage@90 across all three datasets and three models. Use paired bootstrap testing (10,000 resamples) for statistical significance.

**Experiment 2: Ablation Study**
Evaluate the contribution of each uncertainty component:
- C2UD-RS: Only retrieval sensitivity
- C2UD-CD: Only context discrimination
- C2UD-PA: Only parametric agreement
- C2UD-RS+CD: Without parametric agreement
- C2UD-RS+PA: Without context discrimination (reduces to a two-condition method like CRUX)
- C2UD-full: All three components

The comparison of C2UD-RS+PA (two-condition) vs. C2UD-full (three-condition) directly tests the value of the irrelevant context condition.

**Experiment 3: Calibration Analysis**
Compare ECE and Brier scores across methods. Analyze calibration reliability diagrams.

**Experiment 4: Failure Mode Diagnosis**
Demonstrate that C2UD's decomposed scores correctly identify the failure mode:
- On queries where retrieval fails (low-quality passages), RS should be low and CD should be low.
- On queries where the model hallucinates despite good context, CD should be low while RS is high.
- On queries where parametric knowledge overrides context, PA should show disagreement with high RS.
We validate on 100 manually annotated queries with labeled failure modes.

**Experiment 5: Targeted Intervention**
Show that using C2UD's decomposition to trigger targeted interventions improves end-to-end accuracy compared to uniform strategies:
- **C2UD-intervene**: Use decomposed scores to select intervention type (re-retrieval for low CD, more passages for low RS, abstain for parametric conflict).
- **Uniform re-retrieval**: Re-retrieve for all uncertain queries.
- **Uniform abstention**: Abstain on all uncertain queries.
- **Oracle intervention**: Upper bound using ground-truth failure labels.

This experiment is limited to one model (Llama 3.1 8B) and one dataset (NQ) to fit within the compute budget.

### Expected Results

We expect:
1. C2UD will achieve 2-4 point AUROC improvement over CRUX and 3-5 points over non-contrastive baselines on selective prediction, with the advantage largest on PopQA (where retrieval quality varies most).
2. Context discrimination (CD) will be the most important component in the ablation, as it captures a signal not available to two-condition methods.
3. C2UD-RS+PA (two-condition ablation) will perform comparably to CRUX, validating that the improvement comes specifically from the third condition.
4. C2UD will maintain competitive performance across all three model families.
5. Targeted interventions based on decomposed scores will outperform uniform strategies by 2-3% in end-to-end accuracy.

### Computational Budget

Each query requires 3 forward passes (with context, without context, with irrelevant context). For ~3000 test examples across 3 datasets, 3 models:
- C2UD: 3 passes x 3000 queries x 3 models = 27,000 forward passes
- CRUX baseline: 5 samples x 3000 queries x 3 models = 45,000 forward passes (for semantic clustering)
- Semantic Entropy: 5 samples x 3000 queries x 3 models = 45,000 forward passes
- Other baselines: ~1 pass each
- Experiment 5 (intervention): ~3000 additional forward passes (1 model, 1 dataset)

With batched inference on A6000 (Llama 8B at ~50 tokens/s, Phi-3 at ~100 tokens/s), estimated total: ~6-7 GPU-hours. This fits within the 8-hour budget.

## Success Criteria

### Confirming the hypothesis
- C2UD achieves statistically significant improvement (p < 0.05, paired bootstrap) in AUROC over the best baseline (including CRUX) on at least 2 of 3 datasets.
- The ablation shows each component contributes, with the three-condition C2UD-full outperforming the two-condition C2UD-RS+PA ablation.
- Decomposed scores correctly diagnose failure modes (validated manually on 100 samples).

### Refuting the hypothesis
- If C2UD does not outperform CRUX or if the three-condition design provides no benefit beyond two conditions, the hypothesis would be refuted. In this case, we would report the negative result and analyze why the irrelevant context condition fails to add signal.

## References

1. Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR 2023*. arXiv:2302.09664.
2. Farquhar, S., Kuhn, L., et al. (2024). Detecting Hallucinations in Large Language Models Using Semantic Entropy. *Nature*, 630, 625-630.
3. Kossen, J., Gal, Y., & Rainforth, T. (2024). Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. *NeurIPS 2024*. arXiv:2406.15927.
4. Nikitin, A., et al. (2024). Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities. *NeurIPS 2024*.
5. Gao, X., Zhang, J., Mouatadid, L., & Das, K. (2024). SPUQ: Perturbation-Based Uncertainty Quantification for Large Language Models. *EACL 2024*. arXiv:2403.02509.
6. Wang, X., Wei, J., Schuurmans, D., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*. arXiv:2203.11171.
7. Soudani, H., Kanoulas, E., & Hasibi, F. (2025). Why Uncertainty Estimation Methods Fall Short in RAG: An Axiomatic Analysis. *ACL 2025 Findings*. arXiv:2505.07459.
8. Chen, L., Zhang, R., Guo, J., Fan, Y., & Cheng, X. (2024). Controlling Risk of Retrieval-Augmented Generation: A Counterfactual Prompting Framework. arXiv:2409.16146.
9. Shi, W., Han, X., Lewis, M., Tsvetkov, Y., Zettlemoyer, L., & Yih, W. (2024). Trusting Your Evidence: Hallucinate Less with Context-Aware Decoding. *NAACL 2024*. arXiv:2305.14739.
10. Liu, J., Wang, R., Zong, Q., et al. (2026). Noise-Aware Verbal Confidence Calibration for Robust Large Language Models in RAG Systems. arXiv:2601.11004.
11. Huang, Z., et al. (2025). QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for Dynamic Retrieval-Augmented Generation. arXiv:2512.19134.
12. Yuan, M., Zhang, S., & Kao, B. (2026). A Context-Aware Dual-Metric Framework for Confidence Estimation in Large Language Models. arXiv:2508.00600.
13. Fadeeva, E., Rubashevskii, A., Piatrashyn, D., Vashurin, R., Dhuliawala, S., Shelmanov, A., Baldwin, T., Nakov, P., Sachan, M., & Panov, M. (2025). Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation. arXiv:2505.21072.
14. Soudani, H., et al. (2025). Uncertainty Quantification for Retrieval-Augmented Reasoning. arXiv:2510.11483.
15. Kwiatkowski, T., et al. (2019). Natural Questions: A Benchmark for Question Answering Research. *TACL*.
16. Joshi, M., Choi, E., Weld, D., & Zettlemoyer, L. (2017). TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. *ACL 2017*.
17. Mallen, A., et al. (2023). When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories. *ACL 2023*. arXiv:2212.10511.
18. Izacard, G., et al. (2022). Unsupervised Dense Information Retrieval with Contrastive Learning. *TMLR*. arXiv:2112.09118.
19. Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
20. Wen, B., et al. (2025). Know Your Limits: A Survey of Abstention in Large Language Models. *TACL*. arXiv:2407.18418.
21. Chen, J., et al. (2025). Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey. arXiv:2503.15850.
