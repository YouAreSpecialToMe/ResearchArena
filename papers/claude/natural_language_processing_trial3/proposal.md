# Know When to Look: Parametric-Retrieval Agreement as a Calibration Signal for Retrieval-Augmented Generation

## Introduction

Retrieval-Augmented Generation (RAG) has become the dominant paradigm for building knowledge-intensive NLP systems, enabling large language models (LLMs) to ground their responses in external evidence. By retrieving relevant passages from a knowledge base and conditioning generation on them, RAG systems substantially reduce hallucinations and improve factual accuracy compared to parametric-only LLMs (Lewis et al., 2020; Guu et al., 2020).

However, a critical yet underexplored question remains: **does retrieval actually improve the model's ability to know when it is right or wrong?** Confidence calibration -- the alignment between a model's expressed or implicit confidence and its actual correctness -- is essential for safe deployment, selective prediction (knowing when to abstain), and downstream decision-making. While extensive work has studied calibration in standard LLMs (Kadavath et al., 2022; Kuhn et al., 2023), the effect of retrieval augmentation on calibration has received surprisingly little systematic attention.

We identify a key observation: when an LLM is presented with retrieved context, it tends to become overconfident in its answers regardless of whether the retrieval was helpful or misleading (Xie et al., 2024; Ozaki et al., 2025). The URAG benchmark (Nguyen et al., 2026) recently confirmed this at scale, showing that retrieval depth and confidence cues can amplify confident errors and hallucinations across domains. This creates a dangerous failure mode where the model is confidently wrong precisely when retrieval introduces noise or contradictory information.

**Key Insight.** We propose that the *agreement* between an LLM's parametric answer (generated without retrieval) and its retrieval-augmented answer provides a powerful calibration signal for *post-generation confidence estimation*. When both knowledge sources converge on the same answer, confidence should be high. When they diverge -- indicating either a knowledge conflict, noisy retrieval, or genuine uncertainty -- confidence should be low.

**Hypothesis.** The Parametric-Retrieval Agreement (PRA) score captures a complementary uncertainty signal that, when combined with existing confidence measures, yields state-of-the-art selective prediction performance for RAG systems.

### Relationship to PAIRS and Prior Work on Parametric-Retrieval Comparison

Recent concurrent work has explored comparing parametric and retrieval-augmented outputs. Most notably, PAIRS (Li et al., 2025) uses a parametric-verified dual-generation strategy where agreement between direct and pseudo-context-augmented answers determines whether to bypass retrieval entirely. SUGAR (Zubkova et al., 2025) uses semantic entropy from parametric generation as a retrieval trigger. Self-Routing RAG (Zhao et al., 2025) casts retrieval as knowledge source selection within a single forward pass.

Our work differs from all of these in a fundamental way: **we target post-generation calibration and selective prediction, not retrieval triggering or routing**. PAIRS, SUGAR, and Self-Routing RAG ask "should I retrieve?" *before* generation; PRA-Score asks "how confident should I be in this RAG answer?" *after* generation. This distinction matters because:

1. **Different optimization target**: PAIRS optimizes retrieval efficiency (reducing unnecessary retrievals); PRA-Score optimizes answer reliability (identifying when the RAG answer is wrong).
2. **Complementary rather than competing**: In a deployed RAG system, PAIRS could decide *whether* to retrieve, and PRA-Score could then assess *confidence* in the final answer -- the two mechanisms serve different stages of the pipeline.
3. **Different signals**: PAIRS compares a direct answer with a pseudo-context answer (both parametric); PRA-Score compares a parametric answer with a *real* retrieval-augmented answer, capturing actual retrieval quality effects.

### Comparison Table: How PRA-Score Differs from Related Methods

| Method | Target Task | Pipeline Stage | Required Access | Training Required | Signal Type |
|--------|-------------|----------------|-----------------|-------------------|-------------|
| **PAIRS** (Li et al., 2025) | Retrieval triggering | Pre-retrieval | Output only | None | Parametric vs. pseudo-context agreement |
| **SUGAR** (Zubkova et al., 2025) | Retrieval triggering | Pre-retrieval | Output only | None | Semantic entropy of parametric outputs |
| **Self-Routing RAG** (Zhao et al., 2025) | Retrieval routing | Pre-retrieval | Weights (fine-tuned) | SFT on routing data | Source selection logits |
| **CBDR** (Ren et al., 2025) | Dynamic retrieval & reranking | Pre/post-retrieval | Hidden states | None | Hidden-state confidence delta |
| **NAACL** (Liu et al., 2026) | Verbal confidence calibration | Post-generation | Weights (fine-tuned) | SFT on ~2K examples | Noise-aware verbal confidence |
| **PRA-Score** (Ours) | Selective prediction / calibration | Post-generation | Output only | None (individual) / ~500 labels (combined) | Parametric vs. real-RAG output agreement |

Key differentiators of PRA-Score:
- **Only method targeting post-generation selective prediction** using parametric-retrieval comparison
- **Black-box compatible**: works with API-only models (unlike CBDR, Self-Routing RAG, NAACL)
- **Individual signals are fully training-free** (the meta-predictor combination step uses ~500 labels, analogous to Platt scaling)

## Proposed Approach

### Overview

We introduce **PRA-Score** (Parametric-Retrieval Agreement Score), a confidence metric for RAG systems that requires no fine-tuning of the underlying LLM. The method works as follows:

1. **Parametric Generation**: Given a question q, generate an answer a_p using the LLM without any retrieved context
2. **Retrieval-Augmented Generation**: Retrieve top-k passages from a knowledge base and generate an answer a_r conditioned on the retrieved context
3. **Agreement Computation**: Compute PRA(q) = sim(a_p, a_r), where sim is a semantic similarity measure
4. **Confidence Score**: Use PRA(q) as a confidence score for the RAG answer a_r

### On the "Training-Free" Claim

We clarify the scope of this claim: the individual PRA-Score signals (EM, F1, NLI variants) are fully training-free -- they require no labeled data, no fine-tuning, and no learned parameters beyond pre-existing models (e.g., an off-the-shelf NLI classifier). The combination step (Experiment 3), which fits a logistic regression meta-predictor on ~500 labeled examples, is a lightweight supervised calibration step. This is analogous to how Platt scaling or temperature scaling are used to calibrate any classifier -- the underlying signal is training-free, but optimal combination of multiple signals benefits from a small calibration set. We report both standalone PRA-Score results (fully training-free) and combined results (with lightweight calibration) to make this distinction transparent.

### Agreement Measures

We investigate three agreement measures of increasing sophistication:

- **Exact Match (PRA-EM)**: Binary signal -- do the parametric and retrieval answers match exactly? Simple but limited to short-form QA.
- **Token F1 (PRA-F1)**: Unigram overlap between a_p and a_r, capturing partial agreement for longer answers.
- **NLI-Based (PRA-NLI)**: Use an off-the-shelf NLI model (DeBERTa-v3-large fine-tuned on MNLI) to compute the probability that a_p entails a_r. This captures semantic agreement even when surface forms differ.
- **Token Probability Delta (PRA-TPD)**: Compute the difference in mean token log-probability between the parametric and RAG answers. This bridges toward CBDR's hidden-state approach while remaining output-accessible (requires logprob access but not hidden states). Available for open-weight models and some APIs (OpenAI, Anthropic).

### Combination with Existing Signals

PRA-Score captures a fundamentally different type of uncertainty than existing methods:

- **Token probability** measures the model's internal confidence during generation
- **Verbalized confidence** measures the model's self-assessed certainty
- **Self-consistency** measures agreement across multiple samples from the same distribution
- **PRA-Score** measures agreement across two different *knowledge sources* (parametric vs. external)

We combine PRA-Score with existing signals via a simple logistic regression meta-predictor trained on a small validation set (~500 examples), testing whether the combination outperforms any individual signal. This lightweight calibration step is standard practice in selective prediction (Kamath et al., 2020) and does not involve any fine-tuning of the LLM itself.

### Inference Cost Analysis

PRA-Score requires two full forward passes per query: one parametric and one retrieval-augmented. This approximately doubles inference cost (ignoring retrieval latency, which is already incurred in a RAG system). We address this limitation honestly:

**When the extra pass is worth the cost:**
- **High-stakes applications** (medical QA, legal, financial) where a wrong answer is far costlier than doubled latency
- **Selective prediction pipelines** where the system must decide whether to answer or abstain -- the cost of a bad answer (user harm, loss of trust) far exceeds one extra generation
- **Batch processing** scenarios where latency is less critical than reliability
- **Systems already running multiple samples** (e.g., self-consistency with 5 samples) -- PRA adds only 1 extra pass vs. self-consistency's 4 extra passes

**When it is NOT worth the cost:**
- **Latency-critical real-time applications** (chatbots with <500ms SLA) where doubling generation time is prohibitive
- **Simple factoid questions** where the model is almost always correct and calibration adds little value
- **Cost-sensitive high-throughput systems** where even small per-query cost increases matter at scale

We report latency measurements alongside accuracy metrics so practitioners can make informed cost-benefit decisions.

### Key Contributions

1. **A new confidence signal for RAG**: PRA-Score is the first method to use parametric-retrieval agreement specifically for post-generation calibration and selective prediction. Unlike PAIRS (which targets retrieval efficiency), SUGAR (which targets retrieval triggering), or NAACL (which requires fine-tuning), PRA-Score targets answer reliability assessment while remaining training-free at the individual signal level.

2. **Model-agnostic and lightweight**: Unlike Self-RAG (Asai et al., 2024) which requires fine-tuning with special reflection tokens, CBDR (Ren et al., 2025) which requires access to hidden states, or NAACL (Liu et al., 2026) which requires supervised fine-tuning on ~2K examples, PRA-Score works with any LLM including black-box APIs. The additional cost is one extra forward pass per query.

3. **Diagnostic value**: PRA-Score not only provides calibration but also diagnoses *why* the model might be wrong -- when PRA is low, it indicates knowledge conflicts, retrieval noise, or knowledge gaps -- enabling targeted mitigation.

4. **Scale analysis**: We test across model scales (7B to 70B) to understand whether PRA-Score's discriminative power degrades as parametric knowledge improves with scale.

## Related Work

### Retrieval-Augmented Generation
RAG systems ground LLM generation in retrieved external knowledge (Lewis et al., 2020). Active retrieval methods like FLARE (Jiang et al., 2023) trigger retrieval based on generation confidence. Self-RAG (Asai et al., 2024) trains models with reflection tokens for self-assessment. Adaptive-RAG (Jeong et al., 2024) uses a classifier to route queries by complexity. Self-Routing RAG (Zhao et al., 2025) casts selective retrieval as knowledge source selection within a single left-to-right pass, learning to verbalize parametric knowledge when retrieval is unnecessary.

### Parametric vs. Retrieved Knowledge Comparison
PAIRS (Li et al., 2025) introduces parametric-verified dual-generation, comparing a direct answer with a pseudo-context-augmented answer to decide whether to bypass retrieval. If the answers agree, retrieval is skipped. SUGAR (Zubkova et al., 2025) uses semantic entropy from parametric generation as a retrieval trigger. Both methods target retrieval efficiency (reducing unnecessary retrievals). CBDR (Ren et al., 2025) uses hidden-state confidence deltas to measure whether retrieval enhanced confidence, but requires access to model internals. Our work uses the parametric-retrieval comparison for a distinct purpose: assessing confidence in the *final* RAG answer for selective prediction. PRA-Score compares against a *real* retrieval-augmented answer (not a pseudo-context), targets calibration metrics (AUROC, ECE) rather than retrieval rates, and works at the output level (black-box compatible).

### Uncertainty Estimation in LLMs
Token probability-based methods use sequence likelihood or entropy as confidence measures (Kadavath et al., 2022). Semantic entropy (Kuhn et al., 2023) clusters sampled answers by meaning. Self-consistency (Wang et al., 2023) uses majority voting across multiple reasoning chains. Confidence-Informed Self-Consistency (CISC) (Taubenfeld et al., 2025) weights chains by confidence scores to improve majority voting efficiency. SelfCheckGPT (Manakul et al., 2023) detects hallucinations via multi-sample consistency. These methods operate within a single generation mode; PRA-Score uniquely compares across two modes (with/without retrieval).

### Uncertainty Benchmarking and Calibration in RAG
URAG (Nguyen et al., 2026) provides the first comprehensive benchmark for uncertainty quantification in RAG systems, evaluating 8 RAG methods across 5 domains. Their key finding -- that simple modular RAG methods achieve better accuracy-uncertainty trade-offs than complex pipelines, and that retrieval can amplify overconfident errors -- directly motivates our work. PRA-Score provides a practical signal for detecting exactly these overconfident retrieval failures. NAACL (Liu et al., 2026) proposes noise-aware verbal confidence calibration for RAG, using supervised fine-tuning on ~2K examples to teach models intrinsic noise awareness. Unlike NAACL, PRA-Score requires no fine-tuning and works with black-box models. Ozaki et al. (2025) confirm that RAG increases overconfidence in the medical domain, finding that certain models can judge whether retrieved documents relate to the correct answer -- our PRA-Score formalizes this intuition as a general-purpose calibration signal.

### Knowledge Conflicts in RAG
When retrieved content contradicts parametric knowledge, LLMs face knowledge conflicts (Xu et al., 2024). Recent work detects conflicts in retrieved documents (Xie et al., 2024) or uses hidden states to measure confidence enhancement from retrieval (Ren et al., 2025). Our approach is simpler -- it compares output-level answers -- and targets calibration rather than conflict resolution.

### Selective Prediction in NLP
Selective prediction allows models to abstain on uncertain inputs to maintain high precision on answered questions (Geifman & El-Yaniv, 2017; Kamath et al., 2020). The abstention survey by Wen et al. (2025) identifies three perspectives on abstention. Our work contributes a new confidence signal specifically designed for the RAG setting.

## Experiments

### Datasets
- **Natural Questions (NQ)** (Kwiatkowski et al., 2019): Open-domain QA with Wikipedia-derived questions. We use the open-domain test split (~3,610 questions).
- **TriviaQA** (Joshi et al., 2017): Trivia questions requiring broad knowledge. We use the unfiltered test set (~11,313 questions).
- **PopQA** (Mallen et al., 2023): Entity-centric QA with varying popularity, enabling analysis by entity frequency. We use the full set (~14,267 questions).

### Models
- **Llama-3-8B-Instruct** (Meta, 2024): State-of-the-art open-weight instruction-tuned model (8B parameters, fits in 16-bit on A6000)
- **Mistral-7B-Instruct-v0.3** (Mistral AI, 2024): Competitive open-weight model with different training recipe (7B parameters, fits in 16-bit on A6000)
- **Llama-3-70B-Instruct** (Meta, 2024): Larger model loaded via 4-bit quantization (GPTQ/AWQ, ~35GB VRAM) to test whether PRA-Score's discriminative power degrades when parametric knowledge is substantially stronger

The 70B model serves as a critical test: if PRA-Score works well at 7-8B but loses discriminative power at 70B (where the model more often knows the answer parametrically), this reveals an important limitation. Conversely, if it remains effective, it demonstrates robustness across scales.

### Retrieval Setup
- **Retriever**: Contriever (Izacard et al., 2022) fine-tuned on MS MARCO
- **Knowledge Base**: Wikipedia (Dec 2021 dump, following standard practice)
- **Top-k passages**: k=5, with 100-word passages
- We also test BM25 as an alternative retriever to assess robustness

### Baselines for Confidence/Calibration

1. **P(answer)**: Normalized sequence probability of the generated answer tokens
2. **Entropy**: Mean token entropy during generation
3. **Verbalized Confidence (VC)**: Prompt the model to output a confidence score (0-100)
4. **Self-Consistency (SC)**: Agreement rate across 5 sampled generations (temperature=0.7)
5. **SelfCheckGPT-NLI**: NLI-based self-consistency across 5 samples
6. **Token Probability Delta (TPD)**: Difference in mean token log-probability between parametric and RAG passes (without output-level agreement -- isolates the probability signal from the agreement signal)

### Metrics

- **AUROC**: Area under ROC curve for predicting answer correctness from confidence. Higher = better at distinguishing correct from incorrect answers.
- **AUPRC**: Area under precision-recall curve (important for imbalanced scenarios).
- **ECE**: Expected Calibration Error (10 bins). Lower = better calibrated.
- **Selective Accuracy @ Coverage X%**: Accuracy when the model answers only its top X% most confident predictions.
- **Latency**: Wall-clock time per query for each method (to support cost-benefit analysis).

### Experiment Plan

**Exp 1: RAG Calibration Analysis** (Diagnostic)
- Compare calibration metrics (ECE, AUROC) for parametric-only vs. RAG across all datasets and models
- Test hypothesis H1: RAG degrades calibration, especially when retrieval is noisy
- Connect findings to URAG's observation that retrieval amplifies overconfident errors

**Exp 2: PRA-Score Evaluation** (Main Result)
- Compare PRA-Score (EM, F1, NLI, TPD variants) against all baselines on AUROC, AUPRC
- Test hypothesis H2: PRA-Score provides a strong individual calibration signal
- Report results for all three model scales (7B, 8B, 70B)
- Include the TPD baseline to bridge toward CBDR's hidden-state approach

**Exp 3: Signal Combination** (Main Result)
- Train a logistic regression meta-predictor on a held-out validation set (~500 examples) combining PRA-Score with each baseline signal
- Test hypothesis H3: PRA-Score is complementary to existing methods
- Note: This is the only supervised step; individual PRA-Score signals in Exp 2 are fully training-free

**Exp 4: Analysis by Entity Popularity** (Ablation)
- Using PopQA's entity frequency annotations, analyze PRA-Score effectiveness for rare vs. common entities
- Hypothesis: PRA is most informative for medium-popularity entities (where parametric knowledge is partial)

**Exp 5: Scale Analysis** (Ablation)
- Compare PRA-Score effectiveness across 7B, 8B, and 70B models
- Hypothesis: PRA-Score discriminative power may decrease for 70B (stronger parametric knowledge leads to more agreement cases), but the signal remains useful when combined with other features
- Report the distribution of agreement/disagreement cases across scales

**Exp 6: Cost-Benefit and Further Ablations**
- Report latency per query for each method (PRA-Score, SC, SelfCheckGPT, baselines)
- Compute a cost-normalized metric: AUROC improvement per additional inference pass
- Effect of retriever quality (Contriever vs. BM25)
- Effect of number of retrieved passages (k=1,3,5,10)
- Effect of agreement measure (EM vs. F1 vs. NLI vs. TPD)
- Sensitivity to generation temperature

### Expected Results

We expect:
1. RAG improves accuracy but degrades calibration, especially for questions where retrieval is misleading (consistent with URAG findings)
2. PRA-Score (NLI variant) achieves competitive or superior AUROC compared to token probability and verbalized confidence baselines
3. PRA-Score captures complementary information: combining PRA + P(answer) outperforms either alone
4. PRA-Score is most informative for medium-popularity entities and when retrieval quality varies
5. At 70B scale, the agreement rate increases (more parametric answers are correct), potentially reducing PRA-Score's standalone discriminative power but maintaining complementarity when combined
6. PRA-Score (1 extra pass) provides better AUROC-per-pass than self-consistency (4 extra passes)

### Computational Budget
- 7B/8B models: ~30K questions x 2 passes x 2 models = ~120K generations, ~0.5s each = ~17 hours -> subsample to 5K per dataset for main experiments (~5 hours)
- 70B model (4-bit): ~5K questions (NQ subset), ~1.5s per generation = ~4 hours
- NLI model (DeBERTa-v3-large): negligible overhead
- Self-consistency baselines: 5 samples per question (batched)
- Total estimated compute: ~7-8 hours on 1x A6000

## Success Criteria

**Strong confirmation**: PRA-Score (NLI variant) achieves top-2 AUROC among all individual confidence signals on at least 2/3 datasets, AND combining PRA with the best existing signal improves AUROC by >= 2 points.

**Moderate confirmation**: PRA-Score captures complementary information (combination improves over best individual signal) even if PRA alone is not the strongest signal.

**Refutation**: PRA-Score provides no additional information beyond what existing methods capture, or the parametric-retrieval agreement is too noisy to be useful.

## References

- Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR 2024.
- Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. NeurIPS 2017.
- Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M.W. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. ICML 2020.
- Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., & Grave, E. (2022). Unsupervised Dense Information Retrieval with Contrastive Learning. TMLR 2022.
- Jeong, S., Baek, J., Cho, S., Hwang, S.J., & Park, J.C. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. NAACL 2024.
- Jiang, Z., Xu, F.F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y., Callan, J., & Neubig, G. (2023). Active Retrieval Augmented Generation. EMNLP 2023.
- Joshi, M., Choi, E., Weld, D.S., & Zettlemoyer, L. (2017). TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. ACL 2017.
- Kadavath, S., Conerly, T., Askell, A., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Kamath, A., Jia, R., & Liang, P. (2020). Selective Question Answering under Domain Shift. ACL 2020.
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. ICLR 2023.
- Kwiatkowski, T., Palomaki, J., Redfield, O., et al. (2019). Natural Questions: A Benchmark for Question Answering Research. TACL 2019.
- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
- Li, Y., et al. (2025). PAIRS: Parametric-Verified Adaptive Information Retrieval and Selection for Efficient RAG. arXiv:2508.04057.
- Liu, J., Wang, R., Zong, Q., Wang, Y., Qian, C., Zeng, Q., Zheng, T., Shi, H., Guo, D., Xu, B., Li, C., & Song, Y. (2026). NAACL: Noise-AwAre Verbal Confidence Calibration for LLMs in RAG Systems. arXiv:2601.11004.
- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H. (2023). When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories. ACL 2023.
- Manakul, P., Liusie, A., & Gales, M.J.F. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. EMNLP 2023.
- Nguyen, V., Dang, C., Zhang, J., Tran, H., Tran, M., Chau, T., Le, T., Cheng, L., & Wang, S. (2026). URAG: A Benchmark for Uncertainty Quantification in Retrieval-Augmented Large Language Models. arXiv:2603.19281.
- Ozaki, S., Kato, Y., Feng, S., Tomita, M., Hayashi, K., Hashimoto, W., Obara, R., Oyamada, M., Hayashi, K., Kamigaito, H., & Watanabe, T. (2025). Understanding the Impact of Confidence in Retrieval Augmented Generation: A Case Study in the Medical Domain. BioNLP Workshop @ ACL 2025.
- Ren, R., Wang, Y., & Qu, Y. (2025). Rethinking LLM Parametric Knowledge as Post-retrieval Confidence for Dynamic Retrieval and Reranking (CBDR). arXiv:2509.06472.
- Taubenfeld, A., Sheffer, T., Ofek, E., Feder, A., Goldstein, A., Gekhman, Z., & Yona, G. (2025). Confidence Improves Self-Consistency in LLMs. ACL Findings 2025.
- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.
- Wen, B., et al. (2025). Know Your Limits: A Survey of Abstention in Large Language Models. TACL 2025.
- Xie, J., Zhang, K., Chen, J., Lou, R., & Su, Y. (2024). Knowledge Conflicts for LLMs: A Survey. EMNLP 2024.
- Xu, Z., Jiang, Z., & Neubig, G. (2024). ConflictBank: A Benchmark for Evaluating Knowledge Conflicts in LLMs. NeurIPS 2024 Datasets and Benchmarks Track.
- Zhao, R., et al. (2025). Self-Routing RAG: Binding Selective Retrieval with Knowledge Verbalization. arXiv:2504.01018.
- Zubkova, H., Park, J.-H., & Lee, S.-W. (2025). SUGAR: Leveraging Contextual Confidence for Smarter Retrieval. arXiv:2501.04899.
