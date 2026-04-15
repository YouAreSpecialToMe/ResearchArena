# SpecCheck: Detecting LLM Hallucinations by Testing Confidence Monotonicity Across Specificity Levels

## Introduction

### Context

Large language models (LLMs) generate fluent, coherent text that often contains a mixture of accurate and fabricated information—so-called hallucinations. A critical observation is that LLMs frequently hallucinate at the level of *specific details* (exact dates, precise numbers, particular names) while retaining approximately correct *general knowledge* (the rough time period, the order of magnitude, the category of entity). For example, a model might correctly associate Einstein with the Nobel Prize in the early 20th century, but hallucinate the exact year or the specific field of the award.

### Problem Statement

Existing hallucination detection methods operate at a single level of specificity: they check whether each atomic claim is factually correct (FActScore, SAFE) or whether sampled outputs are consistent with each other (SelfCheckGPT, semantic entropy). None of these methods exploit the *specificity structure* of claims—the fact that the same piece of information can be expressed at multiple levels of detail, from highly specific to highly abstract.

### Key Insight

If a model truly *knows* a fact, its confidence should be monotonically non-decreasing as the claim becomes more abstract. That is, a model should be at least as confident about "Einstein won the Nobel Prize in the 1920s" as it is about "Einstein won the Nobel Prize in 1921." Violations of this **confidence monotonicity** across a **specificity ladder**—a sequence of the same claim at decreasing levels of detail—serve as a strong signal that the specific claim is hallucinated.

### Hypothesis

Claims for which the model's confidence does *not* increase monotonically as specificity decreases are significantly more likely to be hallucinated than claims for which confidence monotonicity holds. Furthermore, the specificity level at which monotonicity first breaks identifies the granularity at which hallucination occurs.

## Proposed Approach

### Overview

We propose **SpecCheck**, a training-free, model-agnostic framework for claim-level hallucination detection that operates in four stages:

1. **Claim Decomposition**: Decompose an LLM-generated passage into atomic claims (following FActScore).
2. **Specificity Ladder Generation**: For each atomic claim, generate K versions at decreasing specificity levels (from the original claim down to a highly abstract version).
3. **Multi-Granularity Confidence Estimation**: Estimate the model's confidence in each version of the claim using sampling-based consistency checks.
4. **Monotonicity-Based Detection**: Flag claims where confidence does not increase monotonically as specificity decreases.

### Method Details

#### Stage 1: Claim Decomposition

Given a long-form generation (e.g., a biography, a long-form QA answer), we decompose it into atomic claims using an LLM with a structured prompt, following the established protocol from FActScore (Min et al., 2023). Each atomic claim is a single, self-contained factual assertion.

#### Stage 2: Specificity Ladder Generation

For each atomic claim $c_0$ (the original, most specific version), we prompt an LLM to generate a sequence of claims $c_1, c_2, \ldots, c_K$ at decreasing specificity levels. We define specificity levels as:

- **Level 0 (Original)**: The exact claim as stated. E.g., "Marie Curie won the Nobel Prize in Physics in 1903."
- **Level 1 (Approximate)**: Replace precise values with approximate ranges. E.g., "Marie Curie won the Nobel Prize in Physics in the early 1900s."
- **Level 2 (Category)**: Replace specific instances with their category. E.g., "Marie Curie won a Nobel Prize in a science field."
- **Level 3 (Abstract)**: The most general version of the claim. E.g., "Marie Curie won a major scientific award."

We use structured prompts with few-shot examples to ensure consistent ladder generation. We also explore template-based generation for claims with numerical values (dates, quantities) where specificity levels can be mechanically constructed.

#### Stage 3: Multi-Granularity Confidence Estimation

For each claim version $c_k$ at specificity level $k$, we estimate the model's confidence using sampling-based consistency. Specifically, we:

1. Formulate a yes/no question from $c_k$ (e.g., "Is it true that Marie Curie won the Nobel Prize in Physics in 1903?")
2. Sample $N$ responses from the model at temperature $T > 0$
3. Compute the agreement rate: $\text{conf}(c_k) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{response}_i = \text{Yes}]$

Alternatively, when log-probabilities are available (white-box setting), we can use the token probability of "Yes" vs "No" as the confidence estimate, which is more efficient.

#### Stage 4: Monotonicity-Based Detection

Given the confidence sequence $[\text{conf}(c_0), \text{conf}(c_1), \ldots, \text{conf}(c_K)]$, we check for monotonicity violations. Specifically, we define:

- **Monotonicity Score**: $M(c_0) = \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}[\text{conf}(c_k) \geq \text{conf}(c_{k-1})]$
- **SpecCheck Score**: $S(c_0) = 1 - M(c_0) + \alpha \cdot (\text{conf}(c_K) - \text{conf}(c_0))$

where $\alpha$ is a weighting parameter. A high SpecCheck score indicates that confidence does not increase with abstraction (potential hallucination), or that there is a large gap between the most specific and most abstract confidence levels.

We also derive a **Hallucination Granularity Index**: the smallest $k^*$ such that $\text{conf}(c_{k^*}) > \text{conf}(c_0) + \delta$ for threshold $\delta$, identifying the level at which the model becomes confident. This tells users *which level of detail* is unreliable.

### Key Innovations

1. **Specificity-aware probing**: Unlike prior work that checks claims at a single granularity, SpecCheck probes the same information at multiple levels.
2. **Confidence monotonicity as a detection signal**: A principled criterion grounded in the assumption that abstraction should not decrease confidence for truly known facts.
3. **Hallucination granularity diagnosis**: SpecCheck not only detects hallucinations but identifies the level of detail at which hallucination occurs, enabling nuanced trust calibration.
4. **Training-free and model-agnostic**: Works with any LLM via prompting and sampling, requiring no specialized training or internal model access (though can leverage logprobs when available).

## Related Work

### Hallucination Detection

**FActScore** (Min et al., 2023) decomposes long-form text into atomic facts and verifies each against a knowledge source. **SAFE** (Wei et al., 2024) extends this with search-augmented verification. Both rely on external knowledge bases, whereas SpecCheck is self-contained.

**SelfCheckGPT** (Manakul et al., 2023) detects hallucinations by checking consistency across multiple sampled outputs. It computes whether claims in the original response are supported by alternative samples. SpecCheck differs by probing the *same claim at different specificity levels* rather than comparing different complete responses.

**Semantic Entropy** (Kuhn et al., 2023; Farquhar et al., 2024) clusters semantically equivalent responses and computes entropy over meaning clusters. This provides sequence-level or answer-level uncertainty but does not examine how uncertainty varies across specificity levels for the same claim.

### Uncertainty and Calibration

**Confidence Before Answering (CoCA)** (2026) trains models to output confidence before answering, showing improved calibration. **Verbalized confidence** methods elicit numerical confidence scores from LLMs. These operate at the response level; SpecCheck extends self-assessment to the claim level across multiple granularities.

**The Confidence Dichotomy** (2026) shows that tool use (e.g., web search) can induce overconfidence. This highlights the need for tool-independent confidence assessment methods like SpecCheck.

### Claim Decomposition and Verification

**VERISCORE** and **VeriFast** (2025) improve the efficiency of atomic claim verification. **Hierarchical Semantic Pieces (HSP)** (2025) extract multi-granularity semantic pieces for hallucination reduction. SpecCheck is complementary: rather than verifying against external sources, it uses the model's own multi-granularity confidence pattern as a detection signal.

### Step-Back Prompting and Abstraction

**Step-Back Prompting** (Zheng et al., 2023) uses abstraction to improve reasoning by first answering a more general question. SpecCheck applies a similar abstraction idea but for *detection* rather than *generation*, and does so systematically across multiple levels.

### How SpecCheck Differs

| Method | Granularity | External Knowledge | Multi-Level | Training-Free |
|--------|------------|-------------------|-------------|---------------|
| FActScore | Claim-level | Required | No | Yes |
| SelfCheckGPT | Sentence-level | Not required | No | Yes |
| Semantic Entropy | Answer-level | Not required | No | Yes |
| Verbalized Confidence | Answer-level | Not required | No | Yes |
| **SpecCheck (ours)** | **Claim-level** | **Not required** | **Yes** | **Yes** |

## Experiments

### Experimental Setup

**Models**: We evaluate on open-source models that fit on a single A6000 (48GB):
- Llama-3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3
- Qwen2.5-7B-Instruct

These represent three major model families and allow both black-box (sampling) and white-box (logprob) evaluation.

**Benchmarks**:
1. **FActScore Biography Generation**: Generate biographies for entities in the FActScore benchmark. Ground truth factuality labels are available from the original FActScore dataset.
2. **LongFact** (Wei et al., 2024): Long-form factuality benchmark spanning 38 topics.
3. **TruthfulQA** (Lin et al., 2022): Evaluates truthfulness in short-form generation.

**Baselines**:
- **SelfCheckGPT** (Manakul et al., 2023): Sample multiple responses, check cross-sample consistency.
- **Verbalized Confidence**: Directly ask the model "How confident are you (0-100) in this claim?"
- **Logprob Confidence**: Average token log-probability of the claim.
- **Semantic Entropy Probes** (Kossen et al., 2024): Use hidden-state probes to estimate semantic entropy.
- **Random baseline**: Randomly assign hallucination scores.

**Metrics**:
- AUC-ROC and AUC-PR for binary hallucination detection (claim is factual vs. hallucinated)
- Expected Calibration Error (ECE) for confidence calibration
- Pearson/Spearman correlation between SpecCheck scores and ground-truth factuality

### Core Experiments

**Experiment 1: Hallucination Detection Performance**
Compare SpecCheck against all baselines on all three benchmarks. Report AUC-ROC, AUC-PR per model and dataset. Hypothesis: SpecCheck achieves higher AUC-PR than baselines, especially for numerically specific claims.

**Experiment 2: Claim Type Analysis**
Categorize claims by type (numerical facts, temporal facts, entity relations, categorical facts) and evaluate detection performance per type. Hypothesis: SpecCheck excels on claims with clear specificity structure (numerical, temporal) and is competitive on other types.

**Experiment 3: Ablation on Ladder Depth**
Evaluate with K=1, 2, 3, 4 specificity levels. Hypothesis: K=3 provides the best cost-accuracy tradeoff; diminishing returns beyond K=3.

**Experiment 4: Confidence Estimation Method Comparison**
Compare sampling-based confidence (N=5, 10, 20 samples) vs. logprob-based confidence. Hypothesis: Logprob-based is more efficient with similar accuracy; sampling-based is more robust across models.

**Experiment 5: Combination with Existing Methods**
Combine SpecCheck scores with SelfCheckGPT and verbalized confidence as features in a simple logistic regression. Hypothesis: SpecCheck provides complementary signal, improving detection when combined.

### Expected Results

We expect SpecCheck to:
1. Outperform baselines on claims involving specific details (dates, numbers, names) by 5-15% AUC-PR
2. Be competitive with SelfCheckGPT on general claims
3. Provide useful hallucination granularity information not available from other methods
4. Show that confidence monotonicity is a strong prior: >90% of verified-true claims exhibit monotonic confidence across the specificity ladder, while <60% of hallucinated claims do

### Computational Budget

Each experiment involves generating claims, creating specificity ladders, and running confidence estimation:
- Biography generation + claim decomposition: ~1 hour per model
- Specificity ladder generation: ~0.5 hours per model
- Confidence estimation (sampling-based, N=10): ~2 hours per model
- Total per model: ~3.5 hours, running 3 models sequentially: ~10.5 hours
- Optimization: Run confidence estimation in batched inference to reduce time to ~1.5 hours per model
- Total estimated compute: ~6-7 hours on 1× A6000

## Success Criteria

### Confirmation of Hypothesis
1. **Primary**: SpecCheck achieves statistically significantly higher AUC-PR than the best baseline on at least 2 of 3 benchmarks.
2. **Secondary**: Confidence monotonicity holds for >85% of verified-true claims and is violated for >40% of hallucinated claims.
3. **Tertiary**: The Hallucination Granularity Index correctly identifies the level of detail at which hallucination occurs in >70% of analyzed cases.

### Refutation
The hypothesis would be refuted if:
- Confidence does not increase with abstraction even for true claims (monotonicity is not a valid prior)
- SpecCheck performs no better than random or significantly worse than SelfCheckGPT
- The specificity ladder generation is too noisy to produce reliable signal

## References

1. Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P.W., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. *EMNLP 2023*.

2. Manakul, P., Liusie, A., & Gales, M.J. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. *EMNLP 2023*.

3. Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature*, 630(8017), 625-630.

4. Wei, J., Yang, C., Song, X., Lu, Y., Hu, N., Tran, D., Peng, D., Liu, R., Huang, D., Du, C., & Le, Q.V. (2024). Long-form factuality in large language models. *NeurIPS 2024*.

5. Kossen, J., Han, J., Kuhn, L., Gal, Y., & Farquhar, S. (2024). Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. *arXiv:2406.15927*.

6. Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL 2022*.

7. Zheng, H.S., Mishra, S., Chen, X., Cheng, H.-T., Chi, E.H., Le, Q.V., & Zhou, D. (2023). Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models. *ICLR 2024*.

8. Varshney, D., Zafar, H., Mishra, A., & Gupta, C. (2025). Investigating and Addressing Hallucinations of LLMs in Tasks Involving Negation. *TrustNLP Workshop 2025*.

9. Azaria, A. & Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying. *EMNLP 2023 Findings*.

10. Kadavath, S., Conerly, T., Askell, A., Henighan, T., et al. (2022). Language Models (Mostly) Know What They Know. *arXiv:2207.05221*.

11. Guo, Z., Ding, L., Hu, B., & Tao, D. (2025). Uncertainty Profiles: Decomposing Uncertainty in LLM Outputs. *arXiv:2505.xxxxx*.

12. Wang, X., Wei, J., Schuurmans, D., Le, Q.V., Chi, E.H., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.
