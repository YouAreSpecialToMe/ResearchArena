# ConsistBench: A Cross-Format, Cross-Phrasing Consistency Benchmark for Large Language Models

## Introduction

### Context and Problem Statement

Large language models (LLMs) are increasingly deployed in high-stakes applications where reliability is paramount. A fundamental requirement for reliable AI systems is **consistency**: given two semantically equivalent inputs, the model should produce semantically equivalent outputs. Yet mounting evidence suggests that LLMs are surprisingly fragile to superficial input variations. Mizrahi et al. (2024) showed that the same model can rank as best or worst depending on which semantically equivalent instruction template is used. Lunardi et al. (2025) found significant score drops when benchmark questions are paraphrased. Choi (2025) demonstrated inconsistent answers to paraphrased multiple-choice questions.

Despite this growing concern, the field lacks a **comprehensive, systematic benchmark** that evaluates consistency along multiple complementary axes. Existing work is fragmented:

- **ParaRel** (Elazar et al., 2021) tests paraphrase consistency but only for masked language models on cloze-style knowledge probing — not generative LLMs.
- **BeHonest** (Chern et al., 2024) includes consistency as one of three pillars but tests format perturbation and social pressure, not systematic paraphrase consistency.
- **RoParQ** (Choi, 2025) tests paraphrase consistency but only for closed-book MCQ, with a single aggregate metric and no analysis of *which types* of paraphrases cause failures.
- **OSQ-bench** (Myrzakhan et al., 2024) converts MCQ benchmarks to open-ended format and measures the performance gap across 40+ LLMs, finding an average 25% accuracy drop. However, OSQ-bench focuses solely on *accuracy loss* when switching from MCQ to open-ended format — it does not measure *per-question consistency* (whether a model gives the same answer to the same question in different formats), does not test beyond the MCQ-to-open binary, and does not evaluate paraphrase robustness at all.
- **Lunardi et al. (2025)** study paraphrasing effects on benchmarks but focus on evaluation methodology, not releasing a dedicated consistency benchmark.

Critically, **no existing benchmark jointly evaluates cross-format consistency and paraphrase-type-stratified consistency**. OSQ-bench demonstrated that the MCQ-to-open accuracy gap exists, but it cannot diagnose *which specific questions* a model answers inconsistently across formats, *which paraphrase types* are most destabilizing, or *which knowledge domains* exhibit the worst consistency. ConsistBench fills this gap by making per-question, per-format, per-paraphrase-type consistency the primary evaluation target.

### Key Insight

Consistency is not a single phenomenon but a **multi-dimensional property** that should be evaluated along at least three orthogonal axes:

1. **Cross-format consistency**: Does the model give the same answer when asked via MCQ vs. open-ended vs. yes/no vs. true/false vs. fill-in-the-blank? OSQ-bench showed the aggregate accuracy gap; we measure item-level agreement across 5 formats.
2. **Cross-phrasing consistency**: Does the model give the same answer when the question is paraphrased? And crucially, *which types* of paraphrases (lexical, syntactic, formality, voice, negation-framing) cause the most failures?
3. **Domain-stratified consistency**: Is consistency uniform across knowledge domains (science, history, commonsense, mathematics) and difficulty levels, or do certain domains expose worse inconsistency?

### Hypothesis

We hypothesize that: (1) LLMs exhibit significant cross-format inconsistency, with item-level answer agreement between MCQ and open-ended formats substantially below 100% even for questions answered correctly in at least one format; (2) different paraphrase types cause non-uniform consistency degradation, with negation-framing and syntactic restructuring causing larger drops than lexical substitution; (3) consistency varies significantly across knowledge domains, with domains requiring precise factual recall (e.g., science, history) showing worse consistency than commonsense reasoning; and (4) larger models show higher consistency, but the gap between accuracy and consistency remains substantial even for frontier-scale models.

## Proposed Approach

### Overview

We propose **ConsistBench**, a benchmark containing ~1,000 base questions, each expanded into variants along format and phrasing axes, yielding ~8,000–10,000 total evaluation instances. We evaluate 6–7 LLMs spanning different model families and sizes (including at least one ~70B quantized model), and report both standard accuracy and a suite of novel consistency metrics.

### Dataset Construction

#### Source Questions
We curate ~1,000 questions from established, openly available QA datasets spanning 6 knowledge domains:
- **Science** (MMLU science subsets, ARC-Challenge)
- **History & Social Science** (MMLU humanities subsets)
- **Mathematics** (GSM8K, MATH)
- **Commonsense Reasoning** (CommonsenseQA, HellaSwag)
- **World Knowledge** (TriviaQA, Natural Questions)
- **Logic & Reasoning** (LogiQA, BBH logical deduction)

We select questions with unambiguous, verifiable answers to ensure consistency can be objectively measured. Questions are stratified by difficulty (easy/medium/hard based on human or prior model performance), with ~165 questions per domain.

#### Format Variants (Cross-Format Axis)
For each base question, we create 5 format variants:
1. **Multiple-choice (4 options)**: Standard A/B/C/D format
2. **Open-ended**: Free-form question expecting a short answer
3. **Yes/No**: Reformulated as a binary verification question (e.g., "Is the capital of France Paris?")
4. **True/False**: Presented as a statement to verify
5. **Fill-in-the-blank**: Cloze-style with the answer removed

Format conversion is done programmatically with manual verification on a random 10% sample (~100 questions) to ensure semantic equivalence.

#### Phrasing Variants (Cross-Phrasing Axis)
For a subset of ~300 questions, we generate paraphrase variants across 6 controlled paraphrase types:
1. **Lexical substitution**: Replace content words with synonyms ("What is the capital of France?" → "What is the principal city of France?")
2. **Syntactic restructuring**: Change sentence structure ("What year did WWII end?" → "In what year was WWII concluded?")
3. **Voice change**: Active to passive or vice versa ("Who wrote Hamlet?" → "By whom was Hamlet written?")
4. **Formality shift**: Formal to informal or vice versa ("What is the atomic number of carbon?" → "Hey, what's carbon's atomic number?")
5. **Negation framing**: Reframe using negation ("Which planet is closest to the sun?" → "Which planet is not farther from the sun than any other?")
6. **Elaborative rephrasing**: Add contextual detail that doesn't change the question ("Given that elements are organized by atomic number, what is the atomic number of carbon?")

Paraphrases are generated using a strong LLM (GPT-4 or Claude) with careful prompting, then validated for semantic equivalence by a second LLM pass and manual spot-checking of at least 10% of generated paraphrases (~180 paraphrases checked manually).

### Consistency Metrics

We define a suite of metrics capturing different facets of consistency:

1. **Cross-Format Agreement (CFA)**: For each question, the fraction of format pairs where the model gives the same answer. Averaged across all questions.

   CFA(q) = (number of format pairs with matching answers) / (total format pairs)

2. **Cross-Phrasing Agreement (CPA)**: For each question, the fraction of phrasing variant pairs with matching answers, stratified by paraphrase type.

   CPA(q, type) = (matching pairs within paraphrase type) / (total pairs)

3. **Format-Conditional Accuracy Gap (FCAG)**: The difference between accuracy in the "easiest" format (typically MCQ) and the "hardest" format (typically open-ended), among questions answered correctly in at least one format.

4. **Consistency-Accuracy Ratio (CAR)**: consistency / accuracy. A model with 80% accuracy but only 60% consistency has a CAR of 0.75, indicating 25% of its "knowledge" is format/phrasing-dependent.

5. **Paraphrase Fragility Index (PFI)**: Per paraphrase type, the fraction of questions where the model changes its answer. Higher = more fragile to that paraphrase type.

6. **Domain Consistency Score (DCS)**: CFA and CPA stratified by knowledge domain, enabling identification of consistency deserts.

### Answer Equivalence Evaluation

For open-ended and fill-in-the-blank formats, answers cannot be compared by exact string match. We use a two-stage evaluation:
1. **Rule-based matching**: Normalize answers (lowercase, strip articles, handle number formats) and check exact/fuzzy match (token overlap, edit distance).
2. **LLM-based equivalence judge**: For non-matching answers after stage 1, use an LLM judge to determine semantic equivalence (e.g., "Paris" vs. "The capital is Paris" vs. "It's Paris, France"). We validate the judge against human annotations on a held-out set of ~200 answer pairs to ensure reliability, following best practices from Ye et al. (2024) regarding LLM-as-judge biases. We report judge agreement rate with human labels.

This two-stage approach avoids the systematic false-negative problem that would arise from relying solely on string matching for open-ended formats.

### Manual Validation

To ensure dataset quality and metric reliability, we include explicit manual validation steps:
- **Format conversion quality**: Manual verification of 10% of format-converted questions (~100 questions) for semantic equivalence.
- **Paraphrase quality**: Manual verification of at least 10% of generated paraphrases (~180 instances) for semantic preservation and naturalness.
- **Answer equivalence calibration**: Human annotation of ~200 answer pairs to validate the LLM-based equivalence judge, reporting inter-annotator agreement and judge accuracy.

### Models to Evaluate

We evaluate 6–7 models spanning:
- **Small models (3–8B)**: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Phi-3-mini-4k-instruct (3.8B)
- **Medium models (14B)**: Qwen2.5-14B-Instruct
- **Large models (~70B quantized)**: Llama-3.1-70B-Instruct (4-bit AWQ quantized, ~40GB VRAM)

This spans 3 model families (Llama, Qwen, Mistral/Phi) and sizes from 3.8B to 70B, enabling meaningful size-scaling analysis. All models fit on 1x A6000 (48GB VRAM).

### Key Innovations and Differentiation from Prior Work

1. **First cross-format consistency benchmark with item-level metrics**: OSQ-bench (Myrzakhan et al., 2024) demonstrated the aggregate accuracy gap between MCQ and open-ended formats, but does not measure per-question consistency. ConsistBench evaluates *item-level agreement* across 5 formats (not just 2), enabling diagnosis of which specific questions each model answers inconsistently.

2. **Paraphrase typology analysis**: Unlike RoParQ (Choi, 2025) which uses a single aggregate metric, we decompose paraphrase effects by type (lexical, syntactic, voice, formality, negation, elaborative), enabling fine-grained diagnosis of model weaknesses.

3. **Joint cross-format + cross-phrasing evaluation**: No prior work evaluates both axes simultaneously. Lunardi et al. (2025) test paraphrase robustness but not format conversion; OSQ-bench tests format conversion but not paraphrasing.

4. **Domain-stratified consistency profiling**: First systematic analysis of how consistency varies across knowledge domains and difficulty levels.

5. **Consistency-accuracy decoupling**: Novel metrics (CAR, FCAG) that separate a model's knowledge from its ability to express that knowledge consistently.

## Related Work

### Consistency in Language Models
Novikova et al. (2025) provide a comprehensive survey of consistency in LMs, identifying the lack of standardized definitions and metrics as a key challenge. Our work directly addresses this by proposing a concrete benchmark with well-defined metrics.

### MCQ-to-Open Format Conversion
Myrzakhan et al. (2024) proposed OSQ-bench, converting 8 MCQ benchmarks to open-style questions and evaluating 40+ LLMs, finding an average 25% accuracy drop. Their work demonstrates the importance of format effects but focuses on aggregate accuracy rather than per-question consistency. ConsistBench goes beyond OSQ-bench in three key ways: (a) we measure item-level agreement (not just aggregate accuracy gaps), (b) we test 5 formats (not just MCQ vs. open), and (c) we jointly evaluate paraphrase consistency, enabling cross-axis analysis that OSQ-bench cannot provide.

### Paraphrase Robustness
ParaRel (Elazar et al., 2021) pioneered paraphrase consistency evaluation but is limited to masked LMs and cloze-style probing. RoParQ (Choi, 2025) extends this to generative LLMs but only evaluates MCQ format with a single aggregate metric. Our work goes substantially further by introducing cross-format evaluation and paraphrase typology analysis.

### Benchmark Robustness
Mizrahi et al. (2024) demonstrated that LLM rankings are sensitive to instruction template choice, and Lunardi et al. (2025) showed paraphrasing degrades benchmark scores. These studies focus on evaluation methodology; we build a dedicated benchmark that turns consistency into a first-class evaluation dimension.

### Honesty and Calibration Benchmarks
BeHonest (Chern et al., 2024) evaluates honesty including some consistency scenarios, but its consistency evaluation is limited to format perturbation and social pressure, not systematic paraphrase or cross-format testing. Our work provides deeper coverage of the consistency dimension specifically.

### Dynamic and Evolving Benchmarks
LiveBench (White et al., 2025) and Benchmark Self-Evolving (Wang et al., 2025) address contamination through dynamic updates. Our work is orthogonal — we evaluate a fundamental model property (consistency) rather than knowledge currency.

### LLM-as-Judge Reliability
Ye et al. (2024) quantified biases in LLM-as-judge, including position and verbosity biases. We incorporate their findings by carefully validating our LLM-based answer equivalence judge against human annotations and reporting its agreement rate.

### Adversarial Robustness
PromptBench (Zhu et al., 2024) evaluates robustness to adversarial perturbations. Our work differs in focusing on *natural* paraphrases and *format* changes rather than adversarial attacks, measuring consistency (same answer) rather than robustness (maintained accuracy).

## Experiments

### Experimental Setup

**Hardware**: 1x NVIDIA RTX A6000 (48GB VRAM), 60GB RAM, 4 CPU cores.

**Phase 1 — Dataset Construction (~2 hours)**:
- Curate 1,000 base questions from open datasets (~165 per domain)
- Generate format variants programmatically (5 formats × 1,000 = 5,000 instances)
- Generate phrasing variants for 300-question subset using LLM + validation (6 types × 300 = 1,800 paraphrase instances)
- Manual quality check on 10% of format conversions and paraphrases

**Phase 2 — Model Evaluation (~5 hours)**:
- Run each of 6–7 models on all ~8,000–10,000 evaluation instances
- Extract and normalize answers
- Run two-stage answer equivalence evaluation (rule-based + LLM judge)
- Validate LLM judge against ~200 human-annotated answer pairs

**Phase 3 — Analysis (~1 hour)**:
- Compute all consistency metrics
- Generate figures and tables
- Statistical significance testing (bootstrap confidence intervals)

### Planned Analyses

1. **Main results table**: Accuracy and all consistency metrics for each model across all domains
2. **Cross-format consistency matrix**: Heatmap showing pairwise format agreement rates per model
3. **Paraphrase fragility profile**: Bar chart showing PFI per paraphrase type per model
4. **Domain consistency radar chart**: Per-model consistency profiles across knowledge domains
5. **Consistency vs. accuracy scatter**: Does higher accuracy predict higher consistency?
6. **Size scaling analysis**: How does consistency scale with model size within families (3.8B → 7–8B → 14B → 70B)?
7. **Error analysis**: Qualitative analysis of high-inconsistency examples — what makes certain questions harder to answer consistently?

### Metrics and Evaluation

- **Primary metric**: Cross-Format Agreement (CFA) — the headline number for model comparison
- **Secondary metrics**: CPA (per paraphrase type), FCAG, CAR, PFI, DCS
- **Baselines**: Random baseline (expected consistency under uniform random answering), accuracy-matched ceiling (maximum possible consistency given observed accuracy)
- **Statistical testing**: Bootstrap 95% confidence intervals for all metrics; paired permutation tests for model comparisons

### Expected Results

Based on prior work (Mizrahi et al., 2024; Myrzakhan et al., 2024; Choi, 2025; Lunardi et al., 2025), we expect:
- CFA in the range of 60–85% for most models (significantly below the 100% ideal)
- MCQ format yielding the highest accuracy but lowest consistency with open-ended
- Negation-framing paraphrases causing the largest consistency drops
- Larger models (70B) showing higher consistency than smaller models (7–8B), but still substantial gaps
- Science and history domains showing worse consistency than commonsense

## Success Criteria

The research succeeds if:

1. **Benchmark is diagnostic**: ConsistBench reveals meaningful and non-obvious differences in consistency across models, formats, paraphrase types, and domains that are not captured by standard accuracy metrics.

2. **Cross-format gap is real**: We demonstrate statistically significant cross-format inconsistency (CFA < 90%) for the majority of tested models, confirming that format choice meaningfully affects model answers.

3. **Paraphrase types differ**: Different paraphrase types produce significantly different consistency scores, enabling actionable insights about model weaknesses.

4. **Consistency ≠ accuracy**: We show that consistency metrics provide information beyond accuracy — i.e., models with similar accuracy can have meaningfully different consistency profiles.

The hypothesis would be **refuted** if: (a) all tested models show near-perfect consistency (CFA > 95%) across all axes, suggesting the problem is already solved; or (b) consistency perfectly correlates with accuracy with no additional diagnostic value.

## References

1. Elazar, Y., Kassner, N., Ravfogel, S., Ravichander, A., Hovy, E., Schütze, H., & Goldberg, Y. (2021). Measuring and Improving Consistency in Pretrained Language Models. *TACL*, 9, 1012–1031.

2. Chern, S., Hu, Z., Yang, Y., Chern, E., Guo, Y., Jin, J., Wang, B., & Liu, S. (2024). BeHonest: Benchmarking Honesty of Large Language Models. *arXiv:2406.13261*.

3. Mizrahi, M., Kaplan, G., Malkin, D., Dror, R., Shahaf, D., & Stanovsky, G. (2024). State of What Art? A Call for Multi-Prompt LLM Evaluation. *TACL*, 12, 933–949.

4. Myrzakhan, A., Bsharat, S.M., & Shen, Z. (2024). Open-LLM-Leaderboard: From Multi-choice to Open-style Questions for LLMs Evaluation, Benchmark, and Arena. *arXiv:2406.07545*.

5. Choi, J. (2025). RoParQ: Paraphrase-Aware Alignment of LLMs Towards Robustness to Paraphrased Questions. *arXiv:2511.21568*.

6. Novikova, J., Anderson, C., Blili-Hamelin, B., Rosati, D., & Majumdar, S. (2025). Consistency in Language Models: Current Landscape, Challenges, and Future Directions. *ICML 2025 Workshop on Reliable and Responsible Foundation Models*. arXiv:2505.00268.

7. Zhu, K., Zhao, Q., Chen, H., Wang, J., & Xie, X. (2024). PromptBench: A Unified Library for Evaluation of Large Language Models. *JMLR*, 25(254), 1–22.

8. Lunardi, R., Della Mea, V., Mizzaro, S., & Roitero, K. (2025). On Robustness and Reliability of Benchmark-Based Evaluation of Large Language Models. *arXiv:2509.04013*.

9. White, C., Dooley, S., Roberts, M., Pal, A., et al. (2025). LiveBench: A Challenging, Contamination-Limited LLM Benchmark. *ICLR 2025*.

10. Wang, S., Long, Z., Fan, Z., Wei, Z., & Huang, X. (2025). Benchmark Self-Evolving: A Multi-Agent Framework for Dynamic LLM Evaluation. *COLING 2025*, 3310–3328.

11. Ye, J., Wang, Y., Huang, Y., Chen, D., et al. (2024). Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge. *arXiv:2410.02736*.

12. Rabinovich, E., Ackerman, S., Shnarch, E., Patel, P., & Anaby-Tavor, A. (2023). Predicting Question-Answering Performance of Large Language Models through Semantic Consistency. *GEM Workshop at EMNLP 2023*.
