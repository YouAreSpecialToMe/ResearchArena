# SkillStack: A Procedurally Generated Benchmark for Measuring Compositional Cognitive Skill Gaps in Large Language Models

## Introduction

### Context and Problem Statement

Large Language Models (LLMs) have achieved impressive scores across individual benchmarks testing specific capabilities — arithmetic on GSM8K, knowledge on MMLU, reasoning on ARC. However, real-world tasks rarely require a single cognitive skill in isolation. Writing a data analysis report demands arithmetic, logical reasoning, and natural language generation simultaneously. Debugging code requires code tracing, logical deduction, and pattern recognition together. The critical question is: **can LLMs compose multiple cognitive skills effectively, or do their capabilities degrade when skills must be combined?**

Recent evidence suggests a severe **compositional capability gap**. Apple's GSM-Symbolic work (Mirzadeh et al., 2024) showed that adding even a single irrelevant clause to math problems causes up to 65% accuracy drops, suggesting LLMs rely on pattern matching rather than genuine reasoning. The Skill-Mix evaluation (Yu et al., 2023) demonstrated that LLMs struggle to combine language skills (e.g., metaphor + irony) in text generation, with combinatorial explosion making memorization impossible. Research on compositional multi-tasking found that top-tier LLMs experience 20–40% accuracy drops on tasks requiring simultaneous execution of multiple skills, even when each individual skill is near-perfect.

Yet no existing benchmark **systematically measures** how cognitive skill composition degrades across controlled combinations, with verifiable answers and procedural generation for contamination resistance.

### Key Insight

We propose that the compositional capability gap can be precisely quantified by decomposing complex tasks into **atomic cognitive primitives** and systematically evaluating all pairwise and triple combinations. By using procedurally generated instances with verifiable ground-truth answers, we create a contamination-proof benchmark that reveals exactly which skill combinations LLMs struggle with and how much of their apparent "reasoning" is genuine composition versus holistic pattern matching.

### Hypothesis

LLM performance on tasks requiring K composed cognitive skills degrades super-linearly with K, and this degradation varies systematically across skill types, model families, and model scales. Specifically, we hypothesize that: (1) the composition gap is larger for skills that require sequential dependencies (e.g., arithmetic → comparison) than for parallel skills (e.g., counting + string manipulation); (2) larger models have smaller composition gaps, but the gap does not vanish even at frontier scale; and (3) chain-of-thought prompting partially closes the composition gap by decomposing composed tasks into sequential single-skill steps.

## Proposed Approach

### Overview

**SkillStack** is a procedurally generated benchmark that evaluates LLMs on 8 atomic cognitive primitives and their systematic pairwise (28) and triple (56) combinations, totaling 92 distinct evaluation categories. Each category contains 50 procedurally generated instances with automatically verifiable answers, yielding ~4,600 test instances per evaluation run. The benchmark can generate unlimited fresh instances, making it inherently contamination-resistant.

### Cognitive Primitives (8 Skills)

We define 8 atomic cognitive skills that span the core capabilities tested across existing LLM benchmarks:

1. **Arithmetic (AR)**: Multi-step integer arithmetic (addition, multiplication, division with remainder). *Example*: "What is (47 × 13) + 289?"
2. **Comparison (CO)**: Ordering and comparing quantities, dates, or magnitudes. *Example*: "Which is larger: 3/7 or 5/12?"
3. **Counting (CN)**: Counting items satisfying specific predicates in a given set. *Example*: "In the list [apple, banana, avocado, apricot, cherry], how many items start with 'a'?"
4. **Logical Deduction (LD)**: Syllogistic reasoning, if-then chains, constraint propagation. *Example*: "If all Zorbians are Plexites, and no Plexites are Dravians, are any Zorbians Dravians?"
5. **Spatial Reasoning (SP)**: Relative positions, directions, and spatial relationships. *Example*: "Alice is north of Bob. Bob is east of Carol. What direction is Alice from Carol?"
6. **String Manipulation (ST)**: Character-level operations — reversing, extracting substrings, counting characters. *Example*: "What is the 4th character of the reverse of 'BENCHMARK'?"
7. **Temporal Reasoning (TE)**: Before/after relationships, duration calculations, scheduling. *Example*: "Event A starts at 2:30 PM and lasts 1 hour 45 minutes. Event B starts at 3:00 PM. Do they overlap?"
8. **Set Operations (SE)**: Union, intersection, difference, membership, subset checking. *Example*: "Set X = {1, 3, 5, 7}. Set Y = {2, 3, 5, 8}. What is |X ∩ Y|?"

### Composition Levels

- **Level 1 (Single Skill)**: 8 skills × 50 instances = 400 test items. These establish each model's baseline per-skill capability.
- **Level 2 (Pairwise Composition)**: C(8,2) = 28 pairs × 50 instances = 1,400 test items. Each item requires applying two skills in sequence or in parallel.
  - *Example (AR + CO)*: "Alice has 3 × 17 apples. Bob has 2 × 29 apples. Who has more?" (Requires: arithmetic to compute 51 and 58, then comparison)
  - *Example (ST + CN)*: "In the word 'MISSISSIPPI', how many times does the reverse of 'SS' appear?" (Requires: string reversal to get 'SS', then counting)
- **Level 3 (Triple Composition)**: C(8,3) = 56 triples × 50 instances = 2,800 test items. Each item requires three skills.
  - *Example (AR + TE + CO)*: "Train A departs at 9:15 AM traveling at 60 mph. Train B departs at 9:45 AM traveling at 80 mph. After 2 hours from Train A's departure, which train has traveled farther?" (Requires: temporal reasoning for elapsed times, arithmetic for distances, comparison)

### Procedural Generation Framework

Each test item is generated by a **template engine** with randomized parameters:

1. **Skill templates**: For each skill, we define 5–10 template structures with variable slots (numbers, names, items, etc.)
2. **Composition templates**: For each skill pair/triple, we define templates that naturally integrate both skills, with clear dependency structures
3. **Answer computation**: A deterministic solver computes the ground-truth answer from the template parameters
4. **Difficulty control**: Parameters are sampled to maintain comparable difficulty across compositions (e.g., number ranges are calibrated so arithmetic difficulty is consistent whether tested alone or in composition)
5. **Validation**: Each generated instance is verified by the solver, and instances with ambiguous or overly complex answers are filtered out

The procedural generation ensures:
- **Contamination resistance**: New instances can be generated at any time
- **Statistical power**: Large sample sizes per category enable reliable measurements
- **Controlled difficulty**: Component skill difficulty is held constant across composition levels, isolating the composition effect

### Key Metrics

1. **Per-Skill Accuracy (PSA)**: Accuracy on Level 1 items for each of the 8 skills
2. **Composed Accuracy (CA)**: Accuracy on Level 2/3 items for each skill combination
3. **Composition Gap (CG)**: The difference between expected and actual composed performance:
   - CG(S_i, S_j) = min(PSA(S_i), PSA(S_j)) − CA(S_i, S_j)
   - A positive CG indicates that composition degrades performance beyond what individual skill weakness would predict
4. **Composition Efficiency (CE)**: CE = CA / min(PSA over component skills). Values below 1.0 indicate composition overhead.
5. **Skill Interaction Matrix**: A heatmap showing CG for all 28 pairwise combinations, revealing which skill pairs compose well vs. poorly
6. **Composition Scaling Exponent**: Fitting CA = PSA^α across composition levels to measure whether degradation is linear (α=1), sub-linear, or super-linear in the number of composed skills

## Related Work

### Compositional Evaluation of LLMs

**Skill-Mix** (Yu et al., 2023; ICLR 2024) is the closest related work. It tests LLMs' ability to combine k language skills (e.g., metaphor, alliteration, humor) in text generation, using LLM-as-judge evaluation. SkillStack differs in three key ways: (1) we test *cognitive/reasoning* skills rather than *language/stylistic* skills; (2) our answers are automatically verifiable (exact match or numerical), eliminating judge bias; (3) we use procedural generation for contamination resistance.

**Can Models Learn Skill Composition from Examples?** (Zhao et al., 2024; NeurIPS 2024) extends Skill-Mix to study whether fine-tuning on k=2,3 skill compositions transfers to k=4,5. Their finding that composition can be learned but doesn't fully transfer motivates our systematic measurement of the composition gap across cognitive skills.

### Robustness and Perturbation Benchmarks

**GSM-Symbolic** (Mirzadeh et al., 2024; ICLR 2025) demonstrates that LLMs' math performance degrades significantly with symbolic substitution and irrelevant clause insertion, suggesting pattern matching rather than genuine reasoning. SkillStack generalizes this insight beyond mathematics to systematic multi-skill composition.

**GSM-DC** (Yang et al., 2025; EMNLP 2025) constructs controlled distractors for math reasoning, showing accuracy degrades as distractors increase. While GSM-DC focuses on distraction robustness within a single skill (math), SkillStack measures the orthogonal dimension of *skill composition*.

### Benchmark Quality and Measurement

**MetaBench** (Filippo et al., 2024; ICLR 2025) compresses 6 LLM benchmarks to 3% of their size using item-level analysis from 5000+ models, finding high inter-benchmark correlations suggesting a single underlying factor. SkillStack is complementary — it tests whether this "monolithic factor" masks fine-grained compositional weaknesses.

**Lost in Benchmarks? (PSN-IRT)** (Zhou et al., 2025; AAAI 2026) applies Item Response Theory to diagnose benchmark quality. Their finding of poor measurement properties in existing benchmarks motivates our design of a benchmark with controlled difficulty and procedural generation.

**LiveBench** (White et al., 2024; NeurIPS 2024) addresses contamination through monthly-updated questions from recent sources. SkillStack takes a different anti-contamination approach: procedural generation from templates, which is more scalable and doesn't depend on external information sources.

### Cognitive Evaluation Frameworks

**BIG-Bench** (Srivastava et al., 2023) provides 204 tasks across diverse cognitive domains but lacks systematic control over skill composition — tasks are independently authored without shared cognitive primitives. SkillStack provides a structured, systematic alternative.

**How and Why LLMs Generalize** (Bai et al., 2025) decomposes reasoning into atomic cognitive behaviors (calculation, retrieval, simulation, enumeration, diagnostic) and studies how these are affected by fine-tuning. While this is an analysis paper studying training dynamics, SkillStack provides a reusable benchmark with a complementary set of cognitive primitives focused on composition.

## Experiments

### Experimental Setup

**Models**: We will evaluate 10–15 models spanning:
- Small open models: Qwen2.5-1.5B, Phi-3-mini-3.8B
- Medium open models: Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B
- Large open models: Llama-3.1-70B (quantized), Qwen2.5-14B, Qwen2.5-32B (quantized)
- Reasoning-tuned models: DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-Coder-7B
- Instruction-tuned vs. base model pairs for controlled comparisons

**Prompting Strategies**:
- Zero-shot direct answering
- Zero-shot chain-of-thought (CoT)
- 3-shot with Level 1 examples only (to test if single-skill examples help composition)

**Evaluation Protocol**:
- Exact match for string/set answers
- Numerical tolerance (±0.01) for arithmetic answers
- Yes/No/Name match for comparison/logical answers
- Each model evaluated on the same 4,600 instances (seeded random generation for reproducibility)
- 95% confidence intervals computed via bootstrap resampling

### Key Analyses

1. **Composition Gap Analysis**: Compute CG for all 28 pairwise and 56 triple combinations across all models. Visualize as heatmaps (skills × skills) per model family and size.

2. **Scaling Laws for Composition**: Plot CE against model size (parameters) for each skill combination to identify whether composition improves with scale and at what rate.

3. **Chain-of-Thought Effect**: Compare direct vs. CoT prompting on composed tasks. Hypothesis: CoT helps more for sequential compositions (where skills have dependencies) than parallel compositions.

4. **Skill Interaction Taxonomy**: Cluster skill pairs by their CG profiles across models to identify "easy" compositions (small gap) vs. "hard" compositions (large gap). Investigate what structural properties predict composition difficulty.

5. **Instruction Tuning Effect**: Compare base vs. instruction-tuned model pairs to measure whether instruction tuning specifically improves skill composition or only improves individual skills.

6. **Contamination Resistance Validation**: Generate two independent sets of 4,600 instances and verify that model rankings and composition gaps are consistent across sets (test-retest reliability).

### Expected Results

Based on prior findings:
- Individual skill accuracies of 70–95% for frontier-scale models
- Pairwise composition accuracies 15–30% lower than the minimum of individual skills
- Triple composition accuracies 30–50% lower
- CoT reducing the composition gap by 30–50% for sequential compositions but <10% for parallel ones
- Larger models showing smaller but non-zero composition gaps
- Specific "hard pairs" (e.g., spatial + temporal, logic + counting) showing particularly large gaps

## Success Criteria

The research will be considered successful if:

1. **Primary**: We demonstrate a statistically significant (p < 0.01) composition gap across at least 80% of skill pairs for at least 5 evaluated models, confirming that skill composition is a genuine and widespread weakness.

2. **Secondary**: We identify at least 3 qualitatively distinct patterns in the Skill Interaction Matrix (e.g., "easy", "medium", "hard" composition clusters), providing actionable insights for model developers.

3. **Tertiary**: We show that the composition gap varies meaningfully (>10 percentage points) across model families, sizes, or prompting strategies, demonstrating that SkillStack reveals information not captured by existing benchmarks.

4. **Validation**: Test-retest reliability (correlation between two independently generated instance sets) exceeds r = 0.90 for composition gap measurements.

The hypothesis would be **refuted** if: composition gaps are consistently small (<5%) across models and skill combinations, suggesting that LLMs have already achieved robust skill composition. This would itself be an important finding.

## References

1. Yu, D., Kaur, S., Gupta, A., Brown-Cohen, J., Goyal, A., & Arora, S. (2023). "Skill-Mix: a Flexible and Expandable Family of Evaluations for AI Models." ICLR 2024. arXiv:2310.17567.

2. Zhao, H., Kaur, S., Yu, D., Goyal, A., & Arora, S. (2024). "Can Models Learn Skill Composition from Examples?" NeurIPS 2024. arXiv:2409.19808.

3. Mirzadeh, I., Alizadeh, K., Shahrokhi, H., Tuzel, O., Bengio, S., & Farajtabar, M. (2024). "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models." ICLR 2025. arXiv:2410.05229.

4. Yang, M., Huang, E., Zhang, L., Surdeanu, M., Wang, W. Y., & Pan, L. (2025). "How Is LLM Reasoning Distracted by Irrelevant Context? An Analysis Using a Controlled Benchmark." EMNLP 2025. arXiv:2505.18761.

5. Srivastava, A., et al. (2023). "Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models." Transactions on Machine Learning Research. arXiv:2206.04615.

6. Zhou, J., Huang, K., et al. (2025). "Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory." AAAI 2026 (Oral). arXiv:2505.15055.

7. White, C., et al. (2024). "LiveBench: A Challenging, Contamination-Limited LLM Benchmark." NeurIPS 2024 D&B Track. arXiv:2406.19314.

8. Bai, H., Sun, Y., Hu, W., Qiu, S., Huan, M. Z., Song, P., Nowak, R., & Song, D. (2025). "How and Why LLMs Generalize: A Fine-Grained Analysis of LLM Reasoning from Cognitive Behaviors to Low-Level Patterns." arXiv:2512.24063.

9. Filippo, P. G., et al. (2024). "MetaBench — A Sparse Benchmark of Reasoning and Knowledge in Large Language Models." ICLR 2025. arXiv:2407.12844.

10. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168.
