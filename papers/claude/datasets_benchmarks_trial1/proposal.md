# FlipBench: Measuring Directional Reasoning Asymmetry in Large Language Models

## Introduction

### Context and Motivation

Large language models (LLMs) achieve impressive scores on a wide range of reasoning benchmarks spanning logic, mathematics, code, and commonsense inference. Yet a growing body of evidence suggests that high benchmark accuracy may not reflect genuine, robust reasoning. The *Reversal Curse* (Berglund et al., 2024) demonstrated that LLMs trained on "A is B" fail to infer "B is A," revealing a fundamental directional asymmetry in factual knowledge retrieval. Subsequent work has shown that autoregressive models exhibit systematic biases favoring forward reasoning over backward reasoning in planning tasks (Yu et al., 2024), and that only 8% of the 1,598 papers studying LLM reasoning examine backward chaining (Empowering LLMs with Logical Reasoning survey, 2025).

However, a critical gap remains: while individual papers have probed directional asymmetry in narrow settings (factual recall, planning, or isolated logic rules), **no systematic, cross-domain benchmark exists that uses matched forward-backward problem pairs to quantify directional reasoning asymmetry**. Without such a benchmark, we cannot answer fundamental questions: Is the asymmetry universal across reasoning types? Does it scale with problem difficulty? Do reasoning-optimized models (e.g., those trained with reinforcement learning on reasoning tasks) reduce the gap?

### Problem Statement

Current reasoning benchmarks evaluate models in a single direction—typically forward (given premises, derive conclusion). This one-directional evaluation conflates genuine reasoning ability with surface-level pattern matching. A model that can derive "If A then B; A is true → B is true" (modus ponens) but fails at the logically equivalent "If A then B; B is false → A is false" (modus tollens) has not truly learned conditional reasoning—it has learned a template.

### Key Insight

Genuine reasoning should be **direction-invariant**: if a model truly understands a logical relationship, mathematical operation, or causal link, it should be able to reason about it in both forward and backward directions with comparable accuracy. The gap between forward and backward performance on matched problem pairs—the **Directional Reasoning Gap (DRG)**—serves as a diagnostic signal that separates genuine understanding from pattern matching.

### Hypothesis

LLMs exhibit a systematic directional reasoning asymmetry that is (1) present across multiple reasoning domains, (2) amplified by problem complexity, (3) partially reducible by reasoning-focused training (e.g., RL-trained models), and (4) domain-dependent in magnitude. We hypothesize that the DRG is largest in domains with strong forward-reasoning training signal (logic, code) and smallest in domains where forward and backward problems look structurally similar (arithmetic).

## Proposed Approach

### Overview

We introduce **FlipBench**, a benchmark of ~1,200 matched forward-backward reasoning pairs across four domains, each with multiple difficulty levels. Every problem instance consists of a **forward variant** and a **backward variant** that test the same underlying knowledge or reasoning operation but in opposite directions. All problems are programmatically generated with verified ground-truth answers, ensuring scalability and correctness.

### Domain Design

#### Domain 1: Propositional Logic (300 pairs)

**Forward**: Given a set of propositional rules and facts, derive whether a target proposition is true or false using forward chaining (modus ponens).

**Backward**: Given the same rules and a target proposition's truth value, determine which initial facts must hold using backward chaining (modus tollens, contraposition).

- Difficulty levels: 1-hop (single rule application), 2-hop (chained rules), 3-hop (multi-step chaining with distractors)
- Generation: Randomly generate rule sets over a vocabulary of propositions, ensure unique solutions, create matched pairs

Example:
- Forward: "If P then Q. If Q then R. P is true. Is R true?"
- Backward: "If P then Q. If Q then R. R is false. Is P true or false?"

#### Domain 2: Arithmetic Reasoning (300 pairs)

**Forward**: Given a mathematical expression or word problem with all operands specified, compute the result.

**Backward**: Given the same expression structure but with the result known and one operand missing, find the missing operand.

- Difficulty levels: single-operation, two-operation chains, three-operation chains with mixed operators
- Generation: Sample random integers, operations, and target values; verify unique solvability

Example:
- Forward: "A store sells apples at $3 each. You buy 7 apples and pay with a $25 bill. How much change do you get?"
- Backward: "A store sells apples at $3 each. You buy some apples and pay with a $25 bill. You get $4 in change. How many apples did you buy?"

#### Domain 3: Relational/Causal Reasoning (300 pairs)

**Forward**: Given a set of relational facts (family trees, organizational hierarchies, or simple causal chains), derive a target relation.

**Backward**: Given the same relational structure but with the target relation known, identify the entity that satisfies it.

- Difficulty levels: single-hop relations, two-hop transitive relations, three-hop with multiple entity types
- Generation: Randomly generate entity-relation graphs, sample forward/backward queries with unique answers

Example:
- Forward: "Alice is Bob's mother. Bob is Carol's father. What is Alice's relationship to Carol?"
- Backward: "Alice is Bob's mother. Someone is Carol's grandmother. Carol's father is Bob. Who is Carol's grandmother?"

#### Domain 4: Function Computation (300 pairs)

**Forward**: Given a simple function (expressed in Python) and an input, predict the output.

**Backward**: Given the same function and a target output, determine what input produces that output.

- Difficulty levels: single-operation functions, composed functions (2-step), composed functions (3-step) with conditionals
- Generation: Generate invertible functions with integer domains; verify unique inverse solutions

Example:
- Forward: `def f(x): return 2*x + 3`. What is f(5)?
- Backward: `def f(x): return 2*x + 3`. For what value of x does f(x) = 13?

### Key Design Principles

1. **Matched difficulty**: Forward and backward variants of the same pair involve identical reasoning depth. The backward variant is not inherently "harder" in terms of logical steps—it simply reverses the direction of inference.

2. **Verified ground truth**: All instances are programmatically generated with deterministic, verifiable answers. No human annotation is needed, eliminating label noise.

3. **Controlled generation**: Each domain uses templated generation with randomized parameters, enabling arbitrary dataset scaling and minimizing memorization risk.

4. **Difficulty calibration**: Each domain includes 3 difficulty levels (100 pairs each) to study how the DRG scales with complexity.

### Metrics

- **Forward Accuracy (FA)**: Accuracy on forward variants
- **Backward Accuracy (BA)**: Accuracy on backward variants
- **Directional Reasoning Gap (DRG)**: FA − BA (the core metric)
- **Consistency Rate (CR)**: Fraction of pairs where the model answers both forward and backward correctly
- **DRG per difficulty level**: To study how the gap scales with complexity
- **DRG per domain**: To identify which reasoning types exhibit the largest asymmetry

## Related Work

### The Reversal Curse

Berglund et al. (2024) showed that autoregressive LLMs trained on "A is B" cannot infer "B is A," demonstrating a fundamental asymmetry in factual knowledge storage. Follow-up work by Golovneva et al. (2024) analyzed and proposed mitigations, while Allen-Zhu & Li (2024) connected the phenomenon to transformer binding limitations. Our work extends this from factual recall to **reasoning**: we ask whether the directional asymmetry that plagues knowledge retrieval also affects logical inference, mathematical reasoning, and code understanding.

### Logical Reasoning Benchmarks

LogicBench (Parmar et al., 2024) systematically evaluates 25 logical reasoning patterns including modus ponens and modus tollens, but does not create matched pairs designed to measure the directional gap on the same underlying problem. SATBench (Wei et al., 2025) generates logic puzzles from SAT formulas. ZebraLogic (Lin et al., 2025) tests constraint satisfaction at varying complexity. Our contribution is orthogonal: we focus specifically on directional consistency rather than overall reasoning ability.

### Forward vs. Backward Reasoning in LLMs

Yu et al. (2024) found that LLMs struggle with backward planning due to autoregressive generation biases. Kim et al. (2025) demonstrated that only 8% of LLM reasoning research examines backward chaining. The survey by Wang et al. (2025) on empowering LLMs with logical reasoning notes the forward bias but does not provide a systematic benchmark. Our work fills this gap by creating a dedicated benchmark resource.

### Benchmark Quality and Meta-Evaluation

Benchmark2 (Qian et al., 2026) proposed meta-evaluation metrics for LLM benchmarks. PSN-IRT (Polo et al., 2025) uses Item Response Theory to diagnose benchmark item quality. Benchmark Profiling (Kim et al., 2025) uses mechanistic interpretability to identify which abilities benchmarks test. Our work complements these by introducing a new diagnostic dimension—directional reasoning consistency—that existing benchmarks do not measure.

### Reasoning Robustness

ReasonBENCH (Wang et al., 2025) measures stochastic instability (variance across multiple runs). VAR-MATH (2025) tests consistency under variable substitution for math problems. The paraphrase robustness literature tests rewording sensitivity. FlipBench is complementary: we test **structural invariance** (same reasoning, different direction) rather than surface-level perturbations.

### Generation-Verification Gap

Variation in Verification (Yan et al., 2025) and Shrinking the Generation-Verification Gap (Snell et al., 2025) study the asymmetry between generating and verifying solutions. While related, our work differs in focus: we compare two forms of generation (forward and backward) rather than generation vs. verification.

## Experiments

### Models

We will evaluate 6-8 open-weight models spanning different sizes and training paradigms:

1. **Llama-3.1-8B-Instruct** — medium-size general-purpose
2. **Llama-3.1-70B-Instruct** — large general-purpose (if feasible with quantization)
3. **Qwen2.5-7B-Instruct** — medium alternative family
4. **Qwen2.5-32B-Instruct** — larger alternative family
5. **DeepSeek-R1-Distill-Qwen-7B** — reasoning-optimized (RL-trained)
6. **DeepSeek-R1-Distill-Qwen-32B** — larger reasoning-optimized
7. **Phi-3.5-mini-instruct** — small model baseline
8. **Mistral-7B-Instruct-v0.3** — additional medium baseline

This selection covers: multiple model families, size scaling (7B → 32B → 70B), and standard vs. reasoning-optimized training.

### Evaluation Protocol

- Zero-shot prompting with a standardized prompt template per domain
- For each problem, the model is asked to provide a final answer in a structured format
- Answers are parsed and compared to ground truth via exact match (numeric) or logical equivalence (propositional)
- Temperature set to 0 for deterministic outputs
- Each model is evaluated on all 2,400 instances (1,200 pairs × 2 directions)

### Planned Analyses

1. **Overall DRG by domain**: Which domains show the largest forward-backward gap?
2. **DRG vs. difficulty**: How does the gap scale with problem complexity within each domain?
3. **DRG vs. model size**: Do larger models exhibit smaller gaps?
4. **Standard vs. reasoning-optimized**: Do RL-trained reasoning models (DeepSeek-R1) show reduced asymmetry?
5. **Error analysis**: What types of errors are specific to backward reasoning? (e.g., failure to apply contraposition, inability to invert operations)
6. **Cross-domain correlation**: Do models that are more symmetric in one domain also tend to be more symmetric in others?
7. **Consistency analysis**: What fraction of pairs do models get fully correct (both directions)?

### Expected Results

Based on preliminary evidence from the literature:
- **Logic domain**: Large DRG (20-40%) due to known difficulty with modus tollens and backward chaining
- **Arithmetic domain**: Moderate DRG (10-25%) as inverse operations are well-represented in training data but multi-step inverses are harder
- **Relational domain**: Large DRG (15-35%) due to challenges with backward traversal of relation chains
- **Function domain**: Moderate-to-large DRG (15-30%) as function inversion requires compositional reasoning
- **Reasoning-optimized models**: ~30-50% smaller DRG than standard models, especially in logic and function domains
- **Scaling effect**: DRG decreases by ~5-10% when moving from 7B to 32B+ models

## Success Criteria

### Primary (confirms hypothesis)
- A statistically significant DRG (>5 percentage points) is observed in at least 3 out of 4 domains across the majority of models tested
- The DRG increases with problem difficulty in at least 2 domains
- The benchmark reveals meaningful differences between model families and training paradigms

### Secondary (strengthens contribution)
- Reasoning-optimized models (DeepSeek-R1) show significantly smaller DRG than standard instruction-tuned models
- The DRG varies substantially across domains, suggesting it captures domain-specific reasoning properties
- Consistency rates (both directions correct) are significantly lower than the minimum of forward/backward accuracy, revealing that directional asymmetry is not merely a uniform accuracy offset

### Refutation criteria
- If DRG is consistently <3% across all domains and models, this would suggest that directional asymmetry is not a significant issue for current LLMs (a positive finding in itself, though it would reframe the paper's contribution)

## References

1. Berglund, L., Tong, M., Kaufmann, M., Balesni, M., Stickland, A. C., Korbak, T., & Evans, O. (2024). The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A." *ICLR 2024*. arXiv:2309.12288.

2. Golovneva, O., Allen-Zhu, Z., Weston, J., & Sukhbaatar, S. (2024). An Analysis and Mitigation of the Reversal Curse. *EMNLP 2024*. arXiv:2410.18808.

3. Allen-Zhu, Z. & Li, Y. (2025). Is the Reversal Curse a Binding Problem? Uncovering Limitations of Transformers from a Basic Generalization Failure. arXiv:2504.01928.

4. Parmar, M., Patel, N., Varshney, N., Nakamura, M., Luo, M., Masud, S., Mitra, A., & Baral, C. (2024). LogicBench: Towards Systematic Evaluation of Logical Reasoning Ability of Large Language Models. *ACL 2024*. arXiv:2404.15522.

5. Wei, A., et al. (2025). SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas. *EMNLP 2025*.

6. Lin, B. Y., et al. (2025). ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning. arXiv:2502.01100.

7. Yu, S., et al. (2024). Thinking Forward and Backward: Effective Backward Planning with Large Language Models. arXiv:2411.01790.

8. Wang, H., et al. (2025). Empowering LLMs with Logical Reasoning: A Comprehensive Survey. arXiv:2502.15652.

9. Polo, F. M., et al. (2025). Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory. arXiv:2505.15055.

10. Qian, Q., Huang, C., et al. (2026). Benchmark^2: Systematic Evaluation of LLM Benchmarks. arXiv:2601.03986.

11. Kim, D., et al. (2025). Benchmark Profiling: Mechanistic Diagnosis of LLM Benchmarks. *EMNLP 2025*. arXiv:2510.01232.

12. Wang, X., et al. (2025). ReasonBENCH: Benchmarking the (In)Stability of LLM Reasoning. arXiv:2512.07795.

13. Yan, S., et al. (2025). Variation in Verification: Understanding Verification Dynamics in Large Language Models. arXiv:2509.17995.

14. Snell, C., et al. (2025). Shrinking the Generation-Verification Gap with Weak Verifiers. arXiv:2506.18203.

15. White, C., et al. (2025). LiveBench: A Challenging, Contamination-Limited LLM Benchmark. arXiv:2406.19314.

16. Li, J., et al. (2025). A Survey on Data Contamination for Large Language Models. arXiv:2502.14425.

17. Zhang, R., et al. (2025). Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks. arXiv:2511.04689.

18. Gema, A. P., et al. (2025). Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their Effect on Model Performance. arXiv:2410.18889.
