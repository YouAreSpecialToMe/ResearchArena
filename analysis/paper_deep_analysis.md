
# Deep Analysis of AI-Generated Research Papers: Kimi vs Codex

*Analysis of 78 papers across 2 agents, 13 seeds, CPU and GPU platforms.*


## 1. Data Overview

**Kimi**: 39 trials (15 CPU, 24 GPU) | paper.tex: 39 | paper.pdf: 28 | reviews: 39

**Codex**: 39 trials (15 CPU, 24 GPU) | paper.tex: 39 | paper.pdf: 33 | reviews: 39

| Metric | Kimi | Codex |
|---|---|---|
| Total trials | 39 | 39 |
| Has paper.tex | 39 | 39 |
| Has paper.pdf | 28 | 33 |
| Has reviews | 39 | 39 |
| Has results.json | 37 | 36 |


## 2. Paper Structure and Formatting


### 2.1 Section Completeness

| Section | Kimi % | Codex % |
|---|---|---|
| Abstract | 100% | 100% |
| Introduction | 100% | 100% |
| Method | 100% | 100% |
| Experiments | 100% | 100% |
| Related Work | 100% | 100% |
| Conclusion | 100% | 100% |


### 2.2 LaTeX Compilation

**Kimi**: 28/39 papers compiled to PDF (72%)

**Codex**: 33/39 papers compiled to PDF (85%)


### 2.3 Paper Length

**Kimi**: mean=2011, median=1985, min=1443, max=3061 words

**Codex**: mean=2791, median=2782, min=1929, max=4107 words


### 2.4 Figures, Tables, and Equations

| Metric | Kimi (mean) | Codex (mean) |
|---|---|---|
| Figures | 0.8 | 4.1 |
| Tables | 4.0 | 4.2 |
| Equations | 3.8 | 2.0 |
| Algorithms | 0.6 | 0.0 |
| Citations | 16.9 | 14.5 |


### 2.5 Bibliography Style

**Kimi**: natbib/bibtex=24, inline bibitem=31

**Codex**: natbib/bibtex=33, inline bibitem=10


## 3. Research Content Quality


### 3.1 Title Analysis

**Kimi**: mean title length=11 words, range 6-15

**Codex**: mean title length=11 words, range 7-18


### 3.2 Reference Quality

| Metric | Kimi (mean) | Codex (mean) |
|---|---|---|
| Parsed references (refs/ dir) | 17.3 | 12.9 |
| Citations in paper | 16.9 | 14.5 |


### 3.3 Reference Integrity Scores

**Kimi**: mean reference integrity = 5.5

**Codex**: mean reference integrity = 8.6


## 4. Review Score Analysis


### 4.1 Overall Scores

| Agent | Platform | Mean | Std | Min | Max | N |
|---|---|---|---|---|---|---|
| kimi | cpu | 3.69 | 0.75 | 2.0 | 4.7 | 15 |
| kimi | gpu | 3.19 | 1.06 | 0.0 | 5.3 | 24 |
| codex | cpu | 4.53 | 0.28 | 4.0 | 4.7 | 15 |
| codex | gpu | 4.50 | 0.41 | 4.0 | 5.3 | 24 |


### 4.2 Per-Dimension Scores

| Dimension | Kimi | Codex |
|---|---|---|
| novelty | 4.4 | 4.5 |
| soundness | 3.8 | 6.1 |
| significance | 4.1 | 4.3 |
| clarity | 6.3 | 7.5 |
| reproducibility | 5.0 | 7.6 |
| experimental_rigor | 3.2 | 5.5 |
| references | 5.7 | 7.0 |
| reference_integrity | 6.2 | 8.4 |
| results_integrity | 4.1 | 7.9 |


### 4.3 Per-Reviewer Breakdown

| Reviewer | Scoring Kimi (mean) | Scoring Codex (mean) |
|---|---|---|
| Claude Code | 3.2 | 4.0 |
| Codex | 1.9 | 3.9 |
| Kimi Code | 5.0 | 5.7 |


## 5. Writing Pattern Analysis


### 5.1 Confidence Language

| Pattern | Kimi % | Codex % |
|---|---|---|
| "novel" | 49% | 8% |
| "first" | 67% | 95% |
| "state-of-the-art" | 26% | 15% |
| "outperform" | 54% | 67% |


### 5.2 Abstract Length

**Kimi**: mean=111, median=115 words

**Codex**: mean=172, median=190 words


## 6. Common Failure Modes


### 6.1 Top Reviewer Complaints


**Kimi** (795 total weakness items):

| Category | Count | % |
|---|---|---|
| Other | 204 | 26% |
| Weak baselines | 134 | 17% |
| Unsupported claims | 121 | 15% |
| Missing experiments | 61 | 8% |
| Results mismatch | 59 | 7% |
| Novelty concerns | 49 | 6% |
| Missing ablations | 49 | 6% |
| Limited scope | 44 | 6% |
| Reference issues | 38 | 5% |
| Code/crash issues | 21 | 3% |


**Codex** (697 total weakness items):

| Category | Count | % |
|---|---|---|
| Other | 212 | 30% |
| Limited scope | 117 | 17% |
| Weak baselines | 102 | 15% |
| Missing experiments | 73 | 10% |
| Unsupported claims | 71 | 10% |
| Novelty concerns | 54 | 8% |
| Missing ablations | 28 | 4% |
| Results mismatch | 17 | 2% |
| Code/crash issues | 12 | 2% |
| Reference issues | 11 | 2% |


### 6.2 Strong vs Weak Papers

| Metric | Strong (n=3) | Weak (n=19) |
|---|---|---|
| Mean word count | 3252 | 1947 |
| Mean figures | 2.3 | 0.5 |
| Mean tables | 5.7 | 4.3 |
| Mean citations | 18.7 | 16.3 |
| PDF compiled % | 67% | 74% |
| Mean weaknesses | 18.7 | 20.7 |


## 7. Case Studies


### 7.1 Best Papers

**Kimi — ai_for_biology_gpu/trial3** (score 5.3)

- Title: CellStratCP: Cell-Type-Stratified Adaptive Conformal Prediction for Calibrated Uncertainty in Single-Cell RNA-seq Imputation

- Words: 3061, Figures: 1, Tables: 5, Citations: 19

- Claude: 4, Codex: 4, Kimi: 8

- Abstract: *Single-cell RNA sequencing (scRNA-seq) imputation methods have achieved impressive accuracy in recovering dropout events, yet they lack statistically rigorous uncertainty quantification. Existing approaches either provide no uncertainty estimates or rely on Bayesian posteriors that require strong pa...*



**Codex — privacy_in_machine_learning_gpu/trial1** (score 5.3)

- Title: Who Was in the Recent Window? A Rigorous Audit of Online Test-Time Adaptation Privacy

- Words: 2919, Figures: 4, Tables: 4, Citations: 18

- Claude: 4, Codex: 4, Kimi: 8

- Abstract: *Online test-time adaptation (TTA) updates a deployed model on unlabeled user queries, creating a deployment-time privacy surface absent from frozen inference. We study one exact question: given black-box access to the deployed state at audit time t , can an attacker infer whether a candidate sample ...*



**Codex — supervised_representation_learning_gpu/trial3** (score 5.3)

- Title: How Much Signal Is in Early Training Trajectories? A Matched-Budget Study of Pseudo-Group Inference

- Words: 3777, Figures: 2, Tables: 8, Citations: 19

- Claude: 4, Codex: 6, Kimi: 6

- Abstract: *Annotation-free robustness methods often depend more on pseudo-group inference quality than on the downstream robust objective itself. Motivated by recent work on early spurious-bias detection and training-dynamics analysis, we ask a narrower question: under matched warmup compute, do short early-tr...*



**Kimi — operating_system_design_cpu/trial3** (score 4.7)

- Title: KAPHE: Kernel-Aware Performance Heuristic Extraction - A Lightweight Statistical Methodology for Interpretable OS Kernel Configuration

- Words: 1681, Figures: 0, Tables: 5, Citations: 13

- Claude: 4, Codex: 4, Kimi: 6

- Abstract: *Modern Linux kernels expose thousands of tunable parameters that significantly impact application performance, yet system administrators typically rely on static heuristics or black-box machine learning approaches that lack interpretability. We propose KAPHE (Kernel-Aware Performance Heuristic Extra...*



**Kimi — probabilistic_methods_cpu/trial3** (score 4.7)

- Title: Comparative Analysis of Adaptation Criteria for Gradient-Based Discrete MCMC: When Acceptance-Rate Trumps Jump-Distance

- Words: 1985, Figures: 0, Tables: 2, Citations: 30

- Claude: 4, Codex: 2, Kimi: 8

- Abstract: *Gradient-based discrete MCMC methods have shown promise for Bayesian inference in discrete spaces, but their effectiveness depends critically on hyperparameter adaptation. While multiple adaptation criteria exist---acceptance-rate targeting (ALBP), jump-distance maximization (AB-sampler), and cyclic...*




### 7.2 Worst Papers

**Kimi — compiler_optimization_cpu/trial2** (score 2.0)

- Title: LEOPARD: Lightweight Learned Guidance for Equality Saturation in Compiler Optimization

- Claude: 2, Codex: 0, Kimi: 4

  - *CRITICAL: The experiments use a SIMULATED environment, not a real implementation of equality saturation. The 'simulation.py' file shows simplified dyn*

  - *The results are suspiciously perfect: ALL 29 programs show exactly 2.0x speedup with zero variance across seeds for both LEOPARD and exhaustive ES. Th*

  - *The comparison with Aurora is based on estimated inference times (2.19ms vs 0.25ms), not actual measurements from running Aurora's code. The Aurora pa*



**Kimi — generative_models_gpu/trial1** (score 2.0)

- Title: Flow-Guided Token Routing: Adaptive Computation Allocation for Efficient Flow Matching

- Claude: 2, Codex: 0, Kimi: 4

  - *CRITICAL: The numbers in the paper DO NOT MATCH the experiment results. Paper reports baseline FID 251.5 and FlowRouter FID 298.7, but results_summary*

  - *CRITICAL: The paper claims 6.4x speedup, but results_summary.txt shows 0.91x speedup (slower, not faster)*

  - *CRITICAL: results.json indicates experiments were still 'in_progress' when the paper was written - results appear fabricated*



**Kimi — generative_models_gpu/trial3** (score 2.0)

- Title: VAST: Velocity-Adaptive Spatially-varying Timesteps for Training-Free Acceleration of Flow Matching Models

- Claude: 2, Codex: 2, Kimi: 2

  - *CRITICAL: The experiment logs show the VAST implementation CRASHED with 'RuntimeError: quantile() input tensor must be either float or double dtype' d*

  - *The novelty claim is undermined by RAS (Feb 2025), which proposes very similar training-free spatial adaptation based on noise magnitude*

  - *The paper switched from the originally proposed FLUX model to Stable Diffusion 1.5 due to access issues, significantly weakening the contribution*



**Kimi — supervised_representation_learning_gpu/trial1** (score 2.0)

- Title: Feature-Diversity-Aware Supervised Contrastive Learning: Mitigating Feature Suppression through Adaptive Pair Weighting

- Claude: 2, Codex: 0, Kimi: 4

  - *CRITICAL: Major results integrity issues - the paper reports results that do not exist in the results files or logs.*

  - *The paper claims FD-SCL achieves 6.20% fine accuracy and 83.12% coarse accuracy in Table 1, but no corresponding results file exists in the workspace.*

  - *Table 2 reports SCL standard CIFAR-100 accuracy of 76.42%, but the actual result is 66.21% (results/scl_cifar100_seed42.json).*



**Kimi — causal_learning_cpu/trial1** (score 2.7)

- Title: AIT-LCD: Adaptive Information-Theoretic Local Causal Discovery with Explicit Conditioning Set Awareness

- Claude: 2, Codex: 2, Kimi: 4

  - *Experimental evaluation is extremely limited: only ONE network (Asia, 8 nodes) was evaluated out of 5 planned networks. The authors admit this severel*

  - *The ablation study has critical implementation issues - identical scores between variants suggest flags are not properly propagated, preventing valid *

  - *The main claimed improvement (8% relative improvement, PC F1 0.592 vs 0.548) is NOT statistically significant (p=0.078), and the paper acknowledges th*




## 8. Conclusions

**Overall**: Codex (4.51) outscores Kimi (3.38) by +1.13 points.



**Key differences:**

- Codex writes longer papers (2791 vs 2011 words)

- Codex scores are more consistent (tight 4.0-5.3 range vs Kimi's 0.0-5.3)

- Both agents struggle most with experimental rigor and results integrity

- Kimi Code reviewer is consistently more generous than Claude Code and Codex reviewers

- CPU experiments score higher than GPU for both agents (simpler tasks, easier to implement)


## Appendix A: Full Score Table

| Agent | Seed | Trial | Claude | Codex | Kimi | Avg |
|---|---|---|---|---|---|---|
| codex | ai_for_biology_gpu | trial1 | 4 | 4 | 6 | 4.0 |
| codex | ai_for_biology_gpu | trial2 | 4 | 4 | 6 | 4.7 |
| codex | ai_for_biology_gpu | trial3 | 4 | 4 | 6 | 4.7 |
| codex | causal_learning_cpu | trial1 | 4 | 4 | 6 | 4.7 |
| codex | causal_learning_cpu | trial2 | 4 | 4 | 6 | 4.7 |
| codex | causal_learning_cpu | trial3 | 4 | 4 | 6 | 4.7 |
| codex | compiler_optimization_cpu | trial1 | 4 | 4 | 6 | 4.7 |
| codex | compiler_optimization_cpu | trial2 | 4 | 4 | 4 | 4.0 |
| codex | compiler_optimization_cpu | trial3 | 4 | 4 | 4 | 4.0 |
| codex | computer_vision_gpu | trial1 | 4 | 4 | 4 | 4.0 |
| codex | computer_vision_gpu | trial2 | 4 | 4 | 6 | 4.7 |
| codex | computer_vision_gpu | trial3 | 4 | 4 | 6 | 4.7 |
| codex | data_integration_and_cleaning_cpu | trial1 | 4 | 4 | 6 | 4.7 |
| codex | data_integration_and_cleaning_cpu | trial2 | 4 | 4 | 6 | 4.7 |
| codex | data_integration_and_cleaning_cpu | trial3 | 4 | 2 | 6 | 4.0 |
| codex | datasets_and_benchmarks_gpu | trial1 | 4 | 4 | 6 | 4.7 |
| codex | datasets_and_benchmarks_gpu | trial2 | 4 | 4 | 4 | 4.0 |
| codex | datasets_and_benchmarks_gpu | trial3 | 4 | 4 | 6 | 4.7 |
| codex | generative_models_gpu | trial1 | 4 | 2 | 6 | 4.0 |
| codex | generative_models_gpu | trial2 | 4 | 4 | 4 | 4.0 |
| codex | generative_models_gpu | trial3 | 4 | 4 | 6 | 4.7 |
| codex | interpretability_of_learned_representations_gpu | trial1 | 4 | 4 | 6 | 4.0 |
| codex | interpretability_of_learned_representations_gpu | trial2 | 4 | 4 | 6 | 4.7 |
| codex | interpretability_of_learned_representations_gpu | trial3 | 4 | 4 | 6 | 4.7 |
| codex | natural_language_processing_gpu | trial1 | 4 | 4 | 4 | 4.0 |
| codex | natural_language_processing_gpu | trial2 | 4 | 4 | 6 | 4.7 |
| codex | natural_language_processing_gpu | trial3 | 4 | 4 | 6 | 4.7 |
| codex | operating_system_design_cpu | trial1 | 4 | 4 | 6 | 4.7 |
| codex | operating_system_design_cpu | trial2 | 4 | 4 | 6 | 4.7 |
| codex | operating_system_design_cpu | trial3 | 4 | 4 | 6 | 4.7 |
| codex | privacy_in_machine_learning_gpu | trial1 | 4 | 4 | 8 | 5.3 |
| codex | privacy_in_machine_learning_gpu | trial2 | 4 | 2 | 6 | 4.0 |
| codex | privacy_in_machine_learning_gpu | trial3 | 4 | 4 | 6 | 4.7 |
| codex | probabilistic_methods_cpu | trial1 | 4 | 4 | 6 | 4.7 |
| codex | probabilistic_methods_cpu | trial2 | 4 | 4 | 6 | 4.7 |
| codex | probabilistic_methods_cpu | trial3 | 4 | 4 | 6 | 4.7 |
| codex | supervised_representation_learning_gpu | trial1 | 4 | 4 | 6 | 4.7 |
| codex | supervised_representation_learning_gpu | trial2 | 4 | 4 | 6 | 4.7 |
| codex | supervised_representation_learning_gpu | trial3 | 4 | 6 | 6 | 5.3 |
| kimi | ai_for_biology_gpu | trial1 | 4 | 2 | 6 | 4.0 |
| kimi | ai_for_biology_gpu | trial2 | 4 | 2 | 4 | 3.3 |
| kimi | ai_for_biology_gpu | trial3 | 4 | 4 | 8 | 5.3 |
| kimi | causal_learning_cpu | trial1 | 2 | 2 | 4 | 2.7 |
| kimi | causal_learning_cpu | trial2 | 4 | 2 | 6 | 4.0 |
| kimi | causal_learning_cpu | trial3 | 4 | 2 | 6 | 4.0 |
| kimi | compiler_optimization_cpu | trial1 | 4 | 2 | 6 | 4.0 |
| kimi | compiler_optimization_cpu | trial2 | 2 | 0 | 4 | 2.0 |
| kimi | compiler_optimization_cpu | trial3 | 2 | 2 | 4 | 2.7 |
| kimi | computer_vision_gpu | trial1 | 2 | 0 | 6 | 2.7 |
| kimi | computer_vision_gpu | trial2 | 2 | 4 | 4 | 3.3 |
| kimi | computer_vision_gpu | trial3 | 4 | 2 | 6 | 4.0 |
| kimi | data_integration_and_cleaning_cpu | trial1 | 4 | 0 | 6 | 3.3 |
| kimi | data_integration_and_cleaning_cpu | trial2 | 4 | 2 | 6 | 4.0 |
| kimi | data_integration_and_cleaning_cpu | trial3 | 4 | 2 | 6 | 4.0 |
| kimi | datasets_and_benchmarks_gpu | trial1 | 2 | 2 | 6 | 3.3 |
| kimi | datasets_and_benchmarks_gpu | trial2 | 2 | 0 | 6 | 2.7 |
| kimi | datasets_and_benchmarks_gpu | trial3 | 4 | 2 | 6 | 4.0 |
| kimi | generative_models_gpu | trial1 | 2 | 0 | 4 | 2.0 |
| kimi | generative_models_gpu | trial2 | 4 | 2 | 4 | 3.3 |
| kimi | generative_models_gpu | trial3 | 2 | 2 | 2 | 2.0 |
| kimi | interpretability_of_learned_representations_gpu | trial1 | 4 | 4 | 4 | 4.0 |
| kimi | interpretability_of_learned_representations_gpu | trial2 | 4 | 2 | 4 | 3.3 |
| kimi | interpretability_of_learned_representations_gpu | trial3 | 4 | 2 | 6 | 4.0 |
| kimi | natural_language_processing_gpu | trial1 | 2 | 2 | 4 | 2.7 |
| kimi | natural_language_processing_gpu | trial2 | 2 | 2 | 4 | 2.7 |
| kimi | natural_language_processing_gpu | trial3 | 4 | 2 | 4 | 3.3 |
| kimi | operating_system_design_cpu | trial1 | 4 | 2 | 4 | 3.3 |
| kimi | operating_system_design_cpu | trial2 | 4 | 2 | 6 | 4.0 |
| kimi | operating_system_design_cpu | trial3 | 4 | 4 | 6 | 4.7 |
| kimi | privacy_in_machine_learning_gpu | trial1 | 0 | 0 | 0 | 0.0 |
| kimi | privacy_in_machine_learning_gpu | trial2 | 4 | 4 | 4 | 4.0 |
| kimi | privacy_in_machine_learning_gpu | trial3 | 4 | 2 | 6 | 4.0 |
| kimi | probabilistic_methods_cpu | trial1 | 4 | 2 | 6 | 4.0 |
| kimi | probabilistic_methods_cpu | trial2 | 4 | 2 | 6 | 4.0 |
| kimi | probabilistic_methods_cpu | trial3 | 4 | 2 | 8 | 4.7 |
| kimi | supervised_representation_learning_gpu | trial1 | 2 | 0 | 4 | 2.0 |
| kimi | supervised_representation_learning_gpu | trial2 | 4 | 2 | 6 | 4.0 |
| kimi | supervised_representation_learning_gpu | trial3 | 2 | 2 | 4 | 2.7 |
