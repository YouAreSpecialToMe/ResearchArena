# Difficulty-Aware Unlearning: Closing the Per-Sample Privacy Gap in Machine Unlearning

## Introduction

### Context
Machine unlearning — the process of removing the influence of specific training data from a trained model — has become a critical capability driven by privacy regulations like GDPR's "right to be forgotten" (Bourtoule et al., 2021). A key challenge is *verifying* that unlearning actually worked: did the model truly forget the requested data, or does it retain residual information? The dominant verification approach uses Membership Inference Attacks (MIAs) as a proxy test: if an attacker cannot distinguish whether a sample was in the original training set by examining the unlearned model, then unlearning is deemed successful (Chen et al., 2021; Kurmanji et al., 2023).

### Problem Statement
Recent work has established that aggregate MIA-based unlearning evaluations provide a false sense of privacy. Hayes et al. (2025) demonstrated this convincingly by distinguishing *population U-MIAs* (a single attacker for all examples) from *per-example U-MIAs* (a dedicated attacker per example), showing that the latter reveals significant privacy failures masked by aggregate metrics. Concurrently, the unlearning difficulty literature has shown that unlearning success varies dramatically across individual samples: Rizwan et al. (2024) identified four instance-level factors that make unlearning difficult, Zhao et al. (2024) showed that forget set characteristics substantially affect algorithm performance, and Cheng et al. (2026) proposed circuit-level difficulty metrics showing that hard-to-unlearn samples rely on deeper model pathways.

**The gap we address**: While prior work has diagnosed the problem — that per-sample unlearning quality varies and aggregate metrics are misleading — the question of *how to defend against it* remains largely open. Hayes et al. (2025) focus on better evaluation but propose no defense. Zhao et al. (2024) propose RUM, a meta-algorithm that partitions forget sets by characteristics, but do not address the connection to MIA-based privacy auditing or target the hardest samples specifically. No existing work provides a unified framework that (1) uses principled difficulty calibration from the MIA literature to stratify unlearning evaluation AND (2) uses the same difficulty signal to drive an adaptive defense that equalizes privacy protection across difficulty strata.

### Key Insight
Watson et al. (2022) established that difficulty calibration dramatically improves MIA accuracy. We observe that this same calibration framework can be *repurposed* for unlearning: rather than using difficulty to improve attacks, we use it to (a) expose where unlearning fails via stratified evaluation, and (b) drive an adaptive defense (DAU) that allocates more unlearning effort to vulnerable samples. The specific technical contribution is bridging Watson et al.'s calibration framework into the unlearning evaluation and defense pipeline — a connection not made by prior work on per-example attacks (Hayes et al., 2025) or unlearning difficulty (Rizwan et al., 2024; Zhao et al., 2024).

### Hypothesis
Current machine unlearning methods exhibit *difficulty-dependent privacy gaps*: they provide weaker privacy protection for hard (atypical) samples than for easy (typical) samples. A difficulty-calibrated evaluation framework will quantify these gaps more precisely than existing per-example attacks, and a difficulty-aware unlearning algorithm (DAU) can close the gap — providing uniform privacy protection across difficulty strata with minimal utility loss.

## Proposed Approach

### Overview
We propose **Difficulty-Calibrated Unlearning Auditing (DCUA)** with two main contributions:

1. **Stratified Evaluation Protocol** (diagnostic contribution): Partition the forget set into difficulty quintiles using reference-model calibration (Watson et al., 2022), and evaluate unlearning quality per stratum. This is computationally cheaper than Hayes et al.'s per-example U-MIA (which requires per-example shadow models) while providing similar diagnostic power through principled stratification.

2. **Difficulty-Aware Unlearning (DAU)** (defense contribution — primary novelty): An adaptive wrapper for any gradient-based unlearning method that scales the unlearning signal inversely with sample difficulty, providing stronger forgetting guarantees for vulnerable (hard) samples. This is the first defense specifically designed to equalize privacy protection across difficulty strata.

### Method Details

#### Component 1: Difficulty Estimation
We train K reference models on data from the same distribution as the target model's training data (but disjoint from the forget set). For each sample x in the forget set, we compute:

- **Reference loss**: The average loss of reference models on x: `d(x) = (1/K) Σ_k L(f_k, x)`
- **Difficulty score**: We normalize reference losses to obtain difficulty quintiles Q1 (easiest) through Q5 (hardest).

This mirrors the reference-model approach used in LiRA (Carlini et al., 2022) but applied to characterize sample difficulty rather than to perform the attack itself. Compared to Hayes et al.'s per-example U-MIA, which trains separate shadow models per example, our approach amortizes the reference model cost: K models serve all samples, making it significantly more computationally efficient.

#### Component 2: Stratified Evaluation Protocol
For each difficulty quintile Qi, we compute:

- **Per-stratum MIA-AUC**: The area under the ROC curve for a membership inference attack restricted to samples in Qi.
- **Per-stratum Forgetting Rate**: The fraction of samples in Qi for which MIA fails (below a threshold).
- **Difficulty Gap (DG)**: The difference in MIA-AUC between the hardest and easiest quintiles: `DG = AUC(Q5) - AUC(Q1)`. A large DG indicates difficulty-dependent privacy protection.
- **Worst-Quintile AUC (WQ-AUC)**: MIA-AUC on the hardest quintile Q5 only. This is the metric that matters most for privacy, since the hardest samples are the most vulnerable.

We compare these metrics against both aggregate MIA-AUC and Hayes et al.'s per-example U-MIA to quantify how much standard evaluation overestimates unlearning quality.

#### Component 3: Difficulty-Aware Unlearning (DAU) — Primary Contribution
We modify standard unlearning objectives with a per-sample difficulty weight:

For gradient ascent (GA) unlearning:
```
L_DAU = Σ_{x ∈ D_forget} w(x) · L(f, x)
```
where `w(x) = 1 + α · (d(x) - d_mean) / d_std` is a difficulty-dependent weight that amplifies the unlearning signal for hard samples.

For knowledge distillation-based methods (SCRUB), we increase the "disagreement" loss weight for hard samples:
```
L_SCRUB-DAU = Σ_{x ∈ D_forget} w(x) · KL(f_student(x) || Uniform) + Σ_{x ∈ D_retain} KL(f_student(x) || f_teacher(x))
```

The hyperparameter α controls the strength of difficulty calibration. We perform sensitivity analysis over α ∈ {0.5, 1.0, 2.0, 5.0}.

**Why DAU is different from prior defenses**: Zhao et al.'s RUM (2024) partitions forget sets by characteristics (e.g., class membership, memorization) and applies different algorithms to each partition. DAU instead operates within a single algorithm by reweighting the loss — it is a continuous, fine-grained adjustment rather than a discrete partition-and-dispatch approach. Sepahvand et al. (2025) calibrate noise in DP-based unlearning using per-instance privacy bounds; DAU is algorithm-agnostic and works with any gradient-based unlearning method without requiring DP machinery.

## Related Work

### Machine Unlearning Evaluation: The Per-Sample Gap
Hayes et al. (2025) is the closest prior work to our evaluation contribution. They showed that *population U-MIAs* (same attacker for all examples) dramatically underestimate privacy risk compared to *per-example U-MIAs* (dedicated attacker per example). Their key finding — that aggregate metrics create a false sense of privacy — motivates our work. **How we differ**: (1) Our stratified evaluation uses difficulty calibration from reference models (Watson et al., 2022) as the stratification mechanism, which is more principled and computationally cheaper than training per-example shadow models; (2) We go beyond diagnosis to propose DAU, a defense that actively closes the per-sample privacy gap; (3) We introduce specific metrics (WQ-AUC, DG) for quantifying the difficulty-dependent gap. We include Hayes et al.'s per-example U-MIA as an evaluation baseline to directly demonstrate the complementary strengths of our approach.

### Unlearning Difficulty
A growing literature studies what makes unlearning hard at the instance level:
- **Rizwan et al. (2024)** identified four factors making unlearning difficult (memorization, atypicality, loss landscape, gradient alignment) through instance-level analysis across algorithms and datasets. Their factors are algorithm-independent, depending only on the target model and training data. Our difficulty estimation via reference models captures a complementary signal (population-level hardness) and is specifically designed to connect to MIA vulnerability.
- **Zhao et al. (NeurIPS 2024)** showed that forget set characteristics substantially affect unlearning difficulty and proposed RUM, which partitions forget sets into homogeneous subsets and applies existing algorithms per subset. RUM improves unlearning quality but operates at the subset level; DAU provides continuous per-sample reweighting and specifically targets MIA-detectable privacy gaps.
- **Cheng et al. (2026, preprint)** proposed Circuit-guided Unlearning Difficulty (CUD), a mechanistic metric using circuit-level signals to predict per-sample unlearning difficulty. Their work provides complementary theoretical grounding; our difficulty scores could potentially be replaced by CUD scores in future work.

### Difficulty Calibration in MIA
Watson et al. (2022) established that difficulty calibration dramatically improves MIA by adjusting membership scores based on sample hardness. He et al. (2024) identified limitations of pure difficulty calibration and proposed RAPID. Shi et al. (2024) introduced LDC-MIA with learned difficulty classifiers. Carlini et al. (2022) developed LiRA, the state-of-the-art likelihood ratio attack that implicitly accounts for difficulty through reference model comparisons. **Our contribution is to apply difficulty calibration not as an attack tool but as an evaluation stratification and defense mechanism for unlearning.**

### Machine Unlearning Algorithms
Bourtoule et al. (2021) introduced SISA training for exact unlearning through data sharding. Golatkar et al. (2020) proposed Fisher information-based scrubbing for approximate unlearning. Kurmanji et al. (2023) introduced SCRUB, a teacher-student framework that alternates between forgetting and retaining. These methods are evaluated primarily using aggregate metrics — our work shows these metrics are misleading and proposes DAU to improve their per-sample privacy guarantees.

### Unlearning Verification and Auditing
Chen et al. (2021) demonstrated that machine unlearning can paradoxically *increase* privacy risk by creating detectable differences between original and unlearned models. Naderloui et al. (2025) proposed RULI, a per-sample attack framework for unlearning evaluation. Sun et al. (2026, preprint, arXiv:2602.01150) proposed SMIA, a training-free statistical auditing approach. Our framework is complementary to these attack methods — SMIA or RULI could be used as the per-stratum audit within our DCUA framework.

### Per-Instance Privacy and Unlearning
Sepahvand et al. (2025) leveraged per-instance differential privacy bounds to characterize unlearning difficulty and achieve better utility-unlearning tradeoffs. Their work focuses on theoretical analysis of noisy fine-tuning; DAU is algorithm-agnostic and works with any gradient-based unlearning method.

### How Our Work Differs
| Aspect | Hayes et al. (SaTML 2025) | Rizwan et al. (2024) | Zhao et al. (NeurIPS 2024) | Cheng et al. (2026) | **Ours (DCUA + DAU)** |
|---|---|---|---|---|---|
| Focus | Per-example evaluation | Instance difficulty factors | Forget set characteristics | Mechanistic difficulty | **Evaluation + defense** |
| Difficulty-aware | Per-example shadow models | 4 difficulty factors | Forget set partitioning | Circuit-level scores | **Reference-model calibration** |
| Defense proposed | No | No | RUM (partition-dispatch) | No | **DAU (continuous reweighting)** |
| Computational cost | High (per-example) | Analysis only | Moderate | Analysis only | **Low (amortized reference models)** |
| MIA connection | U-MIA evaluation | No direct connection | No direct connection | No direct connection | **Bridges MIA calibration → defense** |

## Experiments

### Setup
**Datasets:**
- CIFAR-10 (60K images, 10 classes) — standard vision benchmark
- CIFAR-100 (60K images, 100 classes) — harder classification task with more difficulty variation
- Purchase-100 (Kaggle "acquire-valued-shoppers-challenge" dataset, processed following Shokri et al. (2017): binarized purchase histories for 197K customers over 100 product categories) — tabular data benchmark

**Models:**
- ResNet-18 for CIFAR-10/100
- 3-layer MLP (512-256-128) for Purchase-100

**Unlearning Algorithms (baselines):**
1. **Retrain**: Gold standard — retrain from scratch without forget set
2. **Fine-tune (FT)**: Continue training on retain set only
3. **Gradient Ascent (GA)**: Maximize loss on forget set
4. **Random Labels (RL)**: Fine-tune on forget set with random labels
5. **SCRUB**: Teacher-student distillation (Kurmanji et al., 2023)
6. **NegGrad+KD**: Negative gradient with knowledge distillation

**Forget Set Configuration:**
- Random forget sets of size 500, 1000, 2500 samples (from 50K training set)
- Class-level forget sets (remove entire class)

**MIA Methods:**
- Loss-based threshold attack
- LiRA (Carlini et al., 2022) with 8 shadow models
- Difficulty-calibrated attacks (Watson et al., 2022)
- **Per-example U-MIA (Hayes et al., 2025)** — included as evaluation baseline to directly compare our stratified approach against per-example attacks

**Seeds:** 3 random seeds per configuration for error bars.

### Evaluation Metrics
1. **Standard metrics**: Aggregate MIA-AUC, Unlearning Accuracy (UA), Retain Accuracy (RA), Test Accuracy (TA)
2. **DCUA metrics**: Per-quintile MIA-AUC, Worst-Quintile AUC (WQ-AUC), Difficulty Gap (DG)
3. **Defense metrics**: WQ-AUC reduction from DAU vs. standard unlearning; comparison of DAU vs. RUM (Zhao et al., 2024)

### Expected Results
1. **Revelation**: Aggregate MIA-AUC will show most unlearning methods performing close to Retrain (gold standard), but WQ-AUC will reveal that the hardest quintile (Q5) has significantly higher MIA success — consistent with Hayes et al.'s per-example findings but quantified through our stratified lens.
2. **Difficulty Gap**: We expect DG > 0.15 for most approximate unlearning methods, meaning the privacy protection varies substantially across difficulty strata.
3. **DAU improvement**: Our difficulty-aware unlearning should reduce WQ-AUC by 5-15% absolute compared to standard versions, with minimal (<1%) degradation in retain/test accuracy. We expect DAU to outperform RUM on the hardest quintile specifically, since DAU continuously reweights while RUM uses discrete partitions.
4. **Efficiency**: Our stratified evaluation should require ~K reference models (K=8) total, vs. O(n) shadow models for Hayes et al.'s per-example U-MIA, demonstrating a practical advantage.

### Ablation Studies
1. **Number of reference models K**: Test K ∈ {2, 4, 8, 16} for difficulty estimation quality
2. **Number of difficulty strata**: Compare quintiles vs. terciles vs. deciles
3. **DAU strength α**: Sensitivity analysis over α ∈ {0.5, 1.0, 2.0, 5.0}
4. **Forget set size**: How does the difficulty gap change with forget set size?
5. **DAU vs. RUM**: Direct comparison of continuous reweighting vs. partition-dispatch on WQ-AUC

## Success Criteria

### Confirming the hypothesis
The hypothesis is confirmed if:
1. WQ-AUC is significantly higher (>0.05 absolute) than aggregate MIA-AUC for at least 3 out of 5 unlearning methods, demonstrating that aggregate metrics overestimate unlearning quality.
2. The Difficulty Gap (DG) is statistically significant (p < 0.05) across datasets, confirming that unlearning quality depends on sample difficulty.
3. DAU reduces WQ-AUC compared to both standard unlearning methods and RUM without substantially degrading model utility (retain accuracy drop < 1%).

### Refuting the hypothesis
The hypothesis would be refuted if:
1. WQ-AUC and aggregate MIA-AUC are statistically indistinguishable across all methods and datasets (difficulty doesn't matter for unlearning).
2. All unlearning methods provide uniform protection across difficulty strata (no gap exists).
3. DAU provides no improvement over standard unlearning on hard samples.

## References

1. Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C., et al. "Machine Unlearning." IEEE Symposium on Security and Privacy, 2021.
2. Carlini, N., Chien, S., Nasr, M., et al. "Membership Inference Attacks From First Principles." IEEE Symposium on Security and Privacy, 2022.
3. Chen, M., Zhang, Z., Wang, T., Backes, M., Humbert, M., Zhang, Y. "When Machine Unlearning Jeopardizes Privacy." ACM CCS, 2021.
4. Cheng, J., et al. "Toward Understanding Unlearning Difficulty: A Mechanistic Perspective and Circuit-Guided Difficulty Metric." arXiv:2601.09624, 2026 (preprint).
5. Golatkar, A., Achille, A., Soatto, S. "Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks." CVPR, 2020.
6. Hayes, J., Shumailov, I., Triantafillou, E., Khalifa, A., Papernot, N. "Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy." IEEE SaTML, 2025.
7. He, Y., Li, B., Wang, Y., Yang, M., Wang, J., Hu, H., Zhao, X. "Is Difficulty Calibration All We Need? Towards More Practical Membership Inference Attacks." ACM CCS, 2024.
8. Kurmanji, M., Triantafillou, P., Hayes, J., Triantafillou, E. "Towards Unbounded Machine Unlearning." NeurIPS, 2023.
9. Naderloui, N., Yan, S., Wang, B., Fu, J., Wang, W.H., Liu, W., Hong, Y. "Rectifying Privacy and Efficacy Measurements in Machine Unlearning: A New Inference Attack Perspective." USENIX Security, 2025.
10. Rizwan, H., Sarvmaili, M., Sajjad, H., Wu, G. "Towards Understanding the Feasibility of Machine Unlearning." arXiv:2410.03043, 2024.
11. Sepahvand, N.M., Thudi, A., Isik, B., et al. "Leveraging Per-Instance Privacy for Machine Unlearning." ICML, 2025.
12. Shi, H., Ouyang, T., Wang, A. "Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks." IEEE Euro S&P, 2024.
13. Shokri, R., Stronati, M., Song, C., Shmatikov, V. "Membership Inference Attacks Against Machine Learning Models." IEEE Symposium on Security and Privacy, 2017.
14. Sun, J., Wei, Z., Zou, J., et al. "Statistical MIA: Rethinking Membership Inference Attack for Reliable Unlearning Auditing." arXiv:2602.01150, 2026 (preprint).
15. Thudi, A., Deza, G., Chandrasekaran, V., Papernot, N. "Unrolling SGD: Understanding Factors Influencing Machine Unlearning." IEEE Euro S&P, 2022.
16. Wang, C.-L., Li, Q., Xiang, Z., Cao, Y., Wang, D. "Towards Lifecycle Unlearning Commitment Management: Measuring Sample-level Approximate Unlearning Completeness." arXiv:2403.12830, 2024.
17. Watson, L., Guo, C., Cormode, G., Sablayrolles, A. "On the Importance of Difficulty Calibration in Membership Inference Attacks." ICLR, 2022.
18. Zhao, K., Kurmanji, M., Barbulescu, G.-O., Triantafillou, E., Triantafillou, P. "What Makes Unlearning Hard and What to Do About It." NeurIPS, 2024.
