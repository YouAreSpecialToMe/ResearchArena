# Are Early Artifact Forecasts Actionable for Membership Privacy? A Budget-Matched Study of Selective Intervention

## Introduction

Membership inference leakage is highly uneven across training records. Recent papers already establish four ingredients that are relevant here: record-level privacy risk can be estimated from training artifacts, vulnerable records often become identifiable early in training, selective protection can outperform indiscriminate protection, and privacy evaluation should report disparity and low-FPR behavior rather than only average AUC. That leaves a narrower question than "invent a new defense family."

This proposal is therefore positioned as a **targeted empirical study** of whether forecasted risk is actionable under a tight compute budget. The contribution is not "selective protection is new." It is a pre-registered, budget-matched test of whether a simple forecast-driven policy improves the right records, under a fixed intervention budget, better than obvious alternatives.

The core question is:

**Do early artifact forecasts identify future privacy outliers well enough that a fixed selective smoothing budget reduces forecast-conditioned worst-case leakage and disparity more effectively than random targeting, static loss-only targeting, or global smoothing under the same runtime regime?**

This framing is intentionally conservative. MIST already argues that only vulnerable instances need extra protection. AdaMixup already adapts mixup strength during training. MIAShield already studies proactive privacy-aware intervention through exclusion oracles. Long et al. already show that well-generalized models still contain individually vulnerable records. Pollock et al. and Chen et al. already show that privacy risk is measurable from artifacts and often forecastable early. Because the prior art is close, this paper should be evaluated primarily as a careful mechanism-and-benchmarking study, not as a novelty-driven defense paper.

## Proposed Approach

### Overview

The method is intentionally simple and auditable:

1. Train with ordinary ERM for a short warm-up while logging per-sample artifacts.
2. Convert those artifacts into a coefficient-free forecast score.
3. Refresh a fixed-risk subset online and spend the same selective smoothing budget on that subset only.

The paper's positive result, if it appears, is not that the intervention itself is surprising. It is that **forecast-driven targeting is measurably more useful than equally cheap alternatives once the budget and protocol are held fixed**.

### Pre-registered protocol

The protocol is identical everywhere in this proposal:

- warm-up: first 15% of training epochs
- risky subset size: top 10% of training records by forecast score
- refresh schedule: every 4 epochs after warm-up
- main-study seeds: exactly 3 seeds for every guaranteed method and every guaranteed metric
- datasets: Purchase100 and CIFAR-10
- success criteria: identical across proposal sections and `idea.json`

No section in this proposal uses a different seed count or refresh schedule.

### Forecast score

During warm-up, the method logs four per-record signals:

- loss-trace interquartile range (LT-IQR)
- mean loss
- inverse margin
- prediction-flip count

Each feature is converted to a percentile rank within the training set. The warm-up forecast score is the equal-weight average of those four ranks:

\[
q_i=\frac{1}{4}\left(\mathrm{rank}(\mathrm{LT\mbox{-}IQR}_i)+\mathrm{rank}(\bar{\ell}_i)+\mathrm{rank}(-\bar{m}_i)+\mathrm{rank}(f_i)\right).
\]

After warm-up, the score is refreshed every 4 epochs using the same four features recomputed over the most recent window and combined with an exponential moving average over the previous score window with decay 0.5. No learned meta-predictor, attack labels, or shadow-model supervision is used.

### Selective intervention

At each refresh, the risky set \(R_t\) is the top 10% of training records by the current score. Only those records receive same-class vicinal smoothing:

\[
\tilde{x}_i=\lambda_i x_i + (1-\lambda_i)x_j,\qquad
\tilde{y}_i=\lambda_i y_i + (1-\lambda_i)y_j,
\]

where \(x_j\) is chosen from the lowest-risk 50% of class \(y_i\) using penultimate-layer nearest neighbors. The mixing coefficient is sampled from \(\mathrm{Beta}(\alpha_i,\alpha_i)\), with \(\alpha_i\) linearly mapped from 0.8 at the 90th risk percentile to 0.35 at the 100th percentile, so higher-risk records are smoothed more aggressively. Non-risky examples follow ordinary ERM.

This makes the intervention budget fixed and inspectable:

- exactly 10% of records are eligible at each refresh
- all targeting baselines use the same 10% budget
- the same refresh schedule is used for every targeting method

### What the paper does and does not claim

The paper does claim:

- a pre-registered test of whether early artifact forecasts are actionable;
- a matched-budget comparison against random and loss-only targeting;
- a lightweight but stronger-than-single-metric privacy evaluation;
- a realistic single-GPU protocol.

The paper does not claim:

- a new defense family;
- superiority to all recent defenses;
- robustness to white-box or heavy multi-shadow attack pipelines;
- that forecast-driven selective intervention is unprecedented in spirit.

If the results show that forecast-driven targeting is not better than loss-only or random targeting, the paper still has value as a negative empirical result about the practical usefulness of forecast signals.

## Related Work

### Instance-level vulnerability and susceptibility prediction

Long et al. (2018) show that even well-generalized models contain individually vulnerable records and propose GMIA procedures that explicitly search for such records. MIST (Li et al., 2024) makes a closely related conceptual claim: defenses should focus on vulnerable instances rather than uniformly protecting all records. Chen et al. (2025) study the dynamics of membership privacy and show that vulnerable records can often be identified before convergence. Pollock et al. (2024) show that record-level risk can be estimated from artifacts such as LT-IQR without shadow-model auditing.

**Positioning:** this proposal is downstream of this line of work. It does not claim to discover instance-level heterogeneity or early predictability. It tests whether those signals are strong enough to drive a fixed online intervention policy under a strict budget.

### Proactive or selective privacy-aware defenses

MIAShield (Jarin and Eshete, 2022) uses preemptive exclusion oracles at inference time, including learning-based exclusion mechanisms. MIST selectively suppresses memorization of vulnerable instances through membership-invariant subspace training. WEMEM (Shang et al., 2025) studies selective defense in iteratively pruned models.

**Positioning:** these papers are the main reason to narrow the novelty claim. This work differs mainly in protocol and scope: it studies a much simpler online trigger in ordinary training, under one fixed intervention budget and one fixed compute budget, to ask whether forecast quality alone makes selective intervention worthwhile.

### Adaptive or global empirical defenses

AdaMixup (Chen et al., 2025) dynamically adjusts mixup strength during training. RelaxLoss (Chen et al., 2022), CCL (Liu et al., 2024), HAMP (Chen and Pattabiraman, 2024), and adversarial regularization (Nasr et al., 2018) are practical empirical defenses that regularize all examples or all outputs.

**Positioning:** AdaMixup adapts global augmentation intensity, but not via a pre-registered per-record forecast-and-target protocol. RelaxLoss and related methods regularize everyone, not a forecasted subset. The proposed study asks whether a selective budget earns more protection on the records that matter most.

### Privacy evaluation and attacks

Carlini et al. (2022) motivate low-FPR likelihood-ratio attacks as the relevant stress test. Wang et al. (2025) show that privacy reporting should emphasize reliability and disparity, not only average attack strength.

**Positioning:** this proposal adopts that evaluation frame directly. The main outcomes are forecast-conditioned worst-decile leakage, disparity, and forecast quality, not just AUC.

## Experiments

### Main study

To fit one RTX A6000 and roughly 8 total hours, the guaranteed study is:

- datasets: Purchase100 and CIFAR-10
- models: 2-layer MLP for Purchase100, ResNet-18 for CIFAR-10
- seeds: exactly 3 seeds for all guaranteed methods
- methods: ERM, global mixup, RelaxLoss, random targeting, loss-only targeting, forecast-driven targeting

All targeting methods use the same warm-up, same 10% intervention budget, same refresh-every-4-epochs schedule, same same-class smoothing operator, and same checkpoint rule.

### Baselines

Guaranteed baselines:

- ERM
- global mixup
- RelaxLoss
- targeted-random: random 10% subset refreshed every 4 epochs
- targeted-loss-only: top 10% by mean loss refreshed every 4 epochs
- targeted-forecast: top 10% by composite artifact score refreshed every 4 epochs

Contingent baselines only if code reuse is genuinely cheap:

- AdaMixup
- MIST

These external baselines are explicitly optional because the paper's central claim is about matched-budget forecast usefulness, not exhaustive leaderboard ranking.

### Exact metric definitions

The proposal pre-registers three key definitions that were ambiguous before.

**1. Forecast-conditioned worst-decile leakage**

- For each method and seed, compute the forecast score \(q_i\) for every member after warm-up, before any post-warm-up intervention is applied.
- Partition member records into deciles by \(q_i\).
- Run the final membership attack on all evaluation points and calibrate the decision threshold using the non-member reference pool at 1% FPR.
- Worst-decile leakage is the attack TPR restricted to member records in the top forecast decile.

This metric is forecast-conditioned by construction; it does not define deciles using final attack scores.

**2. Privacy disparity**

- Using the same forecast deciles as above, compute attack TPR@1%FPR on the top forecast decile and on the pooled middle two deciles (50th-70th percentile, chosen as a more stable "median-risk" group).
- Privacy disparity is the difference:
\[
\Delta_{\mathrm{disp}}=\mathrm{TPR}_{\text{top forecast decile}}-\mathrm{TPR}_{\text{middle two forecast deciles}}.
\]

Lower is better.

**3. Forecast quality**

- Let \(s_i\) be the final attack score for member \(i\).
- Define final privacy outliers as the top 10% of members by \(s_i\).
- Report Spearman correlation between \(q_i\) and \(s_i\), and Precision@10, defined as the fraction of top-10%-by-\(q_i\) members that also land in the top-10%-by-\(s_i\) set.

This separates "forecast accuracy" from "privacy outcome."

### Attack evaluation

The attack suite is intentionally stronger than the earlier LiRA-lite-only sketch, but still feasible:

- primary attack: global-variance offline LiRA-lite / first-principles likelihood-ratio scoring
- stronger robustness check: class-conditional LiRA-lite, with mean and variance fit separately per class from the same reference pool
- secondary attack: loss-threshold attack

Claims are limited to **lightweight black-box score-based attacks**. The paper will not claim robustness to fully tuned many-shadow LiRA or white-box attacks. That limitation is explicit.

### Additional metrics

Primary metrics:

- LiRA-lite TPR@1% FPR
- class-conditional LiRA-lite TPR@1% FPR
- forecast-conditioned worst-decile leakage
- privacy disparity
- test accuracy
- macro-F1 on Purchase100
- privacy gain per extra training minute

Secondary metrics:

- attack AUC
- LiRA-lite TPR@0.1% FPR as exploratory
- Spearman forecast correlation
- Precision@10 for final privacy outliers
- risky-set Jaccard overlap across refreshes
- risky-set turnover rate

### Feasibility

The guaranteed workload is modest:

- 2 datasets x 3 seeds x 6 methods = 36 training runs
- no mandatory external baseline
- no mandatory multi-shadow retraining
- attack recalibration is mostly CPU-side post-processing

The paper no longer treats runtime efficiency as a broad headline claim. Instead it reports one narrow efficiency metric: privacy gain per extra training minute relative to ERM.

## Success Criteria

The study supports the hypothesis if most of the following hold:

1. Forecast-driven targeting improves forecast-conditioned worst-decile leakage or privacy disparity over ERM and global mixup on both datasets, with at most 1.5 absolute points of test-accuracy loss.
2. Forecast-driven targeting outperforms loss-only targeting and random targeting on worst-decile leakage under the same 10% intervention budget.
3. Forecast-driven targeting achieves better privacy gain per extra training minute than at least one global baseline, preferably RelaxLoss.
4. Forecast quality is non-trivial: Spearman correlation is positive and Precision@10 exceeds the random 10% baseline by a meaningful margin.
5. The improvement is visible under both global-variance and class-conditional LiRA-lite, even if effect sizes differ.

The hypothesis is weakened or refuted if:

- loss-only targeting matches or beats forecast-driven targeting;
- improvements appear only in average AUC but not in worst-decile leakage or disparity;
- the forecast set is too unstable to support an intervention story;
- gains disappear under class-conditional calibration;
- targeted intervention harms minority classes or hard examples disproportionately.

## References

1. Joseph Pollock, Igor Shilov, Euodia Dodd, and Yves-Alexandre de Montjoye. *Free Record-Level Privacy Risk Evaluation Through Artifact-Based Methods*. arXiv:2411.05743, 2024.
2. Yuetian Chen, Zhiqi Wang, Nathalie Baracaldo, Swanand Ravindra Kadhe, and Lei Yu. *Evaluating the Dynamics of Membership Privacy in Deep Learning*. arXiv:2507.23291, 2025.
3. Zhiqi Wang, Chengyu Zhang, Yuetian Chen, Nathalie Baracaldo, Swanand Kadhe, and Lei Yu. *Membership Inference Attacks as Privacy Tools: Reliability, Disparity and Ensemble*. arXiv:2506.13972, 2025.
4. Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramèr. *Membership Inference Attacks From First Principles*. IEEE Symposium on Security and Privacy, 2022.
5. Jiacheng Li, Ninghui Li, and Bruno Ribeiro. *MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training*. arXiv:2311.00919, 2024.
6. Ying Chen, Jiajing Chen, Yijie Weng, ChiaHua Chang, Dezhi Yu, and Guanbiao Lin. *AdaMixup: A Dynamic Defense Framework for Membership Inference Attack Mitigation*. arXiv:2501.02182, 2025.
7. Ismat Jarin and Birhanu Eshete. *MIAShield: Defending Membership Inference Attacks via Preemptive Exclusion of Members*. arXiv:2203.00915, 2022.
8. Yunhui Long, Vincent Bindschaedler, Lei Wang, Diyue Bu, Xiaofeng Wang, Haixu Tang, Carl A. Gunter, and Kai Chen. *Understanding Membership Inferences on Well-Generalized Learning Models*. arXiv:1802.04889, 2018.
9. Dingfan Chen, Ning Yu, and Mario Fritz. *RelaxLoss: Defending Membership Inference Attacks without Losing Utility*. ICLR, 2022.
10. Jing Shang, Jian Wang, Kailun Wang, Jiqiang Liu, Nan Jiang, Md. Armanuzzaman, and Ziming Zhao. *Defending Against Membership Inference Attacks on Iteratively Pruned Deep Neural Networks*. NDSS Symposium, 2025.
11. Zhenlong Liu, Lei Feng, Huiping Zhuang, Xiaofeng Cao, and Hongxin Wei. *Mitigating Privacy Risk in Membership Inference by Convex-Concave Loss*. ICML, 2024.
12. Zitao Chen and Karthik Pattabiraman. *Overconfidence is a Dangerous Thing: Mitigating Membership Inference Attacks by Enforcing Less Confident Prediction*. NDSS Symposium, 2024.
13. Milad Nasr, Reza Shokri, and Amir Houmansadr. *Machine Learning with Membership Privacy using Adversarial Regularization*. ACM CCS, 2018.
