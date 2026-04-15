# Matched-Budget Evaluation of Weak-View Residuals for One-Run DP Auditing

## Introduction

Empirical privacy auditing is used to test whether a differentially private training pipeline behaves as privately as its accountant claims. One-run auditing made this practical by packing many neighboring comparisons into a single training run, and later work tightened the conversion from empirical distinguishability to privacy statements. Most recently, Liu et al. (2025) showed that in the final-model-only black-box setting, the per-example score matters substantially: quantile-regression calibration improves one-run auditing over raw loss or confidence.

This proposal asks a deliberately narrow follow-up question:

**After adopting the Liu et al. (2025) single-view quantile baseline, do repeated weak, label-preserving views add measurable privacy evidence in one-run DP auditing when every consumed query is accounted for honestly?**

The intended contribution is incremental and protocol-focused. This is not a new class of membership inference attack, and it is not a claim that multi-view or repeated-query MIA is novel. Prior work already shows that extra queries can help membership inference in other settings, including augmentation-based attacks, label-only attacks that estimate robustness with many queries, and repeated-query sampling attacks. The contribution here is narrower: a matched-budget empirical evaluation of whether weak-view residual scores add any privacy evidence beyond Liu et al. (2025) in the one-run DP-auditing setting.

The result could be positive or negative. If weak-view statistics help only when extra queries are ignored, or if they add nothing once the clean-view score is quantile-calibrated, then the correct conclusion is that the stronger single-view baseline already captures the practically useful signal in this regime. Given the compute budget, the paper should be presented explicitly as a narrow, directional audit rather than a definitive benchmark.

## Proposed Approach

### Scope and claim

The project keeps the standard one-run neighboring-dataset construction, the \(f\)-DP / \(\varepsilon\)-lower-bound conversion, and the target training pipeline fixed. The only change is the candidate score and the fairness protocol used to compare single-view and weak-view auditing.

The paper's claim is therefore modest:

- reuse the Liu et al. single-view quantile residual as the anchor baseline
- add a fixed family of derived weak-view statistics
- evaluate those scores under honest total-query accounting and randomized matched-budget curves

### Threat model

- one-run neighboring-dataset auditing as in Steinke et al. (2023)
- final-model-only black-box access to the trained DP model
- public or external non-member data for calibration, screening, and score selection
- a finite audit pool of member/non-member candidate pairs fixed before evaluation
- a small number of test-time weak augmentations per candidate

No hidden-state access, no gradient access to the audited DP model, no shadow retraining on private data, and no change to the privacy conversion are assumed.

### Data splits and leakage control

To remove the leakage and tuning ambiguity from the previous draft, every learned auxiliary object is trained only on public or external data. For each dataset, split the non-private data into four disjoint components:

1. **Public calibration split.** Used to fit all quantile regressors for clean and derived weak-view statistics.
2. **Public proxy-audit split.** Used to build cheap public-only pseudo-audits for selecting \(\lambda\) against the main privacy metric.
3. **Public transform-screening split.** Used only to test whether candidate augmentations remain label preserving.
4. **Private audit pool.** A fixed finite set of neighboring candidate pairs used for the actual privacy audit. This pool is never used for fitting or tuning.

Any reference model, screening model, or proxy model is trained only on public or external data. No auxiliary model is trained on the private audit candidates or on the target model's private training split.

### Base statistic and residualization

For each queried input \(x\) with label \(y\), compute one scalar base statistic \(b(x, y)\). The main choice is negative loss, with margin as a prespecified sensitivity check if time permits.

Let \(z(x)\) denote non-private difficulty features available from the public splits, such as public-reference confidence, class label, and clean-input loss under a public reference model. Following Liu et al. (2025), fit upper-tail quantile regressors on the public calibration split so that each raw statistic is converted into a residualized score relative to examples of similar public difficulty.

Each derived weak-view statistic gets its own quantile model fit on public data:

- \(r_{\text{clean}}(x)\): residual of the clean-view base statistic
- \(r_{\text{mean}}(x)\): residual of the mean weak-view statistic
- \(r_{\text{worst}}(x)\): residual of the minimum weak-view statistic
- \(r_{\text{var}}(x)\): residual of the variance across weak-view statistics
- \(r_{\text{agree}}(x)\): residual of a prediction-agreement summary across weak views

This avoids the earlier mismatch where multi-view statistics could have been judged by a clean-view calibrator.

### Weak-view score family

For each candidate \(x\), query:

- the clean input \(x\)
- \(K\) weak, label-preserving views \(a_1(x), \dots, a_K(x)\)

The main structured score is:

\[
S_{\text{weak}}(x) = \max \{ r_{\text{clean}}(x),\ r_{\text{worst}}(x),\ \lambda \, r_{\text{agree}}(x) \}.
\]

The project also evaluates two simpler controls:

- **Single-view quantile baseline:** \(S_{\text{clean}}(x) = r_{\text{clean}}(x)\)
- **Weak-view mean-only control:** \(S_{\text{mean}}(x) = r_{\text{mean}}(x)\)

This separates three questions:

1. Does quantile calibration beat raw single-view scoring?
2. Do multiple weak views help at all?
3. Do worst-view or agreement features help beyond simply averaging extra weak-view queries?

### Public-only selection of \(\lambda\)

The primary paper claim is about empirical privacy evidence, not just ROC AUC. Accordingly, \(\lambda\) is selected against the privacy-auditing objective using only public data.

The protocol is:

1. Build 2 to 3 cheap **proxy auditors** per dataset from the public proxy-audit split, for example by training small non-private classifiers or linear probes on disjoint public subsets and treating held-in versus held-out public points as pseudo-member and pseudo-nonmember populations.
2. For each candidate \(\lambda\) on a small grid, run the same matched-budget audit pipeline used in the main experiment on these public pseudo-audits.
3. Choose the \(\lambda\) that maximizes the median empirical \(\varepsilon\) lower bound at a fixed public query budget \(Q_{\text{sel}}\).

As a robustness check, also record the AUC-selected \(\lambda\). If the AUC-tuned and \(\varepsilon\)-tuned choices agree, report that agreement explicitly. If they differ, use the \(\varepsilon\)-tuned choice for the headline result and show the AUC-tuned choice as a sensitivity analysis.

This keeps all tuning public-only while aligning score selection with the actual paper objective.

### Matched-budget protocol

The previous draft mixed candidate count and query count, and it treated a single ordered prefix as the main curve. This proposal removes both issues by fixing one finite audit pool and averaging results over multiple randomized audit-pool orderings.

Let the audit pool contain \(N\) candidate pairs for a given trained checkpoint. If a method audits \(m\) candidates, its total query cost is:

- single-view baseline: \(q_{\text{clean}}(m) = m\)
- weak-view methods using clean plus \(K\) views: \(q_{\text{weak}}(m) = (1 + K)m\)

Thus the clean query is explicitly counted.

For each checkpoint, sample \(R\) random permutations of the same audit pool, with \(R = 20\) by default and \(R = 10\) as the fallback if runtime is tighter than expected. Report the mean curve and dispersion over these orderings.

Two evaluation modes are reported, both averaged over the \(R\) random orderings:

1. **Same-candidate comparison.** All methods are evaluated on the same first \(m\) candidates of each randomized ordering. This shows whether extra weak-view queries can help if one is willing to spend more queries per candidate.
2. **Matched-total-query comparison.** For a query budget \(Q\), compare:
   - the single-view baseline on \(m_{\text{clean}} = \min(N, Q)\)
   - a weak-view method on \(m_{\text{weak}} = \left\lfloor Q / (1+K) \right\rfloor\)

The primary figure is privacy evidence versus consumed queries \(Q\), averaged over random orderings. Prefix-only single-order plots are relegated to appendix diagnostics.

### Augmentation screening

Weak views must be credibly label preserving. Before the main audit, each transform family is screened separately on:

- a public labeled validation split
- public-set FGSM examples generated with the same recipe used for FGSM canaries

The initial transform pool is intentionally conservative:

- small translations
- mild crops
- horizontal flips only where class preserving
- mild color jitter
- low-variance Gaussian noise

A transform family is retained only if, on both public clean data and public FGSM data:

- accuracy under a strong public reference model drops by at most 1 percentage point from the clean input
- at least 98% of transformed examples retain the original label prediction of that reference model

This filter is fixed before the private audit. If very few transforms survive for FGSM canaries, the study reports that limitation instead of forcing a weak-view claim.

### Why weak views might help

The mechanism under test is narrow. After quantile calibration removes much of the clean-view difficulty effect, weak views could still help if membership leakage also manifests as unusually stable local behavior around a point:

- members may remain unusually confident under several weak perturbations
- worst-case weak views may reveal brittleness differences invisible from one clean query
- agreement statistics may capture local consistency not explained by clean-view difficulty

The opposite outcome is also plausible: once the clean-view residual is well calibrated, weak views may be mostly redundant. That is an acceptable negative result.

## Related Work

### One-run auditing and privacy conversion

Steinke, Nasr, and Jagielski (2023) introduced one-run privacy auditing, which provides the neighboring-dataset backbone of this proposal. Mahloujifar, Melis, and Chaudhuri (2025) strengthened one-run privacy conversion using \(f\)-DP analysis. This project leaves both pieces untouched and changes only the black-box score and evaluation protocol.

### Closest baseline: quantile-regression one-run auditing

Liu et al. (2025) is the direct baseline and the main paper extended here. Their central finding is that calibrating a single-view black-box statistic with quantile regression substantially improves one-run auditing. This sharply raises the bar for follow-up work: any extra signal source must beat a calibrated score, not a weak loss baseline. The present proposal is therefore framed as an incremental matched-budget evaluation around that specific baseline.

### Repeated-query and multi-view membership inference

Semi-Leak (He et al., 2022) already showed that repeated augmented black-box queries can help membership inference in semi-supervised learning.

Sampling Attacks (Rahimian, Orekondy, and Fritz, 2020) made repeated queries central in a different way: by repeatedly sampling label-only outputs, it amplifies membership signal even when the victim exposes only hard labels.

Label-Only Membership Inference Attacks (Choquette-Choo et al., 2021) likewise depends on many extra queries to estimate robustness under perturbations, and You Only Query Once (Wu et al., 2024) is explicitly motivated by reducing the high query cost of that repeated-query label-only line.

These papers are important prior art because they already establish that extra queries can improve generic membership inference. But they do **not** already answer the question studied here. They target standard attack accuracy rather than one-run DP auditing, do not convert attack performance into empirical \(\varepsilon\) evidence, do not benchmark against the Liu et al. quantile-regression one-run baseline, and generally do not enforce a fixed-audit-pool matched-query protocol where every clean and augmented query is charged. That is why they constrain the novelty claim but do not subsume the proposed study.

### Other recent auditing directions

Nearly Tight Black-Box Auditing of Differentially Private Machine Learning (Annamalai and De Cristofaro, 2024), Adversarial Sample-Based Approach for Tighter Privacy Auditing in Final Model-Only Scenarios (Yoon et al., 2024), PANORAMIA (Kazmi et al., 2024), Tighter Privacy Auditing of DP-SGD in the Hidden State Threat Model (Cebere et al., 2025), Sequentially Auditing Differential Privacy (Gonzalez et al., 2025), and Optimizing Canaries for One-Run Privacy Auditing of DP-SGD (Wang et al., 2026) strengthen auditing through tighter analysis, stronger canaries, or alternative threat models. The present study is orthogonal: it keeps the standard one-run setup and asks only whether candidate scoring improves under a query-accounted black-box protocol.

## Experiments

### Compute-bounded setup

The experiment matrix is scoped to fit one RTX A6000 and about 8 hours total:

- **Fashion-MNIST:** small DP CNN
- **CIFAR-10:** DP fine-tuning of only the final linear layer of a pretrained ResNet-18

Privacy settings:

- \(\varepsilon \approx 4\), \(\delta = 10^{-5}\)
- \(\varepsilon \approx 8\), \(\delta = 10^{-5}\)

Random seeds:

- 3 seeds

Total target-model jobs:

- 2 datasets x 2 privacy levels x 3 seeds = 12 DP training jobs

All auditing methods reuse these same checkpoints. No method gets extra target retraining. Given the budget, this should be presented as a narrow directional study rather than a broad cross-benchmark claim.

### Canary families

To stay within budget, use only two canary families:

- random in-distribution canaries
- FGSM canaries generated with a non-private surrogate trained only on public or external data

No bilevel canary optimization or canary-search method is introduced.

### Audit pool

For each trained checkpoint, construct one finite audit pool of \(N\) neighboring candidate pairs. The exact \(N\) is selected from feasibility pilot runs, but it is shared across all methods for that checkpoint. Every privacy-evidence curve is then produced from repeated random orderings of this same pool.

This means:

- no method can silently consume more candidate pairs than the audit pool contains
- matched-budget plots compare evidence versus actual consumed queries
- same-candidate and matched-query comparisons are defined on the same underlying audited examples

### Methods compared

1. Single-view raw loss baseline
2. Single-view quantile baseline of Liu et al. (2025)
3. Weak-view mean-only control
4. Structured weak-view residual score

Optional, if runtime permits:

5. Sequential evidence curves for methods 2 through 4 only

### Query schedules

Main weak-view configuration:

- \(K = 4\), so each weak-view candidate costs \(1 + K = 5\) model queries

Ablation:

- \(K \in \{2, 4\}\) by default
- \(K = 8\) only if the pilot runtime leaves margin

### Metrics

Primary metric:

- empirical \(\varepsilon\) lower bound at \(\delta = 10^{-5}\)

Secondary metrics:

- \(f\)-DP trade-off estimate where available
- ROC AUC
- TPR at low FPR
- privacy evidence versus consumed queries
- dispersion across randomized audit-pool orderings
- optional sequential stopping curves versus consumed queries

### Analysis protocol

Because the study uses only 3 seeds and 2 datasets, the paper should avoid strong universal claims. It reports:

- per-setting results for each of the 4 dataset/privacy combinations
- pooled summaries across seeds with paired bootstrap uncertainty
- direction-of-effect counts across settings and seeds
- whether the structured weak-view score beats the mean-only control under equal query cost
- whether the \(\varepsilon\)-tuned and AUC-tuned \(\lambda\) choices agree

This is a better match to the available budget than claiming definitive superiority from a statistically thin matrix.

## Success Criteria

The study is successful if it supports the following restrained conclusion:

1. In a majority of the 4 dataset/privacy settings, the structured weak-view score is directionally better than the Liu et al. single-view quantile baseline on pooled seeds.
2. At least some of that lift remains visible when privacy evidence is plotted against actual consumed queries \(Q\), averaged over randomized audit-pool orderings.
3. The structured weak-view score is usually no worse, and in the stronger settings modestly better, than the weak-view mean-only control under the same \((1+K)\) query budget.
4. The public-only \(\varepsilon\)-tuned selection of \(\lambda\) is stable, or at least the \(\varepsilon\)-tuned and AUC-tuned choices do not materially disagree.
5. The augmentation screen leaves at least one credible weak-view family for both random and FGSM canaries; otherwise the paper reframes its conclusion as a negative feasibility result.

The study is not successful if gains appear only in same-candidate comparisons, disappear under matched-total-query accounting, are driven by one lucky audit ordering, or fail to beat the mean-only weak-view control. In that case, the paper should conclude that repeated weak-view statistics do not add enough beyond the quantile baseline to justify the additional query cost in this one-run auditing regime.

## References

1. Thomas Steinke, Milad Nasr, and Matthew Jagielski. *Privacy Auditing with One (1) Training Run*. arXiv:2305.08846, 2023. https://arxiv.org/abs/2305.08846
2. Saeed Mahloujifar, Luca Melis, and Kamalika Chaudhuri. *Auditing f-Differential Privacy in One Run*. ICML 2025 / PMLR 267, 2025. https://proceedings.mlr.press/v267/mahloujifar25a.html
3. Terrance Liu, Matteo Boglioni, Yiwei Fu, Shengyuan Hu, Pratiksha Thaker, and Zhiwei Steven Wu. *Enhancing One-run Privacy Auditing with Quantile Regression-Based Membership Inference*. arXiv:2506.15349, 2025. https://arxiv.org/abs/2506.15349
4. Xinlei He, Hongbin Liu, Neil Zhenqiang Gong, and Yang Zhang. *Semi-Leak: Membership Inference Attacks Against Semi-supervised Learning*. ECCV 2022 / arXiv:2207.12535. https://arxiv.org/abs/2207.12535
5. Shadi Rahimian, Tribhuvanesh Orekondy, and Mario Fritz. *Sampling Attacks: Amplification of Membership Inference Attacks by Repeated Queries*. arXiv:2009.00395, 2020. https://arxiv.org/abs/2009.00395
6. Christopher A. Choquette-Choo, Florian Tramer, Nicholas Carlini, and Nicolas Papernot. *Label-Only Membership Inference Attacks*. ICML 2021 / PMLR 139, 2021. https://proceedings.mlr.press/v139/choquette-choo21a.html
7. Yutong Wu, Han Qiu, Shangwei Guo, Jiwei Li, and Tianwei Zhang. *You Only Query Once: An Efficient Label-Only Membership Inference Attack*. ICLR 2024. https://openreview.net/forum?id=7WsivwyHrS
8. Meenatchi Sundaram Muthu Selva Annamalai and Emiliano De Cristofaro. *Nearly Tight Black-Box Auditing of Differentially Private Machine Learning*. NeurIPS 2024. https://arxiv.org/abs/2405.14106
9. Sangyeon Yoon, Wonje Jeung, and Albert No. *Adversarial Sample-Based Approach for Tighter Privacy Auditing in Final Model-Only Scenarios*. arXiv:2412.01756, 2024. https://arxiv.org/abs/2412.01756
10. Mishaal Kazmi, Hadrien Lautraite, Alireza Akbari, Qiaoyue Tang, Mauricio Soroco, Tao Wang, Sebastien Gambs, and Mathias Lecuyer. *PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining*. NeurIPS 2024. https://arxiv.org/abs/2402.09477
11. Tudor Cebere, Aurelien Bellet, and Nicolas Papernot. *Tighter Privacy Auditing of DP-SGD in the Hidden State Threat Model*. ICLR 2025. https://arxiv.org/abs/2405.14457
12. Tomas Gonzalez, Mateo Dulce-Rubio, Aaditya Ramdas, and Monica Ribero. *Sequentially Auditing Differential Privacy*. NeurIPS 2025 / arXiv:2509.07055, 2025. https://arxiv.org/abs/2509.07055
13. Zifan Wang, Adam Islam, and Terrance Liu. *Optimizing Canaries for One-Run Privacy Auditing of DP-SGD*. OpenReview / ICLR 2026 submission. https://openreview.net/forum?id=wQjLz8Qx2E
