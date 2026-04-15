# How Much Signal Is in Early Training Trajectories? A Matched-Budget Study of Pseudo-Group Inference

## Introduction

Annotation-free robustness to spurious correlations increasingly appears to be bottlenecked by **pseudo-group inference quality** rather than by the downstream robust objective alone. SPARE shows that early outputs can already expose spurious structure. ExMap, GIC, and related pseudo-group papers argue that more precise inferred groups lead to better worst-group performance. Beyond Distribution Shift shows that spurious-feature learning is visible in training dynamics, not just in endpoint accuracy.

This proposal deliberately reframes the paper as a **controlled empirical study**, not as a new robustness method. The central question is:

**At matched warmup compute, do short early-training trajectories provide better pseudo-group signal than output-only early statistics or warmup-end features?**

The novelty claim is intentionally narrow. I did **not** find an exact prior comparison that matches warmup cost and downstream learner while isolating:

1. output-only early statistics,
2. richer multi-epoch trajectory descriptors, and
3. static warmup-end features.

At the same time, the overlap with prior ingredients is substantial: early outputs, training-dynamics analysis, within-class clustering, and pseudo-group-aware fine-tuning all exist separately in the literature. The paper is strongest if positioned as a carefully controlled benchmark plus one lightweight new inference component, not as a fundamentally new robustness algorithm.

The hypothesis is compute-aware: if trajectory information is useful, it should help most under short warmup budgets such as 2 to 4 epochs, where a single endpoint representation may not yet expose the latent subgroup structure cleanly.

## Proposed Approach

### Overview

The pipeline has three stages:

1. warm up a pretrained classifier for a small number of epochs,
2. infer within-class pseudo-groups from one of several matched-budget signals,
3. plug those pseudo-groups into a shared downstream learner.

The main variable is the pseudo-group signal. The downstream learner is held fixed for the main experiment, then swapped once to test whether the signal quality transfers beyond contrastive training.

### Stage 1: Warmup and signal extraction

Use ERM warmup for `T in {2, 4}` epochs. `T=4` is the default main setting and `T=2` is the matched-budget stress test. Use an ImageNet-pretrained ResNet-18 on Waterbirds and a small CNN on Colored MNIST.

For sample `i` at epoch `t`, let:

- `z_i^(t)` be logits,
- `h_i^(t)` be the L2-normalized penultimate feature,
- `y_i` be the class label.

Using two stochastic augmentations per sample at each checkpoint gives:

- `p_i^(t) = softmax(z_i^(t))[y_i]` : true-class confidence
- `m_i^(t) = z_i^(t)[y_i] - max_{c != y_i} z_i^(t)[c]` : true-class margin
- `l_i^(t) = CE(z_i^(t), y_i)` : per-sample loss
- `u_i^(t) = JS(softmax(z_{i,a}^(t)), softmax(z_{i,b}^(t)))` : augmentation disagreement
- `d_i^(t) = 1 - cos(h_i^(t), h_i^(t-1))` for `t > 1`, else `0` : feature drift

### New trajectory-specific inference component: trajectory-shape encoding

The previous draft only concatenated raw statistics across epochs. That is too close to existing ingredients and too easy to dismiss as “more bookkeeping.” The revised proposal adds one explicit trajectory-specific component:

For each scalar sequence `x_i^(1:T)`, compute a lightweight shape code

`g(x_i) = [x_i^(1), x_i^(T), mean(x_i), std(x_i), delta(x_i), slope(x_i)]`

where:

- `delta(x_i) = x_i^(T) - x_i^(1)`
- `slope(x_i)` is the least-squares linear slope over epoch index

For output trajectories also compute:

- `t_i^corr` = first epoch where the example is correctly classified with positive margin, else `T+1`

This gives a simple temporal summary that preserves ordering information and learning speed while remaining cheap to compute. It is not a new clustering algorithm; it is a new trajectory-aware descriptor inserted before the same clustering stage used for all methods.

### Signals compared

All signals are z-scored within class before clustering. If dimensionality exceeds 32, apply PCA to 16 dimensions within class.

1. **Trajectory signal**
   `s_i^traj = [p_i^(1:T), m_i^(1:T), l_i^(1:T), u_i^(1:T), d_i^(1:T), g(p_i), g(m_i), g(l_i), g(u_i), g(d_i), t_i^corr]`

2. **Output-only control**
   `s_i^out = [p_i^(1:T), m_i^(1:T), l_i^(1:T), g(p_i), g(m_i), g(l_i), t_i^corr]`

   This is the matched SPARE-style control: same warmup checkpoints, same temporal encoder, but no disagreement or feature-drift information.

3. **Final-feature control**
   `s_i^feat = h_i^(T)`

This comparison isolates whether the additional trajectory information beyond output-only early statistics improves pseudo-group fidelity.

### Stage 2: Within-class pseudo-group inference

For each class `y`, cluster `{s_i : y_i = y}` with a diagonal-covariance Gaussian mixture model. The main study uses `K=2`, which matches the canonical majority/minority subgroup structure in Waterbirds and Colored MNIST.

Unlike the previous draft, the method is not conceptually tied to `K=2`. Let `q_i in R^K` be the posterior mixture probabilities for sample `i`. Define the same-class cross-group disagreement score

`r_ij = 1 - q_i^T q_j`.

This expression works for any `K`. The main paper therefore keeps `K=2` for the primary comparison but adds a cheap sensitivity analysis with `K` chosen by BIC over `{2, 3, 4}` for the top two signals on Waterbirds. That partially addresses the fixed-`K` criticism without committing to a broader subgroup-discovery claim than the budget supports.

### Stage 3a: Main downstream learner

The main downstream learner is the same weighted cross-group supervised contrastive objective for every signal:

`L = L_CE + lambda * L_xgSupCon`.

For anchor `i` with same-class positives `P(i)`, use

`L_xgSupCon(i) = - log ( sum_{j in P(i)} w_ij exp(sim(v_i, v_j)/tau) / sum_{k in B \\ {i}} exp(sim(v_i, v_k)/tau) )`

with:

- `v_i` the normalized projection-head feature,
- `w_ij = eps + (1 - eps) * r_ij`,
- `eps = 0.1`.

The downstream objective is intentionally unchanged across the signal conditions. That makes the paper a test of pseudo-group quality, not a test of objective engineering.

### Stage 3b: Non-contrastive downstream control

To show the effect is not specific to xg-SupCon, run one additional downstream learner on Waterbirds:

**Soft pseudo-group reweighted ERM**

Estimate pseudo-group masses within each class as

`pi_{y,k} = mean_{i:y_i=y} q_{i,k}`.

Weight each sample by the inverse expected pseudo-group mass

`w_i^erm = sum_k q_{i,k} / (pi_{y_i,k} + 1e-6)`

and optimize weighted cross-entropy only.

This is simple, cheap, and sufficient to test whether better pseudo-group inference helps a non-contrastive learner too. The paper still does not claim a new robust optimization method.

## Related Work

### SPARE: early outputs as pseudo-group signal

**Identifying Spurious Biases Early in Training through the Lens of Simplicity Bias** (Yang et al., 2024) is the closest reference point. SPARE shows that early outputs already reveal shortcut-driven structure and uses that signal for later importance sampling. The proposed study differs in three ways:

1. it matches warmup compute explicitly across signal choices,
2. it asks whether extra multi-epoch information improves over output-only signals,
3. it evaluates pseudo-group fidelity directly, not only end-task accuracy.

No exact matched-budget trajectory-versus-output-only comparison was found in SPARE or in papers citing similar early-output intuition.

### Beyond Distribution Shift: dynamics as analysis, not pseudo-group inference

**Beyond Distribution Shift: Spurious Features Through the Lens of Training Dynamics** (Murali et al., 2023) argues that spurious-feature learning leaves signatures in training dynamics. That paper is a training-dynamics analysis paper, not a pseudo-group discovery benchmark. It motivates the core premise here: temporal learning behavior may contain subgroup information that endpoint features miss.

### ExMap: richer signal, different mechanism

**ExMap** (Chakraborty et al., 2024) improves annotation-free robustness by clustering explainability heatmaps. Its signal comes from post hoc explanations extracted from a trained model, not from short warmup trajectories. Relative to ExMap, this proposal is cheaper and more controlled but less ambitious: ordinary warmup checkpoints are the only source of information, and the goal is to compare signal quality rather than to propose a more interpretable group-inference mechanism.

### GIC and precise pseudo-group inference

**Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference** (Han and Zou, 2024) makes the strongest argument that pseudo-group precision is the real bottleneck. This proposal operationalizes that claim in a very specific setting: fixed warmup budget, fixed downstream learner, and signal quality as the main experimental variable. It does not attempt to replace GIC-style inference with a broader general method.

### Broader annotation-free robustness literature

**Learning Robust Classifiers with Self-Guided Spurious Correlation Mitigation** (Zheng et al., 2024), **Improving Group Robustness on Spurious Correlation via Evidential Alignment** (Ye et al., 2025), **Correct-N-Contrast** (Zhang et al., 2022), and related work all combine pseudo labels, robust objectives, or contrastive learning to mitigate spurious correlations without group annotations.

The overlap with this proposal is substantial:

- early signals from training,
- unsupervised or weakly supervised group inference,
- pseudo-group-aware robust learning,
- supervised contrastive fine-tuning.

The contribution is therefore best pitched as a **carefully controlled empirical study with a lightweight new trajectory encoder**, not as a new standalone method class.

## Experiments

### Core scope

The paper should make a narrower, more honest claim than the previous draft:

- primary scope: short-warmup pseudo-group inference on canonical spurious-correlation vision benchmarks,
- strongest evidence: Waterbirds,
- secondary evidence: Colored MNIST as a cheap sanity check,
- no claim of broad subgroup-discovery generality in complex real-world datasets.

### Main Waterbirds experiment

Run 3 seeds with `T=4` warmup for:

1. `ERM`
2. `Vanilla SupCon`
3. `Final-feature xg-SupCon`
4. `Output-only xg-SupCon`
5. `Trajectory xg-SupCon`

This is the main comparison table.

### Matched-budget trajectory test

Run 3 seeds on Waterbirds for:

1. `Output-only xg-SupCon`, `T=2`
2. `Trajectory xg-SupCon`, `T=2`
3. `Output-only xg-SupCon`, `T=4`
4. `Trajectory xg-SupCon`, `T=4`

This is the decisive experiment. It directly tests whether richer trajectory information helps when warmup compute is held fixed, and whether the gain is larger in the short-budget regime.

### Downstream-objective transfer

Run 3 seeds on Waterbirds for:

1. `Output-only soft reweighted ERM`
2. `Trajectory soft reweighted ERM`

This isolates whether the pseudo-group signal ranking transfers beyond contrastive learning.

### `K`-sensitivity analysis

Run 1 to 3 seeds on Waterbirds for the top two signals only:

1. fix `K=2`,
2. choose `K` by BIC over `{2, 3, 4}`.

This is not intended to prove general multi-group discovery. It is only a robustness check that the result is not entirely an artifact of forcing two clusters.

### Lightweight sanity dataset

Use **Colored MNIST** only for a small mechanism check:

1. `ERM`
2. `Output-only xg-SupCon`
3. `Trajectory xg-SupCon`

One seed is acceptable if Waterbirds consumes most of the budget.

### Metrics

Primary performance metrics:

- worst-group accuracy,
- average accuracy,
- robustness gap.

Pseudo-group fidelity metrics, computed with oracle group labels on validation only:

- NMI between inferred pseudo-groups and true groups,
- minority-group F1 after best matching,
- average posterior confidence `max_k q_{i,k}`,
- seed stability of pairwise disagreement scores `r_ij`.

Additional diagnostics:

- correlation between pseudo-group fidelity and worst-group accuracy across runs,
- wall-clock warmup time,
- clustering time,
- extra memory for cached trajectories.

### Expected results

The strongest positive outcome is:

1. trajectory signals yield higher pseudo-group fidelity than output-only signals at matched budget,
2. the advantage is larger at `T=2` than at `T=4`,
3. the better signal improves worst-group accuracy under both xg-SupCon and soft reweighted ERM.

The strongest negative outcome is also publishable as an empirical result:

1. output-only early statistics already match trajectory signals once compute is matched,
2. final-feature clustering is competitive with early-training signals,
3. trajectory bookkeeping does not justify its extra complexity.

That negative result would still sharpen the field’s understanding of where annotation-free robustness gains actually come from.

### Compute realism

The experiments must fit on 1x RTX A6000, 60 GB RAM, 4 CPU cores, and roughly 8 hours total. To stay credible:

- keep the backbone fixed to ResNet-18,
- cache warmup statistics in one pass per checkpoint,
- avoid external baseline reproductions unless their code works immediately,
- limit hyperparameter tuning to one validation choice of `lambda` and `tau`,
- run the non-contrastive transfer only on Waterbirds,
- treat Colored MNIST as optional if the primary Waterbirds matrix runs long.

This resource profile is consistent with a disciplined empirical paper, but not with a broad multi-dataset benchmark or heavy method-development cycle.

## Success Criteria

The hypothesis is supported if:

1. `Trajectory` beats `Output-only` on at least one pseudo-group fidelity metric at both `T=2` and `T=4` on Waterbirds.
2. The trajectory advantage is larger at `T=2`, supporting the short-budget claim.
3. The signal ranking persists under both xg-SupCon and soft reweighted ERM.
4. The result does not disappear under the cheap `K`-sensitivity check.

The hypothesis is weakened if:

1. output-only signals match or beat trajectories after warmup cost is matched,
2. final warmup-end features perform similarly,
3. the gains only appear under xg-SupCon and vanish under weighted ERM,
4. rankings are unstable across seeds,
5. the method requires oracle group labels for checkpoint choice, clustering choice, or hyperparameter selection.

## References

1. Yu Yang, Eric Gan, Gintare Karolina Dziugaite, and Baharan Mirzasoleiman. *Identifying Spurious Biases Early in Training through the Lens of Simplicity Bias*. AISTATS, 2024.
2. Nihal Murali, Aahlad Puli, Ke Yu, Rajesh Ranganath, and Kayhan Batmanghelich. *Beyond Distribution Shift: Spurious Features Through the Lens of Training Dynamics*. TMLR, 2023.
3. Rwiddhi Chakraborty, Adrian Sletten, and Michael C. Kampffmeyer. *ExMap: Leveraging Explainability Heatmaps for Unsupervised Group Robustness to Spurious Correlations*. CVPR, 2024.
4. Yujin Han and Difan Zou. *Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference*. arXiv:2404.13815, 2024.
5. Guangtao Zheng, Wenqian Ye, and Aidong Zhang. *Learning Robust Classifiers with Self-Guided Spurious Correlation Mitigation*. IJCAI, 2024.
6. Wenqian Ye, Guangtao Zheng, and Aidong Zhang. *Improving Group Robustness on Spurious Correlation via Evidential Alignment*. arXiv:2506.11347, 2025.
7. Michael Zhang, Nimit S. Sohoni, Hongyang R. Zhang, Chelsea Finn, and Christopher Re. *Correct-N-Contrast: A Contrastive Approach for Improving Robustness to Spurious Correlations*. arXiv:2203.01517, 2022.
8. Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. *Supervised Contrastive Learning*. NeurIPS, 2020.
9. Guanwen Qiu, Da Kuang, and Surbhi Goel. *Complexity Matters: Feature Learning in the Presence of Spurious Correlations*. ICML, 2024.
