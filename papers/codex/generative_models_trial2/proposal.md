# Should Early Ranking Targets Be Decomposed for Compositional Text-to-Image Generation?

## Introduction

This proposal is a controlled empirical study, not a new text-to-image method. The question is narrow:

**when a diffusion trajectory is only partially generated, is it better to rank candidate continuations with a decomposed compositional target or with a single scalar target?**

Recent work already shows two nearby facts. First, early diffusion activations can predict final image quality well enough to support selective continuation or seed selection (Guo et al., 2026; Cui et al., 2026). Second, compositional text-to-image work often benefits from scoring or correcting individual conditions rather than relying on one global score (Yu and Gao, 2025; Izadi et al., 2025; Jaiswal et al., 2026). The contribution here is to connect those lines conservatively and test one specific design choice: **for compositional prompts under a fixed inference budget, should early ranking targets be decomposed into atomic constraints?**

The claim is intentionally benchmark-specialized. On compositional benchmarks such as GenEval, many failures are local: an image can look globally plausible while failing exactly one count, attribute-binding, or spatial-relation requirement. If early features preserve signal about those atomic failures, then collapsing them into one scalar target may discard useful information for prune-continue decisions. If not, the decomposed target should not help once model capacity, features, and budget are matched.

**Hypothesis.** Under the same prefix, features, probe capacity, and denoising budget, a decomposed early target over benchmark-aligned atomic constraints will yield better fixed-budget continuation decisions than a matched scalar early target on held-out compositional prompts.

## Proposed Approach

### Overview

The study freezes a standard generator such as Stable Diffusion v1.5 and compares two early-ranking targets:

1. Parse a compositional benchmark prompt into atomic constraints.
2. Sample several seeds and run only a short diffusion prefix.
3. Extract the same early feature vector for every candidate.
4. Train two matched lightweight probes on completed-image labels:
   - a decomposed probe predicting one success probability per atomic constraint
   - a scalar probe predicting one overall compositional-success target
5. Under a fixed compute budget, continue only the top-ranked partial trajectory.

The paper's outcome is the comparison in step 4, not the probe architecture itself.

### Prompt Decomposition

Prompt decomposition is deterministic and benchmark-aligned rather than open-ended. Each prompt is mapped into atomic constraints of three types:

- `object/count`
- `attribute binding`
- `relation`

This keeps the study reproducible and aligned with GenEval-style templates. The parser is explicitly restricted to prompts whose constraints can be extracted unambiguously. That restriction narrows the claim, but it avoids overstating generality.

### Budgeted Prefix-Continuation Schedule

The main schedule is:

- generator: Stable Diffusion v1.5
- resolution: `512 x 512`
- sampler: DDIM
- total denoising steps: `T = 20`
- early prefix: `tau = 4`
- total budget: `B40`

At `B40`:

- sample `N = 6` seeds for `4` steps each: `24` UNet evaluations
- continue `K = 1` seed for the remaining `16` steps: `16` UNet evaluations
- total: `40` UNet evaluations

This is compute-matched against practical baselines such as best-of-2 full completion.

### Shared Early Features

To keep the comparison clean and inexpensive, both probes use the same small feature set:

- latent mean, standard deviation, and absolute-activation statistics from the first `4` denoising steps
- conditional-versus-unconditional denoiser disagreement norms
- lightweight tokenizer-derived prompt heuristics computed from the frozen text encoding inputs
- one step-`4` low-resolution preview embedding from a frozen CLIP vision encoder

The main system intentionally avoids heavier detector-based preview features in the default configuration. That keeps latency low and reduces dependence on the same detector family used by benchmark evaluators.

### Two Matched Targets

The probe architecture is deliberately modest: one shared MLP trunk, with either multiple heads or one scalar head.

**Decomposed target**

- one Bernoulli success label per parsed atomic constraint
- type-specific heads for count, attribute, and relation constraints
- ranking score is the mean predicted success probability across constraints

**Scalar target**

- one scalar label for overall prompt success
- same feature vector
- same trunk depth and width
- same optimizer and calibration pipeline

The only intended difference is the target structure.

### Stronger Supervision

The earlier proxy-heavy supervision story was too weak. This revision makes the labeling scheme substantially stricter.

For each completed training or validation image, the primary labels are the final benchmark-aligned atomic outcomes on the completed image, not detector-derived proxy targets invented only for training. In practice:

- atomic success labels come from the official GenEval-style evaluation pipeline whenever the prompt belongs to that benchmark family
- the scalar target is derived from those same atomic outcomes, for example by averaging or exact-all-correct labeling

Because benchmark labels are still imperfect automatic measurements, the proposal originally planned a **small manually verified subset**:

- `80` completed images sampled from train plus validation
- stratified across count, attribute-binding, and relation templates
- manually checked for atomic constraint satisfaction

If genuine human annotations are available, this subset is used for:

- calibrating threshold choices
- estimating label noise by constraint type
- a sensitivity analysis that retrains or recalibrates both probes using only manually verified labels

If such annotations are unavailable, the study explicitly drops the sensitivity claim rather than backfilling pseudo-human labels. In that case, the final paper claim remains benchmark-specialized and depends only on the automatic official-evaluation pipeline.

## Related Work

### Early quality assessment

Guo et al., *Toward Early Quality Assessment of Text-to-Image Diffusion Models* (2026), and Cui et al., *Diffusion Probe: Generated Image Result Prediction Using CNN Probes* (2026), are the closest methodological neighbors. They show that early diffusion signals can predict final quality well enough to support selective continuation. This proposal does not claim that idea as novel. It tests a narrower question that those papers do not isolate: whether the **target itself** should be decomposed for compositional prompts.

### Condition-level compositional scoring

Yu and Gao, *Improving Compositional Generation with Diffusion Models Using Lift Scores* (2025), is the key novelty boundary. That paper already argues that compositional conditions should often be scored separately. The present study differs in two ways:

- it moves the comparison to the early diffusion prefix rather than final-sample resampling
- it directly compares decomposed versus scalar early targets with matched features and capacity

This is a modest contribution, closer to a careful empirical test than to a new algorithm.

### Heavier test-time compositional methods

Izadi et al. (2025), Jaiswal et al. (2026), Wang et al. (2024), and Feng et al. (2022) improve compositional generation through structured guidance, iterative refinement, planning, or verifier-driven correction. These methods change the generator dynamics or add stronger search loops. The proposed study is lighter: the generator stays fixed, and only the early ranking target changes.

### Benchmark dependence

GenEval (Ghosh et al., 2023) is the main benchmark because it provides explicit count, attribute, and relation failures. T2I-CompBench++ (Huang et al., 2023) is relevant as an optional transfer check, but the proposal does not depend on broad transfer claims. Positive results would support a benchmark-specialized conclusion first: **on prompts with explicit compositional constraints, decomposed early targets may be better continuation targets than scalar early targets.**

### Positioning

The novelty claim is intentionally conservative:

**To the best of the March 22, 2026 literature check, prior work appears to cover early quality assessment and condition-level compositional scoring separately, but not a clean controlled comparison of decomposed versus scalar early ranking targets for fixed-budget continuation on compositional text-to-image benchmarks.**

## Experiments

### Main Experimental Question

**At matched compute, does a decomposed early target outperform a scalar early target for selecting which partial compositional trajectory to continue?**

Everything else is secondary.

### Data Split

The study uses one fixed GenEval-centered split:

- `120` prompts for training
- `40` prompts for validation and calibration
- `80` prompts for the held-out test set

For supervision collection:

- `4` completed trajectories per train prompt: `480` training images
- `4` completed trajectories per validation prompt: `160` validation images

This count is fixed throughout the proposal.

### Baselines

The mandatory baselines are:

1. `Best-of-2 Full Completion` at `B40`, selected by a generic preference model such as PickScore or CLIP similarity
2. `Random Prune-Continue` with the same `6 x 4 + 1 x 16` schedule
3. `Scalar Early Target`, using the same features, prefix, probe capacity, and continuation rule

An `Oracle Best-of-6` analysis is optional and should only be run if measured latency leaves clear headroom inside the eight-hour budget.

### Metrics

Mandatory metrics:

- GenEval overall score on the `80`-prompt held-out set
- GenEval category scores for count, attribute, and relation subsets
- prompt-level bootstrap confidence intervals and paired permutation tests for decomposed-versus-scalar comparisons
- exact UNet evaluation count and wall-clock overhead
- held-out calibration metrics for the probes

Optional metrics, only if measured runtime allows:

- a small frozen transfer slice from T2I-CompBench++
- a small blinded human comparison subset

These are supporting checks, not part of the core acceptance case.

### Runtime Scope

The proposal is scoped to one RTX A6000 and roughly eight hours total. To keep that credible, the mandatory package is intentionally small:

- one generator
- one budget (`B40`)
- one main benchmark
- one decisive baseline comparison
- one minimal feature ablation

Before optional transfer, human evaluation, or oracle analysis, the study must first measure actual prefix and completion latencies on the A6000 and confirm that the core package fits comfortably within budget.

### Minimal Ablations

Only two ablations are required for the main claim:

1. **Target ablation**
   Decomposed versus scalar target with identical features and model size.

2. **Feature ablation**
   Internal diffusion features only versus internal features plus one preview embedding.

Anything beyond this is optional.

### Expected Results

The expected effect should be framed conservatively. A successful outcome would look like:

- a positive held-out GenEval gain for the decomposed target over the scalar target at `B40`
- the same-direction effect across at least two of the three major constraint families
- competitive or better performance than best-of-2 full completion at the same budget
- no evidence that the gain disappears after calibration or, if genuine annotations are available, when checked against the manually verified subset

If the scalar target matches the decomposed target once features and budget are controlled, that is a valid negative result and should be reported as such. Likewise, if the manual-label sensitivity study cannot be run with genuine annotations, the claim should be narrowed rather than treated as validated.

## Success Criteria

### Primary success condition

The proposal is supported only if both of the following hold on the fixed `80`-prompt GenEval test set:

1. The decomposed early target beats the matched scalar early target at `B40`.
2. The effect is large enough to survive prompt-level uncertainty analysis, either through a bootstrap interval above zero or a predeclared practically meaningful margin.

### Supporting evidence

The case is stronger if:

- the decomposed probe is better calibrated for at least one major constraint family
- if genuine human annotations are available, the manually verified subset shows the same ordering as the automatic benchmark labels
- the method also matches or beats best-of-2 full completion at the same budget

### Failure conditions

The proposal should be considered unsupported if any of the following occurs:

- the scalar target matches or beats the decomposed target under matched features and budget
- the improvement appears only under benchmark-coupled selection procedures
- the manually verified subset contradicts the claimed gain, if that subset is available
- non-denoising overhead removes the practical budget advantage

## References

- Cui, Benlei, Bukun Huang, Zhizeng Ye, Xuemei Dong, Tuo Chen, Hui Xue, Dingkang Yang, Longtao Huang, Jingqun Tang, and Haiwen Hong. 2026. *Diffusion Probe: Generated Image Result Prediction Using CNN Probes*. arXiv:2602.23783.
- Feng, Weixi, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. 2022. *Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis*. arXiv:2212.05032.
- Ghosh, Dhruba, Hanna Hajishirzi, and Ludwig Schmidt. 2023. *GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment*. arXiv:2310.11513.
- Guo, Huanlei, Hongxin Wei, and Bingyi Jing. 2026. *Toward Early Quality Assessment of Text-to-Image Diffusion Models*. arXiv:2603.02829.
- Huang, Kaiyi, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. 2023. *T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-image Generation*. arXiv:2307.06350.
- Izadi, Amir Mohammad, Seyed Mohammad Hadi Hosseini, Soroush Vafaie Tabar, Ali Abdollahi, Armin Saghafian, and Mahdieh Soleymani Baghshah. 2025. *Fine-Grained Alignment and Noise Refinement for Compositional Text-to-Image Generation*. arXiv:2503.06506.
- Jaiswal, Shantanu, Mihir Prabhudesai, Nikash Bhardwaj, Zheyang Qin, Amir Zadeh, Chuan Li, Katerina Fragkiadaki, and Deepak Pathak. 2026. *Iterative Refinement Improves Compositional Image Generation*. arXiv:2601.15286.
- Ma, Nanye, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, and Saining Xie. 2025. *Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps*. arXiv:2501.09732.
- Rombach, Robin, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. 2022. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. arXiv:2112.10752.
- Wang, Zhenyu, Enze Xie, Aoxue Li, Zhongdao Wang, Xihui Liu, and Zhenguo Li. 2024. *Divide and Conquer: Language Models can Plan and Self-Correct for Compositional Text-to-Image Generation*. arXiv:2401.15688.
- Yu, Chenning, and Sicun Gao. 2025. *Improving Compositional Generation with Diffusion Models Using Lift Scores*. arXiv:2505.13740.
