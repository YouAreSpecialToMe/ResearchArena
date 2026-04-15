# Do Corruption-Family Text Residuals Help Zero-Shot CLIP? A Controlled Baseline Study

## Introduction

Zero-shot CLIP remains a practical baseline for open-vocabulary recognition, but it is brittle under common corruptions. Recent work already explores stronger remedies through test-time prompt optimization, bimodal adaptation, unlabeled prompt-distribution learning, auto-tuned text classifiers, and projection-style debiasing. Under that landscape, a new "method" claim would be weak and hard to justify on a single-GPU budget.

This proposal therefore reframes the contribution as a controlled baseline-and-analysis paper. The central empirical question is narrow: do explicit corruption-family text residuals carry usable family-specific signal beyond two simpler alternatives, namely generic prompt ensembling and naive corruption/degradation prompts? If they do, a tiny frozen calibration rule may provide a useful accuracy-cost tradeoff. If they do not, that negative result is still meaningful because it clarifies that prompt-side corruption language mostly captures generic quality drift rather than family structure.

The closest risk is overlap with nearby prompt-side and unlabeled adaptation work. An online search and local paper review did not reveal a prior paper evaluating the exact full recipe of paired corruption-family text residuals, explicit family posteriors, and additive frozen calibration with family-specific controls. However, the nearest prior art is clearly AutoCLIP, Frolic, BATCLIP, BendVLM, and PRISM, so the novelty claim must stay narrow: this is a controlled baseline study, not a new adaptation paradigm.

The paper is designed to be falsifiable. It includes a pilot gate, a generic "low-quality image" residual control, and a random family reassignment control. These controls directly test the main soundness concern: whether the proposed residual bank is genuinely family-specific or merely acting as a shared degradation vector.

## Proposed Approach

### Overview

The study uses frozen CLIP and asks whether four hand-specified corruption families can support useful prompt-side calibration:

- `gaussian_noise`
- `motion_blur`
- `fog`
- `jpeg_compression`

These are chosen because they are easy to verbalize, appear in CIFAR-C and ImageNet-C style corruption taxonomies, and span different corruption families without expanding the experiment matrix beyond the budget.

### Clean Baseline

For class `y`, the clean zero-shot prototype is

`c_y = normalize(e("a photo of a {class_y}"))`.

The strong prompt baseline uses a small clean prompt ensemble such as:

- `a photo of a {class}`
- `a blurry photo of a {class}`
- `a close-up photo of a {class}`
- `a centered photo of a {class}`
- `an image of a {class}`

This baseline matters because the proposal must beat or match ordinary prompt strengthening before claiming value from corruption-specific language.

### Family Residual Bank

For each corruption family `z`, form a text residual from paired clean and corrupted prompts:

`r_z = mean_y [e("a photo of a {class_y} with {z}") - e("a clear photo of a {class_y}")]`

This is the main object under study. It is intentionally simple and shared across classes. The point is not to invent a sophisticated parameterization, but to test whether this residual carries family-specific signal at all.

### Stronger Controls

Two controls are built into the design.

Generic degradation control:

`r_generic = mean_y [e("a low-quality photo of a {class_y}") - e("a clear photo of a {class_y}")]`

Random family reassignment control:

- infer or use the same family posterior machinery as the main method;
- then randomly permute family labels before applying residuals;
- keep class prototypes, scaling, and evaluation identical.

If the proposed method does not beat both controls, the paper cannot claim family-specific calibration.

### Optional Projection Diagnostic

An optional class-subspace projection is kept only as a diagnostic:

`r_z^proj = normalize((I - U U^T) r_z)`

where `U` contains top principal directions of the clean class prototype matrix. This is not a claimed method contribution. It is a check on whether removing class-semantic directions sharpens family alignment. If it does not help in the pilot, it is dropped from main tables.

### Family Posterior

For a test image `x`, estimate a lightweight family posterior:

`a_z(x) = sim(v(x), e("an image with {z} corruption"))`

`q(z|x) = softmax(beta * a_z(x))`

This is deliberately weak and fully frozen. The goal is interpretability and family-specific testing, not maximal adaptation power.

### Additive Calibration

Using residual bank element `r_z^*` (projected or unprojected depending on the pilot), define

`t_{y,z} = normalize(c_y + alpha * r_z^*)`

and score with

`s(y|x) = sim(v(x), c_y) + lambda * sum_z q(z|x) [sim(v(x), t_{y,z}) - sim(v(x), c_y)]`.

This gives a single-image, optimization-free, fully frozen calibration rule.

### Narrow Claim

The paper claims only the following:

1. a reproducible corruption-aware prompt baseline for frozen CLIP;
2. a controlled test of whether corruption-family text residuals contain usable family-specific information;
3. evidence for or against family-specificity beyond generic degradation prompts and random family assignment.

If the residual bank fails these tests, the correct outcome is a negative finding, not a weakened method claim.

## Related Work

### CLIP and Prompt-Side Zero-Shot Classification

CLIP established the frozen image-text interface that makes prompt-side robustness studies possible. Visual Classification via Description from Large Language Models showed that richer textual descriptions can improve zero-shot classification, but it targets class semantics rather than corruption-aware nuisance modeling.

### Closest Prior Art

The nearest papers found by online search and local reference review are:

- AutoCLIP: unlabeled auto-tuning of zero-shot text classifiers;
- Frolic: label-free prompt distribution learning and bias correction;
- BATCLIP: bimodal test-time adaptation for corruption robustness;
- BendVLM: test-time debiasing of vision-language embeddings;
- PRISM: projection-based bias mitigation with LLM-guided descriptions.

These papers tighten the novelty boundary substantially. None appears to evaluate the exact combination of paired corruption-family residuals, explicit family posteriors, additive frozen calibration, and the two family-specificity controls proposed here. But they make clear that the contribution is only a narrow baseline study near existing prompt-side and debiasing work.

### How This Proposal Differs

AutoCLIP is the most important executable comparator because it also improves frozen zero-shot classifiers from unlabeled data. The proposed study differs by fixing a tiny interpretable residual bank and asking whether explicit family structure exists at all.

Frolic is a stronger nearby alternative that learns prompt distributions rather than explicit corruption variables. Relative to Frolic, this proposal is less general and probably weaker in raw accuracy, but more falsifiable because its latent variables correspond to named corruption families and are directly stress-tested.

BATCLIP performs substantially heavier adaptation. It is relevant for positioning, but the current budget only supports using it as discussion context rather than a reproduced baseline.

BendVLM and PRISM matter because they show that projection-based debiasing is already active territory. That is why the projection component here is demoted to an optional diagnostic rather than presented as a standalone innovation.

### Novelty Statement

The novelty claim is intentionally modest. Online search did not reveal an exact prior paper for the full recipe studied here, but the proposal sits very close to AutoCLIP, Frolic, BATCLIP, BendVLM, and PRISM. The contribution is therefore a controlled baseline variant with stronger analysis, not a genuinely new method family.

## Experiments

### Main Question

The empirical question is:

Do family-specific text residuals improve corruption robustness beyond:

1. standard clean prompt ensembling,
2. naive family corruption prompts,
3. a generic low-quality/degraded residual,
4. random family reassignment?

### Execution Scope

To fit one RTX A6000 and roughly eight total hours, the study is restricted to:

- one frozen CLIP backbone;
- CIFAR-10-C and CIFAR-100-C as main benchmarks;
- four corruption families only;
- one final hyperparameter setting selected from a tiny unlabeled proxy set;
- no required multi-seed sweeps.

ImageNet-C is optional and only attempted on a small subset if the pilot and CIFAR runs finish early.

### Pilot Gate

The pilot is mandatory and decides whether the method deserves full evaluation. It uses clean CIFAR images plus synthetic versions of the same four corruption families and reports:

- family prediction accuracy of `q(z|x)` against known corruption labels;
- matched-versus-mismatched alignment between average visual corruption shifts and text residuals;
- downstream accuracy gain over zero-shot CLIP on the pilot split;
- comparison against `r_generic`;
- comparison against random family reassignment.

The study proceeds to full benchmark runs only if the proposed residual bank beats both stronger controls on at least one of family identification or downstream corrupted accuracy without harming clean performance materially.

### Benchmarks

Required:

- CIFAR-10 clean and CIFAR-10-C, four families, severities 1-5;
- CIFAR-100 clean and CIFAR-100-C, four families, severities 1-5.

Optional only if time remains:

- ImageNet-C subset with the same four families and a reduced severity/class grid.

### Baselines

Headline baselines:

1. zero-shot CLIP with one standard template;
2. clean prompt ensemble;
3. naive family corruption prompts added directly to class templates;
4. generic degraded-image residual;
5. random family reassignment control;
6. proposed family-residual calibration.

Optional comparator if the pilot is strongly positive and time remains:

7. AutoCLIP, single seed, on the same restricted CIFAR protocol.

BATCLIP, Frolic, BendVLM, and PRISM are positioning references rather than required reproduced baselines under this budget.

### Metrics

Primary metrics:

- mean corruption accuracy across families and severities;
- per-family accuracy;
- clean accuracy change relative to zero-shot CLIP.

Secondary metrics:

- family posterior accuracy on the pilot;
- expected calibration error;
- runtime per dataset pass.

### Ablations

Only the highest-value ablations are retained:

1. family residual versus naive corruption prompt;
2. family residual versus generic degraded residual;
3. correct family assignment versus random reassignment;
4. projected versus unprojected residuals only if the pilot justifies projection.

Oracle corruption-family selection may be reported only as a diagnostic ceiling and will not appear in headline comparisons because it is not deployable.

### Budget and Risk Control

A realistic schedule is:

- setup, prompt construction, and proxy corruption generation: 1 hour;
- pilot plus controls: 1.5 hours;
- CIFAR-10-C and CIFAR-100-C evaluation with ablations: 3 to 3.5 hours;
- tables, plots, and error analysis: 1 hour;
- optional AutoCLIP single-seed run only after a strong pilot: 1.5 to 2 hours.

This schedule is feasible because the main paper does not depend on AutoCLIP, ImageNet-C, or multi-seed tuning.

## Success Criteria

The hypothesis is supported if:

1. the pilot shows that family residuals beat both the generic degraded residual and random family reassignment on family identification or corrupted accuracy;
2. the proposed method improves mean corruption accuracy over zero-shot CLIP on both CIFAR-10-C and CIFAR-100-C;
3. it matches or exceeds clean prompt ensembling on at least one dataset and does not clearly lose on the other;
4. clean accuracy drops by at most 0.5 points;
5. the gains are concentrated in matched families rather than appearing as uniform generic quality shifts.

The hypothesis is weakened or refuted if:

1. the family residual bank performs no better than a generic degraded residual;
2. random family reassignment performs similarly to correct family assignment;
3. the family posterior tracks semantics or image quality broadly rather than corruption family;
4. any gains vanish once compared against strong clean prompt ensembling;
5. projection provides no added value, in which case it is removed from the final story.

## References

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021. https://arxiv.org/abs/2103.00020

Menon, S., and Vondrick, C. (2023). Visual Classification via Description from Large Language Models. ICLR 2023. https://arxiv.org/abs/2210.07183

Shu, M., Nie, W., Huang, D.-A., Yu, Z., Goldstein, T., Anandkumar, A., and Xiao, C. (2022). Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models. NeurIPS 2022. https://arxiv.org/abs/2209.07511

Metzen, J. H., Saranrittichai, P., and Mummadi, C. K. (2024). AutoCLIP: Auto-tuning Zero-Shot Classifiers for Vision-Language Models. Transactions on Machine Learning Research, 2024. https://arxiv.org/abs/2309.16414

Zhu, X., Zhu, B., Tan, Y., Wang, S., Hao, Y., and Zhang, H. (2024). Enhancing Zero-Shot Vision Models by Label-Free Prompt Distribution Learning and Bias Correcting. NeurIPS 2024. https://openreview.net/forum?id=OJximyClit

Maharana, S. K., Zhang, B., Karlinsky, L., Feris, R., and Guo, Y. (2024). Enhancing Robustness of CLIP to Common Corruptions through Bimodal Test-Time Adaptation. arXiv:2412.02837. Later presented as BATCLIP at ICCV 2025. https://arxiv.org/abs/2412.02837

Gerych, W., Zhang, H., Hamidieh, K., Pan, E., Sharma, M., Hartvigsen, T., and Ghassemi, M. (2024). BendVLM: Test-Time Debiasing of Vision-Language Embeddings. NeurIPS 2024. https://arxiv.org/abs/2411.04420

Molahasani, M., Motamedi, A., Greenspan, M., Kim, I.-M., and Etemad, A. (2025). PRISM: Reducing Spurious Implicit Biases in Vision-Language Models with LLM-Guided Embedding Projection. ICCV 2025. https://arxiv.org/abs/2507.08979

Usama, M., Asim, S. A., Ali, S. B., Wasim, S. T., and Mansoor, U. B. (2025). Analysing the Robustness of Vision-Language-Models to Common Corruptions. arXiv:2504.13690. https://arxiv.org/abs/2504.13690

Waseda, F., Sugawara, S., and Echizen, I. (2025). Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models. ACM Multimedia 2025. https://arxiv.org/abs/2507.16257
