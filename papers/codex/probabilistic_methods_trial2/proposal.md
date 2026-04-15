# CHiP-RLCP Fallback Note: Hierarchical Diagonal-GMM Posteriors for Localized Conformal Prediction

## Scope

This workspace does **not** contain the proposed LearnSPN-style probabilistic circuit or exact upward-downward latent posterior inference. The proposal is therefore rewritten to match the implemented method and to avoid drawing SPN-specific conclusions that the code cannot support.

The resulting study is a scoped empirical note:

- joint model: a two-level hierarchical diagonal Gaussian-mixture classifier over `(x, y)`,
- score: `s(x, y) = 1 - p(y | x)`,
- localizer: coarse and fine posterior responsibilities from that hierarchical GMM,
- evaluation goal: compare the hierarchical fallback against simpler flat localizers and an explicit finite-group batch multivalid baseline on fixed external groups.

## Method

### Hierarchical fallback model

The fitted model is a coarse diagonal GMM over `X`, with one finer diagonal GMM per coarse component. Class probabilities are estimated within each fine component with additive smoothing. The resulting predictor defines:

- `p(y | x)` for scoring, and
- coarse and fine posterior memberships for localization.

This is a pragmatic CPU-feasible fallback, not a probabilistic circuit.

### Localized weighted conformal prediction

The main method keeps the same rigid localization family as originally intended:

- coarse posterior memberships,
- fine posterior memberships,
- expected-mass filtering at `>= 20` calibration points,
- overlap similarity `b_i(x) = sum_g min(m_g(x_i), m_g(x))`,
- global fallback weight `lambda = 0.10`,
- randomized local p-value computation with a test pseudo-point.

Because the underlying model is not a PC, the method is reported as a hierarchical-GMM localized conformal baseline rather than evidence for exact PC hierarchies.

### Baselines

The benchmark includes:

- Split CP,
- class-conditional CP,
- kNN localized weighted CP,
- flat GMM localized weighted CP,
- finite-group batch multivalid thresholding on the fixed external groups,
- oracle RLCP on the synthetic latent groups as a diagnostic upper bound.

## Datasets and groups

The benchmark uses three datasets and three seeds `[11, 22, 33]`:

- Synthetic latent hierarchy: coarse groups, fine groups, coarse-by-class intersections.
- Anuran Calls: `Family`, `Genus`, `Family x true class`.
- Mice Protein: `Genotype`, `Treatment`, `Behavior`, `Genotype x Treatment`.

## Ablations

The note still evaluates the four planned ablations of the hierarchical fallback:

- flat-only,
- coarse-only,
- no-fallback,
- uniform-overlap.

## Intended conclusion standard

This note supports only claims that the executed hierarchical-GMM fallback can justify.

- If it improves subgroup metrics over Split CP or kNN but is tied with flat GMM, the conclusion is that hierarchy was not clearly necessary.
- If finite-group batch multivalid thresholding dominates, the conclusion is that explicit subgroup optimization remains stronger once the groups are specified.
- If the fallback misses nominal coverage or loses badly on subgroup metrics, that negative result is reported directly.

No claim is made here about LearnSPN-style PCs, exact SPN inference, or PC-specific advantages.
