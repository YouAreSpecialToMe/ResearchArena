# Proxy Feasibility Study: Object Units or Pixels?

## Stage 2 Framing

The registered Stage 1 study targeted CAT-Seg on Cityscapes, Cityscapes-C, and ACDC with MLMP as contextual comparison. That benchmark was not executable in this workspace. The compatibility check saved to `exp/00_environment/compatibility_check.json` found no runnable CAT-Seg, Cityscapes, ACDC, or MLMP assets.

Stage 2 is therefore reported only as an **executed proxy feasibility study**. It uses Pascal VOC 2012, a DeepLabV3-ResNet50 proxy backbone, synthetic clean/fog/gaussian-noise/snow/dusk/night shifts, and the same substrate-comparison logic. Any ranking in this workspace is a **negative proxy result**, not evidence for the unavailable registered OVSS benchmark.

## Registered Intent Versus Executed Study

The registered intent was to test whether CLIP-verified object-like regions are a better online adaptation substrate than pixels or ungated masks in an OVSS setting under matched update budgets.

Because the required OVSS assets were missing, the executed study narrows that question to a closed-vocabulary proxy:

- same adaptation scaffold across methods
- same one-step-per-image update schedule
- same adapter size and optimizer
- same calibration-derived budget and thresholds
- only the support substrate changes

This keeps the proxy useful for feasibility and auditability, while avoiding any claim that the proxy validates the original OVSS hypothesis.

## Executed Proxy Protocol

### Backbone and Label Space

- Backbone: `torchvision` DeepLabV3-ResNet50
- Trainable component: rank-4 logit adapter
- Resolution: 256x256
- Label space: Pascal VOC 21 classes
- Reported mIoU: foreground mIoU over the 20 non-background classes

### Data

- Calibration split: 24 Pascal VOC train images
- Main matrix: 144 Pascal VOC val images, six 24-image domain blocks
- Reduced slice: 72 images, six 12-image domain blocks
- Orders:
  - A: clean -> fog -> gaussian_noise -> snow -> dusk -> night
  - B: night -> dusk -> snow -> gaussian_noise -> fog -> clean

### Main Methods

1. `frozen`: no adaptation
2. `topb_pixel`: update on top-confidence pixels up to budget `B`
3. `raw_mask`: update on cached proposal regions without CLIP verification
4. `clip_verified`: update on cached proposal regions that pass OVSS-CLIP agreement checks

### Ablations

- `slic`: generic regionization control
- `no_clip`: remove CLIP verification and use OVSS-only labels
- `no_consistency`: set the consistency weight to zero
- threshold sensitivity: margin offsets `-0.05`, `0.00`, `+0.05`
- budget sensitivity: `0.8B`, `1.0B`, `1.2B` for `topb_pixel` and `clip_verified`

All stochastic runs use seeds `13`, `17`, and `23`.

## Calibration And Metrics

Thresholds and the shared budget are frozen from the calibration split only. The executed proxy uses:

- `tau_ovss = 1.0`
- `tau_clip = 1.0`
- calibration-derived margin, entropy, and agreement thresholds
- calibration-derived shared budget `B`

Metric protocol:

- Aggregate mIoU is computed from the dataset confusion matrix over the 20 Pascal VOC foreground classes.
- Per-image mIoU is computed over only the foreground classes present in that image.
- Older zero-filled per-image mIoU values are treated only as diagnostics and are not used for final aggregate reporting.

This metric cleanup is necessary because zero-filling absent classes at the image level makes per-image values artificially tiny and numerically incompatible with dataset-level mIoU.

## Expected Interpretation

This workspace can support only the following conclusion:

- whether the proxy implementation is runnable, calibrated, and internally consistent
- whether CLIP-verified regions beat the matched proxy baselines in this restricted setup

It cannot support:

- any claim that CAT-Seg outperforms alternatives
- any claim that CLIP-verified object units are superior on Cityscapes, Cityscapes-C, or ACDC
- any claim about MLMP relative ranking, because MLMP was not run here

## Success And Failure Criteria

The proxy result is considered supportive only if `clip_verified` clearly outperforms both `topb_pixel` and `raw_mask` under the matched proxy protocol and remains ahead under the reduced-slice ablations.

If gains are negligible, inconsistent, or statistically unconvincing, the correct interpretation is a **negative proxy feasibility result**. That is the default expectation for the executed study unless the rerun produces materially different numbers.
