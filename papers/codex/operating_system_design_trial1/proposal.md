# DuelCache: Bias-Corrected Counterfactual Policy Ranking for One Shared Linux Page Cache

## Introduction

Recent systems such as Cache is King, `cache_ext`, FetchBPF, P2Cache, and PageFlex show that Linux paging and page-cache policy are becoming programmable rather than fixed kernel behavior [1, 2, 3, 4, 5]. That progress exposes a new systems problem: once several eviction policies are implementable on the same machine, how should the OS rank them online when all candidates are observed through one live shared page cache?

Adaptive selection itself is not new. ACME and SOPA already established multi-policy cache adaptation [6, 7]. Small monitored regions are also not new: set dueling already showed that a reserved subset can guide policy choice cheaply [8]. This proposal therefore does **not** claim novelty for adaptive selection, expert portfolios, or monitored sets.

The paper's novelty hinges on a narrower statement:

**In a programmable Linux page cache, inactive policies can only be observed through counterfactual shadows induced by the currently active shared cache, and those shadows are systematically biased enough that explicit bias correction, not just adaptive selection, is needed for accurate policy ranking.**

The contribution succeeds only if all three of the following hold:

1. sampled shadow rankings under one active shared Linux cache are measurably biased;
2. a small sentinel region can estimate and reduce that bias better than uncalibrated recent-window or direct leader-set selection;
3. the corrected ranking signal improves online decisions on replay.

If direct recent-window ranking or direct leader-set dueling matches the corrected method on the same state budget, the paper's main claim fails.

## Proposed Approach

### Overview

DuelCache maintains one active eviction policy in a programmable Linux page-cache substrate and several lightweight inactive shadows. The shadows estimate how alternative policies would have behaved, but they do so from requests filtered through the active cache's residency, eviction history, and inter-tenant contention. DuelCache treats this as a counterfactual estimation problem, not merely a policy-selection problem.

The design is scoped to the stated budget:

- policy portfolio: `LRU`, `MRU`, `LFU-aging`, `LHD`;
- one active policy at a time;
- sampled per-policy shadow state for inactive policies;
- a small reserved sentinel fraction used only for calibration;
- simple hysteresis and fixed switch cost;
- replay-heavy evaluation only; programmable-kernel validation is treated as future work unless a suitable substrate is available in-workspace.

The proposal intentionally avoids learned models, synthetic labels, fairness objectives, and large kernel modifications.

### Counterfactual shadow model

For each inactive policy `p` during epoch `e`, DuelCache maintains a sampled estimate `\hat{M}_e(p)` of miss cost or fault cost. The estimate is derived from sampled ghost entries, coarse recency/frequency metadata, and approximate eviction state. Because accesses are observed under active policy `a_e`, `\hat{M}_e(p)` is not an unbiased estimate of what would happen if `p` actually controlled the whole cache.

The bias comes from three Linux-specific sources:

- **residency filtering**: inactive policies never see misses that the active cache already prevented;
- **shared-cache coupling**: one tenant's accesses alter the residency context seen by all others;
- **sample truncation**: a sampled shadow misses interactions among sampled and unsampled objects.

The paper's first analysis is to measure this bias directly by comparing shadow rankings with exact replay on shortened traces.

### Sentinel-based bias correction

DuelCache reserves a small cache fraction `sigma` as a sentinel region. The sentinel is not used as a direct leader set that decides the winning policy by itself. Instead, it is used to estimate how much each inactive shadow deviates from realizable behavior.

For policy `p` and epoch `e`:

- `ShadowSentinel_e(p)`: shadow-predicted miss cost for `p` inside the sentinel region;
- `RealSentinel_e(p)`: realized miss cost for `p` in that same region using mirrored policy updates.

The estimated calibration residual is

`delta_e(p) = RealSentinel_e(p) - ShadowSentinel_e(p)`.

DuelCache then corrects whole-cache shadow estimates with

`tilde{M}_e(p) = \hat{M}_e(p) + w_e(p) * delta_e(p)`,

where `w_e(p)` is a simple variance-aware scaling term based on recent sentinel stability and sentinel coverage. This keeps the claim narrow: the main technical question is whether a small reserved region can transfer a useful bias estimate from sentinel behavior to the full-cache ranking problem.

### Calibration transfer analysis

The critical soundness risk is that sentinel bias may not transfer to the full cache. The proposal makes this a first-class experiment rather than an assumption.

DuelCache explicitly measures when transfer succeeds and when it fails by varying:

- **sentinel placement**: uniform hash-based placement, tenant-local placement, and contiguous-reserved placement;
- **sentinel fraction**: `0.5%`, `1%`, `2%`, `5%` of cache capacity;
- **tenant skew**: balanced two-tenant pressure versus highly dominant background-scan tenants;
- **phase volatility**: stationary workloads versus abrupt phase changes.

For each setting, the evaluation reports:

- full-cache ranking error before and after calibration;
- calibration residual versus actual full-cache error;
- a calibration-failure rate, defined as epochs where correction worsens ranking;
- representative failure cases, such as highly localized hotspots or severe tenant skew where sentinel bias does not transfer.

This analysis is central to the paper. If transfer is unreliable except in narrow cases, the paper should say so and bound the applicability claim.

### Switching rule

The controller selects

`p*_e = argmin_p [tilde{M}_e(p) + C_switch(p, a_e)]`

with a fixed measured switching penalty

`C_switch(p, a_e) = c0 * 1[p != a_e]`.

It switches only when:

- predicted gain exceeds hysteresis threshold `h`;
- minimum dwell time `d` has elapsed.

The controller is intentionally simple because the paper is not about sophisticated control logic. Any gains should come from better ranking, not from a more elaborate switch policy.

## Related Work

### Adaptive cache selection is prior art

**ACME** introduced adaptive caching with multiple experts at the 2002 Workshop on Distributed Data and Structures (WDAS) [6]. **SOPA** later studied adaptive online selection in *ACM Transactions on Storage* in 2010 [7]. These papers already cover the basic idea of choosing among policies online. DuelCache is only interesting if it shows that programmable Linux page cache introduces a distinct counterfactual bias problem that those selector abstractions do not address directly.

### Monitored regions are prior art

Set-dueling work showed that a small monitored subset can guide policy choice [8]. DuelCache borrows the idea of a reserved subset but uses it for **bias correction of sampled counterfactuals**, not as a direct winner-take-all leader-set signal. The nearest baseline is therefore not "no monitored region" but a strong direct `LeaderSetDuel` selector using the same sentinel budget.

### Programmable Linux memory-management substrates motivate the setting

**Cache is King** and **`cache_ext`** establish that Linux page-cache eviction can be customized with eBPF and that policy choice materially affects workload behavior [1, 2]. **FetchBPF** shows analogous flexibility for Linux prefetching [3]. **P2Cache**, published at HotStorage 2023, demonstrates application-directed page-cache control but does not study online ranking under one active shared cache [4]. **PageFlex**, published at USENIX ATC 2025, delegates Linux paging policies to user space and demonstrates low-overhead policy programmability [5]. These systems justify the substrate, but none of them isolate counterfactual ranking bias under a single live shared Linux page cache as the primary research problem.

### Adaptive single-policy baselines remain relevant

**ARC** is a strong adaptive single-policy baseline that changes recency/frequency behavior without explicit inter-policy switching [9]. DuelCache should outperform ARC only in regimes where switching among qualitatively different experts matters and where the ranking signal is accurate enough to justify switching.

## Experiments

### Implementation scope

The evaluation is deliberately scoped for 2 CPU cores, 128 GB RAM, and about 8 hours total:

- a user-space replay engine handles nearly all effectiveness experiments;
- exact replay is run only on shortened windows for ranking-ground-truth measurement;
- the real-system result uses a minimal programmable Linux substrate and only two policies (`LRU` and `MRU`);
- all workloads are CPU-only and replayable on one machine.

### Policy set and baselines

Policies:

- `LRU`
- `MRU`
- `LFU-aging`
- `LHD`

Adaptive baselines:

- `ARC`
- `RecentWindow`: SOPA-style recent-window selector over the same sampled shadows but no calibration
- `LeaderSetDuel`: direct sentinel-winner selector with the same reserved fraction but no whole-cache correction
- `NoCalibration`: DuelCache without the calibration term
- `DuelCache`

The required comparison is among `RecentWindow`, `LeaderSetDuel`, `NoCalibration`, and `DuelCache` with identical shadow-state budgets.

### Workloads

To reduce risk and avoid label-dependent claims, the paper uses four label-free workload families:

1. `PhaseLoop`: alternating sequential-loop and hotspot-random phases generated from reproducible file traces.
2. `TwoTenantMix`: one latency-sensitive hotspot tenant plus one streaming or cyclic tenant sharing the same cache.
3. `SkewShift`: a single-tenant trace whose popularity distribution shifts between two Zipf exponents and hotspot identities.
4. `StationaryZipf`: negative control with stable popularity and no intended phase change.

Each family runs at two cache sizes, `0.5x` and `0.8x` working-set coverage, with three seeds. Exact counterfactual replay is performed only on short windows sampled from each family.

### Metrics

Main replay metrics:

- weighted miss cost;
- miss ratio;
- regret to the offline best fixed policy on the full trace;
- Kendall tau between estimated and exact expert ordering on shortened windows;
- absolute error in predicted miss cost;
- switch count and unstable-epoch fraction;
- selector CPU time and metadata bytes.

Calibration-transfer metrics:

- correlation between sentinel residual and full-cache shadow error;
- fraction of epochs where calibration improves ranking;
- fraction of epochs where calibration degrades ranking;
- sensitivity to sentinel placement, fraction, and tenant skew.

### Main replay study

The main replay study asks whether bias correction improves policy ranking under one active shared cache.

Expected structure:

1. Measure exact-versus-shadow ranking error on shortened windows to establish the existence of bias.
2. Compare `NoCalibration`, `RecentWindow`, `LeaderSetDuel`, and `DuelCache` on full workloads under the same budget.
3. Evaluate whether improved ranking translates into lower miss cost and lower regret.

### Calibration transfer and failure study

This analysis directly addresses the key soundness concern.

For each workload family, vary:

- sentinel placement;
- sentinel fraction;
- tenant skew;
- epoch length.

Report both success and failure modes. In particular, if calibration helps on balanced interference traces but fails when the sentinel is tenant-local or the background tenant dominates residency, those conditions become explicit limitations rather than hidden assumptions.

### Programmable-substrate scope

This proposal is scoped to replay evidence in the current workspace. A real programmable substrate such as `cache_ext` or PageFlex is useful future validation, but it is not a confirmatory requirement unless the substrate is locally available and stable enough to run faithfully inside the fixed budget.

### Runtime budget

A feasible 8-hour schedule is:

- 4.5 hours for replay experiments and ablations;
- 1.5 hours for shortened exact-counterfactual windows;
- 1 hour reserved only if a local programmable substrate is already available;
- 1 hour for setup failures, reruns, and plotting.

If the kernel-backed effectiveness result proves too unstable, the paper should still preserve the replay results but must explicitly downgrade the systems claim.

## Success Criteria

### Confirmatory outcomes

1. On at least two non-stationary workload families, DuelCache reduces weighted miss cost by at least `5%` versus both `RecentWindow` and `LeaderSetDuel`.
2. On shortened exact windows, calibration improves Kendall tau by at least `0.10` absolute and reduces miss-cost prediction error by at least `15%` versus `NoCalibration`.
3. The calibration-transfer study shows that correction helps in a majority of epochs for at least one balanced multi-tenant and one single-tenant phase-shift workload, and it clearly identifies at least one failure regime where transfer breaks down.
4. On `StationaryZipf`, DuelCache remains within `2%` of the better of `LRU` and `ARC`.
5. Any real programmable-substrate result is treated as optional external validation rather than part of the core confirmatory claim in this replay-scoped study.

### Falsifying outcomes

1. Shared-cache shadow estimates are already well calibrated, so exact replay shows little ranking bias to correct.
2. `RecentWindow` or `LeaderSetDuel` matches DuelCache within confidence intervals on the mixed workloads.
3. Sentinel residuals do not transfer to full-cache ranking error except in contrived settings.
4. Calibration frequently worsens rankings under realistic tenant skew or sentinel placements.
5. If later substrate validation is attempted, no end-to-end benefit or implausible overhead would further narrow the practical claim beyond replay.

These are substantive falsifiers, not minor caveats. If they occur, the paper's main claim should be rejected or sharply narrowed.

## References

[1] Tal Zussman, Ioannis Zarkadas, Jeremy Carin, Andrew Cheng, Hubertus Franke, Jonas Pfefferle, and Asaf Cidon. *Cache is King: Smart Page Eviction with eBPF*. arXiv preprint arXiv:2502.02750, 2025. https://arxiv.org/abs/2502.02750

[2] Tal Zussman, Ioannis Zarkadas, Jeremy Carin, Andrew Cheng, Hubertus Franke, Jonas Pfefferle, and Asaf Cidon. *cache_ext: Customizing the Page Cache with eBPF*. In *Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles (SOSP)*, 2025. https://doi.org/10.1145/3731569.3764820

[3] Xuechun Cao, Shaurya Patel, Soo Yee Lim, Xueyuan Han, and Thomas Pasquier. *FetchBPF: Customizable Prefetching Policies in Linux with eBPF*. In *USENIX Annual Technical Conference (ATC)*, 2024. https://www.usenix.org/conference/atc24/presentation/cao

[4] Dusol Lee, Inhyuk Choi, Chanyoung Lee, Sungjin Lee, and Jihong Kim. *P2Cache: An Application-Directed Page Cache for Improving Performance of Data-Intensive Applications*. In *Proceedings of the 15th Workshop on Hot Topics in Storage and File Systems (HotStorage)*, 2023. https://doi.org/10.1145/3599691.3603408

[5] Anil Yelam, Kan Wu, Zhiyuan Guo, Suli Yang, Rajath Shashidhara, Wei Xu, Stanko Novaković, Alex C. Snoeren, and Kimberly Keeton. *PageFlex: Flexible and Efficient User-space Delegation of Linux Paging Policies with eBPF*. In *USENIX Annual Technical Conference (ATC)*, 2025. https://www.usenix.org/conference/atc25/presentation/yelam

[6] Ismail Ari, Ahmed Amer, Robert B. Gramacy, Ethan L. Miller, Scott A. Brandt, and Darrell D. E. Long. *ACME: Adaptive Caching Using Multiple Experts*. In *Workshop on Distributed Data and Structures (WDAS)*, pages 143-158, 2002. https://ssrc.us/media/pubs/96c7ffa6733e50e6fd43f14d4493118f07e369ab.pdf

[7] Yang Wang, Jiwu Shu, Guangyan Zhang, Wei Xue, and Weimin Zheng. *SOPA: Selecting the Optimal Caching Policy Adaptively*. *ACM Transactions on Storage*, 6(2):7:1-7:18, 2010. https://doi.org/10.1145/1807060.1807064

[8] Moinuddin K. Qureshi, Aamer Jaleel, Yale N. Patt, Simon C. Steely Jr., and Joel Emer. *Set-Dueling-Controlled Adaptive Insertion for High-Performance Caching*. In *Proceedings of the 40th IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 2007.

[9] Nimrod Megiddo and Dharmendra S. Modha. *ARC: A Self-Tuning, Low Overhead Replacement Cache*. In *Proceedings of the 2nd USENIX Conference on File and Storage Technologies (FAST)*, 2003. https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache
