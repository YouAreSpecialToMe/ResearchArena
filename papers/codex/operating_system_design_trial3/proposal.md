# ShadowCache: A Simulator-Backed Trace Study of How Much Observable State Policy Ranking Needs

## Introduction

This rerun makes a narrower claim than the original Stage 1 proposal. The workspace does not contain `filebench`, `fio`, or a mechanism to swap Linux page-cache policies live, and the Python standard library does not expose a practical SQLite VFS registration path. The executable study is therefore:

- real workload execution,
- measured trace capture,
- offline policy replay,
- measured online-policy anchors over the same event streams,
- no live kernel page-cache policy swapping.

The question remains the same in spirit:

**How much observable state is needed to preserve the ranking of a fixed six-policy pool?**

The answer is now scoped to a simulator-backed trace study rather than a live Linux page-cache validation paper.

## Narrowed Claim

For a fixed six-policy comparison pool and three workload families, a compact trace that keeps logical accesses together with write or dirty hints and coarse reclaim epochs preserves policy ranking better than an access-only trace. The richer replay reference is built from measured workload executions plus sparse kernel-visible cache-residency snapshots where feasible, not from a replay-side observer derived from the same access stream.

## Trace Hierarchy

The study keeps the original four primary trace modes:

1. `ExtendedHinted`
   - `logical_seq`
   - `inode_id`
   - `page_index`
   - `op_class`
   - `phase_id`
   - `cache_insert_seen`
   - `cache_evict_seen`
   - `dirty_or_writeback_seen`
   - `reclaim_epoch`

2. `CompactState`
   - `ExtendedHinted` without `cache_insert_seen` and `cache_evict_seen`

3. `NoDirty`
   - `CompactState` without `dirty_or_writeback_seen`

4. `AccessOnly`
   - logical access stream only

This rerun also restores the previously missing supplementary ablation:

5. `NoReclaim`
   - `CompactState` without `reclaim_epoch`

## Workloads

The executed workloads remain:

- `stream_scan`
- `sqlite_zipf`
- `filebench_fileserver`

Their implementations are narrower than originally proposed:

- `stream_scan` is a measured sequential file-corpus scan with periodic hot rereads.
- `sqlite_zipf` is a measured SQLite read-mostly workload whose logical page IDs are derived from fixed-width row packing rather than a custom VFS page logger.
- `filebench_fileserver` is a measured Python fileserver-like mixed workload because `filebench` is unavailable.

The dataset scales are sub-GiB rather than the originally planned 24 GiB, 16 GiB, and 20 GiB. This is intentional and is reported explicitly. The reason is not memory pressure; it is execution efficiency after adding measured cache-residency sampling and the extra ablation and paired-resampling analysis within the 2-core, 8-hour envelope.

## Measured State

The main methodological repair in this rerun is that the richer state is no longer synthesized by an internal observer over the logical trace.

- `cache_insert_seen` and `cache_evict_seen` are derived from measured `mincore` residency transitions for the referenced file page where that mapping is available.
- `reclaim_epoch` comes from coarse `/proc/vmstat` reclaim sampling.
- `dirty_or_writeback_seen` combines application-visible writes with coarse vmstat dirty or writeback movement.

These are still approximations and are discussed as such. They are nevertheless materially stronger than deriving all richer state from the same logical access stream inside the replay code.

## Policies

The fixed six-policy pool is unchanged:

- `LinuxDefault`
- `FIFO`
- `CLOCK`
- `LFU`
- `S3FIFO`
- `Hyperbolic`

`FIFO` remains the simple baseline, `S3FIFO` the strong baseline, and `LinuxDefault` the deployed-systems baseline.

## Experiments

### E1. Main Ranking Study

For each of `3 workloads x 3 seeds x 2 budgets`, replay all six policies under the four primary trace modes.

Primary metrics:

- `Kendall_tau_6`
- `Spearman_rho_6`
- `top1_agreement`
- `top2_set_recall`
- `best_policy_regret`

### E2. Reference Adequacy

Rebuild the richer reference without explicit insert or evict hints and compare it against `ExtendedHinted`.

Metrics:

- `reference_tau_6`
- `reference_rho_6`
- `reference_top1_agreement`
- `reference_top2_recall`
- top-policy change fraction

### E3. Online Anchors

Because live kernel policy swaps are infeasible here, the validation step is narrowed to measured online-policy anchors over the exact workload event streams. For seed `11` and budget ratio `0.40`, each workload family runs all six policies in online mode and compares:

- `ExtendedHinted`
- `CompactState`
- `AccessOnly`

against the measured online ranking.

### E4. Component Ablations

The rerun completes the missing supplementary ablation:

- `NoReclaim`

and reports deltas relative to `CompactState`.

### E5. Uncertainty

The rerun now computes paired bootstrap `95 percent` intervals with `1000` resamples for compact-versus-ablation comparisons over the 18 replay cases and bootstrap intervals over the 3 online anchors for live-agreement metrics.

## What This Paper Does Not Claim

This rerun does not claim:

- live Linux page-cache policy validation,
- a Filebench-based mixed-service benchmark,
- a SQLite VFS page logger,
- or official full-fidelity Linux page-cache trace collection.

Those items were planned originally but are infeasible in this workspace and are now explicitly removed from the claim rather than implied.

## Success Criteria

The success criteria remain close to the original hypothesis, but interpreted under the narrower scope:

1. `CompactState` reaches mean `top2_set_recall >= 0.80` and mean `Kendall_tau_6 >= 0.65` versus `ExtendedHinted`.
2. `AccessOnly` is at least `0.10` worse than `CompactState` on mean `Kendall_tau_6`, `top2_set_recall`, or a paired-delta interval that excludes zero in the wrong direction.
3. The rebuilt no-hints reference keeps mean `reference_top2_recall >= 0.80`.
4. `ExtendedHinted` and `CompactState` each recover the online anchor top-2 set in at least `2 of 3` workload families.
5. The entire study remains within the 8-hour, 2-core, CPU-only budget.

Negative results are acceptable and are reported directly if the compact trace does not materially outperform access-only.
