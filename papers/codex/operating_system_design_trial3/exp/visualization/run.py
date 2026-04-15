from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.pipeline import (
    aggregate_results,
    stage_audits,
    stage_data_preparation,
    stage_live_anchors,
    stage_reference_stability,
    stage_replays,
    stage_visualization,
)


if __name__ == "__main__":
    stats_df, _ = stage_data_preparation()
    replay_df, ranking_df, pruning_df = stage_replays(stats_df)
    ref_df = stage_reference_stability(stats_df, replay_df, ranking_df)
    live_df = stage_live_anchors(stats_df)
    reference_audit_df, spec_df = stage_audits()
    runtime_schedule = stage_visualization(replay_df, ranking_df, live_df, ref_df, stats_df, pruning_df)
    aggregate_results(stats_df, replay_df, ranking_df, pruning_df, ref_df, live_df, reference_audit_df, spec_df, runtime_schedule)
