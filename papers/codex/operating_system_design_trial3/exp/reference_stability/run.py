from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.pipeline import stage_data_preparation, stage_replays, stage_reference_stability


if __name__ == "__main__":
    stats_df, _ = stage_data_preparation()
    replay_df, ranking_df, _ = stage_replays(stats_df)
    stage_reference_stability(stats_df, replay_df, ranking_df)
