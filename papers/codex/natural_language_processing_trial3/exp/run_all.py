from __future__ import annotations

import argparse

from exp.shared.runner import run_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="full",
        choices=["smoke", "full"],
    )
    args = parser.parse_args()

    if args.mode == "smoke":
        for condition in ["imdb_erm_smoke_test", "imdb_latebind_smoke_test"]:
            run_condition(condition)
        return

    order = [
        "imdb_erm_smoke_test",
        "imdb_latebind_smoke_test",
        "imdb_tfidf_lr",
        "imdb_erm",
        "imdb_masker",
        "imdb_jtt",
        "imdb_latebind",
        "imdb_no_late_term",
        "imdb_no_invariance",
        "imdb_ungated_entropy",
        "imdb_lexicon_only_risk",
        "imdb_attribution_only_risk",
        "imdb_random_token_risk",
        "imdb_actor_only_masking",
        "civilcomments_erm",
        "civilcomments_jtt",
        "civilcomments_latebind",
    ]
    for condition in order:
        run_condition(condition)


if __name__ == "__main__":
    main()
