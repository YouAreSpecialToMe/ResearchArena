from exp.shared.run_core import run_method
from exp.shared.utils import project_path, write_json, write_jsonl


def main() -> None:
    configs = [
        ("ablation_assignment_source", dict(assignment_source="detector", use_counterfactual=True, aggregation="geometric", force_non_null=False)),
        ("ablation_counterfactual_scoring", dict(assignment_source="detector_daam", use_counterfactual=False, aggregation="geometric", force_non_null=False)),
        ("ablation_null_and_count", dict(assignment_source="detector_daam", use_counterfactual=True, aggregation="geometric", force_non_null=True)),
        ("ablation_aggregation_rule", dict(assignment_source="detector_daam", use_counterfactual=True, aggregation="mean", force_non_null=False)),
        ("ablation_candidate_budget_assign_and_verify", dict(assignment_source="detector_daam", use_counterfactual=True, aggregation="geometric", force_non_null=False)),
    ]
    all_rows = []
    for method, kwargs in configs:
        split = "candidate_budget" if "candidate_budget" in method else "test"
        k = 8 if "candidate_budget" in method else 4
        rows = run_method(split, method, k=k, **kwargs)
        all_rows.extend(rows)
    write_jsonl(project_path("results", "ablations.jsonl"), all_rows)
    write_json(project_path("exp", "ablations", "results.json"), {"num_rows": len(all_rows)})


if __name__ == "__main__":
    main()
