#!/usr/bin/env python3
"""Re-run missing reviewer scores for trials that have <3 reviews.

Runs the missing reviewer agent, parses the result, updates reviews.json
in the results/ dir, and updates tracker.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from researcharena.stages.review import _run_cli_reviewer, _parse_review_from_output

BASE = Path(__file__).parent.parent

TASKS = [
    # (agent, seed_dir, trial, output_prefix, reviewer_config)
    ("claude", "causal_learning_cpu", "trial2", "claude_t2_causal_learning",
     {"type": "codex", "name": "Codex", "model": "gpt-5.4", "review_timeout": 3600}),
    ("claude", "causal_learning_cpu", "trial3", "claude_t3_causal_learning",
     {"type": "claude", "name": "Claude Code", "model": "claude-opus-4-6", "review_timeout": 3600}),
    ("claude", "compiler_optimization_cpu", "trial2", "claude_t2_compiler_optimization",
     {"type": "codex", "name": "Codex", "model": "gpt-5.4", "review_timeout": 3600}),
    ("claude", "data_integration_and_cleaning_cpu", "trial1", "claude_v5_data_integration_and_cleaning",
     {"type": "codex", "name": "Codex", "model": "gpt-5.4", "review_timeout": 3600}),
    ("claude", "data_integration_and_cleaning_cpu", "trial3", "claude_t3_data_integration_and_cleaning",
     {"type": "codex", "name": "Codex", "model": "gpt-5.4", "review_timeout": 3600}),
    ("kimi", "causal_learning_cpu", "trial3", "kimi_t3_causal_learning",
     {"type": "codex", "name": "Codex", "model": "gpt-5.4", "review_timeout": 3600}),
]


def run_review(agent, seed_dir, trial, output_prefix, reviewer_cfg):
    idea_dirs = sorted([
        d for d in (BASE / "outputs" / output_prefix).iterdir()
        if d.is_dir() and d.name.startswith("idea_") and not d.name.endswith("_review_logs")
    ])
    workspace = idea_dirs[-1]

    print(f"\n=== {agent}/{seed_dir}/{trial}: {reviewer_cfg['name']} on {workspace.name} ===")

    review_dict, result = _run_cli_reviewer(
        agent_cfg=reviewer_cfg,
        workspace=workspace,
        venue="neurips",
        docker_image="researcharena/agent-cpu:latest",
        runtime="local",
        domain="ml",
        ref_feedback="",
    )

    # Try parsing from stdout if direct parse failed
    if not review_dict and result and result.stdout:
        review_dict = _parse_review_from_output(result.stdout)

    if not review_dict:
        print(f"  FAILED: no review produced")
        return False

    score = review_dict.get("overall_score")
    decision = review_dict.get("decision")
    print(f"  Score: {score}, Decision: {decision}")

    # Update reviews.json in results
    reviews_files = sorted((BASE / "results" / agent / seed_dir / trial).glob("code/idea_*/reviews.json"))
    if not reviews_files:
        print(f"  ERROR: no reviews.json found")
        return False

    reviews_file = reviews_files[-1]
    reviews = json.loads(reviews_file.read_text())
    reviews["reviews"].append({
        "source": f"agent:{reviewer_cfg['name']}",
        "overall_score": score,
        "decision": decision,
        "scores": review_dict.get("scores", {}),
        "summary": review_dict.get("summary", ""),
        "strengths": review_dict.get("strengths", []),
        "weaknesses": review_dict.get("weaknesses", []),
        "detailed_feedback": review_dict.get("detailed_feedback", ""),
        "questions_for_authors": review_dict.get("questions_for_authors", []),
        "integrity_check": review_dict.get("integrity_check", ""),
    })
    scores = [r["overall_score"] for r in reviews["reviews"] if r.get("overall_score") is not None]
    reviews["avg_score"] = sum(scores) / len(scores) if scores else 0
    reviews_file.write_text(json.dumps(reviews, indent=2))
    print(f"  Updated reviews.json (avg={reviews['avg_score']:.2f})")

    # Update tracker.json
    tracker_file = BASE / "results" / agent / seed_dir / trial / "tracker" / "tracker.json"
    if tracker_file.exists():
        tracker = json.loads(tracker_file.read_text())
        for action in tracker.get("actions", []):
            if (action.get("stage") == "review"
                and reviewer_cfg["name"] in action.get("action", "")
                and action.get("outcome") == "failure"):
                action["outcome"] = "success"
                action["details"] = f"score={score}, decision={decision}"
                break
        tracker_file.write_text(json.dumps(tracker, indent=2))

    return True


def main():
    success = 0
    failed = 0
    for task in TASKS:
        try:
            if run_review(*task):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Done: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
