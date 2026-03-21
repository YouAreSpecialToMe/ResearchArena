"""Review gates for idea, experiment, and paper stages.

Each gate invokes the agent to review the work in the workspace using
stage-specific criteria. Produces a score + feedback used by the pipeline
to decide whether to proceed or revise.
"""

from __future__ import annotations

import json
from pathlib import Path

from researcharena.utils.agent_runner import invoke_agent


_OUTPUT_FORMAT = (
    "\n\nOutput your review as a JSON object. Print ONLY the JSON:\n"
    "{\n"
    '    "score": <int, one of: 0, 2, 4, 6, 8, 10>,\n'
    '    "summary": "<2-3 sentence summary>",\n'
    '    "strengths": ["<strength1>", ...],\n'
    '    "weaknesses": ["<weakness1>", ...],\n'
    '    "feedback": "<specific, actionable feedback for improvement>"\n'
    "}\n\n"
    "Score scale: 10=exceptional, 8=strong, 6=acceptable with concerns,\n"
    "4=major issues, 2=fundamental flaws, 0=not viable.\n"
)


def run_self_review(
    agent_type: str,
    workspace: Path,
    stage: str,
    agent_config: dict | None = None,
    timeout: int = 900,
    domain: str = "ml",
) -> tuple[float, str, object]:
    """Run a review for the given stage.

    Args:
        agent_type: The CLI agent type (claude, codex, etc.)
        workspace: The workspace directory containing artifacts to review
        stage: One of "idea", "experiment", "paper"
        agent_config: Agent configuration dict
        timeout: Max seconds for the review
        domain: Domain for guideline templates

    Returns:
        (score, feedback, agent_result)
    """
    task = _build_task(stage)

    agent_result = invoke_agent(
        agent_type=agent_type,
        task=task,
        workspace=workspace,
        timeout=timeout,
        agent_config=agent_config,
        readonly=True,
    )

    score, feedback = _parse_output(agent_result)
    return score, feedback, agent_result


def _build_idea_review() -> str:
    return (
        "You are reviewing a research proposal and experiment plan.\n\n"
        "Evaluate the following artifacts in the workspace:\n"
        "  - proposal.md: the research proposal\n"
        "  - plan.json: the experiment plan\n"
        "  - idea.json: the idea summary\n"
        "  - references/: parsed reference papers\n\n"
        "## Evaluation Criteria\n\n"
        "### Novelty (most important)\n"
        "- Does the proposal present a genuinely new idea?\n"
        "- **You MUST search online** (arXiv, Semantic Scholar, Google Scholar)\n"
        "  to check whether similar work already exists\n"
        "- Search the exact proposed method name, the core technique + domain,\n"
        "  and the key components combined\n"
        "- If you find substantially similar published work, score <= 4\n\n"
        "### Soundness\n"
        "- Is the proposed methodology appropriate for the problem?\n"
        "- Are assumptions stated and reasonable?\n"
        "- Could the approach plausibly work?\n\n"
        "### Significance\n"
        "- Does this problem matter to the research community?\n"
        "- Would the results (positive or negative) be useful?\n\n"
        "### Reference Quality\n"
        "- Are key related works cited in the proposal?\n"
        "- Are the references real, verifiable publications?\n"
        "- Search online to verify at least 3 key references actually exist\n"
        "- Check that the references/ directory contains parsed papers — if it\n"
        "  is empty or missing, score <= 4 (this is a required output)\n\n"
        "### Experiment Plan Quality\n"
        "- Is plan.json complete? Does it cover: setup, baselines, main experiments,\n"
        "  ablations, evaluation, and visualization?\n"
        "- Are the planned experiments feasible within the resource constraints?\n"
        "- Is each step detailed enough to follow without ambiguity?\n"
        "- Are at least 2 meaningful baselines planned?\n"
        "- Are success criteria clearly defined?\n"
    )


def _build_experiment_review() -> str:
    return (
        "You are reviewing experiment results.\n\n"
        "Evaluate the following artifacts in the workspace:\n"
        "  - exp/<name>/: per-experiment folders, each with code, results, and logs\n"
        "  - plan.json: the original experiment plan\n"
        "  - figures/: generated figures\n"
        "  - results.json (if present): aggregated results at workspace root\n\n"
        "## Evaluation Criteria\n\n"
        "### Plan Compliance\n"
        "- Were all steps in plan.json executed?\n"
        "- If steps were skipped, is there a documented reason?\n"
        "- Were all planned baselines implemented and run?\n"
        "- Were all planned ablations completed?\n\n"
        "### Experimental Rigor\n"
        "- Are there at least 2 meaningful baselines with fair comparisons?\n"
        "- Are ablation studies present showing each component's contribution?\n"
        "- Are error bars / confidence intervals reported?\n"
        "- Is a fixed random seed used for reproducibility?\n"
        "- Are comparisons fair (same data, same compute budget)?\n\n"
        "### Results Integrity\n"
        "- Do per-experiment results contain actual output (not hardcoded)?\n"
        "- Do experiment logs in exp/*/logs/ show real training/evaluation runs?\n"
        "- Do the numbers in the results match what the logs show?\n"
        "- Does the code in exp/ implement what proposal.md describes?\n\n"
        "### Soundness\n"
        "- Do the results support the claims in proposal.md?\n"
        "- Are there confounding factors that could explain the results?\n"
        "- If the method underperforms baselines, is this acknowledged?\n\n"
        "### Reproducibility\n"
        "- Are all hyperparameters recorded in experiment configs or results?\n"
        "- Is the data described (source, splits, preprocessing)?\n"
        "- Could someone reproduce these results from the code?\n"
    )


def _build_paper_review() -> str:
    return (
        "You are reviewing a research paper before submission.\n\n"
        "Evaluate the following artifacts in the workspace:\n"
        "  - paper.tex / paper.pdf: the paper\n"
        "  - exp/<name>/: per-experiment folders with code, results, and logs\n"
        "  - proposal.md: the original proposal\n"
        "  - plan.json: the experiment plan\n\n"
        "## Evaluation Criteria\n\n"
        "### Paper Writing Quality\n"
        "- Is the paper well-organized with clear structure?\n"
        "- Are contributions explicitly stated in the introduction?\n"
        "- Is the writing clear and concise? Any grammatical issues?\n"
        "- Are figures and tables self-contained with descriptive captions?\n"
        "- Is notation consistent throughout?\n"
        "- Does the abstract accurately summarize the paper?\n"
        "- Is the related work section comprehensive and fairly positioned?\n\n"
        "### Results Consistency (critical)\n"
        "- Compare EVERY number in the paper's tables against the experiment\n"
        "  results in exp/<name>/results.json\n"
        "- Flag ANY mismatch — even small rounding differences\n"
        "- Do figures accurately represent the experimental data?\n"
        "- Are claims in the text supported by the numbers shown?\n\n"
        "### Logical Flow\n"
        "- Does the paper tell a coherent story from motivation to conclusion?\n"
        "- Does the method section clearly explain HOW the approach works?\n"
        "- Are experimental choices (datasets, metrics, baselines) justified?\n"
        "- Do the conclusions follow from the results shown?\n"
        "- Are limitations honestly discussed?\n\n"
        "### Reference Quality\n"
        "- Are all cited references real, verifiable publications?\n"
        "- Search online to spot-check at least 3 references\n"
        "- Are key related works properly cited and discussed?\n\n"
        "### Completeness\n"
        "- Does the paper have all required sections?\n"
        "- Are all experiments from exp/ reported in the paper?\n"
        "- Are ablation results included?\n"
        "- Is there a reproducibility section or appendix?\n"
    )


def _build_task(stage: str) -> str:
    """Build the review task prompt for the given stage."""
    builders = {
        "idea": _build_idea_review,
        "experiment": _build_experiment_review,
        "paper": _build_paper_review,
    }

    return (
        f"=== REVIEW: {stage.upper()} ===\n\n"
        f"{builders[stage]()}\n"
        "Be rigorous. If you find issues, flag them specifically.\n"
        f"{_OUTPUT_FORMAT}"
    )


def _parse_output(agent_result) -> tuple[float, str]:
    """Parse score and feedback from agent output.

    Handles two formats:
    - Plain text (Claude): search for JSON blocks containing 'score'
    - JSONL streaming (Codex): extract text from agent_message items, then
      search for JSON blocks within those messages

    Returns (score, feedback). Defaults to (0, "") on parse failure.
    """
    if not agent_result or not agent_result.stdout:
        return 0.0, "Review produced no output."

    stdout = agent_result.stdout

    # Extract text content — handle JSONL streaming format (Codex, Kimi)
    # by pulling text from agent_message items
    text_parts = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                text_parts.append(line)
                continue
            # Codex JSONL: {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
            if obj.get("type") in ("item.completed", "item.started"):
                item = obj.get("item", {})
                if item.get("type") == "agent_message":
                    text_parts.append(item.get("text", ""))
            # Claude stream-json: {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}}
            elif obj.get("type") == "assistant":
                for block in obj.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
            elif obj.get("type") == "result":
                text_parts.append(obj.get("result", ""))
            # Kimi: {"role":"assistant","content":[{"type":"text","text":"..."}]}
            elif obj.get("role") == "assistant":
                content = obj.get("content", [])
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
            # Direct JSON with score field
            elif "score" in obj:
                text_parts.append(json.dumps(obj))
        except json.JSONDecodeError:
            # Plain text line — include it
            text_parts.append(line)

    text = "\n".join(text_parts) if text_parts else stdout

    # Strip markdown code fences
    import re
    text = re.sub(r'```(?:json)?\s*\n?', '', text)

    # Find JSON blocks containing "score"
    return _extract_score_from_text(text)


def _extract_score_from_text(text: str) -> tuple[float, str]:
    """Find the last JSON object with a 'score' field in the text."""
    import re

    brace_depth = 0
    json_start = -1
    candidates = []

    for i, ch in enumerate(text):
        if ch == '{':
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and json_start >= 0:
                candidates.append(text[json_start:i + 1])
                json_start = -1

    # Try candidates from last to first (final output most likely)
    for candidate in reversed(candidates):
        # Try parsing, with fallback for trailing commas
        for attempt_str in [candidate, re.sub(r',\s*}', '}', re.sub(r',\s*]', ']', candidate))]:
            try:
                data = json.loads(attempt_str)
            except json.JSONDecodeError:
                continue
            if "score" in data:
                score = float(data["score"])
                feedback = data.get("feedback", "")
                weaknesses = data.get("weaknesses", [])
                if weaknesses and not feedback:
                    feedback = "; ".join(weaknesses)
                elif weaknesses:
                    feedback += "\nWeaknesses: " + "; ".join(weaknesses)
                return score, feedback
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    return 0.0, ""
