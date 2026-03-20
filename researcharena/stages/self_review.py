"""Review gates for idea, experiment, and paper stages.

Each gate invokes the agent to review the work in the workspace using
stage-specific subsets of the reviewer guidelines. Produces a score +
feedback used by the pipeline to decide whether to proceed or revise.
"""

from __future__ import annotations

import json
from pathlib import Path

from researcharena.utils.agent_runner import invoke_agent

# Template directory
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def _load_reviewer_guidelines(domain: str = "ml") -> str:
    """Load the full reviewer guidelines for the domain."""
    domain_path = _TEMPLATES_DIR / domain / "reviewer_guidelines.md"
    if domain_path.exists():
        return domain_path.read_text()
    return (_TEMPLATES_DIR / "ml" / "reviewer_guidelines.md").read_text()


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
    reviewer_guidelines = _load_reviewer_guidelines(domain)
    task = _build_task(stage, reviewer_guidelines)

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


def _build_task(stage: str, reviewer_guidelines: str) -> str:
    """Build the review task prompt."""
    stage_tasks = {
        "idea": (
            "You are reviewing a research proposal and experiment plan.\n\n"
            "Evaluate the following artifacts in the workspace:\n"
            "  - proposal.md: the research proposal\n"
            "  - plan.json: the experiment plan\n"
            "  - idea.json: the idea summary\n"
            "  - references/: parsed reference papers\n\n"
            "Focus on these dimensions from the reviewer guidelines:\n"
            "  - Novelty: is the idea genuinely new? Search online to verify.\n"
            "  - Soundness: is the methodology appropriate?\n"
            "  - Significance: does this problem matter?\n"
            "  - References: are citations real and relevant?\n"
            "  - Plan quality: is the experiment plan complete and feasible?\n"
        ),
        "experiment": (
            "You are reviewing experiment results.\n\n"
            "Evaluate the following artifacts in the workspace:\n"
            "  - results.json: experiment results\n"
            "  - exp/: experiment code and per-experiment results\n"
            "  - plan.json: the original experiment plan\n"
            "  - figures/: generated figures\n\n"
            "Focus on these dimensions from the reviewer guidelines:\n"
            "  - Soundness: do results support the claims?\n"
            "  - Experimental rigor: baselines, ablations, error bars, seeds?\n"
            "  - Results integrity: do code and logs match results.json?\n"
            "  - Reproducibility: are details specified?\n"
            "  - Plan compliance: were all planned steps executed?\n"
        ),
        "paper": (
            "You are reviewing a research paper.\n\n"
            "Evaluate the following artifacts in the workspace:\n"
            "  - paper.tex / paper.pdf: the paper\n"
            "  - results.json: experiment results\n"
            "  - exp/: experiment code and logs\n\n"
            "Apply ALL dimensions from the reviewer guidelines below.\n"
            "This is a full review — evaluate novelty, soundness, significance,\n"
            "clarity, reproducibility, rigor, references, and results integrity.\n"
        ),
    }

    output_format = (
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

    return (
        f"=== REVIEW: {stage.upper()} ===\n\n"
        f"{stage_tasks[stage]}\n"
        "Be rigorous. Search the web to verify novelty and references.\n\n"
        "--- REVIEWER GUIDELINES ---\n"
        f"{reviewer_guidelines}\n"
        "--- END GUIDELINES ---\n"
        f"{output_format}"
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

    # Find JSON blocks containing "score"
    return _extract_score_from_text(text)


def _extract_score_from_text(text: str) -> tuple[float, str]:
    """Find the last JSON object with a 'score' field in the text."""
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
        try:
            data = json.loads(candidate)
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
