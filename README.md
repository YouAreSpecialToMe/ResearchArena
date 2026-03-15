# AutoResearch

A benchmark harness for evaluating whether CLI agents (Claude Code, Codex, Aider) can autonomously conduct end-to-end machine learning research — from ideation through experimentation to writing a publishable paper that passes peer review.

## Overview

AutoResearch orchestrates a multi-stage pipeline that tasks an AI agent with:

1. **Ideation** — Generate a novel research idea from a seed topic, producing a structured hypothesis, method, datasets, and metrics
2. **Experimentation** — Implement and run the proposed experiments inside a GPU-enabled Docker container, producing quantitative results and figures
3. **Paper Writing** — Write a complete LaTeX conference paper (NeurIPS format by default)
4. **Review** — Evaluate the paper through multi-source automated review:
   - **Reference verification** against Semantic Scholar and CrossRef (fake citations = automatic rejection)
   - **[paperreview.ai](https://paperreview.ai)** — external Stanford agentic reviewer (optional)
   - **LLM reviewer agents** — independent Claude and GPT-4o reviewers score the paper

The pipeline iterates: papers that score below the acceptance threshold are either revised (with reviewer feedback) or abandoned in favor of a new idea. The best paper across all attempts is tracked and returned.

```
Seed Topic ──► Ideation ──► Experiments ──► Paper ──► Review
                  ▲                          │         │
                  │         abandon idea     │  revise │
                  └──────────────────────────┘◄────────┘
```

## Installation

**Requirements:** Python ≥ 3.10, Docker (with GPU support for experiments)

```bash
# Basic install
pip install -e .

# With paperreview.ai integration + dev tools
pip install -e ".[paperreview,dev]"
```

Set your API keys:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

## Quick Start

```bash
# Run with default config (Claude Sonnet on "efficient fine-tuning methods for LLMs")
autoresearch run --config configs/default.yaml

# Override seed topic and agent
autoresearch run \
  --seed "neural architecture search" \
  --agent claude \
  --model claude-opus-4 \
  --max-ideas 3

# Review an existing paper without running the research pipeline
autoresearch review-only --config configs/default.yaml /path/to/workspace
```

## Configuration

All settings live in `configs/default.yaml`. Key options:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `seed_topic` | — | `"efficient fine-tuning methods for large language models"` | Research topic given to the agent |
| `agent.type` | — | `claude` | Agent under test (`claude`, `codex`, `aider`, `custom`) |
| `agent.model` | — | `claude-sonnet-4-6` | Model to use for the agent |
| `agent.gpus` | — | `1` | GPUs allocated to the Docker container |
| `experiment.max_gpu_hours` | — | `4` | Timeout for experiment stage |
| `paper.template` | — | `neurips` | LaTeX template venue |
| `review.accept_threshold` | — | `6.0` | Score (1–10) required for acceptance |
| `pipeline.max_ideas_per_seed` | — | `5` | Maximum idea attempts before giving up |
| `pipeline.max_global_steps` | — | `30` | Hard cap on total pipeline iterations |

## Project Structure

```
autoresearch/
├── cli.py                    # CLI entry point (click)
├── pipeline.py               # Core pipeline orchestrator & state machine
├── stages/
│   ├── ideation.py           # Stage 1: research idea generation
│   ├── experiment_design.py  # Stage 2: experiment implementation & execution
│   ├── paper_writing.py      # Stage 3: LaTeX paper writing
│   └── review.py             # Stage 4: multi-source paper review
├── templates/
│   └── research_guidelines.md  # Scientific rigor guidelines given to agents
└── utils/
    ├── agent_runner.py       # Docker container orchestration
    ├── config.py             # YAML config loading & merging
    ├── llm.py                # Unified LLM client (Anthropic + OpenAI)
    ├── reference_checker.py  # Citation verification via Semantic Scholar + CrossRef
    └── paperreview.py        # paperreview.ai web automation (Playwright)
```

## Output

Each run produces artifacts under `outputs/runs/`:

```
outputs/runs/
├── idea_01/
│   ├── idea.json              # Structured research idea
│   ├── results.json           # Experiment results
│   ├── figures/               # Generated plots
│   ├── paper.tex / paper.pdf  # The written paper
│   ├── reviews.json           # All review feedback
│   ├── reference_check.json   # Citation verification report
│   └── logs/                  # Agent stdout/stderr
├── idea_02/
│   └── ...
└── summary.json               # Final run summary with scores & best paper
```

## Docker Environment

Agents execute inside a Docker container built from the included `Dockerfile`, based on `nvidia/cuda:12.4.0-devel-ubuntu22.04`. The container comes pre-installed with:

- Python, PyTorch, Transformers, HuggingFace ecosystem
- numpy, pandas, scikit-learn, Jupyter
- texlive (LaTeX compilation)
- CLI agent tools (Claude Code, Codex, Aider)

## How Review Works

The review stage aggregates scores from multiple sources to reduce bias:

1. **Reference checker** — Extracts all citations from the paper and verifies each against Semantic Scholar and CrossRef with fuzzy title matching. Any unverified reference triggers an automatic rejection (score 0).
2. **paperreview.ai** — If configured with email credentials, submits the PDF to Stanford's agentic reviewer and parses the returned scores across 7 dimensions.
3. **LLM reviewers** — Independent model instances (Claude Sonnet + GPT-4o by default) evaluate the paper as conference reviewers, scoring on novelty, clarity, rigor, and significance.

The scores are averaged, and the pipeline decides:
- **Score ≥ threshold** → Accepted
- **Score < threshold by ≤ 2 points, revisions remaining** → Revise paper with feedback
- **Score far below threshold or revisions exhausted** → Abandon idea, try a new one

## License

See `LICENSE` for details.
