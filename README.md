# How Far Are We From True Auto Research?

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Website](https://img.shields.io/badge/Website-ResearchArena-5eead4.svg)](https://youarespecialtome.github.io/ResearchArena/)
[![GitHub stars](https://img.shields.io/github/stars/YouAreSpecialToMe/ResearchArena?style=social)](https://github.com/YouAreSpecialToMe/ResearchArena/stargazers)

An in-depth analysis of frontier CLI agents — **Claude Code (Opus 4.6)**, **Codex (GPT-5.4)**, **Kimi Code (K2.5)** — conducting end-to-end research across diverse fields and compute resources.

---

## At a glance

- **117 agent-generated papers** — 39 per agent (Claude Code, Codex, Kimi Code), 3 trials × 13 seeds, spanning both GPU and CPU domains
- **351 code-aware peer reviews** (3 CLI-agent reviewers per paper) + **117 Stanford Agentic Reviewer scores**
- **Manual annotation of every paper** for results mismatches, implementation mismatches, and fabricated references

➡️ Read the [full write-up](https://youarespecialtome.github.io/ResearchArena/) for scores, per-domain breakdowns, case studies, and the human-inspection findings.

## What this does

Given a seed field (e.g., "computer vision", "compiler optimization"), each CLI agent follows a standardized pipeline:

1. **Ideation** — Generate a research idea and experiment plan; self-review for up to 3 iterations.
2. **Experiments** — Write and execute code, collect results; self-review for up to 3 iterations.
3. **Paper Writing** — Produce a paper; self-review for up to 3 iterations.
4. **Review** — Evaluate via Stanford Agentic Reviewer and triple peer review (all three agents review each paper alongside its code).

## Conferences & areas

Seeds span multiple CS conferences and two compute platforms. Hardware: **1× RTX A6000 (48GB), 4 CPUs, 60GB RAM** (main experiments); **H100 (80GB)** re-run for GPU seeds.

| Platform | Seeds | Target conferences |
|---|---|---|
| **GPU (8)** | AI for Biology, Computer Vision, Datasets & Benchmarks, Generative Models, Interpretability, NLP, Privacy in ML, Supervised Representation Learning | ICLR, NeurIPS, ICML, CVPR, ACL, EMNLP |
| **CPU (5)** | Causal Learning, Compiler Optimization, Data Integration & Cleaning, Operating System Design, Probabilistic Methods | OSDI, SOSP, SIGMOD, VLDB, PLDI, POPL |

## Repo structure

```
ResearchArena/
├── papers/                     # 117 agent-generated papers
│   ├── claude/{seed}_trial{N}/
│   │   ├── paper.pdf, paper.tex, references.bib
│   │   ├── idea.json, plan.json, proposal.md
│   │   ├── reviews.json              # 3 peer reviews
│   │   ├── stanford_review.json      # SAR review
│   │   └── exp/                      # experiment code (.py/.sh)
│   ├── codex/…
│   └── kimi/…
├── blog/                       # website (gh-pages branch mirrors this)
│   ├── index.html, index_zh.html, papers.html
│   └── assets/plots/           # all figures used in the blog
├── analysis/                   # baseline assessment data + annotations
│   ├── iclr2025_baseline/      # 300 ICLR papers, SAR assessments
│   ├── stanford_reviews/       # FARS + ICLR SAR fetches
│   └── sar_annotations_*.json  # manual annotations
├── researcharena/              # the benchmark harness
│   ├── stages/                 # ideation / experiment / paper / review
│   ├── templates/              # domain guidelines (ml/systems/databases/pl/theory/security)
│   └── utils/                  # agent_runner, tracker, checkpoint, …
├── paper_viewer.py             # Streamlit app to browse papers + reviews
├── sar_annotator.py            # Streamlit app for manual SAR annotation
├── configs/                    # YAML configs
├── scripts/                    # SLURM / launcher scripts
└── Dockerfile[.cpu]            # agent containers
```

## Setup

```bash
pip install -e .

# Containers (Docker or Podman)
# GPU image: PyTorch 2.6 + CUDA 12.4 + transformers/datasets/accelerate/…
docker build -t researcharena/agent:latest .
# CPU image: Python 3.11 + scipy/sklearn/networkx/sympy/z3-solver/…
docker build -f Dockerfile.cpu -t researcharena/agent-cpu:latest .

# For rootless podman, add --userns=host:
#   podman build --userns=host -t researcharena/agent:latest .
#   podman build --userns=host -f Dockerfile.cpu -t researcharena/agent-cpu:latest .

# CLI agent binaries (claude, codex, kimi) are not installed in the image —
# agent_runner.py mounts them from the host at runtime, so install them
# locally first (e.g. `npm install -g @anthropic-ai/claude-code`).

# API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export MOONSHOT_API_KEY=sk-...
```

## Usage

### Run a single seed

```bash
researcharena run --seed "computer vision" --agent claude --platform gpu
researcharena run --seed "compiler optimization" --agent codex --platform cpu
```

### Run with checkpoint/resume (recommended for SLURM)

```bash
python -m researcharena.run_resumable \
  --config configs/claude_cpu.yaml \
  --seed "compiler optimization" \
  --platform cpu \
  --workspace outputs/claude_t1_compiler_optimization
```

A `checkpoint.json` is saved after every step; interrupted jobs resume from the last completed step.

### Run 8 parallel experiments on SLURM

```bash
sbatch scripts/slurm_claude_gpu.sh
sbatch scripts/slurm_codex_gpu.sh
sbatch scripts/slurm_kimi_gpu.sh
```

### Browse all 117 papers + reviews (Streamlit)

```bash
streamlit run paper_viewer.py
```

Filter by agent, domain, platform, decision; expand any paper to see its 3 peer reviews (with 9-dimension scores), Stanford AI Review, idea description, and experiment code listing. Includes a "My Notes" tab per paper that persists to `papers/human_comments.json`.

### Manually annotate ambiguous SAR reviews

```bash
streamlit run sar_annotator.py
```

Classify each SAR assessment as Accept / Reject / Unclear; progress auto-saves to `analysis/sar_annotations.json`.

## Review pipeline

Each paper is evaluated by 3 independent CLI agents (excluding the researcher agent) with read-only workspace access. Reviewers score 9 dimensions (novelty, soundness, significance, clarity, reproducibility, experimental rigor, references, reference integrity, results integrity), decide accept/revision/reject, and run online verification of citations against arXiv, Semantic Scholar, and CrossRef.

| Researcher | Reviewers |
|---|---|
| Claude Code | Codex, Kimi Code |
| Codex | Claude Code, Kimi Code |
| Kimi Code | Claude Code, Codex |

Scores use a 0–10 scale. Acceptance threshold: 8. Score 5–7 triggers revision. Score < 5 is rejected.

## Configuration

See [`configs/8xa6000.yaml`](configs/8xa6000.yaml) for the full example. Key knobs:

- `platforms.gpu.resources` / `platforms.cpu.resources` — compute budget
- `agent.max_turns`, `ideation_timeout`, `paper_timeout`
- `self_review.threshold` (default 6) and `max_retries_per_gate` (default 2)
- `review.agents` — list of reviewer CLI agents + models
- `pipeline.max_ideas_per_seed` — how many fresh ideas to try before giving up

## Citation

If you use Research Arena or its data, please cite:

```
@misc{researcharena2026,
  title   = {How Far Are We From True Auto Research?},
  author  = {Zhang, Zhengxin and Wang, Ning and Galhotra, Sainyam and Cardie, Claire},
  year    = {2026},
  note    = {Cornell University. \url{https://youarespecialtome.github.io/ResearchArena/}}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
