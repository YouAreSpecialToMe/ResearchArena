# Research Arena

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Website](https://img.shields.io/badge/Website-ResearchArena-5eead4.svg)](https://youarespecialtome.github.io/ResearchArena/)
[![GitHub stars](https://img.shields.io/github/stars/YouAreSpecialToMe/ResearchArena?style=social)](https://github.com/YouAreSpecialToMe/ResearchArena/stargazers)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-supported-2496ed.svg)](https://www.docker.com/)

**How far are we from true auto-research?** A comprehensive benchmark of off-the-shelf CLI agents (Claude Code, Codex, Kimi Code) conducting end-to-end research across 13 CS domains.

**Blog & results:** [youarespecialtome.github.io](https://youarespecialtome.github.io/ResearchArena/)

- **117 agent-generated papers** (39 per agent, 3 trials × 13 seeds) across both GPU and CPU domains
- **351 peer reviews** (code-aware, 3 reviewers per paper) + **117 Stanford Agentic Reviewer scores**
- **302 human-authored ICLR 2025 papers** (100 accepted, 100 rejected, 102 FARS) evaluated as baseline
- **Manual annotation of 117 papers** for results mismatches, implementation mismatches, and fabricated references

## Key findings

| SAR (manually annotated) | Accept | Reject | Accept % |
|---|---|---|---|
| ICLR Accepted | 76 | 24 | 76.0% |
| ICLR Weighted (32%/68%) | 59.7 | 40.3 | 59.7% |
| ICLR Rejected | 52 | 48 | 52.0% |
| **Claude Code** | 16 | 23 | 41.0% |
| FARS (Analemma) | 22 | 80 | 21.6% |
| **Codex** | 5 | 34 | 12.8% |
| **Kimi Code** | 2 | 37 | 5.1% |

| Integrity issues | Claude | Codex | Kimi |
|---|---|---|---|
| Results mismatch | 6 (15%) | 2 (5%) | 4 (10%) |
| Setting mismatch | 10 (26%) | 1 (3%) | 5 (13%) |
| Both | 12 (31%) | 2 (5%) | 30 (77%) |
| Fake reference | 14 (36%) | 3 (8%) | 28 (72%) |

**Takeaways:**
- Current CLI agents trail human-authored papers substantially on SAR acceptance
- **Claude Code** is the strongest agent overall (41% SAR accept); **Codex** is the most *trustworthy* (lowest integrity issues)
- **Kimi Code** shows the most paper–artifact divergence — 77% of papers have both results and setting mismatches, 72% have fake references
- Computational resources are *not* the main bottleneck; upgrading A6000 → H100 shows no consistent pattern
- SAR misses code-level failures that only a code-aware peer review can catch

See [Case Studies: Paper–Artifact Divergence](https://youarespecialtome.github.io/ResearchArena/#case-studies) for five illustrative failure modes with embedded PDFs.

## What this does

Given a seed field (e.g., "computer vision", "compiler optimization"), the pipeline:

1. **Launches a CLI agent** with appropriate compute access (GPU or CPU), network, and packages
2. **The agent produces a structured research proposal** — `proposal.md`, `plan.json`, and `idea.json`
3. **Self-review gates** check quality at each stage before proceeding
4. **The agent follows its experiment plan** step-by-step, producing `results.json`
5. **The agent writes a LaTeX paper** from the proposal, plan, and results
6. **Other CLI agents review it** — with read-only workspace access, searching online to verify novelty
7. If rejected, the pipeline iterates — refine the idea, retry experiments, or try a new idea entirely

## Conferences & Areas

Seeds span multiple CS conferences and two platforms:

| Platform | Areas | Conferences |
|---|---|---|
| **GPU** | NLP, CV, generative models, interpretability, supervised representation learning, privacy, AI for biology, datasets & benchmarks | ICLR, NeurIPS, ICML, CVPR, ACL, EMNLP |
| **CPU** | Causal learning, compiler optimization, data integration, operating systems, probabilistic methods | OSDI, SOSP, SIGMOD, VLDB, PLDI, POPL |

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

# Containers (Docker mode)
docker pull pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
docker tag pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel researcharena/agent:latest
docker build -f Dockerfile.cpu -t researcharena/agent-cpu:latest .

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

Each paper is evaluated by 3 independent CLI agents (excluding the researcher agent) with read-only workspace access. Reviewers score 9 dimensions (novelty, soundness, significance, clarity, reproducibility, experimental rigor, references, reference integrity, results integrity), decide accept/revision/reject, and run online verification of citations.

| Researcher | Reviewers |
|---|---|
| Claude Code | Codex, Kimi Code |
| Codex | Claude Code, Kimi Code |
| Kimi Code | Claude Code, Codex |

Scores use a 0-10 scale. Acceptance threshold: 8. Score 5-7 triggers revision. Score < 5 is rejected.

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

Released under a permissive license — see [`LICENSE`](LICENSE) once added. Contact the authors for questions.
