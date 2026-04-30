# ResearchArena Paper

This directory contains the NeurIPS-style paper source for:

`ResearchArena: Process-Grounded Evaluation of End-to-End Research by CLI Agents`

## Files

- `main.tex`: paper entry point.
- `sections/`: main text and appendix sections.
- `references.bib`: bibliography.
- `figures/`: copied PNG figures from `assets/plots/`.
- `tables/`: reserved for standalone generated tables if needed later.

## Build

The official `neurips_2026.sty` from the NeurIPS 2026 paper template is included in this directory.

```bash
cd paper
latexmk -pdf main.tex
```

If `latexmk` is unavailable, run:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
