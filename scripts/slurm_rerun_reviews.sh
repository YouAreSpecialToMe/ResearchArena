#!/usr/bin/env bash
#SBATCH -J rerun-reviews
#SBATCH -o outputs/rerun_reviews/slurm_%j.out
#SBATCH -e outputs/rerun_reviews/slurm_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64g
#SBATCH -t 12:00:00
#SBATCH --partition=default_partition

set -euo pipefail

cd /home/nw366/ResearchArena
mkdir -p outputs/rerun_reviews

echo "============================================================"
echo "  Re-running 6 missing reviewer scores"
echo "  Node:      $(hostname)"
echo "  Time:      $(date)"
echo "============================================================"

python3 scripts/rerun_missing_reviews.py

echo "============================================================"
echo "  Done at $(date)"
echo "============================================================"
