#!/usr/bin/env bash
#SBATCH -J codex-cpu-t1
#SBATCH -o outputs/codex_cpu_t1/logs/slurm_%A_%a.out
#SBATCH -e outputs/codex_cpu_t1/logs/slurm_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128g
#SBATCH -t 48:00:00
#SBATCH --partition=default_partition
#SBATCH --array=0-4

set -euo pipefail

cd /home/nw366/ResearchArena
mkdir -p outputs/codex_cpu_t1/logs

CONFIG="configs/codex_cpu.yaml"
SEEDS_FILE="configs/seed_cpu_exp.yaml"

# Read seeds into array
SEEDS=()
while IFS= read -r line; do
  line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  [[ -z "$line" || "$line" == \#* ]] && continue
  SEEDS+=("$line")
done < "$SEEDS_FILE"

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
SLUG=$(echo "$SEED" | tr ' ' '_' | tr '/' '_' | tr '[:upper:]' '[:lower:]')

# Redirect logs
LOG_OUT="outputs/codex_cpu_t1/logs/codex_t1_${SLUG}.out"
LOG_ERR="outputs/codex_cpu_t1/logs/codex_t1_${SLUG}.err"
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

echo "============================================================"
echo "  ResearchArena — Codex CPU Job (t1)"
echo "  Config:    $CONFIG"
echo "  Seed:      $SEED"
echo "  Task ID:   $SLURM_ARRAY_TASK_ID"
echo "  Node:      $(hostname)"
echo "  CPUs:      2"
echo "  Memory:    128 GB"
echo "  Time:      48 hours"
echo "============================================================"

WORKSPACE="outputs/codex_t1_${SLUG}"
mkdir -p "$WORKSPACE"

researcharena run \
  --config "$CONFIG" \
  --seed "$SEED" \
  --platform cpu \
  --workspace "$WORKSPACE"
