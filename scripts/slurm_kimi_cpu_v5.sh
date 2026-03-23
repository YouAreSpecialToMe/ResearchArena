#!/usr/bin/env bash
#SBATCH -J kimi-cpu-v5
#SBATCH -o outputs/kimi_cpu_v5/logs/slurm_%A_%a.out
#SBATCH -e outputs/kimi_cpu_v5/logs/slurm_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128g
#SBATCH -t 48:00:00
#SBATCH --partition=default_partition
#SBATCH --array=0-4

set -euo pipefail

cd /home/nw366/ResearchArena
mkdir -p outputs/kimi_cpu_v5/logs

CONFIG="configs/kimi_cpu.yaml"
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
LOG_OUT="outputs/kimi_cpu_v5/logs/kimi_v5_${SLUG}.out"
LOG_ERR="outputs/kimi_cpu_v5/logs/kimi_v5_${SLUG}.err"
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

echo "============================================================"
echo "  ResearchArena — Kimi CPU Job (v5)"
echo "  Config:    $CONFIG"
echo "  Seed:      $SEED"
echo "  Task ID:   $SLURM_ARRAY_TASK_ID"
echo "  Node:      $(hostname)"
echo "  CPUs:      2"
echo "  Memory:    128 GB"
echo "  Time:      48 hours"
echo "============================================================"

WORKSPACE="outputs/kimi_v5_${SLUG}"
mkdir -p "$WORKSPACE"

researcharena run \
  --config "$CONFIG" \
  --seed "$SEED" \
  --platform cpu \
  --workspace "$WORKSPACE"
