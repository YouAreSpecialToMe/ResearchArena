#!/usr/bin/env bash
#SBATCH -J claude-gpu-research
#SBATCH -o outputs/claude/logs/slurm_%A_%a.out
#SBATCH -e outputs/claude/logs/slurm_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH -t 168:00:00
#SBATCH --partition=rush
#SBATCH --nodelist=rush-compute-02
#SBATCH --array=0-7

set -euo pipefail

cd /home/zz865/pythonProject/autoresearch
mkdir -p outputs/claude/logs

CONFIG="configs/8xa6000.yaml"
SEEDS_FILE="configs/seed_gpu_exp.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Read seeds into array
SEEDS=()
while IFS= read -r line; do
  line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  [[ -z "$line" || "$line" == \#* ]] && continue
  SEEDS+=("$line")
done < "$SEEDS_FILE"

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
SLUG=$(echo "$SEED" | tr ' ' '_' | tr '/' '_' | tr '[:upper:]' '[:lower:]')

# Redirect logs to claude_<research_name>.out/.err
LOG_OUT="outputs/claude/logs/claude_${SLUG}.out"
LOG_ERR="outputs/claude/logs/claude_${SLUG}.err"
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

WORKSPACE="outputs/claude/${SLUG}_${TIMESTAMP}"
mkdir -p "$WORKSPACE"

echo "============================================================"
echo "  ResearchArena — Claude GPU Job"
echo "  Config:    $CONFIG"
echo "  Seed:      $SEED"
echo "  Workspace: $WORKSPACE"
echo "  Task ID:   $SLURM_ARRAY_TASK_ID"
echo "  Node:      $(hostname)"
echo "  GPU:       1× NVIDIA RTX A6000 (48GB)"
echo "  CPUs:      4"
echo "  Memory:    60 GB"
echo "  Time:      168 hours"
echo "============================================================"

researcharena run \
  --config "$CONFIG" \
  --seed "$SEED" \
  --workspace "$WORKSPACE" \
  --platform gpu
