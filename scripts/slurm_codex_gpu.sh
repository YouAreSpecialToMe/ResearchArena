#!/usr/bin/env bash
#SBATCH -J codex-gpu-research
#SBATCH -o outputs/codex/logs/slurm_%A_%a.out
#SBATCH -e outputs/codex/logs/slurm_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH -t 48:00:00
#SBATCH --partition=rush
#SBATCH --nodelist=rush-compute-02
#SBATCH --array=0-7

set -euo pipefail

cd /home/zz865/pythonProject/autoresearch
mkdir -p outputs/codex/logs

CONFIG="configs/8xa6000_codex.yaml"
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
WORKSPACE="outputs/codex/${SLUG}_${TIMESTAMP}"

LOG_OUT="outputs/codex/logs/codex_${SLUG}.out"
LOG_ERR="outputs/codex/logs/codex_${SLUG}.err"
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

echo "============================================================"
echo "  ResearchArena — Codex GPU Job"
echo "  Config:    $CONFIG"
echo "  Seed:      $SEED"
echo "  Workspace: $WORKSPACE"
echo "  Task ID:   $SLURM_ARRAY_TASK_ID"
echo "  Node:      $(hostname)"
echo "  GPU:       1× NVIDIA RTX A6000 (48GB)"
echo "  CPUs:      4"
echo "  Memory:    60 GB"
echo "============================================================"

researcharena run \
  --config "$CONFIG" \
  --seed "$SEED" \
  --workspace "$WORKSPACE" \
  --platform gpu
