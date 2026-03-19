#!/usr/bin/env bash
# launch_codex_gpu.sh — Run 8 Codex researchers in parallel on rush-compute-03.
#
# Usage:
#   bash scripts/launch_codex_gpu.sh
#   bash scripts/launch_codex_gpu.sh --config configs/8xa6000_codex.yaml --seeds-file configs/seed_gpu_exp.yaml
#
# Each researcher gets:
#   - One A6000 GPU (CUDA_VISIBLE_DEVICES=N)
#   - Its own workspace under outputs/codex/<seed>_<timestamp>/
#   - A log file under outputs/codex/logs/

set -euo pipefail

cd /home/zz865/pythonProject/autoresearch

CONFIG="${CONFIG:-configs/8xa6000_codex.yaml}"
SEEDS_FILE="${SEEDS_FILE:-configs/seed_gpu_exp.yaml}"
NUM_GPUS=8
AGENT="codex"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/${AGENT}/logs"
mkdir -p "$LOG_DIR"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG="$2"; shift 2 ;;
    --seeds-file) SEEDS_FILE="$2"; shift 2 ;;
    --num-gpus)   NUM_GPUS="$2"; shift 2 ;;
    *)            echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Read seeds from file (one per line) ──
SEEDS=()
while IFS= read -r line; do
  line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  [[ -z "$line" || "$line" == \#* ]] && continue
  SEEDS+=("$line")
done < "$SEEDS_FILE"

TOTAL=${#SEEDS[@]}
echo "============================================================"
echo "  ResearchArena — Codex GPU Parallel Launch"
echo "  Host:       $(hostname)"
echo "  Config:     $CONFIG"
echo "  Seeds file: $SEEDS_FILE"
echo "  Seeds:      $TOTAL"
echo "  GPUs:       $NUM_GPUS × A6000"
echo "  Researcher: Codex (gpt-5.4)"
echo "  Timestamp:  $TIMESTAMP"
if [[ $TOTAL -gt $NUM_GPUS ]]; then
  BATCHES=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
  echo "  Batches:    $BATCHES (${NUM_GPUS} GPUs, overflow queued)"
fi
echo "============================================================"
echo ""

TOTAL_FAILURES=0
BATCH=0

for (( start=0; start<TOTAL; start+=NUM_GPUS )); do
  BATCH=$((BATCH + 1))
  end=$((start + NUM_GPUS))
  [[ $end -gt $TOTAL ]] && end=$TOTAL
  batch_size=$((end - start))

  echo "── Batch $BATCH: seeds $((start+1))–$end of $TOTAL ($batch_size researchers) ──"

  PIDS=()
  BATCH_SEEDS=()
  for (( i=start; i<end; i++ )); do
    SEED="${SEEDS[$i]}"
    GPU_ID=$((i - start))
    SLUG=$(echo "$SEED" | tr ' ' '_' | tr '/' '_' | tr '[:upper:]' '[:lower:]')
    LOG_FILE="$LOG_DIR/${SLUG}.log"

    echo "  [GPU $GPU_ID] $SEED"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
      researcharena run \
        --config "$CONFIG" \
        --seed "$SEED" \
        --workspace "outputs/${AGENT}/${SLUG}_${TIMESTAMP}" \
        --platform gpu \
      > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    BATCH_SEEDS+=("$SEED")
  done

  echo ""
  echo "  PIDs: ${PIDS[*]}"
  echo "  Waiting for batch $BATCH to finish..."
  echo ""

  BATCH_FAILURES=0
  for j in "${!PIDS[@]}"; do
    PID=${PIDS[$j]}
    SEED="${BATCH_SEEDS[$j]}"
    GPU_ID=$j
    if wait "$PID"; then
      echo "  [GPU $GPU_ID] ✓ Done: $SEED"
    else
      echo "  [GPU $GPU_ID] ✗ FAILED: $SEED (exit code $?)"
      BATCH_FAILURES=$((BATCH_FAILURES + 1))
    fi
  done

  TOTAL_FAILURES=$((TOTAL_FAILURES + BATCH_FAILURES))
  echo ""
done

SUCCEEDED=$((TOTAL - TOTAL_FAILURES))
echo "============================================================"
echo "  Finished: $SUCCEEDED/$TOTAL succeeded"
echo "============================================================"

exit $TOTAL_FAILURES
