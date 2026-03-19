#!/usr/bin/env bash
# launch_parallel.sh — Run researchers in parallel, one per GPU.
#
# Usage:
#   bash scripts/launch_parallel.sh
#   bash scripts/launch_parallel.sh --config configs/8xa6000.yaml --seeds-file configs/seed_gpu_exp.yaml
#
# With 8 GPUs and 10 seeds, runs 2 batches: first 8 in parallel, then remaining 2.
# Each researcher gets:
#   - One A6000 GPU (CUDA_VISIBLE_DEVICES=N)
#   - Its own workspace under outputs/8xa6000/<slug>/
#   - A log file under outputs/8xa6000/logs/

set -euo pipefail

cd /home/zz865/pythonProject/autoresearch

CONFIG="${CONFIG:-configs/8xa6000.yaml}"
SEEDS_FILE="${SEEDS_FILE:-configs/seed_gpu_exp.yaml}"
NUM_GPUS=8
AGENT="claude"
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
echo "  ResearchArena — Parallel Launch"
echo "  Config:     $CONFIG"
echo "  Seeds file: $SEEDS_FILE"
echo "  Seeds:      $TOTAL"
echo "  GPUs:       $NUM_GPUS × A6000"
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
