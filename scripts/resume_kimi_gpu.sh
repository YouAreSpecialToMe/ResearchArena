#!/usr/bin/env bash
# resume_kimi_gpu.sh — Resume killed Kimi GPU experiments, 4 at a time.
#
# Detects which runs need resuming and runs them in batches of 4
# to avoid OOM from running 8 simultaneously.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/8xh100_kimi.yaml}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# Find all idea_01 dirs from the latest batch that need work
WORKSPACES=()
for d in outputs/kimi/*_20260324_*/idea_01; do
    [[ -d "$d" ]] || continue
    # Skip if already has reviews (fully done)
    [[ -f "$d/reviews.json" ]] && continue
    WORKSPACES+=("$d")
done

TOTAL=${#WORKSPACES[@]}
echo "============================================================"
echo "  ResearchArena — Resume Kimi GPU Experiments"
echo "  Config:     $CONFIG"
echo "  To resume:  $TOTAL workspaces"
echo "  Batch size: $BATCH_SIZE"
echo "  Host:       $(hostname)"
echo "============================================================"
echo ""

for (( i=0; i<TOTAL; i++ )); do
    ws="${WORKSPACES[$i]}"
    name=$(basename "$(dirname "$ws")")
    has_results=$([[ -f "$ws/results.json" ]] && echo "results" || echo "-")
    has_paper=$([[ -f "$ws/paper.tex" ]] && echo "paper" || echo "-")
    echo "  [$((i+1))] $name: $has_results $has_paper → $ws"
done
echo ""

TOTAL_FAILURES=0
BATCH=0

for (( start=0; start<TOTAL; start+=BATCH_SIZE )); do
    BATCH=$((BATCH + 1))
    end=$((start + BATCH_SIZE))
    [[ $end -gt $TOTAL ]] && end=$TOTAL
    batch_size=$((end - start))

    echo "── Batch $BATCH: $batch_size researchers ──"

    PIDS=()
    BATCH_NAMES=()
    for (( i=start; i<end; i++ )); do
        ws="${WORKSPACES[$i]}"
        name=$(basename "$(dirname "$ws")")
        GPU_ID=$((i - start))
        LOG_FILE="outputs/kimi/logs/resume_${name}.log"

        echo "  [GPU $GPU_ID] Resuming: $name"

        CUDA_VISIBLE_DEVICES=$GPU_ID \
            researcharena run \
                --config "$CONFIG" \
                --resume "$ws" \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
        BATCH_NAMES+=("$name")
    done

    echo ""
    echo "  PIDs: ${PIDS[*]}"
    echo "  Waiting for batch $BATCH..."
    echo ""

    BATCH_FAILURES=0
    for j in "${!PIDS[@]}"; do
        PID=${PIDS[$j]}
        NAME="${BATCH_NAMES[$j]}"
        if wait "$PID"; then
            echo "  [GPU $j] ✓ Done: $NAME"
        else
            echo "  [GPU $j] ✗ FAILED: $NAME (exit code $?)"
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
