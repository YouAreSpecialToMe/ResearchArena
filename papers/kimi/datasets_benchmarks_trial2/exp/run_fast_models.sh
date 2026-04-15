#!/bin/bash
# Run smaller models first to establish real evaluation pipeline

MODELS=(
    "Qwen2.5-0.5B"
    "Qwen2.5-1.8B"
    "Gemma-2-2B"
)

OUTPUT_DIR="data/model_responses"
LOG_DIR="logs/inference"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Starting FAST model evaluations at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo ""

for model in "${MODELS[@]}"; do
    LOG_FILE="$LOG_DIR/${model}.log"
    
    echo "=========================================="
    echo "Evaluating: $model"
    echo "Start time: $(date)"
    echo "Log file: $LOG_FILE"
    echo "=========================================="
    
    source .venv/bin/activate && python exp/eval_single_model.py "$model" \
        --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    echo "Exit code: $EXIT_CODE"
    echo "End time: $(date)"
    echo ""
    
    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done

echo "Fast model evaluations complete at $(date)"
