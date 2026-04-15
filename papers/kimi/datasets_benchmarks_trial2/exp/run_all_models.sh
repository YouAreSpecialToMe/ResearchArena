#!/bin/bash
# Run all model evaluations sequentially on single GPU

MODELS=(
    "Qwen2.5-0.5B"
    "Qwen2.5-1.8B"
    "Gemma-2-2B"
    "Qwen2.5-7B"
    "Mistral-7B"
    "Llama-3.1-8B"
    "Gemma-2-9B"
    "Qwen2.5-14B"
    "Phi-4"
    "Gemma-2-27B"
    "Qwen2.5-32B"
    "Llama-3.1-70B"
)

OUTPUT_DIR="data/model_responses"
LOG_DIR="logs/inference"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Starting model evaluations at $(date)"
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
    
    # Small delay between models
    sleep 5
done

echo "All evaluations complete at $(date)"
