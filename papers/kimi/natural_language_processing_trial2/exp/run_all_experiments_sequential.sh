#!/bin/bash
# Run all CDHR experiments sequentially

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/natural_language_processing/idea_01
source .venv/bin/activate

MODEL="llama-3.1-8b"
LIMIT=100

echo "=========================================="
echo "Starting CDHR Experiments"
echo "=========================================="

# 1. Baseline CoT on GSM8K
echo "1. Running Baseline CoT on GSM8K..."
python exp/run_efficient_experiments.py \
    --model $MODEL \
    --method cot \
    --dataset data/gsm8k.json \
    --limit $LIMIT \
    --output results/baseline_cot/llama_gsm8k.json

# 2. CDHR on GSM8K
echo "2. Running CDHR on GSM8K..."
python exp/run_efficient_experiments.py \
    --model $MODEL \
    --method cdhr \
    --dataset data/gsm8k.json \
    --limit $LIMIT \
    --output results/cdhr_main/llama_gsm8k.json

# 3. Baseline CoT on MATH
echo "3. Running Baseline CoT on MATH..."
python exp/run_efficient_experiments.py \
    --model $MODEL \
    --method cot \
    --dataset data/math.json \
    --limit $LIMIT \
    --output results/baseline_cot/llama_math.json

# 4. CDHR on MATH
echo "4. Running CDHR on MATH..."
python exp/run_efficient_experiments.py \
    --model $MODEL \
    --method cdhr \
    --dataset data/math.json \
    --limit $LIMIT \
    --output results/cdhr_main/llama_math.json

# 5. CDHR with different theta_v (ablation)
echo "5. Running CDHR ablation: theta_v=0.03..."
python exp/run_efficient_experiments.py \
    --model $MODEL \
    --method cdhr \
    --dataset data/gsm8k.json \
    --limit $LIMIT \
    --theta_v 0.03 \
    --output results/ablation_thresholds/llama_gsm8k_thetav003.json

# 6. CDHR with different theta_v (ablation)
echo "6. Running CDHR ablation: theta_v=0.07..."
python exp/run_efficient_experiments.py \
    --model $MODEL \
    --method cdhr \
    --dataset data/gsm8k.json \
    --limit $LIMIT \
    --theta_v 0.07 \
    --output results/ablation_thresholds/llama_gsm8k_thetav007.json

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

# Generate summary
python -c "
import json
import os

print('\n' + '='*60)
print('EXPERIMENT SUMMARY')
print('='*60)

for result_file in [
    'results/baseline_cot/llama_gsm8k.json',
    'results/cdhr_main/llama_gsm8k.json',
    'results/baseline_cot/llama_math.json',
    'results/cdhr_main/llama_math.json',
]:
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
        m = data['metrics']
        print(f\"\n{data['experiment']}:\")
        print(f\"  Accuracy: {m['accuracy']:.4f}\")
        print(f\"  Avg Tokens: {m['avg_tokens']:.1f}\")
        print(f\"  Avg Latency: {m['avg_latency']:.2f}s\")
        if 'strategy_entropy' in m:
            print(f\"  Strategy Entropy: {m['strategy_entropy']:.3f}\")
    else:
        print(f\"\n{result_file}: NOT FOUND\")
"
