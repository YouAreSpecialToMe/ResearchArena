#!/bin/bash
# Run ESR experiments with properly tuned thresholds
# Based on uncertainty analysis: tau_h=0.5, tau_v=0.01

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/natural_language_processing/idea_01
source .venv/bin/activate

# Create results directory if needed
mkdir -p exp/results

echo "==================================================="
echo "Running ESR experiments with TUNED THRESHOLDS"
echo "tau_h = 0.5 (was 2.5)"
echo "tau_v = 0.01 (was 1.5)"
echo "==================================================="

# Run with seed 42
echo ""
echo ">>> Running with seed 42..."
python exp/run_experiments_fast.py \
    --dataset gsm8k \
    --limit 150 \
    --single_seed 42 \
    --tau_h 0.5 \
    --tau_v 0.01 \
    --model Qwen/Qwen3-1.7B \
    --output_dir exp/results

# Run with seed 123
echo ""
echo ">>> Running with seed 123..."
python exp/run_experiments_fast.py \
    --dataset gsm8k \
    --limit 150 \
    --single_seed 123 \
    --tau_h 0.5 \
    --tau_v 0.01 \
    --model Qwen/Qwen3-1.7B \
    --output_dir exp/results

echo ""
echo "==================================================="
echo "Experiments complete! Checking results..."
echo "==================================================="

# Verify revision rates
for seed in 42 123; do
    file="exp/results/all_methods_seed${seed}_tuned.json"
    if [ -f "$file" ]; then
        echo ""
        echo "Seed $seed results:"
        python3 -c "
import json
with open('$file') as f:
    data = json.load(f)
    results = data.get('results', {})
    if 'esr' in results:
        esr = results['esr']
        print(f\"  ESR Accuracy: {esr['accuracy_mean']:.3f}\")
        print(f\"  ESR Revision Rate: {esr.get('revision_rate_mean', 0):.2%}\")
        if esr.get('revision_rate_mean', 0) > 0:
            print(\"  ✓ ESR is triggering revisions!\")
        else:
            print(\"  ✗ ESR is NOT triggering revisions (check thresholds)\")
" 2>/dev/null || echo "  (Could not parse results)"
    fi
done

echo ""
echo "==================================================="
echo "Tuned experiment results saved to:"
echo "  - exp/results/all_methods_seed42_tuned.json"
echo "  - exp/results/all_methods_seed123_tuned.json"
echo "==================================================="
