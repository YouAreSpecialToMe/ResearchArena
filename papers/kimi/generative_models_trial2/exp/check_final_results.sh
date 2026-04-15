#!/bin/bash
# Check if experiments have completed and generate final results

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/generative_models/idea_01

echo "Checking experiment completion status..."
echo ""

# Count completed experiments
completed=$(ls outputs/*/results.json 2>/dev/null | wc -l)
echo "Completed experiments: $completed"
echo ""

# List completed experiments
echo "Completed experiment results:"
ls -la outputs/*/results.json 2>/dev/null

echo ""
echo "Runner log status:"
tail -5 logs/sequential_runner.log 2>/dev/null

# If experiments are complete, run analysis
if [ "$completed" -ge 7 ]; then
    echo ""
    echo "All experiments complete! Running analysis..."
    python analyze_results.py
fi
