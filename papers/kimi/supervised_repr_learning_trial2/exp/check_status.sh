#!/bin/bash
# Check status of running experiments

echo "===== Experiment Status ====="
echo "Timestamp: $(date)"
echo ""

echo "--- Running Processes ---"
ps aux | grep "train.py" | grep -v grep | wc -l | xargs -I {} echo "Active training processes: {}"
ps aux | grep "train.py" | grep -v grep || echo "No active training processes"
echo ""

echo "--- Completed Results ---"
ls -la results/*.json 2>/dev/null | wc -l | xargs -I {} echo "Results completed: {} / 9"
ls -la results/*.json 2>/dev/null || echo "No results yet"
echo ""

echo "--- Latest Log Entries ---"
tail -20 logs/experiments.log 2>/dev/null || echo "No log file"
echo ""

echo "--- Estimated Progress ---"
COMPLETED=$(ls results/*.json 2>/dev/null | wc -l)
if [ "$COMPLETED" -eq 0 ]; then
    echo "Just started - first experiment running"
elif [ "$COMPLETED" -eq 9 ]; then
    echo "ALL EXPERIMENTS COMPLETE!"
    echo "Run: python analyze_results.py"
else
    REMAINING=$((9 - COMPLETED))
    MINUTES_REMAINING=$((REMAINING * 12))
    echo "Completed: $COMPLETED / 9"
    echo "Remaining: $REMAINING experiments (~${MINUTES_REMAINING} minutes)"
fi
