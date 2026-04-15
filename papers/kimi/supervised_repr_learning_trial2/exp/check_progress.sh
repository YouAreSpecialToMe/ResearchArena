#!/bin/bash
# Check experiment progress

echo "=== Experiment Progress Check ==="
echo "Time: $(date)"
echo ""

echo "Running processes:"
ps aux | grep "python exp" | grep -v grep | wc -l
echo ""

echo "Result files:"
ls -la results/*.json 2>/dev/null | wc -l
echo ""

echo "Latest results:"
for f in results/*.json; do
    if [ -f "$f" ]; then
        echo "File: $f"
        cat "$f" | head -10
        echo "---"
    fi
done

echo ""
echo "Log progress:"
tail -5 logs/master_run.log 2>/dev/null
