#!/bin/bash
# Free /tmp space and run remaining experiments
export TMPDIR=/var/tmp

# Try to free some space in /tmp by removing old files
find /tmp -user $(whoami) -name "*.output" -mmin +10 -exec truncate -s 0 {} \; 2>/dev/null
find /tmp -user $(whoami) -name "*.pyc" -delete 2>/dev/null
find /tmp -user $(whoami) -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null

cd /home/zz865/pythonProject/autoresearch/outputs/claude/run_3/computer_vision/idea_01
conda run -n ar --no-banner python exp/run_remaining.py > logs/run_remaining.log 2>&1
echo "EXIT CODE: $?" >> logs/run_remaining.log
