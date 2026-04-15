#!/bin/bash
# Wrapper to run generate_figures.py with custom TMPDIR to avoid /tmp ENOSPC
export TMPDIR=/home/zz865/pythonProject/autoresearch/outputs/claude/run_3/privacy_in_machine_learning/idea_01/.tmp
mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd /home/zz865/pythonProject/autoresearch/outputs/claude/run_3/privacy_in_machine_learning/idea_01
python exp/generate_figures.py 2>&1 | tee exp/generate_figures.log
echo "EXIT_CODE=$?"
