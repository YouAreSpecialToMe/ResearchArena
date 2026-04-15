#!/bin/bash
cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/natural_language_processing/idea_01
source .venv/bin/activate
echo 'Finalizing results at '$(date) > logs/finalize.log
python exp/finalize_results.py >> logs/finalize.log 2>&1
echo 'Done at '$(date) >> logs/finalize.log
