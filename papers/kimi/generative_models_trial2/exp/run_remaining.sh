#!/bin/bash
set -e

echo "Starting sequential experiment runs at $(date)"

# Wait for current baseline_uniform to finish
echo "Waiting for baseline_uniform seed 42 to complete..."
while ! ls outputs/baseline_uniform_seed42/results.json 1>/dev/null 2>&1; do
    sleep 30
done
echo "baseline_uniform seed 42 complete at $(date)"

# Run remaining experiments sequentially
python exp/baseline_uniform/run.py --seed 123 --epochs 50 --batch_size 32 > logs/experiments/baseline_uniform_s123.log 2>&1 &
PID=$!
wait $PID
echo "baseline_uniform seed 123 complete at $(date)"

python exp/baseline_density/run.py --seed 42 --epochs 50 --batch_size 32 > logs/experiments/baseline_density_s42.log 2>&1 &
PID=$!
wait $PID
echo "baseline_density seed 42 complete at $(date)"

python exp/baseline_density/run.py --seed 123 --epochs 50 --batch_size 32 > logs/experiments/baseline_density_s123.log 2>&1 &
PID=$!
wait $PID
echo "baseline_density seed 123 complete at $(date)"

python exp/distflow_idw/run.py --seed 42 --epochs 50 --batch_size 32 > logs/experiments/distflow_idw_s42.log 2>&1 &
PID=$!
wait $PID
echo "distflow_idw seed 42 complete at $(date)"

python exp/distflow_idw/run.py --seed 123 --epochs 50 --batch_size 32 > logs/experiments/distflow_idw_s123.log 2>&1 &
PID=$!
wait $PID
echo "distflow_idw seed 123 complete at $(date)"

python exp/distflow_idw/run.py --seed 456 --epochs 50 --batch_size 32 > logs/experiments/distflow_idw_s456.log 2>&1 &
PID=$!
wait $PID
echo "distflow_idw seed 456 complete at $(date)"

echo "All experiments complete at $(date)"
