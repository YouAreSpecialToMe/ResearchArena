#!/bin/bash
set -e

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/generative_models/idea_01
source .venv/bin/activate

EPOCHS=30
BATCH=16

echo "=== Running all experiments ==="
echo "Epochs: $EPOCHS, Batch: $BATCH"

# Baseline 1: Standard Flow Matching
echo ""
echo "=== Baseline Uniform Seed 42 ==="
python exp/baseline_uniform/run.py --seed 42 --epochs $EPOCHS --batch_size $BATCH

echo ""
echo "=== Baseline Uniform Seed 123 ==="
python exp/baseline_uniform/run.py --seed 123 --epochs $EPOCHS --batch_size $BATCH

# Baseline 2: Density-Weighted
echo ""
echo "=== Baseline Density Seed 42 ==="
python exp/baseline_density/run.py --seed 42 --epochs $EPOCHS --batch_size $BATCH

echo ""
echo "=== Baseline Density Seed 123 ==="
python exp/baseline_density/run.py --seed 123 --epochs $EPOCHS --batch_size $BATCH

# Main: DistFlow-IDW
echo ""
echo "=== DistFlow-IDW Seed 42 ==="
python exp/distflow_idw/run.py --seed 42 --epochs $EPOCHS --batch_size $BATCH

echo ""
echo "=== DistFlow-IDW Seed 123 ==="
python exp/distflow_idw/run.py --seed 123 --epochs $EPOCHS --batch_size $BATCH

echo ""
echo "=== DistFlow-IDW Seed 456 ==="
python exp/distflow_idw/run.py --seed 456 --epochs $EPOCHS --batch_size $BATCH

# DistFlow-LAW
echo ""
echo "=== DistFlow-LAW Seed 42 ==="
python exp/distflow_law/run.py --seed 42 --epochs $EPOCHS --batch_size $BATCH

echo ""
echo "=== DistFlow-LAW Seed 123 ==="
python exp/distflow_law/run.py --seed 123 --epochs $EPOCHS --batch_size $BATCH

# Ablations
echo ""
echo "=== Ablation No-FiLM Seed 42 ==="
python exp/ablation_no_film/run.py --seed 42 --epochs $EPOCHS --batch_size $BATCH

echo ""
echo "=== Ablation No-Stratify Seed 42 ==="
python exp/ablation_no_stratify/run.py --seed 42 --epochs $EPOCHS --batch_size $BATCH

echo ""
echo "=== All experiments complete ==="
