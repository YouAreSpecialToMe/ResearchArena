#!/bin/bash
# Submit a 1-hour smoke test on Unicorn cluster.
# Single database seed ("query optimization"), 2 CPUs, 128 GB RAM.
# Uses Claude CLI with subscription (no API key needed).
#
# Usage:
#   cd /home/nw366/ResearchArena
#   sbatch --requeue submit_smoke_test.sh
#   squeue --me                                    # check status
#   tail -f outputs/slurm_logs/smoke_test_*.out    # watch output

#SBATCH --job-name=ra_smoke_test
#SBATCH --output=outputs/slurm_logs/smoke_test_%j.out
#SBATCH --error=outputs/slurm_logs/smoke_test_%j.err
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

cd /home/nw366/ResearchArena
mkdir -p outputs/smoke_test_unicorn outputs/slurm_logs

# Ensure PATH includes user-local binaries (claude, researcharena, pip)
export PATH="$HOME/.local/bin:$PATH"

echo "=== ResearchArena Smoke Test ==="
echo "Date:  $(date)"
echo "Node:  $(hostname)"
echo "CPUs:  $SLURM_CPUS_PER_TASK"
echo "Mem:   128 GB"
echo "Seed:  query optimization (databases)"
echo ""

# Run the smoke test
researcharena run -c configs/smoke_test_unicorn.yaml 2>&1

echo ""
echo "=== Smoke test finished ==="
echo "Date: $(date)"
