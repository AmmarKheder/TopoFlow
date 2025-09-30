#!/bin/bash
# Master script to run complete ablation study on LUMI
# Launches all configurations in parallel

echo "=========================================="
echo "TopoFlow Ablation Study"
echo "=========================================="
echo "Launching 4 experiments in parallel:"
echo "  1. Baseline (no innovations)"
echo "  2. + Innovation #1 (Pollutant Cross-Attention)"
echo "  3. + Innovation #1+#2 (+ Hierarchical Physics)"
echo "  4. Full Model (All 3 innovations)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# Submit all jobs
JOB1=$(sbatch scripts/slurm_baseline.sh | awk '{print $4}')
echo "# # #  Baseline submitted: Job ID $JOB1"

JOB2=$(sbatch scripts/slurm_innovation1.sh | awk '{print $4}')
echo "# # #  Innovation #1 submitted: Job ID $JOB2"

JOB3=$(sbatch scripts/slurm_innovation2.sh | awk '{print $4}')
echo "# # #  Innovation #1+#2 submitted: Job ID $JOB3"

JOB4=$(sbatch scripts/slurm_full_model.sh | awk '{print $4}')
echo "# # #  Full Model submitted: Job ID $JOB4"

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Monitor with: squeue -u $USER"
echo "=========================================="
echo ""
echo "Expected total time: ~2 hours"
echo "Results will be in logs/ directory"
echo ""
echo "After completion, compare results with:"
echo "  python scripts/compare_ablation_results.py"
echo ""