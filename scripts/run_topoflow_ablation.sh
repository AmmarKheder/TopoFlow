#!/bin/bash
################################################################################
# TopoFlow Ablation Study - From Scratch
#
# Launches 4 experiments in parallel:
# 1. Wind Scanning Baseline (no innovations)
# 2. Wind + Innovation #1 (Pollutant Cross-Attention)
# 3. Wind + Innovation #1+#2 (+ Hierarchical Physics)
# 4. Wind + ALL 3 Innovations (Full TopoFlow)
#
# Each job: 50 nodes (400 GPUs), 6 epochs, 18h, from scratch
################################################################################

echo "=========================================="
echo "TopoFlow Ablation Study - FROM SCRATCH"
echo "=========================================="
echo "Launching 4 experiments in parallel:"
echo "  1. Wind Scanning Baseline"
echo "  2. + Innovation #1 (Pollutant Cross-Attention)"
echo "  3. + Innovation #1+#2 (+ Hierarchical Physics)"
echo "  4. + ALL 3 Innovations (Full TopoFlow)"
echo "=========================================="
echo "Resources per job:"
echo "  - 50 nodes (400 GPUs)"
echo "  - 6 epochs"
echo "  - 18 hours max"
echo "  - From scratch (no checkpoint)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# Submit all jobs
JOB1=$(sbatch scripts/slurm_wind_baseline.sh | awk '{print $4}')
echo "✓ Wind Baseline submitted: Job ID $JOB1"

JOB2=$(sbatch scripts/slurm_innovation1.sh | awk '{print $4}')
echo "✓ Wind + Innovation #1 submitted: Job ID $JOB2"

JOB3=$(sbatch scripts/slurm_innovation2.sh | awk '{print $4}')
echo "✓ Wind + Innovation #1+#2 submitted: Job ID $JOB3"

JOB4=$(sbatch scripts/slurm_full_model.sh | awk '{print $4}')
echo "✓ Wind + Full Model submitted: Job ID $JOB4"

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Monitor with: squeue -u $USER"
echo "=========================================="
echo ""
echo "Job IDs: $JOB1 $JOB2 $JOB3 $JOB4"
echo ""
echo "Expected completion: ~12-18 hours"
echo "Results will be in logs/ directory"
echo ""
echo "After completion, evaluate with:"
echo "  bash scripts/monitor_and_evaluate.sh $JOB1 $JOB2 $JOB3 $JOB4"
echo ""