#!/bin/bash
################################################################################
# TopoFlow - Automatic Monitoring and Evaluation
#
# This script:
# 1. Monitors 4 ablation study jobs
# 2. Waits for completion
# 3. Automatically launches evaluations
# 4. Compares results
#
# Usage: bash scripts/monitor_and_evaluate.sh JOB1 JOB2 JOB3 JOB4
################################################################################

set -e

# Job IDs
JOB_BASELINE=$1
JOB_INNOV1=$2
JOB_INNOV2=$3
JOB_FULL=$4

if [ -z "$JOB_BASELINE" ]; then
    echo "ERROR: No job IDs provided!"
    echo "Usage: $0 JOB_BASELINE JOB_INNOV1 JOB_INNOV2 JOB_FULL"
    exit 1
fi

echo "=========================================="
echo "TopoFlow Automatic Monitor & Evaluation"
echo "=========================================="
echo "Job IDs:"
echo "  Baseline:      $JOB_BASELINE"
echo "  Innovation #1: $JOB_INNOV1"
echo "  Innovation #2: $JOB_INNOV2"
echo "  Full Model:    $JOB_FULL"
echo "=========================================="
echo ""

# Check job status
check_job_status() {
    local job_id=$1
    squeue -j $job_id -h -o "%T" 2>/dev/null || echo "COMPLETED"
}

# Wait for all jobs to complete
echo "Monitoring jobs... (checks every 2 minutes)"
while true; do
    status_baseline=$(check_job_status $JOB_BASELINE)
    status_innov1=$(check_job_status $JOB_INNOV1)
    status_innov2=$(check_job_status $JOB_INNOV2)
    status_full=$(check_job_status $JOB_FULL)

    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] Status: BASE=$status_baseline | INNOV1=$status_innov1 | INNOV2=$status_innov2 | FULL=$status_full"

    # Check if all completed
    if [ "$status_baseline" = "COMPLETED" ] && \
       [ "$status_innov1" = "COMPLETED" ] && \
       [ "$status_innov2" = "COMPLETED" ] && \
       [ "$status_full" = "COMPLETED" ]; then
        echo ""
        echo "=========================================="
        echo "All jobs COMPLETED!"
        echo "=========================================="
        break
    fi

    # Wait 2 minutes before next check
    sleep 120
done

echo ""
echo "=========================================="
echo "STEP 2: Locating checkpoints..."
echo "=========================================="

# Find latest checkpoint directories
LOGS_DIR="logs/multipollutants_climax_ddp"

# Find version directories for each experiment
find_latest_checkpoint() {
    local name_pattern=$1
    local version_dir=$(ls -dt $LOGS_DIR/version_* 2>/dev/null | \
                        xargs -I {} sh -c 'grep -l "'$name_pattern'" {}/hparams.yaml 2>/dev/null && echo {}' | \
                        head -1)

    if [ -z "$version_dir" ]; then
        # Fallback: find latest version
        version_dir=$(ls -dt $LOGS_DIR/version_* 2>/dev/null | head -1)
    fi

    if [ -n "$version_dir" ]; then
        # Find best checkpoint
        local ckpt=$(ls -t $version_dir/checkpoints/*.ckpt 2>/dev/null | head -1)
        echo "$ckpt"
    else
        echo ""
    fi
}

CKPT_BASELINE=$(find_latest_checkpoint "TopoFlow_Baseline")
CKPT_INNOV1=$(find_latest_checkpoint "TopoFlow_Innovation1")
CKPT_INNOV2=$(find_latest_checkpoint "TopoFlow_Innovation")
CKPT_FULL=$(find_latest_checkpoint "TopoFlow_FULL")

echo "Checkpoints found:"
echo "  Baseline:      $CKPT_BASELINE"
echo "  Innovation #1: $CKPT_INNOV1"
echo "  Innovation #2: $CKPT_INNOV2"
echo "  Full Model:    $CKPT_FULL"
echo ""

# Create evaluation directory
mkdir -p experiments/fast_eval

echo "=========================================="
echo "STEP 3: Running evaluations (1000 samples)"
echo "=========================================="

# Run evaluations in parallel if checkpoints found
eval_jobs=()

if [ -f "$CKPT_BASELINE" ]; then
    echo "Evaluating Baseline..."
    python scripts/evaluate_fast.py \
        --checkpoint "$CKPT_BASELINE" \
        --config configs/config_all_pollutants.yaml \
        --num_samples 1000 \
        --output experiments/fast_eval/baseline.yaml &
    eval_jobs+=($!)
fi

if [ -f "$CKPT_INNOV1" ]; then
    echo "Evaluating Innovation #1..."
    python scripts/evaluate_fast.py \
        --checkpoint "$CKPT_INNOV1" \
        --config configs/config_innovation1.yaml \
        --num_samples 1000 \
        --output experiments/fast_eval/innovation1.yaml &
    eval_jobs+=($!)
fi

if [ -f "$CKPT_INNOV2" ]; then
    echo "Evaluating Innovation #1+#2..."
    python scripts/evaluate_fast.py \
        --checkpoint "$CKPT_INNOV2" \
        --config configs/config_innovation2.yaml \
        --num_samples 1000 \
        --output experiments/fast_eval/innovation2.yaml &
    eval_jobs+=($!)
fi

if [ -f "$CKPT_FULL" ]; then
    echo "Evaluating Full Model..."
    python scripts/evaluate_fast.py \
        --checkpoint "$CKPT_FULL" \
        --config configs/config_full_model.yaml \
        --num_samples 1000 \
        --output experiments/fast_eval/full_model.yaml &
    eval_jobs+=($!)
fi

# Wait for all evaluations
echo ""
echo "Waiting for evaluations to complete..."
for pid in "${eval_jobs[@]}"; do
    wait $pid
done

echo ""
echo "=========================================="
echo "STEP 4: Comparing results"
echo "=========================================="

python scripts/compare_ablation_results.py --results_dir experiments/fast_eval

echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo "Results saved in: experiments/fast_eval/"
echo "View logs in: logs/"
echo ""