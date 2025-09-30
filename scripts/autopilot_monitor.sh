#!/bin/bash
################################################################################
# TopoFlow Autopilot Monitor
#
# Surveillance automatique complète des jobs avec auto-correction
# - Détecte les erreurs
# - Corrige automatiquement
# - Relance les jobs si nécessaire
# - Génère des rapports
################################################################################

set -e

JOBS=("$@")
if [ ${#JOBS[@]} -eq 0 ]; then
    echo "ERROR: No job IDs provided!"
    echo "Usage: $0 JOB1 JOB2 JOB3 JOB4"
    exit 1
fi

LOGFILE="logs/autopilot_$(date +%Y%m%d_%H%M%S).log"
ERROR_COUNT=0
MAX_RETRIES=3

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log "=========================================="
log "TOPOFLOW AUTOPILOT MONITOR STARTED"
log "=========================================="
log "Monitoring jobs: ${JOBS[*]}"
log "Log file: $LOGFILE"
log ""

# Function to check job status
get_job_status() {
    local job_id=$1
    squeue -j $job_id -h -o "%T" 2>/dev/null || echo "COMPLETED"
}

# Function to check for errors in logs
check_job_errors() {
    local job_id=$1
    local log_pattern="logs/topoflow_*_${job_id}.err"

    if ls $log_pattern 1> /dev/null 2>&1; then
        local errlog=$(ls -t $log_pattern | head -1)

        # Check for common errors
        if grep -q "SyntaxError\|ImportError\|ModuleNotFoundError\|CUDA error\|OOM" "$errlog" 2>/dev/null; then
            echo "ERROR_FOUND"
            return 0
        fi

        # Check for SLURM errors
        if grep -q "srun: error\|Segmentation fault\|Killed" "$errlog" 2>/dev/null; then
            echo "SLURM_ERROR"
            return 0
        fi
    fi

    echo "OK"
}

# Function to analyze and fix errors
auto_fix_error() {
    local job_id=$1
    local error_type=$2
    local log_pattern="logs/topoflow_*_${job_id}.err"
    local errlog=$(ls -t $log_pattern | head -1)

    log "🔧 ANALYZING ERROR in job $job_id..."

    # Check for UTF-8 encoding errors
    if grep -q "utf-8.*codec" "$errlog" 2>/dev/null; then
        log "  → UTF-8 encoding error detected"
        log "  → Fixing problematic characters in source files..."

        # Fix UTF-8 issues
        find src/ -name "*.py" -exec sed -i 's/[^\x00-\x7F]/#/g' {} \;

        log "  ✓ UTF-8 issues fixed"
        return 0
    fi

    # Check for import errors
    if grep -q "ImportError\|ModuleNotFoundError" "$errlog" 2>/dev/null; then
        log "  → Import error detected"
        log "  → Checking module dependencies..."

        # Could add automatic pip install here if needed
        log "  ⚠ Manual intervention may be required"
        return 1
    fi

    # Check for OOM
    if grep -q "OOM\|out of memory" "$errlog" 2>/dev/null; then
        log "  → Out of memory error detected"
        log "  → Reducing batch size in configs..."

        # Reduce batch size by half
        for config in configs/config_*.yaml; do
            sed -i 's/batch_size: 2/batch_size: 1/g' "$config"
        done

        log "  ✓ Batch size reduced to 1"
        return 0
    fi

    log "  → Error type unclear, logging for review"
    return 1
}

# Function to relaunch a failed job
relaunch_job() {
    local job_id=$1
    local config_name=$2

    log "🚀 RELAUNCHING job $job_id ($config_name)..."

    # Determine which script to run
    case $config_name in
        *baseline*)
            script="scripts/slurm_wind_baseline.sh"
            ;;
        *innov1*)
            script="scripts/slurm_innovation1.sh"
            ;;
        *innov2*)
            script="scripts/slurm_innovation2.sh"
            ;;
        *full*)
            script="scripts/slurm_full_model.sh"
            ;;
        *)
            log "  ✗ Unknown config: $config_name"
            return 1
            ;;
    esac

    # Submit new job
    new_job=$(sbatch "$script" | awk '{print $4}')
    log "  ✓ New job submitted: $new_job"

    # Update job list
    JOBS[$i]=$new_job

    return 0
}

# Main monitoring loop
iteration=0
all_completed=false

while [ "$all_completed" = false ]; do
    iteration=$((iteration + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    log ""
    log "=== ITERATION $iteration - $timestamp ==="

    all_completed=true

    # Check each job
    for i in "${!JOBS[@]}"; do
        job_id=${JOBS[$i]}
        status=$(get_job_status $job_id)

        log "Job $job_id: $status"

        if [ "$status" != "COMPLETED" ]; then
            all_completed=false

            # Check for errors even if running
            if [ "$status" = "RUNNING" ] || [ "$status" = "PENDING" ]; then
                error_check=$(check_job_errors $job_id)

                if [ "$error_check" != "OK" ]; then
                    log "  ⚠ ERROR DETECTED in running job $job_id!"
                    log "  → Canceling job..."
                    scancel $job_id 2>/dev/null || true
                    sleep 2

                    # Try to fix
                    if auto_fix_error $job_id "$error_check"; then
                        log "  ✓ Error fixed, relaunching..."
                        relaunch_job $job_id "$(squeue -j $job_id -h -o '%j' 2>/dev/null || echo 'unknown')"
                    else
                        log "  ✗ Could not auto-fix, manual intervention needed"
                        ERROR_COUNT=$((ERROR_COUNT + 1))
                    fi
                fi
            fi

            # Check if job failed
            if [ "$status" = "FAILED" ] || [ "$status" = "CANCELLED" ] || [ "$status" = "TIMEOUT" ]; then
                log "  ✗ Job $job_id FAILED!"

                ERROR_COUNT=$((ERROR_COUNT + 1))

                if [ $ERROR_COUNT -lt $MAX_RETRIES ]; then
                    if auto_fix_error $job_id "FAILED"; then
                        log "  → Attempting relaunch (retry $ERROR_COUNT/$MAX_RETRIES)..."
                        relaunch_job $job_id "unknown"
                    fi
                else
                    log "  ✗ Max retries reached, stopping auto-recovery"
                fi
            fi
        fi
    done

    # Status summary
    running_count=0
    completed_count=0
    failed_count=0

    for job_id in "${JOBS[@]}"; do
        status=$(get_job_status $job_id)
        case $status in
            RUNNING|PENDING) running_count=$((running_count + 1)) ;;
            COMPLETED) completed_count=$((completed_count + 1)) ;;
            FAILED|CANCELLED|TIMEOUT) failed_count=$((failed_count + 1)) ;;
        esac
    done

    log ""
    log "📊 STATUS: $completed_count completed | $running_count running | $failed_count failed"

    if [ "$all_completed" = true ]; then
        log ""
        log "=========================================="
        log "🎉 ALL JOBS COMPLETED!"
        log "=========================================="
        break
    fi

    # Wait 2 minutes before next check
    log "⏰ Next check in 2 minutes..."
    sleep 120
done

# Final report
log ""
log "=========================================="
log "AUTOPILOT FINAL REPORT"
log "=========================================="
log "Total iterations: $iteration"
log "Total errors handled: $ERROR_COUNT"
log "Final job IDs: ${JOBS[*]}"
log ""

# Launch evaluation if all succeeded
if [ $failed_count -eq 0 ]; then
    log "🚀 Launching automatic evaluation..."
    bash scripts/monitor_and_evaluate.sh "${JOBS[@]}" > logs/evaluation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    eval_pid=$!
    log "  ✓ Evaluation running in background (PID: $eval_pid)"
else
    log "⚠ Some jobs failed, skipping automatic evaluation"
    log "  Please review logs and relaunch manually"
fi

log ""
log "=========================================="
log "AUTOPILOT MONITOR FINISHED"
log "=========================================="