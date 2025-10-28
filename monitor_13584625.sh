#!/bin/bash
JOB_ID=13584625
LOG_ERR="logs/topoflow_full_finetune_${JOB_ID}.err"
LOG_OUT="logs/topoflow_full_finetune_${JOB_ID}.out"

echo "=========================================="
echo "📊 MONITORING JOB ${JOB_ID}"
echo "=========================================="
echo ""

# Job status
echo "🔍 Job Status:"
squeue -j ${JOB_ID} 2>/dev/null || echo "Job not in queue (might be finished or failed)"
echo ""

# Log sizes
echo "📁 Log Files:"
echo "  OUT: $(wc -l $LOG_OUT 2>/dev/null | awk '{print $1}') lines"
echo "  ERR: $(wc -l $LOG_ERR 2>/dev/null | awk '{print $1}') lines"
echo ""

# Check for checkpoint loading
echo "🔍 Checkpoint Loading:"
grep -i "Loading checkpoint\|Checkpoint loaded\|setup()" $LOG_ERR 2>/dev/null | head -5 || echo "  Not yet loaded"
echo ""

# Check for training progress
echo "🚀 Training Progress:"
grep -E "Epoch|training_step|train_loss|val_loss" $LOG_ERR 2>/dev/null | tail -10 || echo "  Not started yet"
echo ""

# Check last output
echo "📝 Last 10 lines of .out:"
tail -10 $LOG_OUT 2>/dev/null
echo ""

echo "=========================================="
