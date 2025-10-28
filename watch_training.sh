#!/bin/bash
# Watch training progress for job 13772284

JOB_ID=13772284

echo "=============================================="
echo "Training Monitor - Job $JOB_ID"
echo "=============================================="

# Job status
echo -e "\n📊 Job Status:"
squeue -j $JOB_ID 2>/dev/null || echo "Job not in queue (completed or cancelled)"

# Last training progress
echo -e "\n📈 Latest Training Progress:"
tail -50 /scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_${JOB_ID}.out | grep -E "Epoch|train_loss|val_loss|step" | tail -5

# File sizes (to see if still writing)
echo -e "\n📁 Log File Sizes:"
ls -lh /scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_${JOB_ID}.{out,err} 2>/dev/null

# Recent errors (excluding amdgpu warnings)
echo -e "\n⚠️  Recent Errors (if any):"
tail -50 /scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_${JOB_ID}.err 2>/dev/null | grep -v "amdgpu.ids" | grep -E "Error|WARNING|Failed|Traceback" | tail -3 || echo "No recent errors"

echo -e "\n=============================================="
echo "To watch continuously: watch -n 30 $0"
echo "=============================================="
