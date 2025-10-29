#!/bin/bash
# Monitor physics fine-tuning experiment

echo "🔍 PHYSICS FINE-TUNING MONITOR"
echo "================================"

# Get latest job ID
JOB_ID=$(squeue -u $USER -h -o "%i" | head -1)

if [ -z "$JOB_ID" ]; then
    echo "❌ No job running"
    exit 1
fi

echo "📊 Job ID: $JOB_ID"
echo ""

# Job status
echo "📈 Job Status:"
squeue -j $JOB_ID

echo ""
echo "📋 Latest Log Output:"
echo "================================"

# Find latest log
LOG_FILE=$(ls -t logs/physics_finetune_*.out 2>/dev/null | head -1)

if [ -f "$LOG_FILE" ]; then
    echo "Log: $LOG_FILE"
    echo ""
    tail -50 "$LOG_FILE"
else
    echo "⏳ Log file not created yet..."
fi

echo ""
echo "================================"
echo "💡 Commands:"
echo "  - Watch logs: tail -f $LOG_FILE"
echo "  - Cancel job: scancel $JOB_ID"
echo "  - Rerun monitor: bash scripts/monitor_physics.sh"
