#!/bin/bash

JOB_ID=13584722
LOG_DIR="/scratch/project_462000640/ammar/aq_net2/logs"
OUT_FILE="${LOG_DIR}/topoflow_full_finetune_${JOB_ID}.out"
ERR_FILE="${LOG_DIR}/topoflow_full_finetune_${JOB_ID}.err"

echo "=========================================="
echo "Monitoring Job ${JOB_ID} - TopoFlow Fine-Tuning"
echo "=========================================="
echo ""

# Check if job is still running
JOB_STATUS=$(squeue -j ${JOB_ID} -h -o "%T" 2>/dev/null)

if [ -z "$JOB_STATUS" ]; then
    echo "❌ Job ${JOB_ID} is not running"
    echo ""
    echo "Checking job history..."
    sacct -j ${JOB_ID} --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,NodeList
    exit 1
fi

echo "✅ Job Status: $JOB_STATUS"
echo ""

# Get job info
squeue -j ${JOB_ID}
echo ""

# Check log file sizes
if [ -f "$OUT_FILE" ]; then
    OUT_LINES=$(wc -l < "$OUT_FILE")
    echo "📄 Output log: $OUT_LINES lines"
else
    echo "⚠️ Output log not found"
    OUT_LINES=0
fi

if [ -f "$ERR_FILE" ]; then
    ERR_LINES=$(wc -l < "$ERR_FILE")
    echo "📄 Error log: $ERR_LINES lines"
else
    echo "⚠️ Error log not found"
    ERR_LINES=0
fi

echo ""
echo "=========================================="
echo "🔍 RECENT ACTIVITY"
echo "=========================================="
echo ""

# Check for key progress indicators
if [ -f "$ERR_FILE" ]; then
    echo "--- Distributed Status ---"
    if grep -q "All distributed processes registered" "$ERR_FILE"; then
        echo "✅ Distributed: ALL 128 PROCESSES REGISTERED"
    else
        REGISTERED=$(grep -c "Initializing distributed:" "$ERR_FILE" 2>/dev/null || echo "0")
        echo "🔄 Distributed: $REGISTERED/128 processes initializing..."
    fi
    echo ""

    echo "--- Model Loading ---"
    if grep -q "Loading.*checkpoint\|Restoring" "$ERR_FILE" 2>/dev/null; then
        echo "🔄 Checkpoint loading in progress..."
        grep -i "loading\|restoring" "$ERR_FILE" | tail -5
    elif grep -q "Restored all states" "$ERR_FILE" 2>/dev/null; then
        echo "✅ Checkpoint loaded successfully"
    else
        echo "⏳ Waiting for checkpoint loading to start..."
    fi
    echo ""

    echo "--- Training Status ---"
    if grep -q "Epoch\|Training:" "$ERR_FILE" 2>/dev/null; then
        echo "✅ TRAINING STARTED!"
        echo ""
        echo "Latest training logs:"
        grep -E "Epoch|loss|step" "$ERR_FILE" | tail -10
    else
        echo "⏳ Training not started yet (still loading model/data)"
    fi
fi

echo ""
echo "=========================================="
echo "📊 LATEST OUTPUT (last 20 lines)"
echo "=========================================="
if [ -f "$OUT_FILE" ]; then
    tail -20 "$OUT_FILE"
else
    echo "No output yet"
fi

echo ""
echo "=========================================="
echo "⚠️ LATEST ERRORS (last 15 lines)"
echo "=========================================="
if [ -f "$ERR_FILE" ]; then
    tail -15 "$ERR_FILE" | grep -v "amdgpu.ids" | tail -15

    # Count amdgpu.ids warnings
    AMDGPU_COUNT=$(grep -c "amdgpu.ids" "$ERR_FILE" 2>/dev/null || echo "0")
    if [ "$AMDGPU_COUNT" -gt 0 ]; then
        echo ""
        echo "ℹ️  Note: $AMDGPU_COUNT 'amdgpu.ids' warnings (non-critical, job continues)"
    fi
else
    echo "No errors yet"
fi

echo ""
echo "=========================================="
echo "📈 PROGRESS SUMMARY"
echo "=========================================="
echo "Output lines: $OUT_LINES"
echo "Error lines: $ERR_LINES"
echo ""

# Estimate progress
if [ -f "$ERR_FILE" ]; then
    if grep -q "Training:" "$ERR_FILE" 2>/dev/null; then
        echo "🎯 Status: TRAINING IN PROGRESS"
        echo "Expected loss: ~0.35 (fine-tuning from checkpoint)"
    elif grep -q "All distributed processes registered" "$ERR_FILE" 2>/dev/null; then
        echo "🔄 Status: LOADING MODEL/CHECKPOINT"
        echo "This phase can take 5-15 minutes for 128 GPUs"
    else
        echo "🔄 Status: INITIALIZING DISTRIBUTED PROCESSES"
    fi
fi

echo ""
echo "=========================================="
echo "Run this script again to update: ./monitor_13584722.sh"
echo "Or watch continuously: watch -n 30 ./monitor_13584722.sh"
echo "=========================================="
