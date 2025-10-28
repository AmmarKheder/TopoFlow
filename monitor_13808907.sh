#!/bin/bash

JOB_ID=13808907

echo "=========================================="
echo "🔍 MONITORING JOB $JOB_ID"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "🔍 Job $JOB_ID - $(date '+%H:%M:%S')"
    echo "=========================================="
    echo ""

    # Job status
    echo "📊 JOB STATUS:"
    squeue -j $JOB_ID 2>/dev/null || echo "Job completed or not found"
    echo ""

    # Check if output file exists
    if [ -f logs/ELEVATION_MASK_${JOB_ID}.out ]; then
        echo "📄 LAST 30 LINES OF OUTPUT:"
        echo "----------------------------------------"
        tail -30 logs/ELEVATION_MASK_${JOB_ID}.out
        echo ""

        # Check for critical messages
        echo "🔍 CRITICAL CHECKS:"
        echo "----------------------------------------"

        # Check for optimizer loading
        if grep -q "LOADING OPTIMIZER STATE IN configure_optimizers" logs/ELEVATION_MASK_${JOB_ID}.out 2>/dev/null; then
            echo "✅ Optimizer state loading initiated"
            grep "Loaded.*optimizer states" logs/ELEVATION_MASK_${JOB_ID}.out 2>/dev/null | tail -1
        fi

        # Check for training start
        if grep -q "train_loss=" logs/ELEVATION_MASK_${JOB_ID}.out 2>/dev/null; then
            echo "✅ TRAINING STARTED!"
            echo "First train_loss values:"
            grep -oP "train_loss=[\d\.]+" logs/ELEVATION_MASK_${JOB_ID}.out 2>/dev/null | head -5
        fi

        # Check for validation
        if grep -q "val_loss=" logs/ELEVATION_MASK_${JOB_ID}.out 2>/dev/null; then
            echo "✅ VALIDATION RUNNING!"
            grep "val_loss=" logs/ELEVATION_MASK_${JOB_ID}.out 2>/dev/null | tail -3
        fi

    else
        echo "⏳ Waiting for output file to be created..."
    fi

    # Check error file
    if [ -f logs/ELEVATION_MASK_${JOB_ID}.err ]; then
        echo ""
        echo "⚠️  LAST 10 LINES OF ERRORS:"
        echo "----------------------------------------"
        tail -10 logs/ELEVATION_MASK_${JOB_ID}.err | grep -v "amdgpu.ids" | grep -v "^$" || echo "No significant errors"

        # Check for segfault
        if grep -q "Segmentation fault" logs/ELEVATION_MASK_${JOB_ID}.err 2>/dev/null; then
            echo ""
            echo "❌❌❌ SEGMENTATION FAULT DETECTED! ❌❌❌"
            grep "Segmentation fault" logs/ELEVATION_MASK_${JOB_ID}.err
        fi
    fi

    echo ""
    echo "=========================================="
    echo "Press Ctrl+C to stop monitoring"
    echo "Next update in 10 seconds..."
    echo "=========================================="

    sleep 10
done
