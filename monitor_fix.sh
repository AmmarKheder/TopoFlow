#!/bin/bash
# Monitor script to verify the DDP fix works

JOB_ID=13563757
LOG_FILE="logs/topoflow_full_finetune_${JOB_ID}.out"

echo "=========================================="
echo "🔍 MONITORING JOB $JOB_ID FOR FIX VALIDATION"
echo "=========================================="
echo ""

# Wait for log file to appear
echo "⏳ Waiting for log file to appear..."
while [ ! -f "$LOG_FILE" ]; do
    sleep 5
done
echo "✅ Log file found: $LOG_FILE"
echo ""

# Wait for checkpoint loading message
echo "⏳ Waiting for checkpoint loading in setup()..."
while ! grep -q "Loading checkpoint in setup()" "$LOG_FILE" 2>/dev/null; do
    sleep 10
done

echo ""
echo "=========================================="
echo "📊 CHECKPOINT LOADING DETECTED"
echo "=========================================="
echo ""

# Show checkpoint loading section
grep -A 10 "Loading checkpoint in setup()" "$LOG_FILE" | head -15

echo ""
echo "=========================================="
echo "⏳ Waiting for first training step..."
echo "=========================================="
echo ""

# Wait for first training loss
while ! grep -q "train_loss=" "$LOG_FILE" 2>/dev/null; do
    sleep 10
done

echo ""
echo "=========================================="
echo "📊 FIRST TRAINING LOSSES"
echo "=========================================="
echo ""

# Show first 5 training steps
grep "train_loss=" "$LOG_FILE" | head -5

echo ""
echo "=========================================="
echo "⏳ Waiting for first validation..."
echo "=========================================="
echo ""

# Wait for first validation
while ! grep -q "val_loss=" "$LOG_FILE" 2>/dev/null; do
    sleep 30
done

echo ""
echo "=========================================="
echo "📊 FIRST VALIDATION LOSS"
echo "=========================================="
echo ""

# Show first validation
grep "val_loss=" "$LOG_FILE" | head -3

echo ""
echo "=========================================="
echo "🎯 VALIDATION RESULTS"
echo "=========================================="
echo ""

# Extract first train and val loss values
FIRST_TRAIN=$(grep "train_loss=" "$LOG_FILE" | head -1 | grep -oP 'train_loss=\K[0-9.]+')
FIRST_VAL=$(grep "val_loss=" "$LOG_FILE" | head -1 | grep -oP 'val_loss=\K[0-9.]+')

echo "First train_loss: $FIRST_TRAIN"
echo "First val_loss:   $FIRST_VAL"
echo ""

# Check if fix works
if (( $(echo "$FIRST_TRAIN < 1.0" | bc -l) )); then
    echo "✅ SUCCESS! Train loss < 1.0"
    echo "   The checkpoint was loaded correctly in all ranks!"
    echo "   Fix is working! 🎉"
else
    echo "❌ PROBLEM! Train loss >= 1.0"
    echo "   Checkpoint may not have loaded correctly"
    echo "   Check the logs above"
fi

if (( $(echo "$FIRST_VAL < 0.5" | bc -l) )); then
    echo "✅ SUCCESS! Val loss < 0.5"
    echo "   Starting close to baseline (0.3557)!"
    echo "   TopoFlow fine-tuning is on track! 🚀"
else
    echo "❌ PROBLEM! Val loss >= 0.5"
    echo "   Should start close to 0.3557"
    echo "   Investigate the checkpoint loading"
fi

echo ""
echo "=========================================="
echo "Monitor script completed!"
echo "Check full logs: tail -f $LOG_FILE"
echo "=========================================="
