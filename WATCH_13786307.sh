#!/bin/bash
for i in {1..20}; do
    echo "=== CHECK $i/20 - $(date +%H:%M:%S) ==="
    
    # Job status
    squeue -j 13786307 -o "%.18i %.2t %.10M %R" 2>/dev/null || echo "Job finished"
    
    # Check log
    if [ -f logs/ELEVATION_MASK_13786307.out ]; then
        echo "📊 Log: $(wc -l < logs/ELEVATION_MASK_13786307.out) lines"
        
        # Check for MANUAL LOADING
        if grep -q "MANUAL OPTIMIZER STATE LOADING" logs/ELEVATION_MASK_13786307.out; then
            echo "✅✅✅ MANUAL LOADING TROUVÉ!"
            echo ""
            grep -A 12 "MANUAL OPTIMIZER STATE LOADING" logs/ELEVATION_MASK_13786307.out | head -15
            echo ""
            echo "=== TRAIN LOSS ==="
            grep "train_loss=" logs/ELEVATION_MASK_13786307.out | head -5
            break
        elif grep -q "train_loss=" logs/ELEVATION_MASK_13786307.out; then
            echo "⚠️ Training started!"
            grep "train_loss=" logs/ELEVATION_MASK_13786307.out | head -5
            break
        elif grep -q "ERROR\|AttributeError" logs/ELEVATION_MASK_13786307.out; then
            echo "❌ ERROR found:"
            grep -A 5 "ERROR\|AttributeError" logs/ELEVATION_MASK_13786307.out | tail -10
            break
        else
            tail -3 logs/ELEVATION_MASK_13786307.out
        fi
    else
        echo "⏳ Waiting for log..."
    fi
    
    echo ""
    sleep 60
done
