#!/bin/bash
echo "=========================================="
echo "JOB 13787155 - OPTIMIZER LOADING IN configure_optimizers()"
echo "THIS IS THE REAL FIX!"
echo "=========================================="

for i in {1..30}; do
    echo ""
    echo "=== CHECK $i/30 - $(date +%H:%M:%S) ==="
    
    # Job status
    squeue -j 13787155 -o "%.18i %.2t %.10M %R" 2>/dev/null || echo "Job finished/cancelled"
    
    # Check log
    if [ -f logs/ELEVATION_MASK_13787155.out ]; then
        echo "📊 Log: $(wc -l < logs/ELEVATION_MASK_13787155.out) lines"
        
        # Check for optimizer loading in configure_optimizers
        if grep -q "LOADING OPTIMIZER STATE IN configure_optimizers" logs/ELEVATION_MASK_13787155.out; then
            echo "✅✅✅ OPTIMIZER LOADED IN configure_optimizers()!"
            echo ""
            grep -A 10 "LOADING OPTIMIZER STATE IN configure_optimizers" logs/ELEVATION_MASK_13787155.out | head -15
            echo ""
            
            # Check train_loss
            if grep -q "train_loss=" logs/ELEVATION_MASK_13787155.out; then
                echo "=== TRAIN LOSS VALUES ==="
                grep -oP "train_loss=[\d\.]+" logs/ELEVATION_MASK_13787155.out | head -15
                echo ""
                
                # Get first train_loss value
                first_loss=$(grep -oP "train_loss=[\d\.]+" logs/ELEVATION_MASK_13787155.out | head -1 | cut -d'=' -f2)
                echo "First train_loss: $first_loss"
                
                # Check if it's low (< 1.0 = SUCCESS!)
                if (( $(echo "$first_loss < 1.0" | bc -l) )); then
                    echo "🎉🎉🎉 SUCCÈS! train_loss < 1.0 - RESUME WORKS!"
                    break
                else
                    echo "⚠️ train_loss still high (>= 1.0)"
                fi
            fi
        elif grep -q "train_loss=" logs/ELEVATION_MASK_13787155.out; then
            echo "⚠️ Training started without manual loading?"
            grep -oP "train_loss=[\d\.]+" logs/ELEVATION_MASK_13787155.out | head -5
        elif grep -q "ERROR\|Exception\|Traceback" logs/ELEVATION_MASK_13787155.out; then
            echo "❌ ERROR detected:"
            grep -A 5 "ERROR\|Exception" logs/ELEVATION_MASK_13787155.out | tail -10
            break
        else
            tail -3 logs/ELEVATION_MASK_13787155.out
        fi
    else
        echo "⏳ Waiting for log..."
    fi
    
    sleep 60
done
