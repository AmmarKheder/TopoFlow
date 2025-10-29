#!/bin/bash
# Monitor job 13616451 - TopoFlow with elevation_alpha fix

while true; do
    clear
    echo "==============================================="
    echo "Job 13616451 - TopoFlow Elevation Fix Monitor"
    echo "==============================================="
    echo ""
    
    # Job status
    echo "--- Job Status ---"
    squeue -j 13616451 2>/dev/null || echo "Job completed or not found"
    echo ""
    
    # Check if log exists and show relevant parts
    if [ -f logs/ELEVATION_MASK_13616451.out ]; then
        echo "--- Checkpoint Loading (should show FIXED messages) ---"
        grep -A 5 "FIXED: .*elevation_alpha\|FIXED: .*H_scale" logs/ELEVATION_MASK_13616451.out | tail -10
        echo ""
        
        echo "--- First Validation Loss (should be ~0.35, not 0.96) ---"
        grep "val_loss=" logs/ELEVATION_MASK_13616451.out | head -3
        echo ""
        
        echo "--- Latest Progress ---"
        tail -20 logs/ELEVATION_MASK_13616451.out
    else
        echo "Log not created yet - job still pending"
    fi
    
    echo ""
    echo "==============================================="
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 30 seconds..."
    echo "==============================================="
    
    sleep 30
done
