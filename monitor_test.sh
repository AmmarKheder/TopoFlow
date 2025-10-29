#!/bin/bash
# Monitoring du job de test

JOBID=13861044

while true; do
    clear
    echo "========================================"
    echo "🔍 MONITORING JOB $JOBID"
    echo "========================================"
    echo ""

    # Status
    squeue -j $JOBID 2>/dev/null

    if [ $? -ne 0 ]; then
        echo ""
        echo "Job terminé. Dernières lignes:"
        echo ""
        tail -100 logs/TEST_RESUME_${JOBID}.out | grep -E "train_loss|val_loss|TEST|RESUME|✅|❌"
        break
    fi

    echo ""
    echo "========================================"
    echo "📊 DERNIERS LOGS (train_loss):"
    echo "========================================"

    if [ -f logs/TEST_RESUME_${JOBID}.out ]; then
        tail -50 logs/TEST_RESUME_${JOBID}.out | grep -E "train_loss|val_loss|epoch|step" | tail -10
    else
        echo "Pas encore de logs..."
    fi

    sleep 5
done
