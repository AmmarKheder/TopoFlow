#!/bin/bash
# Surveillance permanente du job 13593511 (256 GPUs)

JOB_ID=13593511
LOG_FILE="MONITORING_${JOB_ID}.log"
ERR_LOG="logs/topoflow_full_finetune_${JOB_ID}.err"
OUT_LOG="logs/topoflow_full_finetune_${JOB_ID}.out"

echo "========================================"  | tee $LOG_FILE
echo "🔍 SURVEILLANCE PERMANENTE JOB $JOB_ID" | tee -a $LOG_FILE
echo "256 GPUs (32 nodes)" | tee -a $LOG_FILE
echo "Démarré: $(date)" | tee -a $LOG_FILE
echo "========================================"  | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

CHECK_NUM=0

while true; do
    CHECK_NUM=$((CHECK_NUM + 1))
    echo "=== Check $CHECK_NUM - $(date +%H:%M:%S) ===" >> $LOG_FILE

    STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)

    if [ -z "$STATUS" ]; then
        echo "" | tee -a $LOG_FILE
        echo "❌ JOB TERMINÉ OU ANNULÉ!" | tee -a $LOG_FILE
        sacct -j $JOB_ID --format=JobID,State,ExitCode,Elapsed,Start,End | tee -a $LOG_FILE
        break
    fi

    echo "Status: $STATUS" >> $LOG_FILE

    if [ "$STATUS" = "RUNNING" ]; then
        echo "" | tee -a $LOG_FILE
        echo "✅✅✅ JOB DÉMARRÉ! $(date)" | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE

        # Attendre que les logs apparaissent
        sleep 30

        if [ -f "$ERR_LOG" ]; then
            echo "📄 Logs trouvés - Vérification Python startup..." | tee -a $LOG_FILE

            # Attendre 2 minutes pour Python startup
            for i in {1..12}; do
                if grep -q "Bound process to cuda.*LOCAL_RANK\|D.*MARRAGE AQ_NET2" "$ERR_LOG" 2>/dev/null; then
                    echo "" | tee -a $LOG_FILE
                    echo "🎉🎉🎉 PYTHON DÉMARRÉ AVEC SUCCÈS!" | tee -a $LOG_FILE
                    echo "" | tee -a $LOG_FILE
                    echo "Premiers messages Python:" | tee -a $LOG_FILE
                    grep "Bound process\|D.*MARRAGE\|Configuration:" "$ERR_LOG" | head -10 | tee -a $LOG_FILE
                    echo "" | tee -a $LOG_FILE
                    echo "✅ Job fonctionne! Surveillance terminée." | tee -a $LOG_FILE
                    echo "Suivre l'entraînement: tail -f $ERR_LOG" | tee -a $LOG_FILE
                    exit 0
                elif grep -q "All distributed processes registered" "$ERR_LOG" 2>/dev/null; then
                    echo "⏳ DDP initialisé, attente Python... (check $i/12)" >> $LOG_FILE
                else
                    echo "⏳ Initialisation en cours... (check $i/12)" >> $LOG_FILE
                fi
                sleep 10
            done

            echo "" | tee -a $LOG_FILE
            echo "⚠️  Job tourne mais Python ne démarre pas après 2 min" | tee -a $LOG_FILE
            echo "Dernières 30 lignes de stderr:" | tee -a $LOG_FILE
            tail -30 "$ERR_LOG" | grep -v amdgpu | tee -a $LOG_FILE
            echo "" | tee -a $LOG_FILE
            echo "Possible problème - vérifier manuellement" | tee -a $LOG_FILE
            exit 1
        else
            echo "⏳ Attente création fichiers logs..." | tee -a $LOG_FILE
        fi
    elif [ "$STATUS" = "PENDING" ]; then
        # Check position toutes les 10 minutes (120 checks = 10 min)
        if [ $((CHECK_NUM % 120)) -eq 0 ]; then
            POSITION=$(squeue -p standard-g -t PD | grep -n "$JOB_ID" | cut -d: -f1)
            echo "Position queue: $POSITION" | tee -a $LOG_FILE
        fi
    fi

    sleep 5
done

echo "" | tee -a $LOG_FILE
echo "========================================"  | tee -a $LOG_FILE
echo "Surveillance terminée: $(date)" | tee -a $LOG_FILE
echo "Log complet: $LOG_FILE" | tee -a $LOG_FILE
echo "========================================"  | tee -a $LOG_FILE
