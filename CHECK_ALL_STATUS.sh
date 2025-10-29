#!/bin/bash
# Quick status check script

echo "=============================================="
echo "🔍 TopoFlow Debug - Quick Status Check"
echo "=============================================="
echo "Time: $(date)"
echo ""

echo "📋 YOUR ACTIVE JOBS:"
echo "---"
squeue -u $USER || echo "No jobs in queue"
echo ""

echo "📊 RECENT JOB HISTORY (last 5):"
echo "---"
sacct --starttime=$(date -d '1 day ago' +%Y-%m-%d) --format=JobID,JobName,State,Elapsed,ExitCode -n | grep -E "topoflow|test_ven" | tail -5
echo ""

echo "=============================================="
echo "🧪 TEST JOB 13589754 (Venv Diagnostic - 4 GPUs)"
echo "=============================================="
if [ -f logs/test_venv_13589754.out ]; then
    echo "✅ OUTPUT LOG EXISTS:"
    cat logs/test_venv_13589754.out
    echo ""
    if grep -q "ALL TESTS PASSED" logs/test_venv_13589754.out; then
        echo "✅✅✅ VENV TEST PASSED! Venv propagation works!"
    elif grep -q "FAILED" logs/test_venv_13589754.out; then
        echo "❌ VENV TEST FAILED - Check errors above"
    fi
else
    echo "⏳ Log not created yet (job pending or just started)"
fi
echo ""

echo "=============================================="
echo "🚀 MAIN TEST JOB 13589032 (32 GPUs)"
echo "=============================================="
if [ -f logs/topoflow_full_finetune_13589032.err ]; then
    ERR_LINES=$(wc -l < logs/topoflow_full_finetune_13589032.err)
    echo "📄 Error log: $ERR_LINES lines"
    echo ""

    if grep -q "LOCAL_RANK.*CUDA_VISIBLE_DEVICES" logs/topoflow_full_finetune_13589032.err; then
        echo "✅✅✅ PYTHON STARTED! Job is running!"
        echo ""
        echo "Latest logs:"
        tail -30 logs/topoflow_full_finetune_13589032.err | grep -v amdgpu
    elif grep -q "All distributed processes registered" logs/topoflow_full_finetune_13589032.err; then
        echo "⚠️  Distributed init done, checking if Python started..."
        echo ""
        echo "Last 20 lines (without amdgpu spam):"
        tail -50 logs/topoflow_full_finetune_13589032.err | grep -v amdgpu | tail -20
        echo ""
        STUCK_TIME=$((ERR_LINES / 40))  # Rough estimate: 40 lines per minute
        if [ $STUCK_TIME -gt 5 ]; then
            echo "❌ LIKELY STUCK - No progress for ~$STUCK_TIME minutes after DDP init"
        fi
    else
        echo "🔄 Still initializing distributed processes..."
        tail -10 logs/topoflow_full_finetune_13589032.err | grep -v amdgpu
    fi
else
    echo "⏳ Log not created yet (job pending or just started)"
fi

echo ""
echo "=============================================="
echo "📈 AUTO-MONITOR LOG"
echo "=============================================="
if [ -f AUTO_MONITOR_13589032.log ]; then
    echo "Last 15 lines:"
    tail -15 AUTO_MONITOR_13589032.log
else
    echo "Auto-monitor not started or log not found"
fi

echo ""
echo "=============================================="
echo "📝 QUICK ACTION GUIDE"
echo "=============================================="
echo ""
echo "If VENV TEST PASSED + MAIN TEST shows LOCAL_RANK messages:"
echo "  ✅ SUCCESS! Python is starting correctly"
echo "  → Edit submit_multipollutants_from_6pollutants.sh"
echo "  → Change to 16 nodes (128 GPUs)"
echo "  → sbatch submit_multipollutants_from_6pollutants.sh"
echo ""
echo "If VENV TEST PASSED but MAIN TEST stuck:"
echo "  → Problem is scale-dependent"
echo "  → Try 8 nodes (64 GPUs) first"
echo ""
echo "If VENV TEST FAILED:"
echo "  → Venv propagation is broken"
echo "  → Check test_venv logs for error details"
echo "  → May need Singularity container"
echo ""
echo "If both jobs still PENDING:"
echo "  → Just wait, they're in the queue"
echo "  → Run this script again in 10-15 minutes"
echo ""

echo "=============================================="
echo "For full details, read: STATUS_WHEN_YOU_WAKE_UP.md"
echo "=============================================="
