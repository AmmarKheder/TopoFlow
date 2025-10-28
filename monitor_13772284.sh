#!/bin/bash
# Monitor job 13772284

echo "=== Job Status ==="
squeue -j 13772284

echo ""
echo "=== Last 20 lines of output ==="
tail -20 /scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_13772284.out

echo ""
echo "=== Last 10 lines of errors (excluding amdgpu.ids warnings) ==="
tail -100 /scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_13772284.err | grep -v "amdgpu.ids" | tail -10

echo ""
echo "=== File sizes ==="
ls -lh /scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_13772284.*
