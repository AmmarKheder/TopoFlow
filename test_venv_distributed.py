#!/usr/bin/env python3
"""
Minimal test to verify venv propagates correctly with srun
"""
import os
import sys

print("=" * 60)
print("🧪 VENV PROPAGATION TEST")
print("=" * 60)

# Basic environment info
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")
print(f"SLURM_PROCID: {os.environ.get('SLURM_PROCID', 'NOT SET')}")
print(f"SLURM_LOCALID: {os.environ.get('SLURM_LOCALID', 'NOT SET')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")

# Test PyTorch import
try:
    import torch
    print(f"✅ PyTorch imported: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA device count: {torch.cuda.device_count()}")
        print(f"✅ Current CUDA device: {torch.cuda.current_device()}")
except ImportError as e:
    print(f"❌ FAILED to import PyTorch: {e}")
    sys.exit(1)

# Test PyTorch Lightning import
try:
    import pytorch_lightning as pl
    print(f"✅ Lightning imported: {pl.__version__}")
except ImportError as e:
    print(f"❌ FAILED to import Lightning: {e}")
    sys.exit(1)

# Test distributed setup
try:
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    torch.cuda.set_device(local_rank)
    print(f"✅ Bound to GPU {local_rank}")
    print(f"✅ Device name: {torch.cuda.get_device_name(local_rank)}")
except Exception as e:
    print(f"⚠️  Could not bind to GPU: {e}")

print("=" * 60)
print("🎉 ALL TESTS PASSED - venv is working!")
print("=" * 60)
