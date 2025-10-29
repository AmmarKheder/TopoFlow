#!/usr/bin/env python3
"""
Smart Training Monitor - Auto-adjust if not converging well
Monitors first 10 epochs, adjusts hyperparams if val_loss > 0.5 after epoch 10
"""

import re
import time
import subprocess
from pathlib import Path

JOB_ID = "13268747"
LOG_FILE = Path("/scratch/project_462000640/ammar/aq_net2/logs/topoflow_wind_400gpu_13268747.out")
TARGET_VAL_LOSS = 0.260
EPOCHS_TO_MONITOR = 10

def get_job_state():
    """Get current job state"""
    result = subprocess.run(
        f"sacct -j {JOB_ID} --format=State --noheader".split(),
        capture_output=True, text=True
    )
    return result.stdout.strip().split()[0] if result.stdout else "UNKNOWN"

def extract_val_loss():
    """Extract all val_loss values from log"""
    if not LOG_FILE.exists():
        return []

    content = LOG_FILE.read_text(errors='ignore')
    # Find all val_loss values
    pattern = r'val_loss[=:]?\s*([\d.]+)'
    matches = re.findall(pattern, content)
    return [float(m) for m in matches]

def extract_epoch():
    """Extract current epoch from log"""
    if not LOG_FILE.exists():
        return 0

    content = LOG_FILE.read_text(errors='ignore')
    # Find latest epoch number
    pattern = r'Epoch (\d+):'
    matches = re.findall(pattern, content)
    return int(matches[-1]) if matches else 0

def should_adjust(epoch, val_losses):
    """Decide if we need to adjust hyperparameters"""
    if epoch < EPOCHS_TO_MONITOR:
        return False, "Still monitoring..."

    if not val_losses:
        return True, "No val_loss found after 10 epochs - something wrong!"

    latest_val_loss = val_losses[-1]

    # If val_loss > 0.5 after 10 epochs, not converging well
    if latest_val_loss > 0.5:
        return True, f"val_loss={latest_val_loss:.3f} too high after {epoch} epochs"

    # If val_loss decreasing well, keep going
    if len(val_losses) >= 2:
        improvement = val_losses[-2] - val_losses[-1]
        if improvement > 0.1:
            return False, f"Good progress: val_loss decreasing by {improvement:.3f}"

    return False, f"val_loss={latest_val_loss:.3f} - on track"

def suggest_adjustments(latest_val_loss):
    """Suggest hyperparameter adjustments"""
    adjustments = []

    if latest_val_loss > 1.0:
        # Not converging at all - reduce LR
        adjustments.append({
            'param': 'learning_rate',
            'old': 0.0002,
            'new': 0.0001,
            'reason': 'Loss too high - reduce LR'
        })
    elif latest_val_loss > 0.5:
        # Converging slowly - try different LR or more epochs
        adjustments.append({
            'param': 'epochs',
            'old': 30,
            'new': 50,
            'reason': 'Need more time to converge'
        })
        adjustments.append({
            'param': 'warmup_steps',
            'old': 500,
            'new': 1000,
            'reason': 'Longer warmup for stability'
        })

    return adjustments

def monitor():
    """Main monitoring loop"""
    print("="*60)
    print("Smart Training Monitor Started")
    print(f"Target: val_loss < {TARGET_VAL_LOSS}")
    print(f"Will check after {EPOCHS_TO_MONITOR} epochs")
    print("="*60)

    while True:
        state = get_job_state()
        epoch = extract_epoch()
        val_losses = extract_val_loss()

        print(f"\n[{time.strftime('%H:%M:%S')}] Job {JOB_ID}: {state}, Epoch {epoch}")

        if val_losses:
            print(f"  Latest val_loss: {val_losses[-1]:.4f}")
            if len(val_losses) >= 2:
                trend = "↓" if val_losses[-1] < val_losses[-2] else "↑"
                print(f"  Trend: {trend} (prev: {val_losses[-2]:.4f})")

        # Check if job ended
        if state not in ["RUNNING", "PENDING"]:
            print(f"\n✅ Job ended with state: {state}")
            if val_losses:
                print(f"Final val_loss: {val_losses[-1]:.4f}")
                if val_losses[-1] < TARGET_VAL_LOSS:
                    print(f"🎉 SUCCESS! Achieved target < {TARGET_VAL_LOSS}")
                else:
                    print(f"❌ Did not reach target {TARGET_VAL_LOSS}")
            break

        # Check if adjustments needed
        need_adjust, reason = should_adjust(epoch, val_losses)
        print(f"  Status: {reason}")

        if need_adjust and val_losses:
            print("\n⚠️  ADJUSTMENTS NEEDED!")
            adjustments = suggest_adjustments(val_losses[-1])

            print("\nSuggested changes:")
            for adj in adjustments:
                print(f"  - {adj['param']}: {adj['old']} → {adj['new']}")
                print(f"    Reason: {adj['reason']}")

            print("\n📝 Save these to: RECOMMENDED_ADJUSTMENTS.txt")

            # Write to file
            with open('/scratch/project_462000640/ammar/aq_net2/RECOMMENDED_ADJUSTMENTS.txt', 'w') as f:
                f.write(f"Training Monitor Report\n")
                f.write(f"=====================\n\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Latest val_loss: {val_losses[-1]:.4f}\n")
                f.write(f"Target: {TARGET_VAL_LOSS}\n\n")
                f.write(f"RECOMMENDED ADJUSTMENTS:\n\n")
                for adj in adjustments:
                    f.write(f"{adj['param']}: {adj['old']} → {adj['new']}\n")
                    f.write(f"  Reason: {adj['reason']}\n\n")

                f.write(f"\nUpdate in: configs/config_wind_400gpu.yaml\n")
                f.write(f"Then relaunch: sbatch scripts/slurm_wind_400gpu.sh\n")

            print("Continuing to monitor...")

        time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    monitor()
