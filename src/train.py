import torch
import sys
import torch.nn as nn
import sys
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from collections import defaultdict

def evaluate(model, loader, device):
    print("Evaluating...")
    model.eval()
    eval_start = time.time()
    
    # Accumulate metrics by lead time using streaming approach
    by_lead_time = defaultdict(lambda: {
        'sum_squared_error': 0.0,
        'sum_absolute_error': 0.0,
        'sum_true': 0.0,
        'sum_pred': 0.0,
        'sum_true_squared': 0.0,
        'sum_pred_squared': 0.0,
        'sum_true_pred': 0.0,
        'count': 0
    })
    
    # Overall metrics accumulators
    overall_stats = {
        'sum_squared_error': 0.0,
        'sum_absolute_error': 0.0,
        'sum_true': 0.0,
        'sum_pred': 0.0,
        'sum_true_squared': 0.0,
        'sum_pred_squared': 0.0,
        'sum_true_pred': 0.0,
        'count': 0
    }
    
    with torch.no_grad():
        for batch_idx, (X, y, lead_times) in enumerate(loader):
            X = X.to(device)
            y = y.to(device)
            lead_times = torch.tensor(lead_times, dtype=torch.float32).to(device)
            
            pred = model(X, lead_times)
            
            # Process each sample in the batch
            for i in range(len(pred)):
                pred_flat = pred[i].cpu().numpy().ravel()
                true_flat = y[i].cpu().numpy().ravel()
                lt = float(lead_times[i])
                
                # Calculate streaming statistics for this lead time
                stats = by_lead_time[lt]
                stats['sum_squared_error'] += np.sum((pred_flat - true_flat) ** 2)
                stats['sum_absolute_error'] += np.sum(np.abs(pred_flat - true_flat))
                stats['sum_true'] += np.sum(true_flat)
                stats['sum_pred'] += np.sum(pred_flat)
                stats['sum_true_squared'] += np.sum(true_flat ** 2)
                stats['sum_pred_squared'] += np.sum(pred_flat ** 2)
                stats['sum_true_pred'] += np.sum(true_flat * pred_flat)
                stats['count'] += len(pred_flat)
                
                # Update overall statistics
                overall_stats['sum_squared_error'] += np.sum((pred_flat - true_flat) ** 2)
                overall_stats['sum_absolute_error'] += np.sum(np.abs(pred_flat - true_flat))
                overall_stats['sum_true'] += np.sum(true_flat)
                overall_stats['sum_pred'] += np.sum(pred_flat)
                overall_stats['sum_true_squared'] += np.sum(true_flat ** 2)
                overall_stats['sum_pred_squared'] += np.sum(pred_flat ** 2)
                overall_stats['sum_true_pred'] += np.sum(true_flat * pred_flat)
                overall_stats['count'] += len(pred_flat)
            
            if batch_idx % 20 == 0 and batch_idx > 0:
                print(f"   Batch {batch_idx}/{len(loader)}")
    
    eval_time = time.time() - eval_start
    print(f"Evaluation results ({eval_time:.1f}s):")
    
    # Calculate final metrics for each lead time
    all_metrics = {}
    for lead_time in sorted(by_lead_time.keys()):
        stats = by_lead_time[lead_time]
        n = stats['count']
        
        # Calculate metrics from accumulated statistics
        mae = stats['sum_absolute_error'] / n
        rmse = np.sqrt(stats['sum_squared_error'] / n)
        
        # R² calculation: R² = 1 - (SS_res / SS_tot)
        mean_true = stats['sum_true'] / n
        ss_tot = stats['sum_true_squared'] - n * (mean_true ** 2)
        ss_res = stats['sum_squared_error']
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        all_metrics[f'{int(lead_time/24)}d'] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        print(f"   {int(lead_time/24):2d} day:  RMSE={rmse:6.2f}  MAE={mae:6.2f}  R2={r2:6.3f}  (n={n:,})")
    
    # Overall metrics
    overall_rmse = np.sqrt(overall_stats['sum_squared_error'] / overall_stats['count'])
    
    print(f"   Overall: RMSE={overall_rmse:6.2f}")
    return overall_rmse, all_metrics
def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (X, y, lead_times) in enumerate(loader):
        X = X.to(device)
        y = y.to(device) 
        lead_times = torch.tensor(lead_times, dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        pred = model(X, lead_times)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print(f"   Epoch {epoch}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}, Time: {elapsed:.1f}s")
    
    return total_loss / len(loader)

def resume_from_checkpoint(model, optimizer, checkpoint_dir="checkpoints", force_fresh=False):
    import glob
    if force_fresh:
        print("Starting fresh training (--fresh-start flag used)")
        return 0, None, float("inf")
    checkpoint_files = glob.glob(f"{checkpoint_dir}/checkpoint_epoch_*.pth")
    if not checkpoint_files:
        print("No checkpoint found. Starting fresh training.")
        return 0, None, float("inf")
    
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", checkpoint["val_loss"])
    metrics = checkpoint["metrics"]
    
    print(f"Resumed from epoch {checkpoint['epoch']}, best val loss: {best_val_loss:.4f}")
    return start_epoch, metrics, best_val_loss

def main():
    from config_manager import ConfigManager
    from dataloader import CAQRADataset
    from model import PM25Model
    
    print("Starting AQ_Net2 Training")

    # Check for fresh start flag
    force_fresh = "--fresh-start" in sys.argv
    if force_fresh:
        print("Fresh start requested - ignoring existing checkpoints")
    
    # Load config
    config = ConfigManager("configs/config.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = CAQRADataset(
        data_path=config.data_path,
        years=config.train_years,
        variables=config.input_variables,
        target_variables=config.target_variables,
        lead_times_hours=config.lead_times_hours,
        normalize=config.normalize,
        target_resolution=config.target_resolution
    )
    
    val_dataset = CAQRADataset(
        data_path=config.data_path,
        years=config.val_years,
        variables=config.input_variables,
        target_variables=config.target_variables,
        lead_times_hours=config.lead_times_hours,
        normalize=config.normalize,
        target_resolution=config.target_resolution
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=getattr(config, "eval_batch_size", config.batch_size), shuffle=False, num_workers=2)
    
    # Create model
    print("Creating model...")
    model = PM25Model(
        config=config.config,
        device=device
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # Resume from checkpoint if available
    start_epoch, metrics, best_val_loss = resume_from_checkpoint(model, optimizer, force_fresh=force_fresh)
    if best_val_loss == float("inf"):
        best_val_loss = float("inf")
    
    print(f"Training for {config.num_epochs} epochs (starting from epoch {start_epoch+1})")
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples: {len(val_dataset):,}")
    
    
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{config.num_epochs} ===")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch+1)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, metrics = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"New best model saved (RMSE: {val_loss:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'metrics': metrics
            }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    print(f"\nTraining completed! Best validation RMSE: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
