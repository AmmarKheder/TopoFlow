import torch
import torch.nn as nn
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
    all_predictions = []  # Store (pred, true, lead_time) tuples
    eval_start = time.time()
    
    with torch.no_grad():
        for batch_idx, (X, y, lead_times) in enumerate(loader):
            X = X.to(device)
            y = y.to(device)
            lead_times = torch.tensor(lead_times, dtype=torch.float32).to(device)
            
            pred = model(X, lead_times)
            
            # Store predictions with their lead times
            for i in range(len(pred)):
                all_predictions.append((
                    pred[i].cpu().numpy().ravel(),
                    y[i].cpu().numpy().ravel(), 
                    float(lead_times[i])
                ))
            
            if batch_idx % 20 == 0 and batch_idx > 0:
                print(f"   Batch {batch_idx}/{len(loader)}")
    
    # Group by lead time
    by_lead_time = defaultdict(lambda: {'pred': [], 'true': []})
    for pred, true, lt in all_predictions:
        by_lead_time[lt]['pred'].extend(pred)
        by_lead_time[lt]['true'].extend(true)
    
    eval_time = time.time() - eval_start
    print(f"Evaluation results ({eval_time:.1f}s):")
    
    all_metrics = {}
    for lead_time in sorted(by_lead_time.keys()):
        y_pred = np.array(by_lead_time[lead_time]['pred'])
        y_true = np.array(by_lead_time[lead_time]['true'])
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        all_metrics[f'{int(lead_time/24)}d'] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        print(f"   {int(lead_time/24):2d} day:  RMSE={rmse:6.2f}  MAE={mae:6.2f}  R2={r2:6.3f}  (n={len(y_pred):,})")
    
    # Overall metrics
    all_pred = np.concatenate([np.array(by_lead_time[lt]['pred']) for lt in by_lead_time])
    all_true = np.concatenate([np.array(by_lead_time[lt]['true']) for lt in by_lead_time])
    overall_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    
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

def main():
    from config_manager import ConfigManager
    from dataloader import CAQRADataset
    from model import PM25Model
    
    print("Starting AQ_Net2 Training")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    print("Creating model...")
    model = PM25Model(
        config=config.config,
        device=device
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)
    
    print(f"Training for {config.num_epochs} epochs")
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples: {len(val_dataset):,}")
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
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
                'metrics': metrics
            }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
    
    print(f"\nTraining completed! Best validation RMSE: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
