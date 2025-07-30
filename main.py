import torch.distributed as dist
import os
#!/usr/bin/env python3
"""
AQ_Net2 Project - ClimaX fine-tuning for PM2.5 prediction with Lead Time
Main entry point for training and evaluation
"""
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import sys
from src.config_manager import ConfigManager

def collate_fn(batch):
    """Custom collate function to handle lead_times"""
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    lead_times = torch.tensor([item[2] for item in batch])
    return inputs, targets, lead_times

def setup_environment(config):
    """Setup system environment variables"""
    print("Setting up system environment...")
    system_config = config.get_system_config()
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = system_config['miopen_cache_dir']
    os.environ["MIOPEN_USER_DB_PATH"] = system_config['miopen_cache_dir']
    os.environ["MIOPEN_DISABLE_CACHE"] = "0"
    os.makedirs(os.environ["MIOPEN_CUSTOM_CACHE_DIR"], exist_ok=True)
    os.environ["TIMM_FUSED_ATTN"] = "0"
    print("System environment configured")

def create_datasets(config):
    """Create train, validation and test datasets"""
    sys.path.insert(0, 'src')
    from dataloader import CAQRADataset
    
    data_config = config.get_data_config()
    
    # Training dataset
    print(f"Loading training data ({data_config['train_years']})...")
    train_ds = CAQRADataset(
        data_path=data_config['data_path'],
        years=data_config['train_years'],
        variables=data_config['variables'],
        target_variables=data_config['target_variables'],
        lead_times_hours=data_config['lead_times_hours'],
        normalize=data_config['normalize'],
        target_resolution=data_config['target_resolution']
    )
    
    # Validation dataset
    print(f"Loading validation data ({data_config['val_years']})...")
    val_ds = CAQRADataset(
        data_path=data_config['data_path'],
        years=data_config['val_years'],
        variables=data_config['variables'],
        target_variables=data_config['target_variables'],
        lead_times_hours=data_config['lead_times_hours'],
        normalize=data_config['normalize'],
        target_resolution=data_config['target_resolution']
    )
    
    # Test dataset
    print(f"Loading test data ({data_config['test_years']})...")
    test_ds = CAQRADataset(
        data_path=data_config['data_path'],
        years=data_config['test_years'],
        variables=data_config['variables'],
        target_variables=data_config['target_variables'],
        lead_times_hours=data_config['lead_times_hours'],
        normalize=data_config['normalize'],
        target_resolution=data_config['target_resolution']
    )
    
    return train_ds, val_ds, test_ds

def main():
    # DDP setup
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"DDP initialized - Rank: {dist.get_rank()}, Local rank: {local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="AQ_Net2 Project - PM2.5 Prediction with Lead Time")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Setup environment
    setup_environment(config)
    
    # Create output directories
    Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Create datasets
    train_ds, val_ds, test_ds = create_datasets(config)
    
    # Create data loaders with custom collate function
    train_config = config.get_train_config()
    system_config = config.get_system_config()
    
    train_loader = DataLoader(train_ds, batch_size=train_config['batch_size'], shuffle=True, 
                             num_workers=system_config['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=train_config['batch_size'], shuffle=False, 
                           num_workers=system_config['num_workers'], collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=train_config['batch_size'], shuffle=False, 
                            num_workers=system_config['num_workers'], collate_fn=collate_fn)
    
    print(f"Training dataset: {len(train_ds):,} samples, {len(train_loader)} batches")
    print(f"Validation dataset: {len(val_ds):,} samples, {len(val_loader)} batches")
    print(f"Test dataset: {len(test_ds):,} samples, {len(test_loader)} batches")
    
    # Create model
    from src.model import PM25Model
    model = PM25Model(config.config, device=device)
    
    # Training summary
    config.print_summary()
    
    # Start training
    from src.train import train, evaluate
    best_rmse = train(config.config, model, train_loader, val_loader, device)
    
    # Final test evaluation
    print("\nFinal test evaluation...")
    if os.path.exists('outputs/checkpoints/best_model.pt'):
        print("Loading best model...")
        model.load_state_dict(torch.load('outputs/checkpoints/best_model.pt'))
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nFINAL RESULTS:")
    print("=" * 30)
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")
    print("=" * 30)
    
    print("\nFine-tuning completed successfully!")
    print("Model saved in: outputs/checkpoints/best_model.pt")

if __name__ == '__main__':
    main()
