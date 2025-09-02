

# Clear MIOpen cache to fix GPU issues
import os
os.system("rm -rf /tmp/*.ukdb /tmp/miopen* 2>/dev/null")
os.environ["MIOPEN_USER_DB_PATH"] = "/tmp/miopen_cache"
os.environ["MIOPEN_FIND_MODE"] = "1"
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
def plot_comparison_maps(sample_data, china_mask, target_vars, forecast_hours, output_dir):
    """Generate predicted vs real comparison maps - 24 maps total (6 pollutants × 4 horizons)"""
    print("\n🗺️  Generating 24 comparison maps (6 pollutants × 4 horizons)...")
    
    for horizon in forecast_hours:
        if horizon in sample_data:
            predictions = sample_data[horizon]['predictions']
            targets = sample_data[horizon]['targets']
            
            for v, var in enumerate(target_vars):
                pred_map = predictions[v].numpy()
                target_map = targets[v].numpy()
                
                # Apply mask for visualization
                mask_np = china_mask.cpu().numpy()
                pred_map_masked = np.where(mask_np, pred_map, np.nan)
                target_map_masked = np.where(mask_np, target_map, np.nan)
                
                # Calculate RMSE for this specific map
                valid_pixels = mask_np.astype(bool)
                rmse_map = np.sqrt(np.mean((pred_map[valid_pixels] - target_map[valid_pixels])**2))
                
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
                
                # Color scale
                vmin = np.nanmin([np.nanmin(pred_map_masked), np.nanmin(target_map_masked)])
                vmax = np.nanmax([np.nanmax(pred_map_masked), np.nanmax(target_map_masked)])
                
                # Predicted map
                im1 = ax1.imshow(pred_map_masked, cmap='viridis', vmin=vmin, vmax=vmax)
                ax1.set_title(f'{var.upper()} Predicted - {horizon}d\nRMSE: {rmse_map:.3f} µg/m³')
                plt.colorbar(im1, ax=ax1)
                
                # Ground truth map
                im2 = ax2.imshow(target_map_masked, cmap='viridis', vmin=vmin, vmax=vmax)
                ax2.set_title(f'{var.upper()} Ground Truth - {horizon}d')
                plt.colorbar(im2, ax=ax2)
                
                # Difference map
                diff_map = np.where(mask_np, pred_map - target_map, np.nan)
                max_diff = np.nanmax(np.abs(diff_map))
                im3 = ax3.imshow(diff_map, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
                ax3.set_title(f'Difference - {horizon}d\n(Predicted - Truth)')
                plt.colorbar(im3, ax=ax3)
                
                plt.tight_layout()
                map_file = os.path.join(output_dir, f'map_{var}_{horizon}d.png')
                plt.savefig(map_file, dpi=150, bbox_inches='tight')
                plt.close()
    
    total_maps = len(target_vars) * len(forecast_hours)
    print(f"✅ {total_maps} comparison maps generated!")

def plot_rmse_evolution(results, target_vars, forecast_hours, output_dir):
    """Generate RMSE evolution plots - one per pollutant"""
    print("\n📈 Generating RMSE evolution plots...")
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, var in enumerate(target_vars):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Extract RMSE values for this pollutant
        horizons = []
        rmse_values = []
        
        for horizon in sorted(forecast_hours):
            if horizon in results[var] and 'rmse' in results[var][horizon]:
                horizons.append(horizon)
                rmse_values.append(results[var][horizon]['rmse'])
        
        # Plot curve
        if len(horizons) > 0:
            ax.plot(horizons, rmse_values, 'o-', color=colors[i], linewidth=3, markersize=10)
            ax.set_xlabel('Forecast Horizon (hours)', fontsize=14)
            ax.set_ylabel('RMSE (µg/m³)', fontsize=14)
            ax.set_title(f'{var.upper()} - RMSE Evolution (China Region)', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(forecast_hours)
            
            # Annotate values
            for j, (h, rmse) in enumerate(zip(horizons, rmse_values)):
                ax.annotate(f'{rmse:.3f}', (h, rmse), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        rmse_file = os.path.join(output_dir, f'rmse_evolution_{var}.png')
        plt.savefig(rmse_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ 6 RMSE evolution plots saved!")

#!/usr/bin/env python3
"""
BASELINE EVALUATION COMPLETE - Epoch 9 (val_loss=0.0710) - CHINA REGION ONLY
Calculate RMSE/MAE per pollutant and horizon + Generate comparison maps
"""

import os
import sys
import numpy as np
import torch
import pytorch_lightning as pl
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config_manager import ConfigManager
from dataloader_fixed import create_dataset_fixed
from torch.utils.data import DataLoader
from model_multipollutants import MultiPollutantLightningModule
from dataloader_zarr_optimized import NORM_STATS

def denormalize_data(data, variable):
    """Denormalize according to stats"""
    if variable in NORM_STATS:
        mean, std = NORM_STATS[variable]
        return data * std + mean
    return data

def plot_comparison_maps(predictions, targets, china_mask, target_vars, forecast_hours, output_dir, batch_idx=0):
    """Generate predicted vs real comparison maps"""
    print("\n🗺️  Generating comparison maps...")
    
    # Take first sample from batch
    pred_sample = predictions[batch_idx]  # [V, H, W]
    target_sample = targets[batch_idx]    # [V, H, W]
    
    for v, var in enumerate(target_vars):
        pred_map = pred_sample[v].numpy()
        target_map = target_sample[v].numpy()
        
        # Apply mask for visualization
        mask_np = china_mask.cpu().numpy()
        pred_map_masked = np.where(mask_np, pred_map, np.nan)
        target_map_masked = np.where(mask_np, target_map, np.nan)
        
        # Calculate RMSE for this map
        valid_pixels = mask_np.astype(bool)
        rmse_map = np.sqrt(np.mean((pred_map[valid_pixels] - target_map[valid_pixels])**2))
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Maps
        vmin = np.nanmin([np.nanmin(pred_map_masked), np.nanmin(target_map_masked)])
        vmax = np.nanmax([np.nanmax(pred_map_masked), np.nanmax(target_map_masked)])
        
        im1 = ax1.imshow(pred_map_masked, cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title(f'{var.upper()} Predicted\nRMSE: {rmse_map:.3f} µg/m³')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(target_map_masked, cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_title(f'{var.upper()} Ground Truth')
        plt.colorbar(im2, ax=ax2)
        
        # Difference map
        diff_map = np.where(mask_np, pred_map - target_map, np.nan)
        im3 = ax3.imshow(diff_map, cmap='RdBu_r', vmin=-np.nanmax(np.abs(diff_map)), vmax=np.nanmax(np.abs(diff_map)))
        ax3.set_title(f'Difference\n(Predicted - Truth)')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        map_file = os.path.join(output_dir, f'map_{var}_sample.png')
        plt.savefig(map_file, dpi=150, bbox_inches='tight')
        plt.close()

def plot_rmse_evolution(results, target_vars, forecast_hours, output_dir):
    """Generate RMSE evolution plots - one per pollutant"""
    print("\n📈 Generating RMSE evolution plots...")
    
    # Create separate plot for each pollutant
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, var in enumerate(target_vars):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Extract RMSE values for this pollutant
        horizons = []
        rmse_values = []
        
        for horizon in sorted(forecast_hours):
            if horizon in results[var] and 'rmse' in results[var][horizon]:
                horizons.append(horizon)
                rmse_values.append(results[var][horizon]['rmse'])
        
        # Plot curve
        if len(horizons) > 0:
            ax.plot(horizons, rmse_values, 'o-', color=colors[i], linewidth=3, markersize=10)
            ax.set_xlabel('Forecast Horizon (hours)', fontsize=14)
            ax.set_ylabel('RMSE (µg/m³)', fontsize=14)
            ax.set_title(f'{var.upper()} - RMSE Evolution by Forecast Horizon', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(forecast_hours)
            
            # Annotate values
            for j, (h, rmse) in enumerate(zip(horizons, rmse_values)):
                ax.annotate(f'{rmse:.3f}', (h, rmse), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        rmse_file = os.path.join(output_dir, f'rmse_evolution_{var}.png')
        plt.savefig(rmse_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ {var.upper()} RMSE plot saved")

def main():
    print("🚀 BASELINE EVALUATION COMPLETE - Epoch 9")
    print("📊 6 pollutants × 4 forecast horizons × CHINA REGION ONLY")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/eval_baseline_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda")  # Force GPU usage
    print(f"Device: {device}")
    
    # Load China/Taiwan mask
    china_mask = torch.from_numpy(np.load("mask_china_taiwan_128x256.npy")).bool()
    print(f"📍 China mask loaded: {china_mask.sum()} valid pixels / {china_mask.numel()} ({china_mask.sum()/china_mask.numel()*100:.1f}%)")
    china_mask = china_mask.to(device)
    
    # Configuration
    config_manager = ConfigManager("configs/config_all_pollutants.yaml")
    config = config_manager.config
    
    # Data module
    # Create dataset with fixed indices
    test_dataset = create_dataset_fixed(config, mode="test", fixed_indices_file="data_processed/fixed_eval_indices.json")
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Load model
    print("\n🔄 Loading baseline model (epoch 9)...")
    best_checkpoint = "logs/multipollutants_climax_ddp/version_7/checkpoints/epoch_epoch=09-val_loss_val_loss=0.0710.ckpt"
    
    if os.path.exists(best_checkpoint):
        model = MultiPollutantLightningModule.load_from_checkpoint(best_checkpoint, config=config)
        print("✅ Baseline checkpoint loaded successfully")
    else:
        print("❌ Baseline checkpoint not found!")
        return
    
    model = model.to(device)
    model.eval()
    
    # Target variables
    target_vars = config["data"]["target_variables"]  # ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
    forecast_hours = config["data"]["forecast_hours"]   # [1, 3, 5, 7]
    
    print(f"🎯 Pollutants evaluated: {target_vars}")
    print(f"🎯 Horizons evaluated: {forecast_hours} hours")
    
    # Initialize accumulators per pollutant and horizon
    sum_squared_errors = {}
    sum_absolute_errors = {}
    counts = {}
    
    for var in target_vars:
        sum_squared_errors[var] = {}
        sum_absolute_errors[var] = {}
        counts[var] = {}
        for horizon in forecast_hours:
            sum_squared_errors[var][horizon] = 0.0
            sum_absolute_errors[var][horizon] = 0.0
            counts[var][horizon] = 0
    
    print("\n🔬 Starting evaluation...")
    
    # Store samples for visualization (one per horizon)
    sample_data = {}  # {horizon: {'predictions': tensor, 'targets': tensor}}
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i % 100 == 0:
                print(f"Batch {i}/{len(test_dataloader)}")
            
            # Dataloader returns (inputs, targets, lead_times)
            inputs, targets, lead_times = batch
            variables = config['data']['variables']
            inputs = inputs.to(device)
            lead_times = lead_times.to(device)
            
            predictions = model(inputs, lead_times, variables)
            
            # Store samples for each horizon (for visualization)
            predictions_cpu = predictions.cpu()
            targets_cpu = targets
            lead_times_cpu = lead_times.cpu()
            
            batch_size = predictions_cpu.shape[0]
            for b in range(batch_size):
                horizon_hours = lead_times_cpu[b].item()
                if horizon_hours in forecast_hours and horizon_hours not in sample_data:
                    # Store this sample for visualization
                    sample_pred = predictions_cpu[b].clone()  # [V, H, W]
                    sample_target = targets_cpu[b].clone()    # [V, H, W]
                    
                    # Denormalize for visualization
                    for v, var in enumerate(target_vars):
                        sample_pred[v] = torch.from_numpy(denormalize_data(sample_pred[v].numpy(), var))
                        sample_target[v] = torch.from_numpy(denormalize_data(sample_target[v].numpy(), var))
                    
                    sample_data[horizon_hours] = {
                        'predictions': sample_pred,
                        'targets': sample_target
                    }
            
            # CPU to save memory
            predictions = predictions.cpu()
            targets = targets.cpu()
            lead_times = lead_times.cpu()
            
            batch_size = predictions.shape[0]
            
            for b in range(batch_size):
                horizon_hours = lead_times[b].item()  # Already in hours
                
                if horizon_hours in forecast_hours:
                    for v, var in enumerate(target_vars):
                        # Extract data for this pollutant
                        pred_var = predictions[b, v]  # Shape: [H, W]
                        target_var = targets[b, v]    # Shape: [H, W]
                        
                        # Denormalize
                        pred_denorm = denormalize_data(pred_var, var)
                        target_denorm = denormalize_data(target_var, var)
                        
                        # Calculate errors
                        squared_error = (pred_denorm - target_denorm) ** 2
                        absolute_error = torch.abs(pred_denorm - target_denorm)
                        
                        # Apply China mask + exclude NaN/invalid values
                        china_mask_cpu = china_mask.cpu()
                        valid_mask = ~torch.isnan(squared_error) & ~torch.isnan(absolute_error) & china_mask_cpu
                        if valid_mask.sum() > 0:
                            sum_squared_errors[var][horizon_hours] += squared_error[valid_mask].sum().item()
                            sum_absolute_errors[var][horizon_hours] += absolute_error[valid_mask].sum().item()
                            counts[var][horizon_hours] += valid_mask.sum().item()
    
    # Calculate and display final metrics
    print("\n" + "="*80)
    print("📊 BASELINE RESULTS - EPOCH 9 (val_loss=0.0710) - CHINA REGION ONLY")
    print("="*80)
    
    results = {}
    
    for var in target_vars:
        print(f"\n🏷️  {var.upper()}:")
        results[var] = {}
        
        for horizon in forecast_hours:
            if counts[var][horizon] > 0:
                mse = sum_squared_errors[var][horizon] / counts[var][horizon]
                mae = sum_absolute_errors[var][horizon] / counts[var][horizon]
                rmse = np.sqrt(mse)
                
                results[var][horizon] = {
                    'rmse': rmse,
                    'mae': mae,
                    'count': counts[var][horizon]
                }
                
                print(f"   📈 {horizon}d: RMSE={rmse:.4f} µg/m³, MAE={mae:.4f} µg/m³ (n={counts[var][horizon]:,})")
            else:
                print(f"   ❌ {horizon}d: No data")
    
    print("\n" + "="*80)
    
    # Generate visualizations
    if len(sample_data) > 0:
        print("\n🎨 Generating visualizations...")
        plot_comparison_maps(sample_data, china_mask, target_vars, forecast_hours, output_dir)
        plot_rmse_evolution(results, target_vars, forecast_hours, output_dir)
    
    # Save results
    import json
    results_file = os.path.join(output_dir, "baseline_metrics.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved: {results_file}")
    print("✅ Baseline evaluation completed!")

if __name__ == "__main__":
    main()
