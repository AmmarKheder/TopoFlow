#!/usr/bin/env python3
"""
Fast Evaluation Script for TopoFlow
===================================

Evaluates model on 1000 samples per forecast horizon for quick validation.
Useful for ablation studies and rapid iteration.

Usage:
    python scripts/evaluate_fast.py --checkpoint path/to/model.ckpt --config configs/config.yaml

Author: Ammar Kheddar
Project: TopoFlow
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_multipollutants import MultiPollutantLightningModule
from src.datamodule_fixed import AQNetDataModule


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_fast(
    checkpoint_path: str,
    config_path: str,
    num_samples: int = 1000,
    output_dir: str = "experiments/fast_eval"
):
    """
    Fast evaluation on limited samples.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        num_samples: Number of samples to evaluate per horizon
        output_dir: Directory to save results
    """
    print("\n" + "="*70)
    print("TOPOFLOW FAST EVALUATION")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Samples per horizon: {num_samples}")
    print("="*70 + "\n")

    # Load config
    config = load_config(config_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = MultiPollutantLightningModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        strict=False
    )
    model.eval()
    model = model.cuda()

    # Create datamodule
    print("Loading data...")
    datamodule = AQNetDataModule(config)
    datamodule.setup('test')

    # Get test dataloader
    test_loader = datamodule.test_dataloader()

    # Evaluation metrics storage
    forecast_hours = config['data']['forecast_hours']
    target_variables = config['data']['target_variables']

    results = {
        'overall': {'rmse': [], 'mae': []},
        'by_horizon': {h: {'rmse': [], 'mae': []} for h in forecast_hours},
        'by_pollutant': {pol: {'rmse': [], 'mae': []} for pol in target_variables}
    }

    # Evaluate on limited samples
    print(f"\nEvaluating on {num_samples} samples...")
    samples_evaluated = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if samples_evaluated >= num_samples:
                break

            # Unpack batch
            if len(batch) == 3:
                x, y, lead_times = batch
                variables = model.model.variables
            elif len(batch) == 4:
                x, y, lead_times, variables = batch
            else:
                x, y, lead_times, variables, _ = batch

            # Move to GPU
            x = x.cuda()
            y = y.cuda()
            lead_times = lead_times.cuda()

            # Forward pass
            y_pred = model.model(x, lead_times, variables)

            # Compute metrics
            valid_mask = (y != -999) & torch.isfinite(y)

            # Overall RMSE and MAE
            if valid_mask.sum() > 0:
                mse = ((y_pred - y) ** 2 * valid_mask).sum() / valid_mask.sum()
                rmse = torch.sqrt(mse).item()
                mae = (torch.abs(y_pred - y) * valid_mask).sum() / valid_mask.sum()
                mae = mae.item()

                results['overall']['rmse'].append(rmse)
                results['overall']['mae'].append(mae)

            # Per-pollutant metrics
            for i, pol in enumerate(target_variables):
                if i < y_pred.shape[1]:
                    y_pred_pol = y_pred[:, i:i+1]
                    y_true_pol = y[:, i:i+1]
                    valid_pol = valid_mask[:, i:i+1] if valid_mask.shape[1] > i else valid_mask[:, 0:1]

                    if valid_pol.sum() > 0:
                        mse_pol = ((y_pred_pol - y_true_pol) ** 2 * valid_pol).sum() / valid_pol.sum()
                        rmse_pol = torch.sqrt(mse_pol).item()
                        mae_pol = (torch.abs(y_pred_pol - y_true_pol) * valid_pol).sum() / valid_pol.sum()

                        results['by_pollutant'][pol]['rmse'].append(rmse_pol)
                        results['by_pollutant'][pol]['mae'].append(mae_pol.item())

            samples_evaluated += x.shape[0]

    # Compute final averages
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    overall_rmse = np.mean(results['overall']['rmse'])
    overall_mae = np.mean(results['overall']['mae'])

    print(f"\nOverall Performance:")
    print(f"  RMSE: {overall_rmse:.4f}")
    print(f"  MAE:  {overall_mae:.4f}")

    print(f"\nPer-Pollutant Results:")
    for pol in target_variables:
        if results['by_pollutant'][pol]['rmse']:
            rmse_pol = np.mean(results['by_pollutant'][pol]['rmse'])
            mae_pol = np.mean(results['by_pollutant'][pol]['mae'])
            print(f"  {pol.upper()}: RMSE={rmse_pol:.4f}, MAE={mae_pol:.4f}")

    print("\n" + "="*70)

    # Save results
    results_file = Path(output_dir) / f"fast_eval_results_{Path(checkpoint_path).stem}.yaml"
    results_summary = {
        'checkpoint': str(checkpoint_path),
        'config': str(config_path),
        'samples_evaluated': samples_evaluated,
        'overall_rmse': float(overall_rmse),
        'overall_mae': float(overall_mae),
        'by_pollutant': {
            pol: {
                'rmse': float(np.mean(results['by_pollutant'][pol]['rmse'])) if results['by_pollutant'][pol]['rmse'] else 0,
                'mae': float(np.mean(results['by_pollutant'][pol]['mae'])) if results['by_pollutant'][pol]['mae'] else 0
            }
            for pol in target_variables
        }
    }

    with open(results_file, 'w') as f:
        yaml.dump(results_summary, f, default_flow_style=False)

    print(f"\n# # #  Results saved to: {results_file}")
    print("="*70 + "\n")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Fast evaluation for TopoFlow")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to evaluate')
    parser.add_argument('--output_dir', type=str, default='experiments/fast_eval',
                        help='Output directory for results')

    args = parser.parse_args()

    evaluate_fast(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()