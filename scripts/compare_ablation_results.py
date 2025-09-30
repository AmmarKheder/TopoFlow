#!/usr/bin/env python3
"""
Compare Ablation Study Results
==============================

Analyzes and compares results from different TopoFlow configurations.

Usage:
    python scripts/compare_ablation_results.py --results_dir experiments/fast_eval

Author: Ammar Kheddar
Project: TopoFlow
"""

import argparse
import yaml
from pathlib import Path
import sys


def load_results(results_dir: str):
    """Load all result YAML files from directory."""
    results_dir = Path(results_dir)
    results = {}

    for yaml_file in results_dir.glob("*.yaml"):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            # Extract model name from filename
            model_name = yaml_file.stem.replace('fast_eval_results_', '')
            results[model_name] = data

    return results


def compare_results(results: dict):
    """Compare and print ablation results."""
    print("\n" + "="*80)
    print("TOPOFLOW ABLATION STUDY RESULTS")
    print("="*80)

    if not results:
        print("No results found!")
        return

    # Print header
    print(f"\n{'Model':<40} {'Overall RMSE':<15} {'Overall MAE':<15}")
    print("-"*80)

    # Sort by RMSE
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('overall_rmse', float('inf')))

    baseline_rmse = None

    for model_name, data in sorted_results:
        rmse = data.get('overall_rmse', 0)
        mae = data.get('overall_mae', 0)

        # Track baseline for improvement calculation
        if 'baseline' in model_name.lower() or baseline_rmse is None:
            baseline_rmse = rmse

        # Calculate improvement
        if baseline_rmse and baseline_rmse > 0:
            improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
            improvement_str = f"({improvement:+.1f}%)"
        else:
            improvement_str = ""

        print(f"{model_name:<40} {rmse:<15.4f} {mae:<15.4f} {improvement_str}")

    print("-"*80)

    # Print per-pollutant comparison
    print("\n" + "="*80)
    print("PER-POLLUTANT RESULTS (RMSE)")
    print("="*80)

    # Get all pollutants
    all_pollutants = set()
    for data in results.values():
        all_pollutants.update(data.get('by_pollutant', {}).keys())

    for pol in sorted(all_pollutants):
        print(f"\n{pol.upper()}:")
        print(f"{'Model':<40} {'RMSE':<15}")
        print("-"*60)

        for model_name, data in sorted_results:
            pol_data = data.get('by_pollutant', {}).get(pol, {})
            rmse_pol = pol_data.get('rmse', 0)
            print(f"{model_name:<40} {rmse_pol:<15.4f}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    best_model = sorted_results[0][0]
    best_rmse = sorted_results[0][1].get('overall_rmse', 0)

    if baseline_rmse:
        total_improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
        print(f"Best Model: {best_model}")
        print(f"Best RMSE: {best_rmse:.4f}")
        print(f"Total Improvement over Baseline: {total_improvement:+.1f}%")
    else:
        print(f"Best Model: {best_model}")
        print(f"Best RMSE: {best_rmse:.4f}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare ablation study results")
    parser.add_argument('--results_dir', type=str, default='experiments/fast_eval',
                        help='Directory containing result YAML files')

    args = parser.parse_args()

    results = load_results(args.results_dir)
    compare_results(results)


if __name__ == "__main__":
    main()