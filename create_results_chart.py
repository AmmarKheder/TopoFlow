#!/usr/bin/env python3
"""
Create Professional Results Chart PNG for TopoFlow
==================================================

Creates a single PNG with 6 subplots showing RMSE results
for all pollutants across all forecast horizons.

Author: Ammar Kheddar
Project: TopoFlow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data from /scratch/project_462000640/ammar/aq_net2/logs/wind_aware_v41_rmse_13073257.out
results = {
    'PM2.5': {
        '12h': {'rmse': 10.7224, 'mae': 5.5236},
        '24h': {'rmse': 12.6724, 'mae': 5.4642},
        '48h': {'rmse': 12.2103, 'mae': 5.5340},
        '96h': {'rmse': 13.3960, 'mae': 6.3912}
    },
    'PM10': {
        '12h': {'rmse': 17.8956, 'mae': 8.9252},
        '24h': {'rmse': 20.2899, 'mae': 9.0742},
        '48h': {'rmse': 19.8482, 'mae': 9.1882},
        '96h': {'rmse': 21.6250, 'mae': 10.2679}
    },
    'SO₂': {
        '12h': {'rmse': 2.8916, 'mae': 1.4936},
        '24h': {'rmse': 2.8818, 'mae': 1.4148},
        '48h': {'rmse': 2.8138, 'mae': 1.4100},
        '96h': {'rmse': 3.2688, 'mae': 1.5637}
    },
    'NO₂': {
        '12h': {'rmse': 9.7501, 'mae': 4.4767},
        '24h': {'rmse': 9.1602, 'mae': 4.2392},
        '48h': {'rmse': 9.3257, 'mae': 4.2798},
        '96h': {'rmse': 10.1261, 'mae': 4.7187}
    },
    'CO': {
        '12h': {'rmse': 48.9836, 'mae': 27.3985},
        '24h': {'rmse': 48.0557, 'mae': 26.4823},
        '48h': {'rmse': 48.6473, 'mae': 26.8427},
        '96h': {'rmse': 54.0223, 'mae': 29.1326}
    },
    'O₃': {
        '12h': {'rmse': 24.0613, 'mae': 16.8583},
        '24h': {'rmse': 19.9747, 'mae': 14.0714},
        '48h': {'rmse': 21.1260, 'mae': 14.8734},
        '96h': {'rmse': 21.4414, 'mae': 15.1289}
    }
}

def create_results_chart():
    """Create 6-subplot chart with RMSE and MAE for all pollutants."""

    print("📊 Creating results chart...")

    # Create figure with 2 rows × 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('TopoFlow Performance: RMSE & MAE by Pollutant\n(Test Year 2018 - China Region)',
                 fontsize=22, fontweight='bold', y=0.98)

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Pollutant names and colors
    pollutants = ['PM2.5', 'PM10', 'SO₂', 'NO₂', 'CO', 'O₃']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c', '#2ecc71']

    # Horizons
    horizons = ['12h', '24h', '48h', '96h']
    horizon_x = [12, 24, 48, 96]

    # Create subplot for each pollutant
    for idx, (pollutant, color) in enumerate(zip(pollutants, colors)):
        ax = axes[idx]

        # Extract RMSE and MAE values
        rmse_values = [results[pollutant][h]['rmse'] for h in horizons]
        mae_values = [results[pollutant][h]['mae'] for h in horizons]

        # Plot RMSE (primary y-axis)
        ax.plot(horizon_x, rmse_values, marker='o', linewidth=3, markersize=10,
                color=color, label='RMSE', zorder=3)
        ax.fill_between(horizon_x, rmse_values, alpha=0.2, color=color)

        # Plot MAE (secondary y-axis)
        ax2 = ax.twinx()
        ax2.plot(horizon_x, mae_values, marker='s', linewidth=2.5, markersize=8,
                 color=color, linestyle='--', alpha=0.7, label='MAE', zorder=2)

        # Add value labels for RMSE
        for x, y in zip(horizon_x, rmse_values):
            ax.text(x, y, f'{y:.2f}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color=color)

        # Add value labels for MAE
        for x, y in zip(horizon_x, mae_values):
            ax2.text(x, y, f'{y:.2f}', ha='center', va='top',
                    fontsize=9, color=color, alpha=0.8)

        # Styling
        ax.set_title(f'{pollutant}', fontsize=16, fontweight='bold',
                    color=color, pad=10)
        ax.set_xlabel('Forecast Horizon (hours)', fontsize=11, fontweight='bold')
        ax.set_ylabel('RMSE (µg/m³)', fontsize=11, fontweight='bold', color=color)
        ax2.set_ylabel('MAE (µg/m³)', fontsize=11, fontweight='bold',
                      color=color, alpha=0.7)

        # Grid and limits
        ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
        ax.set_xlim([8, 100])
        ax.set_xscale('log')
        ax.set_xticks(horizon_x)
        ax.set_xticklabels(horizons)

        # Color y-axis labels
        ax.tick_params(axis='y', labelcolor=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                 loc='upper left', fontsize=10, framealpha=0.9)

        # Add performance indicator
        avg_rmse = np.mean(rmse_values)
        if avg_rmse < 5:
            perf = '⭐⭐⭐⭐⭐'
        elif avg_rmse < 10:
            perf = '⭐⭐⭐⭐'
        elif avg_rmse < 15:
            perf = '⭐⭐⭐'
        elif avg_rmse < 25:
            perf = '⭐⭐'
        else:
            perf = '⭐'

        ax.text(0.98, 0.02, perf, transform=ax.transAxes,
               fontsize=14, ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        alpha=0.8, edgecolor=color, linewidth=2))

    # Add footer with statistics
    footer_text = (
        '📍 Test Dataset: 4,000 samples from 2018 | '
        'Region: China (11,317 active pixels, 34.5% coverage) | '
        'Resolution: 128×256 grid | '
        'Validation Loss: 0.3557'
    )
    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=11,
            fontweight='bold', style='italic', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1',
                     alpha=0.9, edgecolor='#34495e', linewidth=1.5))

    # Tight layout
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    # Save as high-resolution PNG
    output_file = 'topoflow_results_6panels.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ Results chart saved: {output_file}")
    print(f"   Resolution: 300 DPI (high quality)")
    print(f"   Size: ~5400×3000 pixels")

    return output_file


def create_combined_bar_chart():
    """Create additional chart: grouped bar chart for all pollutants."""

    print("\n📊 Creating combined bar chart...")

    fig, ax = plt.subplots(figsize=(16, 9))

    pollutants = ['PM2.5', 'PM10', 'SO₂', 'NO₂', 'CO', 'O₃']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c', '#2ecc71']
    horizons = ['12h', '24h', '48h', '96h']

    # Bar positions
    x = np.arange(len(horizons))
    width = 0.14

    # Plot bars for each pollutant
    for i, (pollutant, color) in enumerate(zip(pollutants, colors)):
        rmse_values = [results[pollutant][h]['rmse'] for h in horizons]
        offset = (i - 2.5) * width

        bars = ax.bar(x + offset, rmse_values, width, label=pollutant,
                     color=color, alpha=0.85, edgecolor='white', linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom',
                   fontsize=8, fontweight='bold', color=color)

    # Styling
    ax.set_xlabel('Forecast Horizon', fontsize=14, fontweight='bold')
    ax.set_ylabel('RMSE (µg/m³)', fontsize=14, fontweight='bold')
    ax.set_title('TopoFlow: RMSE Comparison Across All Pollutants and Horizons\n(Test Year 2018)',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=13)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Footer
    footer = '11,317 pixels | 128×256 grid | China Region | Val Loss: 0.3557'
    fig.text(0.5, 0.02, footer, ha='center', fontsize=11, fontweight='bold',
            color='#34495e', style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])

    output_file = 'topoflow_results_grouped_bars.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ Grouped bar chart saved: {output_file}")

    return output_file


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TOPOFLOW RESULTS CHART GENERATOR")
    print("="*70)
    print()

    # Create both charts
    file1 = create_results_chart()
    file2 = create_combined_bar_chart()

    print()
    print("="*70)
    print("🎉 SUCCESS! Charts created:")
    print("="*70)
    print(f"   1. {file1} (6 subplots - detailed)")
    print(f"   2. {file2} (grouped bars - overview)")
    print()
    print("📖 Usage:")
    print("   - Insert into PowerPoint: Insert → Pictures")
    print("   - High resolution (300 DPI) for printing")
    print("   - Transparent background compatible")
    print()
    print("📥 Download:")
    print(f"   scp lumi:{file1} .")
    print(f"   scp lumi:{file2} .")
    print()
