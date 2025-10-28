#!/usr/bin/env python3
"""
Create Clean RMSE Chart for TopoFlow
=====================================

Simple, professional chart showing RMSE evolution by forecast horizon
for all 6 pollutants.

Author: Ammar Kheddar
Project: TopoFlow
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from evaluation results
results = {
    'PM2.5': {
        12: 10.7224,
        24: 12.6724,
        48: 12.2103,
        96: 13.3960
    },
    'PM10': {
        12: 17.8956,
        24: 20.2899,
        48: 19.8482,
        96: 21.6250
    },
    'SO₂': {
        12: 2.8916,
        24: 2.8818,
        48: 2.8138,
        96: 3.2688
    },
    'NO₂': {
        12: 9.7501,
        24: 9.1602,
        48: 9.3257,
        96: 10.1261
    },
    'CO': {
        12: 48.9836,
        24: 48.0557,
        48: 48.6473,
        96: 54.0223
    },
    'O₃': {
        12: 24.0613,
        24: 19.9747,
        48: 21.1260,
        96: 21.4414
    }
}

def create_rmse_evolution_chart():
    """Create clean RMSE evolution chart with 6 subplots."""

    print("📊 Creating RMSE evolution chart...")

    # Create figure with 2 rows × 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('TopoFlow: RMSE Evolution by Forecast Horizon\n(Test Year 2018 - China Region)',
                 fontsize=24, fontweight='bold', y=0.98)

    # Flatten axes
    axes = axes.flatten()

    # Pollutants and colors
    pollutants = ['PM2.5', 'PM10', 'SO₂', 'NO₂', 'CO', 'O₃']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c', '#2ecc71']

    # Forecast horizons (in hours)
    horizons = [12, 24, 48, 96]

    # Create each subplot
    for idx, (pollutant, color) in enumerate(zip(pollutants, colors)):
        ax = axes[idx]

        # Get RMSE values
        rmse_values = [results[pollutant][h] for h in horizons]

        # Plot line with markers
        ax.plot(horizons, rmse_values,
                marker='o',
                linewidth=4,
                markersize=12,
                color=color,
                label=pollutant,
                markeredgecolor='white',
                markeredgewidth=2,
                zorder=3)

        # Fill area under curve
        ax.fill_between(horizons, 0, rmse_values, alpha=0.2, color=color, zorder=1)

        # Add value labels on points
        for x, y in zip(horizons, rmse_values):
            ax.text(x, y, f'{y:.2f}',
                   ha='center',
                   va='bottom',
                   fontsize=11,
                   fontweight='bold',
                   color=color,
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor=color,
                            alpha=0.8))

        # Styling
        ax.set_title(f'{pollutant}',
                    fontsize=18,
                    fontweight='bold',
                    color=color,
                    pad=15)

        ax.set_xlabel('Forecast Horizon (hours)',
                     fontsize=13,
                     fontweight='bold')

        ax.set_ylabel('RMSE (µg/m³)',
                     fontsize=13,
                     fontweight='bold',
                     color=color)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5, zorder=0)
        ax.set_axisbelow(True)

        # Set x-axis
        ax.set_xticks(horizons)
        ax.set_xticklabels(['12h', '24h', '48h', '96h'], fontsize=12)

        # Color y-axis
        ax.tick_params(axis='y', labelcolor=color, labelsize=11)
        ax.tick_params(axis='x', labelsize=12)

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0, top=max(rmse_values) * 1.15)

        # Add min/max annotations
        min_val = min(rmse_values)
        max_val = max(rmse_values)
        min_idx = rmse_values.index(min_val)
        max_idx = rmse_values.index(max_val)

        # Mark best and worst
        ax.scatter([horizons[min_idx]], [min_val],
                  s=300, marker='*', color='green',
                  edgecolors='darkgreen', linewidth=2,
                  zorder=4, label='Best')

        ax.scatter([horizons[max_idx]], [max_val],
                  s=200, marker='v', color='red',
                  edgecolors='darkred', linewidth=2,
                  zorder=4, label='Worst')

        # Legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

    # Add footer
    footer_text = (
        '📍 Region: China (11,317 active pixels, 34.5% coverage) | '
        'Resolution: 128×256 grid | '
        'Test Samples: 4,000 | '
        'Validation Loss: 0.3557'
    )
    fig.text(0.5, 0.01, footer_text,
            ha='center',
            fontsize=12,
            fontweight='bold',
            style='italic',
            color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.5',
                     facecolor='#ecf0f1',
                     alpha=0.9,
                     edgecolor='#34495e',
                     linewidth=2))

    # Tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save
    output_file = 'topoflow_rmse_evolution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ RMSE evolution chart saved: {output_file}")
    print(f"   Resolution: 300 DPI")
    print(f"   Layout: 2×3 subplots")
    print(f"   File size: ~900 KB")

    return output_file


def create_combined_rmse_chart():
    """Create single chart with all pollutants together."""

    print("\n📊 Creating combined RMSE chart...")

    fig, ax = plt.subplots(figsize=(16, 10))

    # Pollutants and colors
    pollutants = ['PM2.5', 'PM10', 'SO₂', 'NO₂', 'CO', 'O₃']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c', '#2ecc71']

    # Forecast horizons
    horizons = [12, 24, 48, 96]

    # Plot each pollutant
    for pollutant, color in zip(pollutants, colors):
        rmse_values = [results[pollutant][h] for h in horizons]

        ax.plot(horizons, rmse_values,
               marker='o',
               linewidth=4,
               markersize=14,
               color=color,
               label=pollutant,
               markeredgecolor='white',
               markeredgewidth=2.5,
               alpha=0.9)

        # Add value at 24h (most important)
        val_24h = results[pollutant][24]
        ax.text(24, val_24h, f'{val_24h:.1f}',
               ha='left',
               va='center',
               fontsize=10,
               fontweight='bold',
               color=color,
               bbox=dict(boxstyle='round,pad=0.2',
                        facecolor='white',
                        edgecolor=color,
                        alpha=0.8))

    # Styling
    ax.set_title('TopoFlow: RMSE Evolution Across All Pollutants\n(Test Year 2018 - China Region)',
                fontsize=22,
                fontweight='bold',
                pad=20)

    ax.set_xlabel('Forecast Horizon (hours)',
                 fontsize=16,
                 fontweight='bold')

    ax.set_ylabel('RMSE (µg/m³)',
                 fontsize=16,
                 fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)

    # X-axis
    ax.set_xticks(horizons)
    ax.set_xticklabels(['12h', '24h', '48h', '96h'], fontsize=14)

    # Y-axis
    ax.tick_params(axis='both', labelsize=13)
    ax.set_ylim(bottom=0)

    # Legend
    ax.legend(loc='upper left',
             fontsize=14,
             framealpha=0.95,
             edgecolor='black',
             fancybox=True,
             shadow=True,
             ncol=2)

    # Add annotation
    ax.text(0.98, 0.02,
           '⭐ Lower is Better',
           transform=ax.transAxes,
           ha='right',
           va='bottom',
           fontsize=12,
           style='italic',
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='yellow',
                    alpha=0.3))

    # Footer
    footer = '11,317 pixels | 128×256 grid | 4,000 samples | Val Loss: 0.3557'
    fig.text(0.5, 0.01, footer,
            ha='center',
            fontsize=13,
            fontweight='bold',
            color='#34495e',
            style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    output_file = 'topoflow_rmse_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ Combined RMSE chart saved: {output_file}")

    return output_file


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TOPOFLOW RMSE CHART GENERATOR")
    print("="*70)
    print()

    # Create both versions
    file1 = create_rmse_evolution_chart()
    file2 = create_combined_rmse_chart()

    print()
    print("="*70)
    print("🎉 SUCCESS! Charts created:")
    print("="*70)
    print(f"   1. {file1} (6 subplots - detailed view)")
    print(f"   2. {file2} (single chart - all pollutants)")
    print()
    print("📖 Features:")
    print("   • Clean design (RMSE only, no MAE)")
    print("   • Horizons in hours (12h, 24h, 48h, 96h)")
    print("   • Evolution curves with fill")
    print("   • Value labels on points")
    print("   • Best/worst markers")
    print("   • 300 DPI high quality")
    print()
    print("📥 Download:")
    print(f"   scp lumi:{file1} .")
    print(f"   scp lumi:{file2} .")
    print()
