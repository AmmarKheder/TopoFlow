#!/usr/bin/env python3
"""
Simple RMSE Charts - Clean and Minimal
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
results = {
    'PM2.5': [10.72, 12.67, 12.21, 13.40],
    'PM10':  [17.90, 20.29, 19.85, 21.63],
    'SO₂':   [2.89, 2.88, 2.81, 3.27],
    'NO₂':   [9.75, 9.16, 9.33, 10.13],
    'CO':    [48.98, 48.06, 48.65, 54.02],
    'O₃':    [24.06, 19.97, 21.13, 21.44]
}

horizons = [12, 24, 48, 96]
pollutants = ['PM2.5', 'PM10', 'SO₂', 'NO₂', 'CO', 'O₃']
colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c', '#2ecc71']

# ===== GRAPH 1: 6 subplots simple =====
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('RMSE by Forecast Horizon', fontsize=20, fontweight='bold')
axes = axes.flatten()

for i, (pol, color) in enumerate(zip(pollutants, colors)):
    ax = axes[i]
    values = results[pol]

    # Simple line plot
    ax.plot(horizons, values, 'o-', color=color, linewidth=3, markersize=10)

    # Title
    ax.set_title(pol, fontsize=16, fontweight='bold', color=color)

    # Labels
    ax.set_xlabel('Hours', fontsize=12)
    ax.set_ylabel('RMSE (µg/m³)', fontsize=12)

    # Grid
    ax.grid(True, alpha=0.3)

    # Values on points
    for x, y in zip(horizons, values):
        ax.text(x, y*1.05, f'{y:.1f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('simple_rmse_6panels.png', dpi=200, bbox_inches='tight')
print("✅ Created: simple_rmse_6panels.png")
plt.close()

# ===== GRAPH 2: All in one =====
fig, ax = plt.subplots(figsize=(12, 8))

for pol, color in zip(pollutants, colors):
    values = results[pol]
    ax.plot(horizons, values, 'o-', color=color, linewidth=3,
            markersize=10, label=pol)

ax.set_title('RMSE Evolution', fontsize=20, fontweight='bold')
ax.set_xlabel('Forecast Horizon (hours)', fontsize=14, fontweight='bold')
ax.set_ylabel('RMSE (µg/m³)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_rmse_all.png', dpi=200, bbox_inches='tight')
print("✅ Created: simple_rmse_all.png")
plt.close()

# ===== GRAPH 3: Bar chart =====
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(horizons))
width = 0.13

for i, (pol, color) in enumerate(zip(pollutants, colors)):
    values = results[pol]
    offset = (i - 2.5) * width
    ax.bar(x + offset, values, width, label=pol, color=color)

ax.set_title('RMSE by Pollutant and Horizon', fontsize=20, fontweight='bold')
ax.set_xlabel('Forecast Horizon', fontsize=14, fontweight='bold')
ax.set_ylabel('RMSE (µg/m³)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['12h', '24h', '48h', '96h'])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('simple_rmse_bars.png', dpi=200, bbox_inches='tight')
print("✅ Created: simple_rmse_bars.png")
plt.close()

print("\n🎉 Done! 3 simple graphs created:")
print("   1. simple_rmse_6panels.png - 6 subplots")
print("   2. simple_rmse_all.png - all together")
print("   3. simple_rmse_bars.png - bar chart")
