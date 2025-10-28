#!/usr/bin/env python3
"""
Generate visualizations for TopoFlow GitHub Pages website
Creates:
1. Wind-following scan order animation (GIF)
2. 32x32 regional grid with wind vectors
3. Elevation bias visualization
4. Architecture diagram
5. Sample predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import xarray as xr
import pickle
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

OUTPUT_DIR = Path("/scratch/project_462000640/ammar/aq_net2/docs/assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_sample_data():
    """Load a sample timestep for visualization"""
    data_path = Path("../data_processed/data_2018_china_masked.zarr")
    if data_path.exists():
        ds = xr.open_zarr(data_path, consolidated=True)
        # Get a sample timestep
        sample = ds.isel(time=100)
        return sample
    return None

def create_wind_reorder_animation():
    """Create animated GIF showing wind-following scan order"""
    print("Creating wind reorder animation...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Grid parameters
    grid_h, grid_w = 8, 16  # Simplified grid for visualization
    num_patches = grid_h * grid_w

    # Simulate wind field (NE direction for demo)
    wind_angle = 45  # degrees
    u = np.cos(np.radians(wind_angle))
    v = np.sin(np.radians(wind_angle))

    # Calculate patch coordinates
    patch_coords = []
    for i in range(num_patches):
        row = i // grid_w
        col = i % grid_w
        patch_coords.append((row, col))

    # Calculate projections along wind direction
    projections = []
    for row, col in patch_coords:
        proj = row * v + col * u  # Projection onto wind vector
        projections.append((proj, row, col))

    # Sort by projection (upwind to downwind)
    projections_sorted = sorted(projections, key=lambda x: x[0])
    wind_order = [(row, col) for _, row, col in projections_sorted]

    # Standard raster order
    raster_order = patch_coords.copy()

    def update(frame):
        for ax in axes:
            ax.clear()

        # Plot 1: Original raster order
        ax1 = axes[0]
        ax1.set_title("Standard Raster Order", fontweight='bold')
        ax1.set_xlim(-0.5, grid_w - 0.5)
        ax1.set_ylim(grid_h - 0.5, -0.5)
        ax1.set_aspect('equal')

        # Draw grid
        for i in range(grid_h + 1):
            ax1.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(grid_w + 1):
            ax1.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        # Highlight patches in raster order up to current frame
        for idx in range(min(frame + 1, num_patches)):
            row, col = raster_order[idx]
            alpha = 0.3 + 0.7 * (idx / num_patches)
            rect = Rectangle((col - 0.4, row - 0.4), 0.8, 0.8,
                            facecolor='blue', alpha=alpha, edgecolor='darkblue')
            ax1.add_patch(rect)
            if idx == frame:
                ax1.text(col, row, str(idx), ha='center', va='center',
                        color='white', fontweight='bold')

        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot 2: Wind field
        ax2 = axes[1]
        ax2.set_title(f"Wind Field (θ={wind_angle}°)", fontweight='bold')
        ax2.set_xlim(-0.5, grid_w - 0.5)
        ax2.set_ylim(grid_h - 0.5, -0.5)
        ax2.set_aspect('equal')

        # Draw grid
        for i in range(grid_h + 1):
            ax2.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(grid_w + 1):
            ax2.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        # Draw wind vectors
        for row in range(0, grid_h, 2):
            for col in range(0, grid_w, 2):
                arrow = FancyArrowPatch((col, row), (col + u*0.8, row + v*0.8),
                                       arrowstyle='->', mutation_scale=15,
                                       color='red', linewidth=2, alpha=0.7)
                ax2.add_patch(arrow)

        ax2.set_xticks([])
        ax2.set_yticks([])

        # Plot 3: Wind-following order
        ax3 = axes[2]
        ax3.set_title("Wind-Following Order (TopoFlow)", fontweight='bold')
        ax3.set_xlim(-0.5, grid_w - 0.5)
        ax3.set_ylim(grid_h - 0.5, -0.5)
        ax3.set_aspect('equal')

        # Draw grid
        for i in range(grid_h + 1):
            ax3.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(grid_w + 1):
            ax3.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        # Highlight patches in wind-following order
        for idx in range(min(frame + 1, num_patches)):
            row, col = wind_order[idx]
            alpha = 0.3 + 0.7 * (idx / num_patches)
            rect = Rectangle((col - 0.4, row - 0.4), 0.8, 0.8,
                            facecolor='green', alpha=alpha, edgecolor='darkgreen')
            ax3.add_patch(rect)
            if idx == frame:
                ax3.text(col, row, str(idx), ha='center', va='center',
                        color='white', fontweight='bold')

        ax3.set_xticks([])
        ax3.set_yticks([])

        plt.tight_layout()

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=num_patches, interval=100, repeat=True)

    # Save as GIF
    output_path = OUTPUT_DIR / "wind_reorder_demo.gif"
    anim.save(output_path, writer='pillow', fps=10)
    print(f"Saved: {output_path}")
    plt.close()

def create_regional_grid_visual():
    """Create 32x32 regional grid visualization"""
    print("Creating 32x32 regional grid visualization...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Full grid
    grid_h, grid_w = 128, 256
    regions_h, regions_w = 32, 32
    region_size_h = grid_h // regions_h
    region_size_w = grid_w // regions_w

    # Draw regional boundaries
    for i in range(0, grid_h + 1, region_size_h):
        ax.axhline(i, color='red', linewidth=1.5, alpha=0.7)
    for j in range(0, grid_w + 1, region_size_w):
        ax.axvline(j, color='red', linewidth=1.5, alpha=0.7)

    # Highlight a few example regions with different wind directions
    example_regions = [
        (10, 10, 30),   # NE wind
        (20, 20, 90),   # E wind
        (15, 5, 135),   # SE wind
    ]

    for region_i, region_j, wind_angle in example_regions:
        center_i = region_i * region_size_h + region_size_h // 2
        center_j = region_j * region_size_w + region_size_w // 2

        # Highlight region
        rect = Rectangle((region_j * region_size_w, region_i * region_size_h),
                        region_size_w, region_size_h,
                        facecolor='yellow', alpha=0.3, edgecolor='orange', linewidth=2)
        ax.add_patch(rect)

        # Draw wind arrow
        u = np.cos(np.radians(wind_angle)) * region_size_w * 0.4
        v = np.sin(np.radians(wind_angle)) * region_size_h * 0.4
        arrow = FancyArrowPatch((center_j, center_i), (center_j + u, center_i - v),
                               arrowstyle='->', mutation_scale=30,
                               color='darkred', linewidth=3)
        ax.add_patch(arrow)
        ax.text(center_j, center_i - region_size_h * 0.6, f"{wind_angle}°",
               ha='center', fontsize=10, fontweight='bold', color='darkred')

    ax.set_xlim(0, grid_w)
    ax.set_ylim(grid_h, 0)
    ax.set_aspect('equal')
    ax.set_title("32×32 Regional Grid with Dynamic Wind-Following Orders", fontsize=14, fontweight='bold')
    ax.set_xlabel("Longitude (grid cells)")
    ax.set_ylabel("Latitude (grid cells)")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='yellow', alpha=0.3, edgecolor='orange', label='Example regions'),
        Patch(facecolor='red', alpha=0, edgecolor='red', label='Regional boundaries (32×32)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "regional_grid_32x32.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_elevation_bias_visual():
    """Create elevation bias visualization"""
    print("Creating elevation bias visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Create synthetic elevation profile
    x = np.linspace(0, 100, 100)
    elevation = 500 + 1500 * np.exp(-((x - 50)**2) / 300)  # Mountain in center

    # Plot 1: Elevation profile
    ax1 = axes[0]
    ax1.fill_between(x, 0, elevation, alpha=0.7, color='brown', label='Terrain')
    ax1.plot(x, elevation, color='#8B4513', linewidth=2)
    ax1.set_xlabel("Distance (km)")
    ax1.set_ylabel("Elevation (m)")
    ax1.set_title("Terrain Profile", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Attention bias heatmap
    ax2 = axes[1]

    # Create attention bias matrix based on elevation differences
    n_patches = 20
    patch_elevations = np.interp(np.linspace(0, 100, n_patches), x, elevation)

    # Compute pairwise elevation differences
    elev_diff = np.abs(patch_elevations[:, None] - patch_elevations[None, :])

    # Attention bias: higher for similar elevations
    attention_bias = np.exp(-elev_diff / 500)  # Decay with elevation difference

    im = ax2.imshow(attention_bias, cmap='RdYlGn', aspect='auto')
    ax2.set_xlabel("Patch j")
    ax2.set_ylabel("Patch i")
    ax2.set_title("Elevation-Aware Attention Bias", fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Attention Weight Modifier")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "elevation_bias_demo.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_architecture_diagram():
    """Create simple architecture diagram"""
    print("Creating architecture diagram...")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    # Define boxes
    boxes = [
        # (x, y, width, height, label, color)
        (0.5, 1.5, 1.2, 2, "Input\n\nMeteo (u,v,T,RH,P)\nStatic (elev, pop)\nPollutants (t-1)", 'lightblue'),
        (2.2, 2, 1.5, 1.5, "Wind Scanner\n\n32×32 Regions\nDynamic Reorder", 'lightgreen'),
        (4.2, 1.5, 1.5, 2, "ViT Encoder\n\n6 Layers\nElevation Bias\nWind-Aware", 'lightyellow'),
        (6.2, 2, 1.2, 1.5, "Decoder\n\n2 Layers\nMulti-Head", 'lightcoral'),
        (7.9, 1.5, 1.5, 2, "Output\n\nPM2.5, PM10\nSO₂, NO₂\nCO, O₃", 'lightblue'),
    ]

    for x, y, w, h, label, color in boxes:
        rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
               fontsize=10, fontweight='bold')

    # Draw arrows
    arrows = [
        (1.7, 2.5, 2.2, 2.75),
        (3.7, 2.75, 4.2, 2.5),
        (5.7, 2.5, 6.2, 2.75),
        (7.4, 2.75, 7.9, 2.5),
    ]

    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=30,
                               color='black', linewidth=2)
        ax.add_patch(arrow)

    ax.set_title("TopoFlow Architecture Pipeline", fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "architecture.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def create_placeholder_predictions():
    """Create placeholder prediction visualizations"""
    print("Creating placeholder prediction images...")

    # Create synthetic data for demonstration
    for pollutant in ['pm25', 'no2']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Generate synthetic spatial field
        x = np.linspace(0, 10, 128)
        y = np.linspace(0, 10, 256)
        X, Y = np.meshgrid(y, x)

        # Ground truth
        Z_true = 50 + 30 * np.sin(X/2) * np.cos(Y/3) + np.random.randn(128, 256) * 5

        # Prediction (similar with small error)
        Z_pred = Z_true + np.random.randn(128, 256) * 3

        # Error
        Z_error = np.abs(Z_true - Z_pred)

        # Plot ground truth
        im1 = axes[0].imshow(Z_true, cmap='YlOrRd', aspect='auto')
        axes[0].set_title("Ground Truth", fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], label=f'{pollutant.upper()} (μg/m³)')

        # Plot prediction
        im2 = axes[1].imshow(Z_pred, cmap='YlOrRd', aspect='auto')
        axes[1].set_title("TopoFlow Prediction", fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], label=f'{pollutant.upper()} (μg/m³)')

        # Plot error
        im3 = axes[2].imshow(Z_error, cmap='Reds', aspect='auto')
        axes[2].set_title("Absolute Error", fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], label='|Error| (μg/m³)')

        fig.suptitle(f"{pollutant.upper()} 48h Forecast", fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"prediction_{pollutant}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

if __name__ == "__main__":
    print("Generating TopoFlow website visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")

    # Generate all visuals
    create_wind_reorder_animation()
    create_regional_grid_visual()
    create_elevation_bias_visual()
    create_architecture_diagram()
    create_placeholder_predictions()

    print("\n✓ All visualizations generated successfully!")
    print(f"Files saved to: {OUTPUT_DIR}")
