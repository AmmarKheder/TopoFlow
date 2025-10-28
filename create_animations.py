#!/usr/bin/env python3
"""
Create Animated GIFs for TopoFlow Innovations
==============================================

Creates two animations:
1. Wind-Guided Scanning: Shows how patches are reordered by wind direction
2. Elevation-Based Attention: Shows attention modulation by terrain

Author: Ammar Kheddar
Project: TopoFlow
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

def create_wind_scanning_animation():
    """
    Create animated GIF showing wind-guided patch scanning.
    Demonstrates how patches are reordered from upwind to downwind.
    """
    print("🌬️ Creating Wind Scanning Animation...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Wind-Guided Patch Scanning Innovation',
                 fontsize=20, fontweight='bold', y=0.98)

    # Create grid
    grid_h, grid_w = 8, 12
    num_patches = grid_h * grid_w

    # Wind angles to animate (in degrees)
    wind_angles = np.linspace(0, 360, 37)[:-1]  # 36 frames (full rotation)

    # Create colormap for ordering
    cmap = plt.cm.viridis

    def update(frame):
        ax1.clear()
        ax2.clear()

        angle = wind_angles[frame]
        angle_rad = np.deg2rad(angle)

        # --- Left Panel: Standard Row-Major Order ---
        ax1.set_title('❌ Standard Row-Major Scanning\n(Arbitrary Order - No Physics)',
                     fontsize=14, fontweight='bold', color='#e74c3c')
        ax1.set_xlim(-0.5, grid_w - 0.5)
        ax1.set_ylim(-0.5, grid_h - 0.5)
        ax1.set_aspect('equal')
        ax1.invert_yaxis()

        # Draw grid with row-major order
        for idx in range(num_patches):
            row = idx // grid_w
            col = idx % grid_w

            color = cmap(idx / num_patches)
            rect = mpatches.Rectangle((col - 0.4, row - 0.4), 0.8, 0.8,
                                     facecolor=color, edgecolor='white', linewidth=2)
            ax1.add_patch(rect)

            # Add patch number
            ax1.text(col, row, str(idx+1), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')

        ax1.set_xlabel('Column Index', fontsize=12)
        ax1.set_ylabel('Row Index', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # --- Right Panel: Wind-Guided Order ---
        ax2.set_title(f'✅ Wind-Guided Scanning\n(Wind Direction: {angle:.0f}° - Physics-Aware)',
                     fontsize=14, fontweight='bold', color='#2ecc71')
        ax2.set_xlim(-0.5, grid_w - 0.5)
        ax2.set_ylim(-0.5, grid_h - 0.5)
        ax2.set_aspect('equal')
        ax2.invert_yaxis()

        # Compute wind-aligned projection for each patch
        projections = []
        for idx in range(num_patches):
            row = idx // grid_w
            col = idx % grid_w

            # Project position onto wind direction
            proj = col * np.cos(angle_rad) + row * np.sin(angle_rad)
            projections.append((proj, idx, row, col))

        # Sort by projection (upwind to downwind)
        projections.sort(key=lambda x: x[0])

        # Draw grid with wind-guided order
        for new_idx, (proj, orig_idx, row, col) in enumerate(projections):
            color = cmap(new_idx / num_patches)
            rect = mpatches.Rectangle((col - 0.4, row - 0.4), 0.8, 0.8,
                                     facecolor=color, edgecolor='white', linewidth=2)
            ax2.add_patch(rect)

            # Add new order number
            ax2.text(col, row, str(new_idx+1), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')

        # Draw wind arrow
        arrow_length = 2.5
        arrow_start_x = grid_w / 2
        arrow_start_y = grid_h / 2
        arrow_end_x = arrow_start_x + arrow_length * np.cos(angle_rad)
        arrow_end_y = arrow_start_y + arrow_length * np.sin(angle_rad)

        ax2.arrow(arrow_start_x, arrow_start_y,
                 arrow_end_x - arrow_start_x, arrow_end_y - arrow_start_y,
                 head_width=0.5, head_length=0.4, fc='red', ec='red',
                 linewidth=3, alpha=0.8, zorder=1000)

        ax2.text(arrow_end_x + 0.5 * np.cos(angle_rad),
                arrow_end_y + 0.5 * np.sin(angle_rad),
                'WIND', fontsize=12, fontweight='bold', color='red',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax2.set_xlabel('Column Index', fontsize=12)
        ax2.set_ylabel('Row Index', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add colorbar legend
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=1, vmax=num_patches))
        sm.set_array([])

        # Add description
        fig.text(0.5, 0.02,
                f'Processing Order: Dark Blue (First) → Yellow (Last) | Frame {frame+1}/36',
                ha='center', fontsize=12, fontweight='bold')

        return []

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(wind_angles),
                        interval=200, blit=True, repeat=True)

    # Save as GIF
    output_file = 'wind_scanning_animation.gif'
    writer = PillowWriter(fps=5)
    anim.save(output_file, writer=writer, dpi=120)
    plt.close()

    print(f"✅ Wind Scanning Animation saved: {output_file}")
    return output_file


def create_elevation_attention_animation():
    """
    Create animated GIF showing elevation-based attention modulation.
    Demonstrates how attention is penalized for uphill transport.
    """
    print("🏔️ Creating Elevation Attention Animation...")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    ax_terrain = fig.add_subplot(gs[0:2, 0])
    ax_attention = fig.add_subplot(gs[0:2, 1])
    ax_formula = fig.add_subplot(gs[2, :])

    fig.suptitle('Elevation-Based Attention Bias',
                 fontsize=22, fontweight='bold', y=0.98)

    # Create terrain profile
    x = np.linspace(0, 10, 100)

    # Create varied terrain
    elevation = (
        200 * np.sin(x * 0.5) +           # Large hills
        100 * np.sin(x * 1.2 + 1) +       # Medium hills
        50 * np.sin(x * 2.5 + 0.5) +      # Small hills
        500                                # Base elevation
    )
    elevation = np.maximum(elevation, 100)  # Minimum 100m

    # Sample points for attention
    num_points = 20
    sample_indices = np.linspace(0, len(x)-1, num_points, dtype=int)
    sample_x = x[sample_indices]
    sample_elevation = elevation[sample_indices]

    # Animation frames: move source point across terrain
    def update(frame):
        ax_terrain.clear()
        ax_attention.clear()
        ax_formula.clear()

        # Source point (moves across terrain)
        source_idx = frame % num_points
        source_x = sample_x[source_idx]
        source_elev = sample_elevation[source_idx]

        # --- Panel 1: Terrain Profile ---
        ax_terrain.set_title('Terrain Profile & Attention Source',
                            fontsize=14, fontweight='bold')

        # Plot terrain
        ax_terrain.fill_between(x, 0, elevation, alpha=0.6, color='#8B4513',
                               label='Terrain')
        ax_terrain.plot(x, elevation, 'k-', linewidth=2)

        # Mark sample points
        ax_terrain.scatter(sample_x, sample_elevation, s=100, c='blue',
                          alpha=0.3, zorder=5, label='Potential Targets')

        # Highlight source point
        ax_terrain.scatter([source_x], [source_elev], s=400, c='red',
                          marker='*', edgecolors='black', linewidth=2,
                          zorder=10, label='Source Patch (Query)')

        # Add horizontal line from source
        ax_terrain.axhline(y=source_elev, color='red', linestyle='--',
                          alpha=0.5, linewidth=2)

        ax_terrain.set_xlabel('Distance (km)', fontsize=12)
        ax_terrain.set_ylabel('Elevation (m)', fontsize=12)
        ax_terrain.set_ylim(0, 1000)
        ax_terrain.grid(True, alpha=0.3)
        ax_terrain.legend(loc='upper right', fontsize=10)

        # --- Panel 2: Attention Weights with Elevation Bias ---
        ax_attention.set_title('Attention Weights (with Elevation Bias)',
                              fontsize=14, fontweight='bold')

        # Compute elevation bias for all targets
        H_scale = 1000  # meters
        alpha = 2.0     # learnable parameter

        attention_weights = []
        colors = []

        for target_idx in range(num_points):
            target_elev = sample_elevation[target_idx]

            # Compute elevation difference
            delta_h = target_elev - source_elev

            # Elevation bias (penalize uphill)
            bias = -alpha * np.maximum(0, delta_h / H_scale)
            bias = np.clip(bias, -10, 0)

            # Simulate base attention (random for demo)
            base_attention = 0.5 + 0.3 * np.random.randn()

            # Final attention score (before softmax)
            attention_score = base_attention + bias

            # Softmax approximation (for visualization)
            attention_weight = np.exp(attention_score)
            attention_weights.append(attention_weight)

            # Color based on uphill/downhill
            if delta_h > 50:
                colors.append('#e74c3c')  # Red for uphill (penalized)
            elif delta_h < -50:
                colors.append('#2ecc71')  # Green for downhill (normal)
            else:
                colors.append('#f39c12')  # Orange for flat

        # Normalize attention weights
        attention_weights = np.array(attention_weights)
        attention_weights = attention_weights / attention_weights.sum()

        # Bar chart of attention weights
        bars = ax_attention.bar(range(num_points), attention_weights,
                               color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Highlight source position
        bars[source_idx].set_color('red')
        bars[source_idx].set_alpha(1.0)
        bars[source_idx].set_linewidth(3)

        ax_attention.set_xlabel('Target Patch Index', fontsize=12)
        ax_attention.set_ylabel('Attention Weight', fontsize=12)
        ax_attention.set_ylim(0, max(attention_weights) * 1.2)
        ax_attention.grid(axis='y', alpha=0.3)

        # Add legend
        red_patch = mpatches.Patch(color='#e74c3c', label='Uphill (Penalized)')
        green_patch = mpatches.Patch(color='#2ecc71', label='Downhill (Normal)')
        orange_patch = mpatches.Patch(color='#f39c12', label='Flat')
        source_patch = mpatches.Patch(color='red', label='Source (Self-Attention)')
        ax_attention.legend(handles=[red_patch, green_patch, orange_patch, source_patch],
                           loc='upper right', fontsize=10)

        # --- Panel 3: Formula Explanation ---
        ax_formula.axis('off')

        # Display formula (plain text to avoid LaTeX issues)
        formula_text = """
ELEVATION BIAS FORMULA:

bias_ij = -α × max(0, (elevation_j - elevation_i) / H_scale)

attention_ij = softmax(Q·K^T / √d + bias_ij)

PHYSICS: Pollutants struggle to climb mountains → Penalize uphill attention

KEY: Applied BEFORE softmax (additive bias) → Preserves normalization

LEARNABLE: α = 2.0 (learned during training, initialized at 0.0)
        """

        ax_formula.text(0.5, 0.5, formula_text, ha='center', va='center',
                       fontsize=13, family='serif',
                       bbox=dict(boxstyle='round,pad=1', facecolor='#ecf0f1',
                                edgecolor='black', linewidth=2))

        # Add frame counter
        fig.text(0.5, 0.01, f'Frame {frame+1}/{num_points} | Source moving across terrain',
                ha='center', fontsize=11, fontweight='bold')

        return []

    # Create animation
    anim = FuncAnimation(fig, update, frames=num_points,
                        interval=500, blit=True, repeat=True)

    # Save as GIF
    output_file = 'elevation_attention_animation.gif'
    writer = PillowWriter(fps=2)
    anim.save(output_file, writer=writer, dpi=120)
    plt.close()

    print(f"✅ Elevation Attention Animation saved: {output_file}")
    return output_file


def create_combined_animation():
    """
    Create a combined animation showing both innovations side by side.
    """
    print("🎬 Creating Combined Animation...")

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle('TopoFlow: Physics-Informed Innovations',
                 fontsize=24, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(1, 2, wspace=0.3)

    # Left: Wind Scanning
    ax_wind = fig.add_subplot(gs[0, 0])

    # Right: Elevation Attention (simplified)
    ax_elev = fig.add_subplot(gs[0, 1])

    grid_h, grid_w = 6, 8
    num_patches = grid_h * grid_w
    wind_angles = np.linspace(0, 360, 25)[:-1]  # 24 frames

    def update(frame):
        ax_wind.clear()
        ax_elev.clear()

        angle = wind_angles[frame]
        angle_rad = np.deg2rad(angle)

        # --- Wind Scanning Panel ---
        ax_wind.set_title('🌬️ Wind-Guided Scanning\n(Upwind → Downwind Order)',
                         fontsize=14, fontweight='bold', color='#3498db')
        ax_wind.set_xlim(-0.5, grid_w - 0.5)
        ax_wind.set_ylim(-0.5, grid_h - 0.5)
        ax_wind.set_aspect('equal')
        ax_wind.invert_yaxis()

        # Compute wind-guided order
        projections = []
        for idx in range(num_patches):
            row = idx // grid_w
            col = idx % grid_w
            proj = col * np.cos(angle_rad) + row * np.sin(angle_rad)
            projections.append((proj, idx, row, col))

        projections.sort(key=lambda x: x[0])

        # Draw patches
        cmap = plt.cm.plasma
        for new_idx, (proj, orig_idx, row, col) in enumerate(projections):
            color = cmap(new_idx / num_patches)
            rect = mpatches.Rectangle((col - 0.45, row - 0.45), 0.9, 0.9,
                                     facecolor=color, edgecolor='white', linewidth=1.5)
            ax_wind.add_patch(rect)

        # Draw wind arrow
        arrow_length = 2.0
        cx, cy = grid_w / 2, grid_h / 2
        dx = arrow_length * np.cos(angle_rad)
        dy = arrow_length * np.sin(angle_rad)

        ax_wind.arrow(cx, cy, dx, dy,
                     head_width=0.4, head_length=0.3, fc='red', ec='red',
                     linewidth=3, alpha=0.9, zorder=1000)

        ax_wind.text(cx + 1.5*dx, cy + 1.5*dy, 'WIND',
                    fontsize=11, fontweight='bold', color='red',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax_wind.axis('off')

        # --- Elevation Attention Panel ---
        ax_elev.set_title('🏔️ Elevation-Based Attention\n(Penalize Uphill Transport)',
                         fontsize=14, fontweight='bold', color='#e74c3c')

        # Create simple terrain
        terrain_x = np.linspace(0, 10, 50)
        terrain_y = 300 * np.sin(terrain_x * 0.8) + 500

        ax_elev.fill_between(terrain_x, 0, terrain_y, alpha=0.5, color='#8B4513')
        ax_elev.plot(terrain_x, terrain_y, 'k-', linewidth=2)

        # Show attention from point A to B
        source_idx = frame % 40
        source_x = terrain_x[source_idx]
        source_y = terrain_y[source_idx]

        # Multiple targets
        for target_idx in range(5, 45, 5):
            target_x = terrain_x[target_idx]
            target_y = terrain_y[target_idx]

            delta_h = target_y - source_y

            # Attention strength (inverse of uphill penalty)
            if delta_h > 0:
                attention = np.exp(-delta_h / 200)  # Penalize uphill
                color = '#e74c3c'  # Red
            else:
                attention = 1.0  # Normal for downhill
                color = '#2ecc71'  # Green

            # Draw attention arrow
            ax_elev.arrow(source_x, source_y, target_x - source_x, target_y - source_y,
                         head_width=30, head_length=0.3, fc=color, ec=color,
                         linewidth=2*attention, alpha=0.3 + 0.5*attention)

        # Mark source
        ax_elev.scatter([source_x], [source_y], s=400, c='blue', marker='*',
                       edgecolors='black', linewidth=2, zorder=10)

        ax_elev.set_xlim(0, 10)
        ax_elev.set_ylim(0, 900)
        ax_elev.set_xlabel('Distance', fontsize=11)
        ax_elev.set_ylabel('Elevation', fontsize=11)
        ax_elev.grid(True, alpha=0.2)

        # Legend
        ax_elev.text(0.5, 800, '← Strong (Downhill)', fontsize=10, color='#2ecc71', fontweight='bold')
        ax_elev.text(7, 800, 'Weak (Uphill) →', fontsize=10, color='#e74c3c', fontweight='bold')

        fig.text(0.5, 0.02, f'Frame {frame+1}/24 | Both innovations working together',
                ha='center', fontsize=12, fontweight='bold')

        return []

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(wind_angles),
                        interval=300, blit=True, repeat=True)

    # Save as GIF
    output_file = 'topoflow_combined_animation.gif'
    writer = PillowWriter(fps=3)
    anim.save(output_file, writer=writer, dpi=150)
    plt.close()

    print(f"✅ Combined Animation saved: {output_file}")
    return output_file


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TOPOFLOW ANIMATION GENERATOR")
    print("="*70)
    print()

    # Create all animations
    wind_file = create_wind_scanning_animation()
    print()

    elev_file = create_elevation_attention_animation()
    print()

    combined_file = create_combined_animation()
    print()

    print("="*70)
    print("🎉 ALL ANIMATIONS CREATED SUCCESSFULLY!")
    print("="*70)
    print()
    print("📁 Generated files:")
    print(f"   1. {wind_file} (Wind-guided scanning)")
    print(f"   2. {elev_file} (Elevation-based attention)")
    print(f"   3. {combined_file} (Both innovations combined)")
    print()
    print("📖 To view:")
    print("   - Copy to local machine: scp lumi:path/*.gif .")
    print("   - Open with any image viewer or web browser")
    print("   - Insert into PowerPoint: Insert → Pictures")
    print()
