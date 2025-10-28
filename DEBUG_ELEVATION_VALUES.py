"""
Debug script to check elevation values received in TopoFlow
"""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

from datamodule_fixed import AQNetDataModule

if __name__ == '__main__':
    # Load config
    with open('configs/config_all_pollutants.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Use single worker to avoid multiprocessing
    config['data']['num_workers'] = 0

    # Create data module
    print("Creating data module...")
    data_module = AQNetDataModule(config)
    data_module.setup('fit')

    # Get one batch
    print("Getting one batch from train dataloader...")
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    x, y, lead_times = batch
    print(f"\n=== BATCH SHAPES ===")
    print(f"x shape: {x.shape}")  # [B, n_vars, H, W]
    print(f"y shape: {y.shape}")
    print(f"lead_times shape: {lead_times.shape}")

    # Find elevation index
    variables = config['data']['variables']
    print(f"\n=== VARIABLES ===")
    print(f"Variables: {variables}")

    if 'elevation' in variables:
        elev_idx = variables.index('elevation')
        print(f"\nElevation index: {elev_idx}")

        # Extract elevation field
        elevation_field = x[:, elev_idx, :, :]  # [B, H, W]
    print(f"\n=== ELEVATION STATISTICS (as received by model) ===")
    print(f"Shape: {elevation_field.shape}")
    print(f"Min: {elevation_field.min().item():.6f}")
    print(f"Max: {elevation_field.max().item():.6f}")
    print(f"Mean: {elevation_field.mean().item():.6f}")
    print(f"Std: {elevation_field.std().item():.6f}")

    # Check if normalized
    print(f"\n=== ANALYSIS ===")
    if abs(elevation_field.mean().item()) < 0.1 and abs(elevation_field.std().item() - 1.0) < 0.2:
        print("✅ Elevation appears to be NORMALIZED (mean≈0, std≈1)")
        print("   This is the output from DataLoader normalization:")
        print("   elevation_normalized = (elevation - 1039.13) / 1931.40")
    else:
        print("⚠️  Elevation does NOT appear normalized")
        print(f"   Mean should be ≈0, got {elevation_field.mean().item():.3f}")
        print(f"   Std should be ≈1, got {elevation_field.std().item():.3f}")

    # Simulate what TopoFlow sees after avg_pool
    print(f"\n=== AFTER PATCH POOLING (patch_size=2) ===")
    from torch.nn.functional import avg_pool2d

    elev_4d = elevation_field.unsqueeze(1)  # [B, 1, H, W]
    elev_patches = avg_pool2d(elev_4d, kernel_size=2, stride=2)  # [B, 1, H/2, W/2]
    elev_patches_flat = elev_patches.squeeze(1).reshape(elevation_field.shape[0], -1)  # [B, N]

    print(f"Patches shape: {elev_patches_flat.shape}")
    print(f"Min patch elevation: {elev_patches_flat.min().item():.6f}")
    print(f"Max patch elevation: {elev_patches_flat.max().item():.6f}")
    print(f"Mean patch elevation: {elev_patches_flat.mean().item():.6f}")
    print(f"Std patch elevation: {elev_patches_flat.std().item():.6f}")

    # Simulate elevation differences
    B, N = elev_patches_flat.shape
    elev_i = elev_patches_flat.unsqueeze(2)  # [B, N, 1]
    elev_j = elev_patches_flat.unsqueeze(1)  # [B, 1, N]
    elev_diff = elev_j - elev_i  # [B, N, N]

    print(f"\n=== ELEVATION DIFFERENCES (elev_j - elev_i) ===")
    print(f"Min diff: {elev_diff.min().item():.6f}")
    print(f"Max diff: {elev_diff.max().item():.6f}")
    print(f"Mean abs diff: {elev_diff.abs().mean().item():.6f}")
    print(f"Std diff: {elev_diff.std().item():.6f}")

    # Simulate current TopoFlow normalization (WRONG)
    H_scale_current = 1000.0
    elev_diff_normalized_wrong = elev_diff / H_scale_current
    print(f"\n=== WITH CURRENT H_scale={H_scale_current} (WRONG!) ===")
    print(f"Normalized diff range: [{elev_diff_normalized_wrong.min().item():.8f}, {elev_diff_normalized_wrong.max().item():.8f}]")
    print(f"Mean abs normalized diff: {elev_diff_normalized_wrong.abs().mean().item():.8f}")
    print("❌ These values are TINY! Even with alpha=1.0, bias will be ~0.001")

    # Simulate corrected normalization
    H_scale_correct = 1.93  # std of elevation in normalized space ≈ 1931.40/1000
    elev_diff_normalized_correct = elev_diff / H_scale_correct
    print(f"\n=== WITH CORRECTED H_scale={H_scale_correct} (CORRECT!) ===")
    print(f"Normalized diff range: [{elev_diff_normalized_correct.min().item():.6f}, {elev_diff_normalized_correct.max().item():.6f}]")
    print(f"Mean abs normalized diff: {elev_diff_normalized_correct.abs().mean().item():.6f}")
    print("✅ These values are reasonable! With alpha=1.0, bias will be ~0.5")

    # Simulate bias with different alphas
    print(f"\n=== SIMULATED BIAS (with ReLU for uphill only) ===")
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        bias_wrong = -alpha * torch.relu(elev_diff_normalized_wrong)
        bias_correct = -alpha * torch.relu(elev_diff_normalized_correct)

        print(f"\nAlpha = {alpha}")
        print(f"  WRONG (H_scale=1000): bias range [{bias_wrong.min().item():.6f}, {bias_wrong.max().item():.6f}], mean_abs={bias_wrong.abs().mean().item():.6f}")
        print(f"  CORRECT (H_scale=1.93): bias range [{bias_correct.min().item():.6f}, {bias_correct.max().item():.6f}], mean_abs={bias_correct.abs().mean().item():.6f}")

    # Denormalization check
    print(f"\n=== IF WE DENORMALIZE BACK TO METERS ===")
    elevation_meters = elevation_field * 1931.40 + 1039.13
    print(f"Min elevation (meters): {elevation_meters.min().item():.1f}m")
    print(f"Max elevation (meters): {elevation_meters.max().item():.1f}m")
    print(f"Mean elevation (meters): {elevation_meters.mean().item():.1f}m")
    print(f"Range: {elevation_meters.max().item() - elevation_meters.min().item():.1f}m")

    # After denormalization, check patch diffs
    elev_meters_4d = elevation_meters.unsqueeze(1)
    elev_meters_patches = avg_pool2d(elev_meters_4d, kernel_size=2, stride=2)
    elev_meters_patches_flat = elev_meters_patches.squeeze(1).reshape(elevation_meters.shape[0], -1)

    elev_meters_i = elev_meters_patches_flat.unsqueeze(2)
    elev_meters_j = elev_meters_patches_flat.unsqueeze(1)
    elev_meters_diff = elev_meters_j - elev_meters_i

    elev_meters_diff_normalized = elev_meters_diff / 1000.0
    print(f"\nAfter denormalization, diff/1000m range: [{elev_meters_diff_normalized.min().item():.3f}, {elev_meters_diff_normalized.max().item():.3f}]")
    print(f"This matches the CORRECT approach with H_scale=1.93!")

else:
    print("\n❌ ERROR: elevation not in variables!")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("The current TopoFlow implementation divides normalized elevation")
print("(mean≈0, std≈1) by H_scale=1000.0, resulting in TINY bias values.")
print("")
print("FIX: Change H_scale from 1000.0 to 1.93 (or ~2.0)")
print("This properly scales normalized elevation differences to match the")
print("physical elevation differences divided by 1000m.")
print("="*80)
