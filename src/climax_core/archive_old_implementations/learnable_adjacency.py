"""
Learnable Adjacency Matrix for Spatial Attention
=================================================

Instead of hard masking, learn soft adjacency matrix that guides attention.
Initialized with physical distance, then refined by data.

Key advantages:
- Physics-informed initialization (distance-based)
- Data-driven refinement (learnable)
- Interpretable (can visualize learned connections)
- No hard blocking (gradient flows)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSparseAdjacency(nn.Module):
    """
    Sparse learnable adjacency matrix for efficient spatial attention.

    Instead of full N×N matrix (67M params for 8192 patches), use sparse
    representation: each patch connects to K nearest neighbors only.

    Args:
        n_patches: Number of spatial patches (e.g., 64×128 = 8192)
        k_neighbors: Number of nearest neighbors to connect (default: 64)
        grid_size: (H, W) spatial grid dimensions
        init_scale: Scale for distance-based initialization
    """

    def __init__(self, n_patches, k_neighbors=64, grid_size=(64, 128), init_scale=50.0):
        super().__init__()
        self.n_patches = n_patches
        self.k_neighbors = k_neighbors
        self.grid_h, self.grid_w = grid_size

        # Compute spatial coordinates for each patch
        coords_h = torch.arange(self.grid_h, dtype=torch.float32)
        coords_w = torch.arange(self.grid_w, dtype=torch.float32)
        mesh_h, mesh_w = torch.meshgrid(coords_h, coords_w, indexing='ij')
        coords = torch.stack([mesh_h, mesh_w], dim=0)  # [2, H, W]
        self.register_buffer('patch_coords', coords.reshape(2, -1).T)  # [N, 2]

        # Compute K nearest neighbors for each patch (fixed, not learned)
        neighbor_indices = self._compute_k_nearest_neighbors(k_neighbors)
        self.register_buffer('neighbor_indices', neighbor_indices)  # [N, K]

        # Initialize adjacency values with distance-based prior
        adjacency_init = self._init_from_distances(neighbor_indices, init_scale)

        # Learnable adjacency values (only K per patch, not N!)
        self.adjacency_values = nn.Parameter(adjacency_init)  # [N, K]

        print(f"📊 Learnable Sparse Adjacency:")
        print(f"  Patches: {n_patches}")
        print(f"  K-neighbors: {k_neighbors}")
        print(f"  Parameters: {n_patches * k_neighbors:,} ({n_patches * k_neighbors / 1e6:.2f}M)")
        print(f"  vs Full matrix: {n_patches * n_patches:,} ({n_patches * n_patches / 1e6:.1f}M)")

    def _compute_k_nearest_neighbors(self, k):
        """Compute K nearest neighbors for each patch based on spatial distance."""
        N = self.n_patches
        coords = self.patch_coords  # [N, 2]

        # Compute pairwise distances [N, N]
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # [N, N, 2]
        distances = torch.norm(diff, dim=2)  # [N, N]

        # Get K nearest neighbors for each patch (including self)
        _, indices = torch.topk(distances, k, dim=1, largest=False)  # [N, K]

        return indices

    def _init_from_distances(self, neighbor_indices, scale):
        """Initialize adjacency values based on spatial distance."""
        N = self.n_patches
        K = self.k_neighbors
        coords = self.patch_coords  # [N, 2]

        # Compute distances to K neighbors
        adjacency_values = torch.zeros(N, K)

        for i in range(N):
            neighbors = neighbor_indices[i]  # [K]
            coord_i = coords[i]  # [2]
            coords_neighbors = coords[neighbors]  # [K, 2]

            # Distance to each neighbor
            distances = torch.norm(coords_neighbors - coord_i, dim=1)  # [K]

            # Convert to adjacency: close = high value, far = low value
            # exp(-d/scale): d=0 → 1.0, d=scale → 0.37, d=3*scale → 0.05
            adjacency_values[i] = torch.exp(-distances / scale)

        return adjacency_values

    def forward(self):
        """
        Returns sparse adjacency bias for attention.

        Returns:
            adjacency: [N, N] sparse tensor (actually dense for simplicity)
        """
        N = self.n_patches
        K = self.k_neighbors

        # Build full matrix from sparse values
        # (In practice, could use torch.sparse for efficiency, but dense is simpler)
        adjacency_full = torch.zeros(N, N, device=self.adjacency_values.device)

        # Fill in the K neighbors for each patch
        for i in range(N):
            neighbors = self.neighbor_indices[i]  # [K]
            values = self.adjacency_values[i]  # [K]
            adjacency_full[i, neighbors] = values

        return adjacency_full  # [N, N]

    def get_statistics(self):
        """Get statistics about learned adjacency for analysis."""
        values = self.adjacency_values  # [N, K]

        stats = {
            'mean': values.mean().item(),
            'std': values.std().item(),
            'min': values.min().item(),
            'max': values.max().item(),
            'num_strong_connections': (values > 0.5).sum().item(),
            'num_weak_connections': (values < 0.1).sum().item(),
        }

        return stats
