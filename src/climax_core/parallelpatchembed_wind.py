import math
import torch
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple
from torch import nn

# Import wind scanning functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wind_scanning_cached import apply_cached_wind_reordering, CachedWindScanning


def _get_conv2d_weights(in_channels, out_channels, kernel_size):
    weight = torch.empty(out_channels, in_channels, *kernel_size)
    return weight


def _get_conv2d_biases(out_channels):
    bias = torch.empty(out_channels)
    return bias


class ParallelVarPatchEmbedWind(nn.Module):
    """
    Variable to Patch Embedding with Wind-Following Scanning.
    
    Key innovation: Reorder patches according to wind direction (upwind # †’ downwind)
    Physical inductive bias for atmospheric transport modeling.
    
    Args:
        max_vars (int): Maximum number of variables
        img_size (int): Image size
        patch_size (int): Patch size
        embed_dim (int): Embedding dimension
        enable_wind_scan (bool): Enable wind-following patch scanning
        u_var_idx (int): Index of u (horizontal wind) variable
        v_var_idx (int): Index of v (vertical wind) variable
        norm_layer (nn.Module, optional): Normalization layer
        flatten (bool, optional): Flatten the output
    """

    def __init__(self, max_vars: int, img_size, patch_size, embed_dim, 
                 enable_wind_scan=True, u_var_idx=0, v_var_idx=1,
                 norm_layer=None, flatten=True):
        super().__init__()
        self.max_vars = max_vars
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        
        # Wind scanning parameters
        self.enable_wind_scan = enable_wind_scan
        self.u_var_idx = u_var_idx  # Index of u wind component
        self.v_var_idx = v_var_idx  # Index of v wind component
        self.grid_h = self.grid_size[0]  # Number of patch rows
        self.grid_w = self.grid_size[1]  # Number of patch columns
        
        print(f"# # # # # # #  Wind-Following Patch Embedding initialized:")
        print(f"   # # # # # #  Grid size: {self.grid_size} ({self.num_patches} patches)")
        print(f"   # # # # # #  Wind scan enabled: {self.enable_wind_scan}")
        print(f"   # # # # # #  U wind var index: {u_var_idx}")
        print(f"   # # # # # #  V wind var index: {v_var_idx}")
        
        # # # # # # # #  Wind scanner cache (per-instance, no global state)
        self.wind_scanner = None  # Lazy initialization
        
        # # # # # # # #  Wind scanner cache (per-instance, no global state)
        self.wind_scanner = None  # Lazy initialization

        # Standard parallel patch embedding weights
        grouped_weights = torch.stack(
            [_get_conv2d_weights(1, embed_dim, self.patch_size) for _ in range(max_vars)], dim=0
        )
        self.proj_weights = nn.Parameter(grouped_weights)
        grouped_biases = torch.stack([_get_conv2d_biases(embed_dim) for _ in range(max_vars)], dim=0)
        self.proj_biases = nn.Parameter(grouped_biases)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.reset_parameters()

    def _ensure_wind_scanner(self, L_expected, num_sectors=16):
        """Ensure wind scanner is initialized for current grid configuration."""
        if (self.wind_scanner is None or 
            self.wind_scanner.grid_h != self.grid_h or
            self.wind_scanner.grid_w != self.grid_w or
            self.wind_scanner.num_patches != L_expected):
            
            print(f"# # # #  Initializing wind scanner cache for {self.grid_h}x{self.grid_w} grid ({L_expected} patches)")
            # CPU-only initialization to avoid device conflicts
            self.wind_scanner = CachedWindScanning(self.grid_h, self.grid_w, num_sectors=num_sectors)
    
    def reset_parameters(self):
        for idx in range(self.max_vars):
            nn.init.kaiming_uniform_(self.proj_weights[idx], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.proj_weights[idx])
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.proj_biases[idx], -bound, bound)

    def forward(self, x, vars=None):
        """
        Forward pass with optional wind-following patch reordering.
        
        Args:
            x: [B, C, H, W] input tensor
            vars: Variable indices to process
        
        Returns:
            proj: [B, V, L, D] patch embeddings (L=num_patches, potentially reordered)
        """
        B, C, H, W = x.shape
        if vars is None:
            vars = list(range(self.max_vars))
        
        # Standard parallel patch embedding
        weights = self.proj_weights[vars].flatten(0, 1)
        biases = self.proj_biases[vars].flatten()
        groups = len(vars)
        
        proj = F.conv2d(x, weights, biases, groups=groups, stride=self.patch_size)
        
        if self.flatten:
            proj = proj.reshape(B, groups, -1, *proj.shape[-2:])  # [B, V, embed_dim, grid_h, grid_w]
            proj = proj.flatten(3).transpose(2, 3)  # [B, V, L, D] where L = grid_h * grid_w
        
        proj = self.norm(proj)
        
        # Apply wind-following reordering if enabled
        if self.enable_wind_scan and self.u_var_idx in vars and self.v_var_idx in vars:
            try:
                # Extract wind components from original input
                u_wind = x[:, self.u_var_idx, :, :]  # [B, H, W]
                v_wind = x[:, self.v_var_idx, :, :]  # [B, H, W]
                
                # Ensure wind scanner is initialized
                B, V, L, D = proj.shape
                assert self.grid_h * self.grid_w == L, f"Grid {self.grid_h}x{self.grid_w} => {self.grid_h*self.grid_w} != L={L}"
                
                self._ensure_wind_scanner(L, num_sectors=16)

                device = proj.device  # Get device from input tensor
                # CRITICAL: Pre-load cache to GPU
                if hasattr(self.wind_scanner, "_move_cache_to_gpu"):
                    self.wind_scanner._move_cache_to_gpu(device)

                
                # Apply wind reordering to patch tokens  
                proj = apply_cached_wind_reordering(proj, u_wind, v_wind, self.grid_h, self.grid_w, self.wind_scanner, regional_mode="32x32")
                
            except Exception as e:
                print(f"# # # # # #  Wind reordering failed: {e}")
                print("   Falling back to standard row-major ordering")
                # Continue with original proj if wind reordering fails
        
        return proj


# Compatibility alias
ParallelVarPatchEmbed = ParallelVarPatchEmbedWind
