"""
Physics-Guided Attention - PATCH LEVEL Implementation
=====================================================

Modification par patch au lieu de par r√©gion :
1. Chaque patch a sa propre elevation (pas de moyennage r√©gional)
2. Calcul direct patch-√# -patch sans regroupement
3. Granularit√© maximale pour l'attention bas√©e sur l'elevation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from timm.models.vision_transformer import Block, Attention


class PhysicsGuidedAttentionPatchLevel(Attention):
    """
    Physics-Guided Attention - Version PATCH LEVEL
    
    L'elevation agit au niveau de chaque patch individuel :
    - R√©solution maximale : chaque patch 2√# 2 pixels a sa propre elevation
    - Calcul direct de masque patch-√# -patch
    - Plus pr√©cis mais potentiellement plus co√ªteux
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        
        # Param√# tre physique learnable pour la force de la barri√# re
        self.elevation_barrier_strength = nn.Parameter(torch.tensor(3.0))
        
        # Configuration patch level - pas de regroupement en r√©gions
        self.patch_size = 2  # Consistant avec l'architecture ClimaX
        
    def forward(self, x, elevation_patches=None):
        """
        Physics-guided attention forward pass avec masque par patch.
        
        Args:
            x: [B, N, C] where N = number of patches
            elevation_patches: [B, N] elevation per patch (not per region!)
        """
        B, N, C = x.shape
        
        # Standard Q, K, V computation
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Standard attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        
        # Standard softmax # Üí attention weights normalis√©s [0,1]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, N, N]
        
        # # # # #  PHYSICS-GUIDED PATCH-LEVEL MULTIPLICATIVE MASK
        if elevation_patches is not None:
            elevation_mask = self.compute_patch_elevation_mask(elevation_patches)  # [B, N, N]
            
            # Expand mask for all heads
            elevation_mask_expanded = elevation_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            # # # #  MULTIPLICATION pour masquage physique
            attn_weights_masked = attn_weights * elevation_mask_expanded  # [B, H, N, N]
            
            # Re-normalisation pour que chaque ligne somme √#  1
            attn_weights = attn_weights_masked / (attn_weights_masked.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Dropout et application
        attn_weights = self.attn_drop(attn_weights)
        
        # Apply attention to values
        x = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    def compute_patch_elevation_mask(self, elevation_patches):
        """
        Compute elevation mask - VERSION PATCH LEVEL
        
        Args:
            elevation_patches: [B, N] elevation per patch (d√©j√#  normalis√© [0,1])
            
        Returns:
            elevation_mask: [B, N, N] mask values # # #  [0,1]
        """
        B, N = elevation_patches.shape
        
        # 1. Compute elevation differences (vectorized outer difference)
        elev_i = elevation_patches.unsqueeze(2)  # [B, N, 1] - patch source
        elev_j = elevation_patches.unsqueeze(1)  # [B, 1, N] - patch destination
        
        # 2. Diff√©rence directionnelle : positif = mont√©e (i# Üíj)
        elevation_diff = elev_j - elev_i  # [B, N, N]
        
        # 3. # # # #  MASQUE MULTIPLICATIF avec sigmoid
        # Transport montant # Üí masque faible (pr√# s de 0) = attention r√©duite
        # Transport descendant # Üí masque fort (pr√# s de 1) = attention pr√©serv√©e
        elevation_mask = torch.sigmoid(-self.elevation_barrier_strength * elevation_diff)
        
        # 4. Clamping pour stabilit√© num√©rique
        elevation_mask = torch.clamp(elevation_mask, min=1e-6, max=1.0)
        
        return elevation_mask  # [B, N, N] # # #  [0,1]


class PhysicsGuidedBlockPatchLevel(nn.Module):
    """
    Block wrapper pour physics-guided attention au niveau patch.
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = PhysicsGuidedAttentionPatchLevel(dim, num_heads, qkv_bias, 0., 0.)
        
        self.drop_path1 = nn.Identity() if drop_path == 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        self.drop_path2 = nn.Identity() if drop_path == 0. else nn.Identity()
    
    def forward(self, x, elevation_patches=None):
        """Forward pass with patch-level elevation."""
        x = x + self.drop_path1(self.attn(self.norm1(x), elevation_patches))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class ElevationPatchProcessor:
    """
    Processor pour calculer l'elevation par patch (pas par r√©gion).
    """
    
    @staticmethod
    def compute_patch_elevations(elevation_field, patch_size=2):
        """
        Compute elevation per patch directly.
        
        Args:
            elevation_field: [B, H, W] elevation data
            patch_size: int, size of each patch (default: 2)
            
        Returns:
            patch_elevations: [B, N] where N = (H//patch_size) * (W//patch_size)
        """
        B, H, W = elevation_field.shape
        device = elevation_field.device
        
        # Nombre de patches
        patches_h = H // patch_size  # 64 pour 128√# 256
        patches_w = W // patch_size  # 128 pour 128√# 256
        num_patches = patches_h * patches_w  # 8192
        
        # Calcul efficace par unfold
        patches = elevation_field.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        # patches: [B, patches_h, patches_w, patch_size, patch_size]
        
        # Moyenne par patch
        patch_elevations = patches.mean(dim=[3, 4])  # [B, patches_h, patches_w]
        patch_elevations = patch_elevations.view(B, -1)  # [B, num_patches]
        
        # Normalisation per batch [0, 1]
        min_elev = patch_elevations.min(dim=1, keepdim=True)[0]
        max_elev = patch_elevations.max(dim=1, keepdim=True)[0]
        
        range_elev = max_elev - min_elev
        range_elev = torch.where(range_elev > 1e-6, range_elev, torch.ones_like(range_elev))
        
        patch_elevations = (patch_elevations - min_elev) / range_elev
        
        return patch_elevations  # [B, N] # # #  [0,1]
    
    @staticmethod 
    def get_patch_coordinates(patch_idx, patches_h, patches_w):
        """
        Get 2D coordinates of a patch from its linear index.
        
        Args:
            patch_idx: int, linear patch index
            patches_h: int, number of patches in height 
            patches_w: int, number of patches in width
            
        Returns:
            (row, col): tuple of 2D coordinates
        """
        row = patch_idx // patches_w
        col = patch_idx % patches_w
        return row, col
    
    @staticmethod
    def get_patch_distance(patch1_idx, patch2_idx, patches_h, patches_w):
        """
        Compute spatial distance between two patches.
        Utile pour ajouter un biais de distance si n√©cessaire.
        """
        row1, col1 = ElevationPatchProcessor.get_patch_coordinates(patch1_idx, patches_h, patches_w)
        row2, col2 = ElevationPatchProcessor.get_patch_coordinates(patch2_idx, patches_h, patches_w)
        
        return math.sqrt((row2 - row1)**2 + (col2 - col1)**2)


def test_patch_level_physics_attention():
    """Test the patch-level physics-guided attention."""
    print("# # # #  Testing PATCH-LEVEL Physics-Guided Attention...")
    
    # Test parameters - ClimaX standard
    batch_size = 2
    H, W = 128, 256  # Image size
    patch_size = 2
    embed_dim = 768
    num_heads = 8
    
    # Calculs des patches
    patches_h = H // patch_size  # 64
    patches_w = W // patch_size  # 128
    num_patches = patches_h * patches_w  # 8192
    
    print(f"# # # #  Configuration:")
    print(f"   Image: {H}√# {W}")
    print(f"   Patch size: {patch_size}√# {patch_size}")
    print(f"   Patches: {patches_h}√# {patches_w} = {num_patches}")
    
    # Create test data
    x = torch.randn(batch_size, num_patches, embed_dim)
    elevation_field = torch.rand(batch_size, H, W)  # Random elevation field
    
    # Compute patch-level elevations
    patch_elevations = ElevationPatchProcessor.compute_patch_elevations(elevation_field, patch_size)
    
    print(f"# # #  Patch elevations shape: {patch_elevations.shape}")
    print(f"# # #  Elevation range: [{patch_elevations.min():.3f}, {patch_elevations.max():.3f}]")
    
    # Test patch-level physics-guided attention
    physics_attn = PhysicsGuidedAttentionPatchLevel(embed_dim, num_heads)
    
    print("# # # #  Forward pass...")
    output = physics_attn(x, patch_elevations)
    
    print(f"# # #  Input shape: {x.shape}")
    print(f"# # #  Output shape: {output.shape}")
    print(f"# # #  Barrier strength: {physics_attn.elevation_barrier_strength.item():.2f}")
    print("# # #  Patch-level physics-guided attention works!")
    
    # Test distance computation
    dist = ElevationPatchProcessor.get_patch_distance(0, num_patches-1, patches_h, patches_w)
    print(f"# # #  Distance from corner to corner: {dist:.1f} patches")
    
    return True


if __name__ == "__main__":
    test_patch_level_physics_attention()
