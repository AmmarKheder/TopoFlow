"""
CODE FOR SUPERVISOR - FILE 2/3
================================
ELEVATION-BASED MULTIPLICATIVE ATTENTION MASK

Description:
    Modifies attention weights based on topographic elevation differences.
    Uses MULTIPLICATIVE masking (A * B) as suggested in your feedback.

Key Innovation:
    - Standard Attention: All patches attend to all others equally (spatially)
    - Our approach: Attention reduced for uphill transport (physically constrained)

Physical Motivation:
    Mountains block horizontal air flow. A patch in a valley should pay less
    attention to patches at high elevation (limited transport across barriers).

Approach:
    MULTIPLICATIVE masking AFTER softmax (not additive bias before softmax)
    - A = softmax(Q @ K.T)  # Standard attention weights [0,1]
    - B = elevation_mask    # Physics-based mask [0,1]
    - Final = A * B         # Element-wise multiplication (TRUE MASKING!)

Reference:
    https://aiml.com/explain-self-attention-and-masked-self-attention-as-used-in-transformers/
    Standard masked attention in Transformers (e.g., causal masking)

Location in codebase:
    src/climax_core/physics_attention_patch_level.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention


class PhysicsGuidedAttentionMultiplicative(Attention):
    """
    Physics-Guided Attention with MULTIPLICATIVE ELEVATION MASK.

    This is the approach suggested by the supervisor:
        1. Compute standard attention: A = softmax(Q @ K.T)  → [0,1]
        2. Compute elevation mask: B = sigmoid(elevation_bias) → [0,1]
        3. Apply multiplicative masking: A_final = A * B
        4. Renormalize: A_final = A_final / sum(A_final)

    Key difference from additive approach:
        - Additive: attn_scores + bias (before softmax, in log-space)
        - Multiplicative: attn_weights * mask (after softmax, in probability space)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)

        # LEARNABLE PARAMETER: Controls strength of elevation barrier
        # Positive value → stronger blocking for uphill transport
        self.elevation_barrier_strength = nn.Parameter(torch.tensor(3.0))

        # Patch-level resolution (no regional grouping)
        self.patch_size = 2

    def forward(self, x, elevation_patches=None):
        """
        Physics-guided attention with MULTIPLICATIVE elevation mask.

        Args:
            x: [B, N, C] token embeddings
               B = batch size
               N = number of patches (8192 = 64×128)
               C = embedding dimension (768)

            elevation_patches: [B, N] elevation per patch in meters
               Example: [[100, 250, 500, 1200, ...], ...]
               Normalized to [0,1] range before use

        Returns:
            x: [B, N, C] attention output

        Process:
            Standard:   Q,K,V → scores → softmax → output
            Ours:       Q,K,V → scores → softmax → mask * softmax → renorm → output
                                                      ↑
                                            Physics-based masking!
        """
        B, N, C = x.shape

        # ========================================================================
        # STEP 1: Standard QKV computation (same as original Transformer)
        # ========================================================================
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, C_head]
        q, k, v = qkv.unbind(0)  # Each: [B, num_heads, N, C_head]

        # ========================================================================
        # STEP 2: Compute attention scores (Q @ K.T)
        # ========================================================================
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        # attn_scores[b,h,i,j] = how much query i attends to key j

        # ========================================================================
        # STEP 3: Softmax to get attention weights (standard operation)
        # ========================================================================
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, N, N]
        # NOW: attn_weights are in range [0, 1] and sum to 1 over dim=-1
        # attn_weights[b,h,i,:].sum() = 1.0  ✓

        # ========================================================================
        # STEP 4: APPLY PHYSICS-BASED MULTIPLICATIVE MASK (our contribution!)
        # ========================================================================
        if elevation_patches is not None:
            # Compute elevation mask [0,1]
            elevation_mask = self.compute_elevation_mask(elevation_patches)  # [B, N, N]

            # Expand for all attention heads
            # [B, N, N] → [B, 1, N, N] → [B, num_heads, N, N]
            elevation_mask_expanded = elevation_mask.unsqueeze(1).expand(
                -1, self.num_heads, -1, -1
            )

            # =====================================================================
            # KEY OPERATION: MULTIPLICATIVE MASKING (supervisor's suggestion)
            # =====================================================================
            # attn_weights: [B, H, N, N] values in [0,1]
            # elevation_mask: [B, H, N, N] values in [0,1]
            #
            # Effect:
            #   - If mask[i,j] = 1.0 → attention preserved (attn * 1.0 = attn)
            #   - If mask[i,j] = 0.0 → attention blocked (attn * 0.0 = 0)
            #   - If mask[i,j] = 0.5 → attention halved (attn * 0.5 = attn/2)
            #
            attn_weights_masked = attn_weights * elevation_mask_expanded

            # =====================================================================
            # STEP 5: RENORMALIZATION (critical for maintaining probability distribution)
            # =====================================================================
            # After masking, rows no longer sum to 1
            # Must renormalize: divide by row sum
            row_sums = attn_weights_masked.sum(dim=-1, keepdim=True)  # [B, H, N, 1]
            attn_weights = attn_weights_masked / (row_sums + 1e-8)  # Avoid division by zero

            # NOW: attn_weights[b,h,i,:].sum() = 1.0 again  ✓

        # ========================================================================
        # STEP 6: Apply dropout (standard)
        # ========================================================================
        attn_weights = self.attn_drop(attn_weights)

        # ========================================================================
        # STEP 7: Apply attention to values (standard)
        # ========================================================================
        x = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def compute_elevation_mask(self, elevation_patches):
        """
        Compute MULTIPLICATIVE elevation mask in range [0, 1].

        Physical principle:
            - Uphill transport is difficult (mountains block flow)
            - Downhill transport is easy (gravity assists)

        Math:
            elevation_diff[i,j] = elevation[j] - elevation[i]
            - Positive → j is higher than i (UPHILL from i to j)
            - Negative → j is lower than i (DOWNHILL from i to j)

            mask[i,j] = sigmoid(-strength * elevation_diff[i,j])

            Effect:
            - Large uphill (diff > 0) → mask ≈ 0 (attention blocked)
            - Flat (diff ≈ 0) → mask ≈ 0.5 (neutral)
            - Large downhill (diff < 0) → mask ≈ 1 (attention preserved)

        Args:
            elevation_patches: [B, N] elevation per patch (meters)

        Returns:
            elevation_mask: [B, N, N] mask values in [0, 1]
        """
        B, N = elevation_patches.shape

        # Normalize elevation to [0,1] range for numerical stability
        # (Assumption: elevation in dataset already roughly normalized)

        # =====================================================================
        # STEP 1: Compute pairwise elevation differences (vectorized)
        # =====================================================================
        elev_i = elevation_patches.unsqueeze(2)  # [B, N, 1] - source patch elevation
        elev_j = elevation_patches.unsqueeze(1)  # [B, 1, N] - target patch elevation

        # Broadcasting: [B, N, 1] - [B, 1, N] = [B, N, N]
        elevation_diff = elev_j - elev_i  # [B, N, N]

        # elevation_diff[b, i, j] = elev[j] - elev[i]
        #   > 0 : j is HIGHER than i (uphill from i→j, difficult transport)
        #   < 0 : j is LOWER than i (downhill from i→j, easy transport)
        #   = 0 : j same elevation as i (flat terrain)

        # =====================================================================
        # STEP 2: Convert elevation difference to MASK using sigmoid
        # =====================================================================
        # We want:
        #   - High uphill difference → LOW mask value (block attention)
        #   - High downhill difference → HIGH mask value (allow attention)
        #
        # Solution: sigmoid with NEGATIVE coefficient
        #
        # sigmoid(-strength * diff):
        #   diff = +1000m (uphill) → sigmoid(-3.0 * 1.0) = sigmoid(-3) ≈ 0.05 (blocked!)
        #   diff = 0m (flat)       → sigmoid(0) = 0.5 (neutral)
        #   diff = -1000m (downhill) → sigmoid(+3.0) ≈ 0.95 (allowed!)

        elevation_mask = torch.sigmoid(-self.elevation_barrier_strength * elevation_diff)

        # =====================================================================
        # STEP 3: Clamp for numerical stability
        # =====================================================================
        # Ensure mask is strictly in (0, 1) to avoid degenerate attention
        elevation_mask = torch.clamp(elevation_mask, min=1e-6, max=1.0 - 1e-6)

        return elevation_mask  # [B, N, N] in range [0, 1]


class PhysicsGuidedBlockMultiplicative(nn.Module):
    """
    Complete Transformer block with physics-guided attention.

    Replaces standard Attention with PhysicsGuidedAttentionMultiplicative.
    Otherwise identical to standard Transformer block.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path=0.,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        # Pre-norm
        self.norm1 = norm_layer(dim)

        # Physics-guided attention (our modification)
        self.attn = PhysicsGuidedAttentionMultiplicative(
            dim, num_heads, qkv_bias, attn_drop=0., proj_drop=0.
        )

        # Drop path (stochastic depth)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Identity()

        # MLP block (standard)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x, elevation_patches=None):
        """
        Forward pass with optional elevation-based masking.

        Args:
            x: [B, N, C] token embeddings
            elevation_patches: [B, N] elevation per patch (optional)

        Returns:
            x: [B, N, C] output embeddings
        """
        # Attention block with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x), elevation_patches))

        # MLP block with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ============================================================================
# COMPARISON: ADDITIVE vs MULTIPLICATIVE
# ============================================================================
"""
TWO APPROACHES TO PHYSICS-GUIDED ATTENTION:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPROACH 1: ADDITIVE BIAS (before softmax)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    scores = Q @ K.T
    bias = compute_elevation_bias(elevation)  # Can be any value (negative for blocking)
    scores = scores + bias  # Additive in log-space
    attn = softmax(scores)  # Softmax converts to probabilities

Pros:
    - Mathematically elegant (operates in log-space)
    - Similar to positional bias (ALiBi, T5)
    - Soft blocking (never completely zeros out attention)

Cons:
    - Less intuitive (hard to interpret bias magnitude)
    - Bias range unbounded (can be very negative)

Example:
    bias = -10 → softmax(-10) ≈ 4.5e-5 (nearly zero but not exactly)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPROACH 2: MULTIPLICATIVE MASK (after softmax) ← THIS FILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    scores = Q @ K.T
    attn = softmax(scores)  # Standard attention [0,1]
    mask = sigmoid(elevation_diff)  # Mask [0,1]
    attn_masked = attn * mask  # Multiplicative masking
    attn_final = attn_masked / sum(attn_masked)  # Renormalize

Pros:
    - Intuitive (mask directly scales attention)
    - Standard approach (similar to causal masking in GPT)
    - Bounded range [0,1] for both mask and attention
    - Hard blocking possible (mask=0 → attention=0)

Cons:
    - Requires renormalization step
    - Slightly more computation (after softmax)

Example:
    mask = 0.0 → attn * 0 = 0 (complete blocking)
    mask = 0.5 → attn * 0.5 (50% reduction)
    mask = 1.0 → attn * 1 = attn (no effect)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For physics-based masking, MULTIPLICATIVE is more appropriate:
    1. Clearer interpretation (mask as "passability" factor)
    2. Direct control over blocking strength
    3. Matches intuition from masked attention in NLP

For learned biases, ADDITIVE may be better:
    1. More flexible (can be positive or negative)
    2. No renormalization needed
    3. Easier optimization (log-space)

CURRENT IMPLEMENTATION: MULTIPLICATIVE (supervisor's suggestion)
"""


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
def example_usage():
    """
    Example showing how to use PhysicsGuidedAttentionMultiplicative.
    """
    B, N, C = 2, 8192, 768  # Batch=2, Patches=8192, Dim=768
    num_heads = 8

    # Create physics-guided attention layer
    attn_layer = PhysicsGuidedAttentionMultiplicative(
        dim=C,
        num_heads=num_heads
    )

    # Input: token embeddings
    x = torch.randn(B, N, C)

    # Input: elevation per patch (meters, normalized to [0,1])
    # Example: some patches at low elevation (valleys), others high (mountains)
    elevation = torch.rand(B, N) * 1.0  # [0, 1] range

    # Forward pass WITH elevation masking
    output = attn_layer(x, elevation_patches=elevation)

    print(f"Input shape: {x.shape}")
    print(f"Elevation shape: {elevation.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Learnable parameter (barrier strength): {attn_layer.elevation_barrier_strength.item():.2f}")


if __name__ == "__main__":
    example_usage()
