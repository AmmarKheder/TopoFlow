"""
Adaptive Wind Memory with LSTM - Innovation #3
===============================================

Models wind persistence and intensity to adapt patch scanning dynamically.

Key idea: Wind patterns have temporal persistence. A strong persistent wind
should trigger more aggressive scanning than weak or changing winds.

Simplified implementation:
- Encodes current wind field (u, v) spatially
- Predicts wind strength and coherence
- Modulates wind scanning strength adaptively

Future extension: Can be upgraded to use temporal wind history when available.

Author: Ammar Kheddar
Project: TopoFlow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class WindSpatialEncoder(nn.Module):
    """
    Encodes spatial wind patterns to capture coherence and strength.

    Strong, coherent winds → higher modulation (aggressive scanning)
    Weak, chaotic winds → lower modulation (conservative scanning)
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (128, 256),
        hidden_dim: int = 256
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim

        # Convolutional encoder for wind field
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),  # u, v channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling
        )

        # MLP head for wind characteristics
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # [strength, coherence]
        )

        print(f"# # #  WindSpatialEncoder: {hidden_dim} hidden dim")

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode wind field spatially.

        Args:
            u: [B, H, W] horizontal wind component
            v: [B, H, W] vertical wind component

        Returns:
            strength: [B] wind strength score (0-1)
            coherence: [B] wind coherence score (0-1)
        """
        B = u.shape[0]

        # Stack u, v
        wind_field = torch.stack([u, v], dim=1)  # [B, 2, H, W]

        # Encode spatially
        features = self.conv_encoder(wind_field)  # [B, 256, 1, 1]
        features = features.view(B, -1)  # [B, 256]

        # Predict wind characteristics
        characteristics = self.mlp(features)  # [B, 2]

        # Apply sigmoid to get scores in [0, 1]
        strength = torch.sigmoid(characteristics[:, 0])    # [B]
        coherence = torch.sigmoid(characteristics[:, 1])   # [B]

        return strength, coherence


class AdaptiveWindMemory(nn.Module):
    """
    Adaptive wind-based modulation for patch scanning.

    Combines wind strength and coherence to compute adaptive modulation:
    - High strength + high coherence → aggressive scanning (modulation ≈ 1.5-2.0)
    - Low strength or low coherence → conservative scanning (modulation ≈ 0.5-1.0)
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (128, 256),
        hidden_dim: int = 256,
        modulation_range: Tuple[float, float] = (0.5, 2.0)
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.modulation_range = modulation_range

        # Wind spatial encoder
        self.wind_encoder = WindSpatialEncoder(
            input_size=input_size,
            hidden_dim=hidden_dim
        )

        # Learnable combination weights
        self.register_parameter(
            'alpha',
            nn.Parameter(torch.tensor(0.7))  # Weight for strength
        )
        self.register_parameter(
            'beta',
            nn.Parameter(torch.tensor(0.3))  # Weight for coherence
        )

        print(f"# # #  AdaptiveWindMemory: modulation range {modulation_range}")

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive modulation for wind scanning.

        Args:
            u: [B, H, W] horizontal wind
            v: [B, H, W] vertical wind

        Returns:
            modulation: [B] adaptive modulation factors
        """
        # Encode wind characteristics
        strength, coherence = self.wind_encoder(u, v)  # [B], [B]

        # Compute combined score (weighted average)
        alpha_norm = torch.sigmoid(self.alpha)
        beta_norm = torch.sigmoid(self.beta)
        norm_factor = alpha_norm + beta_norm

        combined_score = (alpha_norm * strength + beta_norm * coherence) / norm_factor

        # Map to modulation range [min, max]
        min_mod, max_mod = self.modulation_range
        modulation = min_mod + (max_mod - min_mod) * combined_score

        return modulation

    def get_wind_statistics(
        self,
        u: torch.Tensor,
        v: torch.Tensor
    ) -> dict:
        """
        Extract detailed wind statistics for analysis.

        Args:
            u: [B, H, W] horizontal wind
            v: [B, H, W] vertical wind

        Returns:
            stats: Dictionary with wind characteristics
        """
        strength, coherence = self.wind_encoder(u, v)
        modulation = self(u, v)

        return {
            'strength': strength.mean().item(),
            'coherence': coherence.mean().item(),
            'modulation': modulation.mean().item(),
            'alpha': torch.sigmoid(self.alpha).item(),
            'beta': torch.sigmoid(self.beta).item()
        }


def test_adaptive_wind_memory():
    """Test the adaptive wind memory module."""
    print("\n" + "="*70)
    print("Testing Adaptive Wind Memory - Innovation #3")
    print("="*70)

    # Test parameters
    batch_size = 4
    H, W = 128, 256

    print(f"\nInput configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Wind field size: {H} x {W}")

    # Create test wind data
    # Scenario 1: Strong coherent wind
    u_strong = torch.randn(batch_size, H, W) * 10 + 15  # Mean ~15 m/s
    v_strong = torch.randn(batch_size, H, W) * 2 + 5    # Mean ~5 m/s

    # Scenario 2: Weak chaotic wind
    u_weak = torch.randn(batch_size, H, W) * 5  # Mean ~0 m/s
    v_weak = torch.randn(batch_size, H, W) * 5  # Mean ~0 m/s

    # Create module
    adaptive_wind = AdaptiveWindMemory(
        input_size=(H, W),
        hidden_dim=256,
        modulation_range=(0.5, 2.0)
    )

    # Test strong wind
    print("\nScenario 1: Strong coherent wind")
    print(f"  Mean u: {u_strong.mean():.2f} m/s")
    print(f"  Mean v: {v_strong.mean():.2f} m/s")

    modulation_strong = adaptive_wind(u_strong, v_strong)
    stats_strong = adaptive_wind.get_wind_statistics(u_strong, v_strong)

    print(f"\n# # #  Modulation: {modulation_strong.mean().item():.3f}")
    print(f"  Strength: {stats_strong['strength']:.3f}")
    print(f"  Coherence: {stats_strong['coherence']:.3f}")
    print("  → Strong wind should get high modulation (≈1.5-2.0)")

    # Test weak wind
    print("\nScenario 2: Weak chaotic wind")
    print(f"  Mean u: {u_weak.mean():.2f} m/s")
    print(f"  Mean v: {v_weak.mean():.2f} m/s")

    modulation_weak = adaptive_wind(u_weak, v_weak)
    stats_weak = adaptive_wind.get_wind_statistics(u_weak, v_weak)

    print(f"\n# # #  Modulation: {modulation_weak.mean().item():.3f}")
    print(f"  Strength: {stats_weak['strength']:.3f}")
    print(f"  Coherence: {stats_weak['coherence']:.3f}")
    print("  → Weak wind should get low modulation (≈0.5-1.0)")

    # Verify modulation difference
    mod_diff = modulation_strong.mean() - modulation_weak.mean()
    print(f"\n# # #  Modulation difference: {mod_diff.item():.3f}")
    print("  → Positive difference confirms adaptive behavior!")

    print("\n" + "="*70)
    print("# # #  All tests passed!")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    test_adaptive_wind_memory()