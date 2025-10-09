"""
Wrapper pour appliquer physics bias à un bloc standard.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsBiasedBlock(nn.Module):
    """
    Wrapper qui ajoute physics bias à un bloc transformer standard.
    """

    def __init__(self, block):
        super().__init__()
        # CRITICAL: Use object.__setattr__ to avoid registering as submodule
        # This prevents infinite recursion in .apply() while still allowing manual device transfer
        object.__setattr__(self, '_block', block)
        object.__setattr__(self, '_original_attn', block.attn)

        # Replace attention avec version modifiée
        block.attn = self._create_biased_attention(block.attn)

        self.physics_bias = None  # Set externally before forward

    @property
    def block(self):
        return object.__getattribute__(self, '_block')

    def to(self, *args, **kwargs):
        """Override to() to manually transfer _block."""
        super().to(*args, **kwargs)
        # Manually move _block since it's not registered as submodule
        block = object.__getattribute__(self, '_block')
        block.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        """Override cuda() to manually transfer _block."""
        super().cuda(device)
        block = object.__getattribute__(self, '_block')
        block.cuda(device)
        return self

    def cpu(self):
        """Override cpu() to manually transfer _block."""
        super().cpu()
        block = object.__getattribute__(self, '_block')
        block.cpu()
        return self

    def _create_biased_attention(self, original_attn):
        """Create attention that accepts physics bias."""

        class BiasedAttention(nn.Module):
            def __init__(self, attn, wrapper):
                super().__init__()
                self.attn = attn
                self.wrapper = wrapper

            def forward(self, x):
                B, N, C = x.shape

                # Standard QKV
                qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)

                # Attention scores
                attn = (q @ k.transpose(-2, -1)) * self.attn.scale

                # Add physics bias if available
                if self.wrapper.physics_bias is not None:
                    # physics_bias: [B, N, N] -> expand to [B, num_heads, N, N]
                    physics_bias_expanded = self.wrapper.physics_bias.unsqueeze(1).expand(-1, self.attn.num_heads, -1, -1)
                    attn = attn + physics_bias_expanded

                attn = attn.softmax(dim=-1)
                attn = self.attn.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.attn.proj(x)
                x = self.attn.proj_drop(x)
                return x

        return BiasedAttention(original_attn, self)

    def forward(self, x, physics_bias=None):
        """Forward with optional physics bias."""
        # CRITICAL: Ensure _block is on same device as input
        block = object.__getattribute__(self, '_block')
        if x.device != next(block.parameters()).device:
            block.to(x.device)

        self.physics_bias = physics_bias
        return block(x)


def test_wrapper():
    """Test physics bias wrapper."""
    from timm.models.vision_transformer import Block

    print("Testing PhysicsBiasedBlock...")

    # Create standard block
    block = Block(768, 8, 4.0, qkv_bias=True)

    # Wrap it
    biased_block = PhysicsBiasedBlock(block)

    # Test data
    B, N, D = 2, 100, 768
    x = torch.randn(B, N, D)
    bias = torch.randn(B, N, N) * 0.1

    # Forward without bias
    out1 = biased_block(x)
    print(f"Output without bias: {out1.shape}")

    # Forward with bias
    out2 = biased_block(x, physics_bias=bias)
    print(f"Output with bias: {out2.shape}")

    print("✅ Wrapper works!")

if __name__ == "__main__":
    test_wrapper()
