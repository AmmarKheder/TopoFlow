import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple

from .wind_scan import WindScanner


class WindPatchEmbed(nn.Module):

    def __init__(
        self,
        num_vars: int,
        img_size: tuple,
        patch_size: int,
        embed_dim: int,
        enable_wind_scan: bool = True,
        u_var_idx: int = 0,
        v_var_idx: int = 1,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.enable_wind_scan = enable_wind_scan
        self.u_var_idx = u_var_idx
        self.v_var_idx = v_var_idx
        self.grid_h, self.grid_w = self.grid_size

        self.proj_weights = nn.Parameter(
            torch.empty(num_vars, embed_dim, 1, *self.patch_size)
        )
        self.proj_biases = nn.Parameter(torch.empty(num_vars, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

        self.wind_scanner = None
        self._init_weights()

    def _init_weights(self):
        for idx in range(self.num_vars):
            nn.init.kaiming_uniform_(self.proj_weights[idx], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.proj_weights[idx])
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.proj_biases[idx], -bound, bound)

    def _ensure_wind_scanner(self, device):
        if self.wind_scanner is None:
            self.wind_scanner = WindScanner(self.grid_h, self.grid_w, device=device)

    def forward(self, x, var_ids=None):
        B, C, H, W = x.shape
        if var_ids is None:
            var_ids = list(range(self.num_vars))

        weights = self.proj_weights[var_ids].flatten(0, 1)
        biases = self.proj_biases[var_ids].flatten()
        groups = len(var_ids)

        proj = F.conv2d(x, weights, biases, groups=groups, stride=self.patch_size)
        proj = proj.reshape(B, groups, -1, *proj.shape[-2:])
        proj = proj.flatten(3).transpose(2, 3)
        proj = self.norm(proj)

        if self.enable_wind_scan and self.u_var_idx in var_ids and self.v_var_idx in var_ids:
            u_idx = list(var_ids).index(self.u_var_idx) if hasattr(var_ids, '__iter__') else self.u_var_idx
            v_idx = list(var_ids).index(self.v_var_idx) if hasattr(var_ids, '__iter__') else self.v_var_idx

            u_wind = x[:, u_idx, :, :]
            v_wind = x[:, v_idx, :, :]

            self._ensure_wind_scanner(proj.device)
            proj = self.wind_scanner.reorder(proj, u_wind, v_wind)

        return proj
