from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, trunc_normal_

from .topoflow import TopoFlowBlock, compute_patch_coords, compute_patch_elevations
from .wind_embed import WindPatchEmbed
from .utils.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class TopoFlowModel(nn.Module):

    def __init__(
        self,
        variables: list,
        img_size: tuple = (128, 256),
        patch_size: int = 2,
        embed_dim: int = 768,
        depth: int = 6,
        decoder_depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        drop_rate: float = 0.1,
        enable_wind_scan: bool = True,
        enable_topoflow: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = tuple(variables)
        self.enable_wind_scan = enable_wind_scan
        self.enable_topoflow = enable_topoflow

        self.patch_embed = WindPatchEmbed(
            num_vars=len(variables),
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_wind_scan=enable_wind_scan,
        )
        self.num_patches = self.patch_embed.num_patches

        self.var_embed = nn.Parameter(
            torch.zeros(1, len(variables), embed_dim), requires_grad=True
        )
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
        )
        self.lead_time_embed = nn.Linear(1, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i == 0 and self.enable_topoflow:
                self.blocks.append(
                    TopoFlowBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=drop_rate,
                        drop_path=dpr[i],
                    )
                )
            else:
                self.blocks.append(
                    Block(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        drop_path=dpr[i],
                        norm_layer=nn.LayerNorm,
                    )
                )
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, len(self.variables) * patch_size**2))
        self.head = nn.Sequential(*self.head)

        self._init_weights()

    def _init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(
            self.var_embed.shape[-1], np.arange(len(self.variables))
        )
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        for i in range(len(self.patch_embed.proj_weights)):
            w = self.patch_embed.proj_weights[i].data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @lru_cache(maxsize=None)
    def _get_var_ids(self, vars_tuple, device):
        all_vars = list(self.variables)
        ids = np.array([all_vars.index(var) for var in vars_tuple])
        return torch.from_numpy(ids).to(device)

    def _unpatchify(self, x: torch.Tensor):
        p = self.patch_size
        c = len(self.variables)
        h = self.img_size[0] // p
        w = self.img_size[1] // p

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def _aggregate_variables(self, x: torch.Tensor):
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)
        x = x.squeeze()
        return x.unflatten(dim=0, sizes=(b, l))

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)

        elevation_field = None
        if self.enable_topoflow and "elevation" in variables:
            elev_idx = variables.index("elevation")
            elevation_field = x[:, elev_idx, :, :]

        var_ids = self._get_var_ids(variables, x.device)
        x = self.patch_embed(x, var_ids)

        var_embed = self.var_embed[:, var_ids, :]
        x = x + var_embed.unsqueeze(2)
        x = self._aggregate_variables(x)

        x = x + self.pos_embed
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1)).unsqueeze(1)
        x = x + lead_time_emb
        x = self.pos_drop(x)

        coords_2d = None
        elevation_patches = None
        if self.enable_topoflow:
            coords_2d = compute_patch_coords(self.img_size, self.patch_size, x.device)
            coords_2d = coords_2d.expand(x.shape[0], -1, -1)
            if elevation_field is not None:
                elevation_patches = compute_patch_elevations(
                    elevation_field, self.patch_size
                )

        for i, blk in enumerate(self.blocks):
            if i == 0 and self.enable_topoflow and elevation_patches is not None:
                x = blk(x, coords_2d, elevation_patches)
            else:
                x = blk(x)

        return self.norm(x)

    def forward(self, x, lead_times, variables, target_variables=None):
        out = self.forward_encoder(x, lead_times, variables)
        preds = self.head(out)
        preds = self._unpatchify(preds)

        if target_variables is not None:
            if isinstance(variables, list):
                variables = tuple(variables)
            target_ids = self._get_var_ids(tuple(target_variables), preds.device)
            preds = preds[:, target_ids]

        return preds
