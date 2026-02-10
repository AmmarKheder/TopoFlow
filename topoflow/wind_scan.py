import math
import torch
import torch.nn.functional as F

from .utils.scan_orders import wind_band_hilbert


class WindScanner:

    def __init__(self, grid_h: int, grid_w: int, num_sectors: int = 16, device="cuda"):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_patches = grid_h * grid_w
        self.num_sectors = num_sectors
        self.device = device

        self.sector_orders = self._precompute_sector_orders()

    def _precompute_sector_orders(self):
        orders = {}
        for sector in range(self.num_sectors):
            angle = 2 * math.pi * sector / self.num_sectors
            order = wind_band_hilbert(self.grid_h, self.grid_w, angle)
            orders[sector] = torch.tensor(order, dtype=torch.long, device=self.device)
        return orders

    def _get_wind_sector(self, u_wind, v_wind):
        u_mean = F.avg_pool2d(
            u_wind.unsqueeze(1), kernel_size=(self.grid_h, self.grid_w)
        ).squeeze()
        v_mean = F.avg_pool2d(
            v_wind.unsqueeze(1), kernel_size=(self.grid_h, self.grid_w)
        ).squeeze()

        angle = torch.atan2(v_mean, u_mean)
        angle = (angle + 2 * math.pi) % (2 * math.pi)
        sector = (angle / (2 * math.pi) * self.num_sectors).long() % self.num_sectors
        return sector

    def reorder(self, tokens, u_wind, v_wind):
        B, V, L, D = tokens.shape
        sectors = self._get_wind_sector(u_wind, v_wind)

        reordered = torch.zeros_like(tokens)
        for b in range(B):
            sector = sectors[b].item() if sectors.dim() == 1 else sectors.item()
            order = self.sector_orders[sector]
            reordered[b] = tokens[b, :, order, :]

        return reordered

    def to(self, device):
        self.device = device
        for sector in self.sector_orders:
            self.sector_orders[sector] = self.sector_orders[sector].to(device)
        return self
