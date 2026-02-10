import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .model import TopoFlowModel


class TopoFlowLightningModule(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = TopoFlowModel(
            variables=config["data"]["variables"],
            img_size=config["model"]["img_size"],
            patch_size=config["model"]["patch_size"],
            embed_dim=config["model"]["embed_dim"],
            depth=config["model"]["depth"],
            decoder_depth=config["model"]["decoder_depth"],
            num_heads=config["model"]["num_heads"],
            mlp_ratio=config["model"]["mlp_ratio"],
            drop_path=config["model"].get("drop_path", 0.1),
            drop_rate=config["model"].get("drop_rate", 0.1),
            enable_wind_scan=config["model"].get("parallel_patch_embed", True),
            enable_topoflow=config["model"].get("use_physics_mask", True),
        )

        self.variables = config["data"]["variables"]
        self.target_variables = config["data"]["target_variables"]
        self.register_buffer("china_mask", self._create_mask())

    def _create_mask(self):
        H, W = self.config["model"]["img_size"]
        mask_path = self.config.get("data", {}).get("china_mask_path")

        if mask_path:
            try:
                m = np.load(mask_path).astype(np.float32)
                return torch.from_numpy((m > 0.5).astype(np.float32))
            except:
                pass

        mask = torch.zeros(H, W, dtype=torch.float32)
        mask[30:100, 45:180] = 1.0
        return mask

    def _masked_mse(self, pred, target, mask):
        mask = mask.expand_as(pred)
        diff = (pred - target) ** 2
        denom = mask.sum()
        if denom == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return (diff * mask).sum() / denom

    def forward(self, x, lead_times, variables):
        return self.model(x, lead_times, variables, self.target_variables)

    def training_step(self, batch, batch_idx):
        x, y, lead_times = batch[:3]
        variables = batch[3] if len(batch) > 3 else self.variables

        pred = self(x, lead_times, variables)
        if y.size(1) != pred.size(1):
            y = y[:, : pred.size(1)]

        mask = self.china_mask.to(dtype=torch.bool, device=y.device).unsqueeze(0).unsqueeze(0)
        loss = self._masked_mse(pred, y, mask)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, lead_times = batch[:3]
        variables = batch[3] if len(batch) > 3 else self.variables

        pred = self(x, lead_times, variables)
        if y.size(1) != pred.size(1):
            y = y[:, : pred.size(1)]

        mask = self.china_mask.to(dtype=torch.bool, device=y.device).unsqueeze(0).unsqueeze(0)
        loss = self._masked_mse(pred, y, mask)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, lead_times = batch[:3]
        variables = batch[3] if len(batch) > 3 else self.variables

        pred = self(x, lead_times, variables)
        mask = (y != -999) & torch.isfinite(y)

        mse = self._masked_mse(pred, y, mask)
        rmse = torch.sqrt(mse)

        self.log("test_loss", mse, prog_bar=True, sync_dist=True)
        self.log("test_rmse", rmse, prog_bar=True, sync_dist=True)
        return {"test_loss": mse, "test_rmse": rmse}

    def configure_optimizers(self):
        lr = self.config["train"]["learning_rate"]
        wd = self.config["train"]["weight_decay"]

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["train"].get("epochs", 100),
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
