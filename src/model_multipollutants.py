import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import List, Optional, Union, Tuple

from .climax_core.arch import ClimaX


def ensure_tuple(variables):
    """
    Convertit récursivement les listes en tuples pour rendre compatible avec @lru_cache
    """
    if isinstance(variables, list):
        return tuple(ensure_tuple(item) for item in variables)
    elif isinstance(variables, tuple):
        # Si c'est un tuple de listes répétitives comme (['u', 'u', ...], ['v', 'v', ...])
        # On le convertit en tuple simple de variables uniques  
        if len(variables) > 0 and isinstance(variables[0], list):
            # Prendre la première variable de chaque liste
            return tuple(var_list[0] if isinstance(var_list, list) and len(var_list) > 0 else var_list for var_list in variables)
        return variables
    return variables


class MultiPollutantModel(nn.Module):
    """
    Multi-Pollutant Model basé sur ClimaX pour prédire 6 polluants simultanément
    Supporte: PM2.5, PM10, SO2, NO2, CO, O3
    """
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.config = config
        self.variables = tuple(config["data"]["variables"])
        self.target_variables = tuple(config["data"]["target_variables"])
        self.img_size = config["model"]["img_size"]
        self.patch_size = config["model"]["patch_size"]
        
        print(f"🎯 Target variables: {self.target_variables}")
        print(f"📊 Input variables: {self.variables}")
        
        # Architecture ClimaX complète
        self.climax = ClimaX(
            default_vars=self.variables,
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=config["model"]["embed_dim"],
            depth=config["model"]["depth"],
            decoder_depth=config["model"]["decoder_depth"],
            num_heads=config["model"]["num_heads"],
            mlp_ratio=config["model"]["mlp_ratio"],
        )
        
        # Index des polluants cibles dans la liste des variables
        self.target_indices = []
        for target in self.target_variables:
            if target in self.variables:
                idx = self.variables.index(target)
                self.target_indices.append(idx)
                print(f"✅ {target} found at index {idx}")
            else:
                raise ValueError(f"Target variable '{target}' not found in input variables")
                
        self.target_indices = torch.tensor(self.target_indices)
        print(f"🎯 Target indices: {self.target_indices.tolist()}")

    def forward(self, x, lead_times, variables, out_variables=None):
        """
        Forward pass pour multi-polluants
        Args:
            x: [B, V, H, W] - batch, variables, height, width
            lead_times: [B, T] - lead times pour chaque sample
            variables: liste des noms de variables d'entrée
        Returns:
            [B, N_targets, H, W] où N_targets = nombre de polluants cibles
        """
        if out_variables is None:
            out_variables = self.variables
            
        # Forward pass complet de ClimaX
        # Utiliser la vraie fonction ensure_tuple comme l'ancien modèle
        out_transformers = self.climax.forward_encoder(x, lead_times, ensure_tuple(variables))
        preds = self.climax.head(out_transformers)  # [B, L, V*p*p]
        preds = self.climax.unpatchify(preds)  # [B, V, H, W]
        
        # Extraire les polluants cibles
        target_preds = preds[:, self.target_indices, :, :]  # [B, N_targets, H, W]
        
        return target_preds


class MultiPollutantLightningModule(pl.LightningModule):
    """
    Module PyTorch Lightning pour l'entraînement multi-polluants
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = MultiPollutantModel(config)
        self.target_variables = config["data"]["target_variables"]
        self.n_targets = len(self.target_variables)
        
        # Métriques de suivi
        self.train_losses = []
        self.val_losses = []
        
        print(f"🏗️ Multi-Pollutant Lightning Module initialized")
        print(f"🎯 Targets: {self.target_variables}")
        print(f"📊 Number of targets: {self.n_targets}")

        # Masque Chine (gardé pour compatibilité)
        self.register_buffer("china_mask", self._create_china_mask())

    def _create_china_mask(self):
        """
        Crée un masque pour la région Chine
        """
        H, W = self.config["model"]["img_size"]
        
        # Région approximative de la Chine sur la grille 128x256
        # Longitude: 70E-140E (colonnes ~45-180 sur 256)
        # Latitude: 15N-55N (lignes ~30-100 sur 128)
        mask = torch.zeros(H, W)
        mask[30:100, 45:180] = 1.0
        
        print(f"🇨🇳 China mask created: {mask.sum().item():.0f}/{H*W} pixels")
        return mask

    def forward(self, x, lead_times, variables, out_variables=None):
        return self.model(x, lead_times, tuple(variables) if isinstance(variables, list) else variables, out_variables)
    
    def training_step(self, batch, batch_idx):
        # Auto-detect batch format
        if len(batch) == 4:
            x, y, lead_times, variables = batch
        else:
            x, y, lead_times = batch
            variables = self.config["data"]["variables"]
            
        y_pred = self(x, lead_times, variables)  # [B, N_targets, H, W]
        
        # Assurer que y a la bonne dimension
        if y.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
            y = y.unsqueeze(1)
        elif y.dim() == 4 and y.size(1) != self.n_targets:
            # Si y n'a qu'un seul canal mais on prédit plusieurs polluants
            if y.size(1) == 1:
                print("⚠️ Warning: Ground truth has 1 channel but predicting multiple pollutants")
        
        # Loss MSE pour tous les polluants
        if y.size(1) == y_pred.size(1):  # Même nombre de canaux
            loss_full = F.mse_loss(y_pred, y, reduction="none")  # [B, N_targets, H, W]
        else:
            # Si y n'a qu'un canal, on ne calcule la loss que sur le premier polluant (PM2.5)
            loss_full = F.mse_loss(y_pred[:, 0:1, :, :], y, reduction="none")
        
        # Appliquer le masque Chine
        mask_exp = self.china_mask.unsqueeze(0).unsqueeze(0).expand_as(loss_full)
        loss = (loss_full * mask_exp).sum() / mask_exp.sum()
        
        # Log loss total + loss par polluant
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        if y.size(1) == y_pred.size(1):
            for i, pollutant in enumerate(self.target_variables):
                pollutant_loss = (loss_full[:, i:i+1, :, :] * mask_exp[:, i:i+1, :, :]).sum() / mask_exp[:, i:i+1, :, :].sum()
                self.log(f'train_loss_{pollutant}', pollutant_loss, sync_dist=True)
        
        self.train_losses.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Auto-detect batch format
        if len(batch) == 4:
            x, y, lead_times, variables = batch
        else:
            x, y, lead_times = batch
            variables = self.config["data"]["variables"]
            
        y_pred = self(x, lead_times, variables)  # [B, N_targets, H, W]
        
        # Assurer que y a la bonne dimension
        if y.dim() == 3:
            y = y.unsqueeze(1)
        
        # Loss de validation
        if y.size(1) == y_pred.size(1):
            val_loss_full = F.mse_loss(y_pred, y, reduction="none")
        else:
            val_loss_full = F.mse_loss(y_pred[:, 0:1, :, :], y, reduction="none")
        
        # Appliquer le masque Chine
        mask_exp = self.china_mask.unsqueeze(0).unsqueeze(0).expand_as(val_loss_full)
        val_loss = (val_loss_full * mask_exp).sum() / mask_exp.sum()
        
        # Log validation loss
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        
        if y.size(1) == y_pred.size(1):
            for i, pollutant in enumerate(self.target_variables):
                pollutant_val_loss = (val_loss_full[:, i:i+1, :, :] * mask_exp[:, i:i+1, :, :]).sum() / mask_exp[:, i:i+1, :, :].sum()
                self.log(f'val_loss_{pollutant}', pollutant_val_loss, sync_dist=True)
        
        self.val_losses.append(val_loss.item())
        return val_loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["train"]["learning_rate"],
            weight_decay=self.config["train"]["weight_decay"]
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["train"]["epochs"],
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def on_train_epoch_end(self):
        """Log epoch metrics"""
        if self.train_losses:
            avg_train_loss = np.mean(self.train_losses)
            self.log('epoch_train_loss', avg_train_loss, sync_dist=True)
            self.train_losses = []

    def on_validation_epoch_end(self):
        """Log validation epoch metrics"""
        if self.val_losses:
            avg_val_loss = np.mean(self.val_losses)
            self.log('epoch_val_loss', avg_val_loss, sync_dist=True)
            self.val_losses = []
