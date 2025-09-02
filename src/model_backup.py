import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.climax_core.arch import ClimaX


# Fonction pour masque Chine
def create_china_mask(height, width):
    """Charger le masque Chine précis"""
    import torch
    try:
        mask = torch.load("china_mask_fixed.pt", map_location="cpu")
        # Redimensionner si nécessaire
        if mask.shape != (height, width):
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0), 
                size=(height, width), 
                mode="bilinear", 
                align_corners=False
            ).squeeze(0).squeeze(0)
        return mask
    except:
        # Fallback au masque rectangulaire
        mask = torch.zeros(height, width)
        mask[20:120, 50:200] = 1.0
        return mask

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

class PM25Model(nn.Module):
    """
    Modèle PM2.5 basé sur l'architecture ClimaX complète
    """
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.config = config
        self.variables = tuple(config["data"]["variables"])  # Variables d'entrée (maintenant avec pm25)
        self.target_variables = tuple(config["data"]["target_variables"])  # Variables de sortie (PM2.5)
        self.img_size = config["model"]["img_size"]
        self.patch_size = config["model"]["patch_size"]
        
        # Architecture ClimaX complète - prédit toutes les variables
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
        
        # Index de PM2.5 dans la liste des variables
        # Variables : ['u', 'v', 'temp', 'rh', 'psfc', 'pm10', 'so2', 'no2', 'co', 'o3', 'lat2d', 'lon2d', 'pm25']
        # PM2.5 est à l'index 12 (le dernier)
        self.pm25_idx = self.variables.index('pm25')
        print(f"PM2.5 index in variables: {self.pm25_idx}")

    def forward(self, x, lead_times, variables, out_variables=None):
        """
        Forward pass utilisant ClimaX complet
        x: [B, V, H, W] - batch, variables, height, width
        lead_times: [B, T] - lead times pour chaque sample
        variables: liste des noms de variables d'entrée
        """
        if out_variables is None:
            out_variables = self.variables  # Prédit toutes les variables
            
        # Forward pass complet de ClimaX
        # ClimaX fait : encoder -> head -> unpatchify
        out_transformers = self.climax.forward_encoder(x, lead_times, ensure_tuple(variables))  # [B, L, D]
        preds = self.climax.head(out_transformers)  # [B, L, V*p*p]
        preds = self.climax.unpatchify(preds)  # [B, V, H, W]
        
        # Extraire seulement PM2.5
        pm25_pred = preds[:, self.pm25_idx:self.pm25_idx+1, :, :]  # [B, 1, H, W]
        
        return pm25_pred

class PM25LightningModule(pl.LightningModule):
    """
    Module PyTorch Lightning pour l'entraînement du modèle PM2.5
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = PM25Model(config)
        
        # Métriques
        # Masque Chine
        self.register_buffer("china_mask", create_china_mask(config["model"]["img_size"][0], config["model"]["img_size"][1]))
        self.train_losses = []
        self.val_losses = []

    def forward(self, x, lead_times, variables, out_variables=None):
        return self.model(x, lead_times, tuple(variables) if isinstance(variables, list) else variables, out_variables)
    
    def training_step(self, batch, batch_idx):
        # Auto-detect batch format
        if len(batch) == 4:
            x, y, lead_times, variables = batch
        else:
            x, y, lead_times = batch
            variables = self.config["data"]["variables"]
        y_pred = self(x, lead_times, variables)
        
        # Loss (MSE pour PM2.5)
        # Fix dimensions if needed
        if y.dim() == 3:
            y = y.unsqueeze(1)  # Ajoute le canal manquant : [B, H, W] -> [B, 1, H, W]
        # Loss avec masque Chine
        loss_full = F.mse_loss(y_pred, y, reduction="none")
        mask_exp = self.china_mask.unsqueeze(0).unsqueeze(0).expand_as(loss_full)
        loss = (loss_full * mask_exp).sum() / mask_exp.sum()
        
        # Logging
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.train_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Auto-detect batch format
        if len(batch) == 4:
            x, y, lead_times, variables = batch
        else:
            x, y, lead_times = batch
            variables = self.config["data"]["variables"]
        y_pred = self(x, lead_times, variables)
        
        # Loss (MSE pour PM2.5)
        # Fix dimensions if needed
        if y.dim() == 3:
            y = y.unsqueeze(1)  # Ajoute le canal manquant : [B, H, W] -> [B, 1, H, W]
        # Loss avec masque Chine
        loss_full = F.mse_loss(y_pred, y, reduction="none")
        mask_exp = self.china_mask.unsqueeze(0).unsqueeze(0).expand_as(loss_full)
        loss = (loss_full * mask_exp).sum() / mask_exp.sum()
        
        # Logging
        self.log('val_loss', loss, sync_dist=True)
        self.val_losses.append(loss.item())
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["train"]["learning_rate"],
            weight_decay=self.config["train"]["weight_decay"]
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["train"]["epochs"],
            eta_min=self.config["train"]["learning_rate"] * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def on_train_epoch_end(self):
        if self.train_losses:
            avg_loss = sum(self.train_losses) / len(self.train_losses)
            self.log('avg_train_loss', avg_loss, sync_dist=True)
            self.train_losses.clear()
    
    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_loss = sum(self.val_losses) / len(self.val_losses)
            self.log('avg_val_loss', avg_loss, sync_dist=True)
            self.val_losses.clear()
