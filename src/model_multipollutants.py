
import glob
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Tuple

from src.climax_core.arch import ClimaX


def ensure_tuple(variables):
    if isinstance(variables, list):
        return tuple(ensure_tuple(item) for item in variables)
    elif isinstance(variables, tuple):
        if len(variables) > 0 and isinstance(variables[0], list):
            return tuple(var_list[0] if isinstance(var_list, list) and len(var_list) > 0 else var_list
                         for var_list in variables)
        return variables
    return variables


class MultiPollutantModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.variables = tuple(config["data"]["variables"])
        self.target_variables = tuple(config["data"]["target_variables"])
        self.img_size = config["model"]["img_size"]
        self.patch_size = config["model"]["patch_size"]

        print(f"# # # #  Target variables: {self.target_variables}")
        print(f"# # # #  Input variables: {self.variables}")

        # Backbone ClimaX
        self.climax = ClimaX(
            default_vars=self.variables,
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=config["model"]["embed_dim"],
            depth=config["model"]["depth"],
            decoder_depth=config["model"]["decoder_depth"],
            num_heads=config["model"]["num_heads"],
            mlp_ratio=config["model"]["mlp_ratio"],
            scan_order=config.get("model", {}).get("scan_order", "hilbert"),
            parallel_patch_embed=config.get("model", {}).get("parallel_patch_embed", False),
        )

        # Indices des variables cibles
        idxs = []
        for target in self.target_variables:
            if target in self.variables:
                idxs.append(self.variables.index(target))
                print(f"# # #  {target} found")
            else:
                raise ValueError(f"Target variable '{target}' not in inputs")
        self.register_buffer("target_indices", torch.tensor(idxs, dtype=torch.long))

    def forward(self, x, lead_times, variables, out_variables=None):
        if out_variables is None:
            out_variables = self.variables
        outs = self.climax.forward_encoder(x, lead_times, ensure_tuple(variables))
        preds = self.climax.head(outs)
        preds = self.climax.unpatchify(preds)  # [B, V, H, W]
        return preds.index_select(1, self.target_indices.to(preds.device))


class MultiPollutantLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = MultiPollutantModel(config)
        self.target_variables = config["data"]["target_variables"]
        self.n_targets = len(self.target_variables)

        self.train_losses, self.val_losses = [], []

        print(f"# # # # # # #  Multi-Pollutant Lightning Module initialized")
        print(f"# # # #  Targets: {self.target_variables}")

        # Masque Chine/Taiwan
        self.register_buffer("china_mask", self._create_china_mask())

    # ----------------- MASK -----------------
    def _create_china_mask(self):
        H, W = self.config["model"]["img_size"]
        try:
            path = self.config.get("data", {}).get(
                "china_mask_path",
                "/scratch/project_462000640/ammar/aq_net2/mask_china_taiwan_128x256.npy",
            )
            m = np.load(path).astype(np.float32)
            assert m.shape == (H, W), f"china_mask {m.shape} != {(H,W)}"
            m = (m > 0.5).astype(np.float32)
            t = torch.from_numpy(m)
            cover = int(t.sum().item())
            print(f"# # # #  China/Taiwan mask loaded: {cover}/{t.numel()} pixels ({100.0*cover/t.numel():.2f}%)")
            return t
        except Exception as e:
            print(f"# # # # # #  china_mask load failed: {e} # Üí fallback rectangle")
            t = torch.zeros(H, W, dtype=torch.float32)
            t[30:100, 45:180] = 1.0
            return t

    # ----------------- FORWARD -----------------
    def forward(self, x, lead_times, variables, out_variables=None):
        return self.model(x, lead_times, tuple(variables) if isinstance(variables, list) else variables, out_variables)

    # ----------------- LOSS -----------------
    def _masked_mse(self, y_pred, y, valid_mask_bool):
        mask = valid_mask_bool.expand_as(y_pred)   # [B,C,H,W]
        diff2 = (y_pred - y) ** 2
        denom = mask.sum()
        if denom == 0:
            return torch.tensor(0.0, device=y.device, requires_grad=True)
        return (diff2 * mask).sum() / denom


    def _masked_rmse(self, y_pred, y, valid_mask_bool):
        """Calculer RMSE masqu√©e pour les pixels valides."""
        mse = self._masked_mse(y_pred, y, valid_mask_bool)
        return torch.sqrt(mse)
    # ----------------- TRAIN -----------------
    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            x, y, lead_times, variables = batch
        else:
            x, y, lead_times = batch
            variables = self.config["data"]["variables"]

        if y.dim() == 3:
            y = y.unsqueeze(1)

        y_pred = self(x, lead_times, variables)

        if y.size(1) != y_pred.size(1):
            y = y[:, :1]
            y_pred = y_pred[:, :1]

        china = self.china_mask.to(dtype=torch.bool, device=y.device).unsqueeze(0).unsqueeze(0)
        valid = china

        loss = self._masked_mse(y_pred, y, valid)
        if not torch.isfinite(loss):
            self.print("# # # #  Non-finite train loss; replacing with 0")
            loss = torch.tensor(0.0, device=y.device, requires_grad=True)

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)

        # log par polluant - FIXED
        with torch.no_grad():
            for i, name in enumerate(self.target_variables[:y_pred.size(1)]):
                # Utiliser le masque pour ce polluant sp√©cifique
                vi = valid.expand(y_pred.shape[0], 1, -1, -1)  # [B, 1, H, W]
                y_pred_i = y_pred[:, i:i+1, :, :]  # [B, 1, H, W]
                y_i = y[:, i:i+1, :, :]            # [B, 1, H, W]
                di = vi.sum()
                if di > 0:
                    ploss = (((y_pred_i - y_i)**2) * vi.float()).sum() / di
                    self.log(f'train_loss_{name}', ploss, sync_dist=True)

        self.train_losses.append(loss.item())
        return loss

    # ----------------- VAL -----------------
    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            x, y, lead_times, variables = batch
        else:
            x, y, lead_times = batch
            variables = self.config["data"]["variables"]

        if y.dim() == 3:
            y = y.unsqueeze(1)

        y_pred = self(x, lead_times, variables)

        if y.size(1) != y_pred.size(1):
            y = y[:, :1]
            y_pred = y_pred[:, :1]

        china = self.china_mask.to(dtype=torch.bool, device=y.device).unsqueeze(0).unsqueeze(0)
        valid = china

        val_loss = self._masked_mse(y_pred, y, valid)
        val_rmse = self._masked_rmse(y_pred, y, valid)
        val_rmse = self._masked_rmse(y_pred, y, valid)
        if not torch.isfinite(val_loss):
            self.print("# # # #  Non-finite val loss; replacing with 0")
            val_loss = torch.tensor(0.0, device=y.device)

        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)

        # log par polluant - FIXED
        with torch.no_grad():
            for i, name in enumerate(self.target_variables[:y_pred.size(1)]):
                # Utiliser le masque pour ce polluant sp√©cifique
                vi = valid.expand(y_pred.shape[0], 1, -1, -1)  # [B, 1, H, W]
                y_pred_i = y_pred[:, i:i+1, :, :]  # [B, 1, H, W]
                y_i = y[:, i:i+1, :, :]            # [B, 1, H, W]
                di = vi.sum()
                if di > 0:
                    ploss = (((y_pred_i - y_i)**2) * vi.float()).sum() / di
                    self.log(f'val_loss_{name}', ploss, sync_dist=True)

        self.val_losses.append(val_loss.item())
        return val_loss

    # ----------------- OPTIM -----------------
    def configure_optimizers(self):
        """
        Fine-tuned optimizer with different learning rates for different components.
        Wind-aware transfer learning optimization.
        """
        base_lr = self.config["train"]["learning_rate"]
        
        # Different learning rates for different components
        param_groups = []
        
        # 1. Pre-trained ViT blocks: Lower LR (fine-tuning)
        vit_blocks_params = []
        for param in self.model.climax.blocks.parameters():
            vit_blocks_params.append(param)
        if vit_blocks_params:
            param_groups.append({
                'params': vit_blocks_params, 
                'lr': base_lr * 0.1,  # 10x lower for pre-trained parts
                'name': 'vit_blocks'
            })
        
        # 2. Wind-aware patch embedding: Higher LR (new/improving component)
        wind_embed_params = []
        if hasattr(self.model.climax, 'token_embeds'):
            for param in self.model.climax.token_embeds.parameters():
                wind_embed_params.append(param)
        if wind_embed_params:
            param_groups.append({
                'params': wind_embed_params,
                'lr': base_lr * 2.0,  # 2x higher for wind scanning
                'name': 'wind_embedding'
            })
        
        # 3. Head layers: Medium LR (task-specific adaptation)
        head_params = []
        if hasattr(self.model.climax, 'head'):
            for param in self.model.climax.head.parameters():
                head_params.append(param)
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': base_lr * 0.5,  # 0.5x for head adaptation
                'name': 'head'
            })
        
        # 4. All other parameters: Base LR
        handled_params = set()
        for group in param_groups:
            for param in group['params']:
                handled_params.add(id(param))
        
        other_params = []
        for param in self.parameters():
            if id(param) not in handled_params:
                other_params.append(param)
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'name': 'others'
            })
        
        print(f"# # # #  Fine-tuned optimizer configuration:")
        for group in param_groups:
            print(f"   {group['name']}: LR={group['lr']:.2e} ({len(group['params'])} param groups)")
        
        opt = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config["train"]["weight_decay"]
        )
        
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.config["train"]["epochs"], eta_min=1e-6
        )
        
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


    # ----------------- TEST -----------------
    def test_step(self, batch, batch_idx):
        """Test step ULTRA SIMPLIFI√#  - QUI MARCHE!"""
        
        # Parsing batch robuste
        if len(batch) == 3:
            x, y, lead_times = batch
            variables = self.model.variables
        elif len(batch) == 4:
            x, y, lead_times, variables = batch
        else:
            x, y, lead_times, variables, _ = batch
        
        # Valid mask simple
        valid = (y != -999) & torch.isfinite(y)
        
        # Prediction
        y_pred = self.model(x, lead_times, variables)
        
        # M√©triques de base SEULEMENT
        test_loss = self._masked_mse(y_pred, y, valid)
        test_rmse = self._masked_rmse(y_pred, y, valid)
        
        # Log simple
        self.log("test_loss", test_loss, prog_bar=True, sync_dist=True)
        self.log("test_rmse", test_rmse, prog_bar=True, sync_dist=True)
        
        # M√©triques par polluant (optionnel, simple)
        for i, pollutant in enumerate(self.target_variables):
            if i < y_pred.shape[1]:  # S√©curit√© dimension
                y_pred_pol = y_pred[:, i:i+1]  
                y_true_pol = y[:, i:i+1]
                valid_pol = valid[:, i:i+1] if valid.shape[1] > i else valid[:, 0:1]
                
                rmse_pol = self._masked_rmse(y_pred_pol, y_true_pol, valid_pol)
                self.log(f"test_rmse_{pollutant}", rmse_pol, prog_bar=False, sync_dist=True)
        
        return {"test_loss": test_loss, "test_rmse": test_rmse}


    def on_test_epoch_end(self):
        """Appel√© √#  la fin de l'√©valuation test."""
        print("\n" + "="*60)
        print("# # # #  √# VALUATION TEST TERMIN√# E (2018)")
        print("="*60)
        
        # R√©cup√©rer les m√©triques logg√©es
        logged_metrics = self.trainer.logged_metrics
        
        print("\n# # # #  R√# SULTATS PAR POLLUANT:")
        for pollutant in self.target_variables:
            rmse_key = f"test_rmse_{pollutant}"
            if rmse_key in logged_metrics:
                rmse_val = logged_metrics[rmse_key].item()
                print(f"  {pollutant.upper()}: RMSE = {rmse_val:.4f}")
        
        print("\n# # # #  R√# SULTATS PAR HORIZON:")
        forecast_hours = self.config.get('data', {}).get('forecast_hours', [12, 24, 48, 96])
        for h in forecast_hours:
            rmse_key = f"test_rmse_h{h}"
            if rmse_key in logged_metrics:
                rmse_val = logged_metrics[rmse_key].item()
                print(f"  {h}h: RMSE = {rmse_val:.4f}")
        
        print("\n# # # #  R√# SULTATS D√# TAILL√# S (POLLUANT √#  HORIZON):")
        for pollutant in self.target_variables:
            print(f"\n  {pollutant.upper()}:")
            for h in forecast_hours:
                rmse_key = f"test_rmse_{pollutant}_h{h}"
                if rmse_key in logged_metrics:
                    rmse_val = logged_metrics[rmse_key].item()
                    print(f"    {h}h: RMSE = {rmse_val:.4f}")
        
        print("\n" + "="*60)



    def on_fit_end(self):
        """Appel√© √#  la fin de l'entra√Ænement pour lancer test automatique."""
        print("\n" + "="*60)
        print("# # # #  ENTRA√# NEMENT TERMIN√#  - LANCEMENT TEST AUTOMATIQUE")
        print("="*60)
        
        try:
            # Obtenir le checkpoint directory du trainer
            if hasattr(self.trainer, 'checkpoint_callback') and self.trainer.checkpoint_callback:
                checkpoint_dir = Path(self.trainer.checkpoint_callback.dirpath)
            else:
                # Fallback: chercher dans default_root_dir
                log_dir = Path(self.trainer.default_root_dir)
                checkpoint_dirs = list(log_dir.glob("**/checkpoints"))
                if checkpoint_dirs:
                    checkpoint_dir = checkpoint_dirs[-1]  # Plus r√©cent
                else:
                    print("# ù#  Impossible de trouver le dossier checkpoints")
                    return
            
            print(f"# # # #  Dossier checkpoints: {checkpoint_dir}")
            
            # Trouver le meilleur checkpoint
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if not checkpoints:
                print("# ù#  Aucun checkpoint trouv√©")
                return
                
            print(f"# # # #  Checkpoints trouv√©s: {len(checkpoints)}")
            
            # Parser pour trouver le meilleur
            best_checkpoint = None
            best_metric = float('inf')
            
            for ckpt in checkpoints:
                filename = ckpt.name
                if 'val_rmse' in filename:
                    try:
                        metric_str = filename.split('val_rmse_')[1].split('.ckpt')[0]
                        metric_val = float(metric_str)
                        if metric_val < best_metric:
                            best_metric = metric_val
                            best_checkpoint = ckpt
                    except:
                        continue
                elif 'val_loss' in filename:
                    try:
                        metric_str = filename.split('val_loss_')[1].split('.ckpt')[0]
                        metric_val = float(metric_str)
                        if metric_val < best_metric:
                            best_metric = metric_val
                            best_checkpoint = ckpt
                    except:
                        continue
            
            if best_checkpoint is None:
                best_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                print("# # # # # #  Prise du checkpoint le plus r√©cent")
            
            print(f"# # # Ü Meilleur checkpoint: {best_checkpoint.name}")
            print(f"# # # #  M√©trique: {best_metric:.4f}")
            
            # Supprimer les autres checkpoints (garder seulement le meilleur)
            for ckpt in checkpoints:
                if ckpt != best_checkpoint:
                    try:
                        ckpt.unlink()
                        print(f"# # # # # # #   Supprim√©: {ckpt.name}")
                    except:
                        pass
            
            print(f"# # #  Gard√© uniquement: {best_checkpoint.name}")
            
            # Lancer le test automatiquement
            print("\n# # # #  LANCEMENT TEST AUTOMATIQUE...")
            
            # Charger le mod√# le depuis le meilleur checkpoint  
            from src.model_multipollutants import MultiPollutantLightningModule
            model = MultiPollutantLightningModule.load_from_checkpoint(str(best_checkpoint), config=self.config)
            
            # Cr√©er nouveau trainer pour test (m√# me config GPU)
            import pytorch_lightning as pl
            test_trainer = pl.Trainer(
                devices=self.trainer.num_devices,
                accelerator=self.trainer.accelerator,
                strategy=self.trainer.strategy if hasattr(self.trainer, 'strategy') else None,
                logger=False,  # Pas besoin de logger pour test
                enable_checkpointing=False,
                enable_model_summary=False
            )
            
            # DataModule pour test
            from src.datamodule import AQNetDataModule  
            data_module = AQNetDataModule(self.config)
            
            # Lancer test
            test_trainer.test(model, data_module)
            
            print("\n# # #  TEST AUTOMATIQUE TERMIN√# !")
            print("="*60)
            
        except Exception as e:
            print(f"# ù#  Erreur test automatique: {e}")
            import traceback
            traceback.print_exc()


    # ----------------- EPOCH LOGS -----------------
    def on_train_epoch_end(self):
        if self.train_losses:
            self.log('epoch_train_loss', float(np.mean(self.train_losses)), sync_dist=True)
            self.train_losses = []

    def on_validation_epoch_end(self):
        if self.val_losses:
            self.log('epoch_val_loss', float(np.mean(self.val_losses)), sync_dist=True)
            self.val_losses = []
