#!/usr/bin/env python3
"""
Supprime l'optimizer state d'un checkpoint pour garder SEULEMENT les poids du modèle.
Utile quand la structure du modèle a légèrement changé.
"""

import torch
import sys
from pathlib import Path

def strip_optimizer(input_path, output_path):
    print(f"📥 Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')

    print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")

    # Garder seulement ce qui est nécessaire pour les poids du modèle
    new_checkpoint = {
        'state_dict': checkpoint['state_dict'],  # Les poids du modèle
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'pytorch-lightning_version': checkpoint.get('pytorch-lightning_version', ''),
    }

    # Supprimer l'optimizer et lr_scheduler
    removed = []
    if 'optimizer_states' in checkpoint:
        removed.append('optimizer_states')
    if 'lr_schedulers' in checkpoint:
        removed.append('lr_schedulers')

    print(f"❌ Removed: {removed}")
    print(f"✅ Kept: {list(new_checkpoint.keys())}")

    print(f"💾 Saving to: {output_path}")
    torch.save(new_checkpoint, output_path)

    # Vérification
    verify = torch.load(output_path, map_location='cpu')
    print(f"✅ Verification - keys in new checkpoint: {list(verify.keys())}")
    print(f"✅ Model state_dict has {len(verify['state_dict'])} keys")

    # Taille
    orig_size = Path(input_path).stat().st_size / 1024 / 1024
    new_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"📊 Size: {orig_size:.1f} MB → {new_size:.1f} MB (saved {orig_size-new_size:.1f} MB)")

if __name__ == "__main__":
    input_ckpt = Path("/scratch/project_462000640/ammar/aq_net2/logs/multipollutants_climax_ddp/version_144/checkpoints/best-val_loss_val_loss=0.2931-step_step=408.ckpt")
    output_ckpt = input_ckpt.parent / f"weights_only_{input_ckpt.name}"

    strip_optimizer(input_ckpt, output_ckpt)

    print("\n" + "="*80)
    print("✅ DONE!")
    print("="*80)
    print(f"\n💡 Update your config to use:")
    print(f"   checkpoint_path: {output_ckpt}")
