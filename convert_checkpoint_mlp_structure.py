#!/usr/bin/env python3
"""
Script pour convertir le checkpoint version_144 de la structure Mlp (fc1/fc2)
vers la structure nn.Sequential (0/3).

Usage:
    python convert_checkpoint_mlp_structure.py

Input:  version_144/checkpoints/best-val_loss_val_loss=0.2931-step_step=408.ckpt
Output: version_144/checkpoints/converted_best-val_loss_val_loss=0.2931-step_step=408.ckpt
"""

import torch
import sys
from pathlib import Path

def convert_mlp_keys(state_dict, prefix="state_dict"):
    """
    Convertit les clés MLP de fc1/fc2 vers 0/3

    Conversion:
        model.climax.blocks.X.mlp.fc1.weight → model.climax.blocks.X.mlp.0.weight
        model.climax.blocks.X.mlp.fc1.bias   → model.climax.blocks.X.mlp.0.bias
        model.climax.blocks.X.mlp.fc2.weight → model.climax.blocks.X.mlp.3.weight
        model.climax.blocks.X.mlp.fc2.bias   → model.climax.blocks.X.mlp.3.bias
    """
    new_dict = {}
    conversions = []

    for key, value in state_dict.items():
        new_key = key

        # Convertir fc1 → 0
        if ".mlp.fc1." in key:
            new_key = key.replace(".mlp.fc1.", ".mlp.0.")
            conversions.append(f"  {key} → {new_key}")

        # Convertir fc2 → 3
        elif ".mlp.fc2." in key:
            new_key = key.replace(".mlp.fc2.", ".mlp.3.")
            conversions.append(f"  {key} → {new_key}")

        new_dict[new_key] = value

    return new_dict, conversions


def convert_optimizer_state(optimizer_states):
    """
    Convertit les clés dans l'optimizer state pour matcher les nouveaux noms de paramètres

    L'optimizer stocke les states par index de param_group, donc on doit aussi
    convertir les noms dans la partie 'param_groups'
    """
    # Les optimizer states sont stockés par index de paramètre
    # On doit juste s'assurer que les param_groups matchent

    new_optimizer_states = []
    conversions = []

    for opt_state in optimizer_states:
        new_opt_state = {}

        # Copier tous les champs
        for key, value in opt_state.items():
            if key == 'param_groups':
                # Convertir les noms de paramètres dans param_groups
                new_param_groups = []
                for group in value:
                    new_group = group.copy()
                    if 'params' in new_group:
                        # Les params sont juste des indices, pas besoin de changer
                        new_param_groups.append(new_group)
                    else:
                        new_param_groups.append(new_group)
                new_opt_state[key] = new_param_groups
            else:
                # Copier tel quel (state, etc.)
                new_opt_state[key] = value

        new_optimizer_states.append(new_opt_state)

    return new_optimizer_states, conversions


def main():
    # Chemins
    input_ckpt = Path("/scratch/project_462000640/ammar/aq_net2/logs/multipollutants_climax_ddp/version_144/checkpoints/best-val_loss_val_loss=0.2931-step_step=408.ckpt")
    output_ckpt = input_ckpt.parent / f"converted_{input_ckpt.name}"

    print("="*80)
    print("🔧 CONVERSION CHECKPOINT MLP STRUCTURE")
    print("="*80)
    print(f"\n📥 Input:  {input_ckpt}")
    print(f"📤 Output: {output_ckpt}")

    if not input_ckpt.exists():
        print(f"\n❌ ERROR: Input checkpoint not found: {input_ckpt}")
        sys.exit(1)

    if output_ckpt.exists():
        print(f"\n⚠️  WARNING: Output checkpoint already exists: {output_ckpt}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Charger le checkpoint
    print("\n⏳ Loading checkpoint...")
    checkpoint = torch.load(input_ckpt, map_location='cpu')

    print(f"✅ Checkpoint loaded")
    print(f"   Keys in checkpoint: {list(checkpoint.keys())}")

    # Convertir state_dict du modèle
    print("\n🔄 Converting model state_dict...")
    if 'state_dict' in checkpoint:
        new_state_dict, conversions = convert_mlp_keys(checkpoint['state_dict'])
        checkpoint['state_dict'] = new_state_dict

        print(f"✅ Converted {len(conversions)} keys:")
        for conv in conversions[:10]:  # Afficher les 10 premières
            print(conv)
        if len(conversions) > 10:
            print(f"   ... and {len(conversions) - 10} more")
    else:
        print("⚠️  No 'state_dict' found in checkpoint")

    # Convertir optimizer state
    print("\n🔄 Converting optimizer state...")
    if 'optimizer_states' in checkpoint:
        new_opt_states, opt_conversions = convert_optimizer_state(checkpoint['optimizer_states'])
        checkpoint['optimizer_states'] = new_opt_states
        print(f"✅ Optimizer states processed")
    else:
        print("⚠️  No 'optimizer_states' found in checkpoint")

    # Sauvegarder le nouveau checkpoint
    print(f"\n💾 Saving converted checkpoint to: {output_ckpt}")
    torch.save(checkpoint, output_ckpt)

    # Vérification
    print("\n🔍 Verifying converted checkpoint...")
    verify_ckpt = torch.load(output_ckpt, map_location='cpu')

    # Compter les clés converties
    mlp_0_keys = [k for k in verify_ckpt['state_dict'].keys() if '.mlp.0.' in k]
    mlp_3_keys = [k for k in verify_ckpt['state_dict'].keys() if '.mlp.3.' in k]
    mlp_fc1_keys = [k for k in verify_ckpt['state_dict'].keys() if '.mlp.fc1.' in k]
    mlp_fc2_keys = [k for k in verify_ckpt['state_dict'].keys() if '.mlp.fc2.' in k]

    print(f"✅ Verification:")
    print(f"   Keys with '.mlp.0.': {len(mlp_0_keys)}")
    print(f"   Keys with '.mlp.3.': {len(mlp_3_keys)}")
    print(f"   Keys with '.mlp.fc1.': {len(mlp_fc1_keys)} (should be 0)")
    print(f"   Keys with '.mlp.fc2.': {len(mlp_fc2_keys)} (should be 0)")

    if mlp_fc1_keys or mlp_fc2_keys:
        print("\n❌ ERROR: Still found fc1/fc2 keys after conversion!")
        sys.exit(1)

    print("\n" + "="*80)
    print("✅ CONVERSION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📄 Converted checkpoint: {output_ckpt}")
    print(f"   Size: {output_ckpt.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n💡 Update your config to use:")
    print(f"   checkpoint_path: {output_ckpt}")
    print()

if __name__ == "__main__":
    main()
