"""
Vérifier TOUS les blocs (0-5) entre checkpoint et modèle actuel
"""
import torch
import sys
sys.path.insert(0, 'src')
import yaml

print("="*100)
print("🔍 COMPARAISON COMPLÈTE: TOUS LES BLOCS")
print("="*100)

# Load checkpoint
ckpt_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = ckpt['state_dict']

# Fix prefix
ckpt_keys = {}
for key, value in state_dict.items():
    if key.startswith('model.'):
        ckpt_keys[key[6:]] = value
    else:
        ckpt_keys[key] = value

# Create model
from model_multipollutants import MultiPollutantModel
with open('configs/config_all_pollutants.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = MultiPollutantModel(config)
model_params = dict(model.named_parameters())

# Get number of blocks
num_blocks = config['model']['depth']
print(f"\nNombre de blocs: {num_blocks}")

# Check each block
for block_idx in range(num_blocks):
    print(f"\n{'='*100}")
    print(f"BLOC {block_idx}:")
    print(f"{'='*100}")

    # Checkpoint keys for this block
    ckpt_block_keys = sorted([k for k in ckpt_keys.keys() if f'blocks.{block_idx}.' in k])

    # Model keys for this block
    model_block_keys = sorted([k for k in model_params.keys() if f'blocks.{block_idx}.' in k])

    print(f"\n   Checkpoint: {len(ckpt_block_keys)} clés")
    print(f"   Modèle actuel: {len(model_block_keys)} clés")

    # Normalize keys (remove block index for comparison)
    ckpt_normalized = set([k.replace(f'climax.blocks.{block_idx}', 'blocks.X') for k in ckpt_block_keys])
    model_normalized = set([k.replace(f'climax.blocks.{block_idx}', 'blocks.X') for k in model_block_keys])

    # Find differences
    only_in_ckpt = ckpt_normalized - model_normalized
    only_in_model = model_normalized - ckpt_normalized
    common = ckpt_normalized & model_normalized

    print(f"   Clés communes: {len(common)}")

    if only_in_ckpt:
        print(f"\n   ❌ Clés SEULEMENT dans checkpoint:")
        for k in sorted(only_in_ckpt):
            print(f"      {k}")

    if only_in_model:
        print(f"\n   ⚠️  Clés SEULEMENT dans modèle:")
        for k in sorted(only_in_model):
            print(f"      {k}")

    # Check attention specifically
    ckpt_attn_keys = [k for k in ckpt_block_keys if '.attn.' in k]
    model_attn_keys = [k for k in model_block_keys if '.attn.' in k]

    print(f"\n   Attention layer:")
    print(f"     Checkpoint: {len(ckpt_attn_keys)} clés")
    print(f"     Modèle: {len(model_attn_keys)} clés")

    if len(ckpt_attn_keys) != len(model_attn_keys):
        print(f"     ❌ DIFFÉRENCE DANS L'ATTENTION!")

        print(f"\n     Checkpoint attention keys:")
        for k in sorted(ckpt_attn_keys):
            print(f"       {k}")

        print(f"\n     Modèle attention keys:")
        for k in sorted(model_attn_keys):
            print(f"       {k}")
    else:
        print(f"     ✅ Même nombre de clés d'attention")

    # Summary for this block
    if len(only_in_ckpt) == 0 and len(only_in_model) == 0:
        print(f"\n   ✅ BLOC {block_idx}: PARFAITEMENT COMPATIBLE")
    else:
        print(f"\n   ❌ BLOC {block_idx}: INCOMPATIBLE!")

# Final summary
print(f"\n{'='*100}")
print(f"RÉSUMÉ FINAL:")
print(f"{'='*100}")

all_compatible = True
for block_idx in range(num_blocks):
    ckpt_block_keys = set([k.replace(f'climax.blocks.{block_idx}', 'blocks.X') for k in ckpt_keys.keys() if f'blocks.{block_idx}.' in k])
    model_block_keys = set([k.replace(f'climax.blocks.{block_idx}', 'blocks.X') for k in model_params.keys() if f'blocks.{block_idx}.' in k])

    if ckpt_block_keys != model_block_keys:
        all_compatible = False
        print(f"   ❌ Bloc {block_idx}: INCOMPATIBLE")
    else:
        print(f"   ✅ Bloc {block_idx}: Compatible")

if all_compatible:
    print(f"\n✅✅✅ TOUS LES BLOCS SONT COMPATIBLES!")
    print(f"\nLe problème de val_loss élevé (8.0 vs 0.356) vient d'AILLEURS:")
    print(f"   - Normalisation des données différente")
    print(f"   - Preprocessing différent")
    print(f"   - Autre bug dans le forward pass ou loss computation")
else:
    print(f"\n❌❌❌ CERTAINS BLOCS SONT INCOMPATIBLES!")
    print(f"\nLe checkpoint version_47 ne peut PAS être utilisé avec le code actuel.")

print(f"{'='*100}")
