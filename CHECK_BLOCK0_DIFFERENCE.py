"""
Vérifier si le bloc 0 dans le checkpoint est différent des autres blocs
et si le modèle actuel a la même structure
"""
import torch
import sys
sys.path.insert(0, 'src')
import yaml

print("="*100)
print("🔍 COMPARAISON: Bloc 0 du checkpoint vs modèle actuel")
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

# Get all block 0 attention keys
print("\n1️⃣ CHECKPOINT - Bloc 0 attention keys:")
print("-" * 100)
block0_keys = sorted([k for k in ckpt_keys.keys() if 'blocks.0.attn' in k])
for k in block0_keys:
    shape = ckpt_keys[k].shape if hasattr(ckpt_keys[k], 'shape') else 'scalar'
    print(f"   {k:60s} {shape}")

# Get all block 1 attention keys for comparison
print("\n2️⃣ CHECKPOINT - Bloc 1 attention keys (pour comparaison):")
print("-" * 100)
block1_keys = sorted([k for k in ckpt_keys.keys() if 'blocks.1.attn' in k])
for k in block1_keys:
    shape = ckpt_keys[k].shape if hasattr(ckpt_keys[k], 'shape') else 'scalar'
    print(f"   {k:60s} {shape}")

# Compare structures
print("\n3️⃣ COMPARAISON:")
print("-" * 100)
print(f"   Bloc 0 a {len(block0_keys)} clés")
print(f"   Bloc 1 a {len(block1_keys)} clés")

if len(block0_keys) == len(block1_keys):
    print(f"   ✅ Même nombre de clés")
else:
    print(f"   ⚠️  Nombre de clés différent!")

    # Find differences
    block0_set = set([k.replace('blocks.0', 'blocks.X') for k in block0_keys])
    block1_set = set([k.replace('blocks.1', 'blocks.X') for k in block1_keys])

    only_in_0 = block0_set - block1_set
    only_in_1 = block1_set - block0_set

    if only_in_0:
        print(f"\n   ⚠️  Clés seulement dans bloc 0:")
        for k in sorted(only_in_0):
            print(f"      {k}")

    if only_in_1:
        print(f"\n   ⚠️  Clés seulement dans bloc 1:")
        for k in sorted(only_in_1):
            print(f"      {k}")

# Now check current model
print("\n4️⃣ MODÈLE ACTUEL - Bloc 0:")
print("-" * 100)

from model_multipollutants import MultiPollutantModel
with open('configs/config_all_pollutants.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = MultiPollutantModel(config)
model_params = dict(model.named_parameters())

model_block0_keys = sorted([k for k in model_params.keys() if 'blocks.0.attn' in k])
for k in model_block0_keys:
    shape = model_params[k].shape
    print(f"   {k:60s} {shape}")

print("\n5️⃣ MODÈLE ACTUEL - Bloc 1 (pour comparaison):")
print("-" * 100)
model_block1_keys = sorted([k for k in model_params.keys() if 'blocks.1.attn' in k])
for k in model_block1_keys:
    shape = model_params[k].shape
    print(f"   {k:60s} {shape}")

print("\n6️⃣ COMPARAISON MODÈLE ACTUEL:")
print("-" * 100)
print(f"   Bloc 0 a {len(model_block0_keys)} clés")
print(f"   Bloc 1 a {len(model_block1_keys)} clés")

if len(model_block0_keys) == len(model_block1_keys):
    print(f"   ✅ Même nombre de clés dans le modèle actuel")
else:
    print(f"   ❌ PROBLÈME: Bloc 0 différent du bloc 1 dans le modèle actuel!")

    # Find differences
    model_block0_set = set([k.replace('blocks.0', 'blocks.X') for k in model_block0_keys])
    model_block1_set = set([k.replace('blocks.1', 'blocks.X') for k in model_block1_keys])

    only_in_model_0 = model_block0_set - model_block1_set
    only_in_model_1 = model_block1_set - model_block0_set

    if only_in_model_0:
        print(f"\n   ⚠️  Clés seulement dans bloc 0 du modèle actuel:")
        for k in sorted(only_in_model_0):
            print(f"      {k}")

    if only_in_model_1:
        print(f"\n   ⚠️  Clés seulement dans bloc 1 du modèle actuel:")
        for k in sorted(only_in_model_1):
            print(f"      {k}")

print("\n7️⃣ ANALYSE FINALE:")
print("-" * 100)

# Check if checkpoint block 0 matches model block 0
ckpt_block0_normalized = set([k.replace('blocks.0', 'blocks.X') for k in block0_keys])
model_block0_normalized = set([k.replace('climax.blocks.0', 'blocks.X').replace('blocks.0', 'blocks.X') for k in model_block0_keys])

if ckpt_block0_normalized == model_block0_normalized:
    print("   ✅ Le bloc 0 du checkpoint MATCH le bloc 0 du modèle actuel")
    print("   ✅ Architecture compatible!")
else:
    print("   ❌ INCOMPATIBILITÉ: Le bloc 0 du checkpoint NE MATCH PAS le modèle actuel!")

    only_in_ckpt_block0 = ckpt_block0_normalized - model_block0_normalized
    only_in_model_block0 = model_block0_normalized - ckpt_block0_normalized

    if only_in_ckpt_block0:
        print(f"\n   Clés dans checkpoint bloc 0 mais pas dans modèle:")
        for k in sorted(only_in_ckpt_block0):
            print(f"      {k}")

    if only_in_model_block0:
        print(f"\n   Clés dans modèle bloc 0 mais pas dans checkpoint:")
        for k in sorted(only_in_model_block0):
            print(f"      {k}")

    print("\n   🚨 PROBLÈME: Le modèle actuel a une architecture TopoFlow même avec use_physics_mask=false!")
    print("   📝 ACTION REQUISE: Vérifier climax.py pour voir comment le bloc 0 est créé")

print("="*100)
