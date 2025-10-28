"""
TEST: Charger l'optimizer state MANUELLEMENT avec mapping par nom
"""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantLightningModule

print("="*100)
print("TEST: CHARGEMENT MANUEL DE L'OPTIMIZER STATE")
print("="*100)

# Load config
with open('configs/config_all_pollutants.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Simplify
config['train']['devices'] = 1
config['train']['num_nodes'] = 1

# Checkpoint path
ckpt_path = 'logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt'

# 1. Créer le modèle
print("\n1. Créer le modèle...")
model = MultiPollutantLightningModule(config)

# 2. Charger le checkpoint
print("\n2. Charger le checkpoint...")
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# 3. Charger les poids du modèle
print("\n3. Charger state_dict (poids)...")
result = model.load_state_dict(checkpoint['state_dict'], strict=False)
print(f"   Missing keys: {len(result.missing_keys)}")
print(f"   Unexpected keys: {len(result.unexpected_keys)}")

# 4. Créer l'optimizer (comme configure_optimizers())
print("\n4. Créer l'optimizer...")
opt_config = model.configure_optimizers()
optimizer = opt_config['optimizer']
scheduler = opt_config['lr_scheduler']['scheduler']

print(f"   Optimizer créé avec {len(optimizer.param_groups)} param_groups")

# 5. LE TRUC CRITIQUE: Créer un mapping nom -> paramètre
print("\n5. Créer mapping nom → paramètre...")
name_to_param = {}
for name, param in model.named_parameters():
    name_to_param[name] = param

print(f"   {len(name_to_param)} paramètres nommés")

# 6. Créer un mapping param_id_old → nom (depuis checkpoint)
print("\n6. Analyser le checkpoint pour mapper ID → nom...")

# On doit deviner quels params correspondent à quels IDs
# En supposant que l'ordre dans state_dict correspond à l'ordre des IDs
state_dict_keys = [k for k in checkpoint['state_dict'].keys() if 'model.climax' in k]
state_dict_keys.sort()  # Important: même ordre

print(f"   {len(state_dict_keys)} clés dans checkpoint state_dict")

# Les 91 premiers paramètres trainables correspondent aux IDs 0-90
trainable_keys = []
for key in state_dict_keys:
    if 'weight' in key or 'bias' in key:  # Paramètres trainables
        trainable_keys.append(key)

print(f"   {len(trainable_keys)} paramètres trainables")

# 7. Créer le mapping ID_checkpoint → nom_actuel
print("\n7. Créer mapping ID checkpoint → nom actuel...")
id_to_name = {}
for i, key in enumerate(trainable_keys):
    if i < 91:  # Les IDs dans le checkpoint vont de 0 à 90
        id_to_name[i] = key

print(f"   Mapping créé pour {len(id_to_name)} paramètres")

# 8. Essayer de charger l'optimizer state avec ce mapping
print("\n8. Charger optimizer state avec mapping...")
ckpt_opt_state = checkpoint['optimizer_states'][0]

# Pour chaque param_group dans le checkpoint
for group_idx, ckpt_group in enumerate(ckpt_opt_state['param_groups']):
    print(f"\n   Group {group_idx}:")
    ckpt_param_ids = ckpt_group['params']
    print(f"     IDs dans checkpoint: {len(ckpt_param_ids)} params")

    # Trouver les noms correspondants
    mapped_names = []
    for ckpt_id in ckpt_param_ids:
        if ckpt_id in id_to_name:
            mapped_names.append(id_to_name[ckpt_id])

    print(f"     Noms mappés: {len(mapped_names)} params")
    if mapped_names:
        print(f"       Exemples: {mapped_names[:3]}")

# 9. Charger les états individuels (exp_avg, exp_avg_sq, step)
print("\n9. Charger les états optimizer (momentum, variance)...")
loaded_states = 0
for ckpt_id, ckpt_state in ckpt_opt_state['state'].items():
    if ckpt_id in id_to_name:
        param_name = id_to_name[ckpt_id]
        if param_name in name_to_param:
            loaded_states += 1

print(f"   {loaded_states}/{len(ckpt_opt_state['state'])} états chargés")

print("\n" + "="*100)
print("RÉSULTAT:")
print("="*100)
print("Si ce mapping fonctionne, on peut l'intégrer dans on_load_checkpoint()")
print("pour forcer le chargement manuel de l'optimizer state.")
print("="*100)
