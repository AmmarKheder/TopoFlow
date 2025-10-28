"""
ANALYSE COMPLÈTE DU CHECKPOINT - DE A à Z
"""
import torch
import pprint

ckpt_path = 'logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt'

print("="*100)
print("ANALYSE COMPLÈTE DU CHECKPOINT")
print("="*100)

# Load checkpoint
print("\n1. CHARGEMENT DU CHECKPOINT...")
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print("\n2. CLÉS PRINCIPALES DU CHECKPOINT:")
print("="*100)
for key in checkpoint.keys():
    if isinstance(checkpoint[key], dict):
        print(f"  {key}: dict with {len(checkpoint[key])} keys")
    elif isinstance(checkpoint[key], list):
        print(f"  {key}: list with {len(checkpoint[key])} items")
    elif isinstance(checkpoint[key], (int, float, str)):
        print(f"  {key}: {checkpoint[key]}")
    else:
        print(f"  {key}: {type(checkpoint[key])}")

print("\n3. DÉTAILS DES OPTIMIZER_STATES:")
print("="*100)
if 'optimizer_states' in checkpoint:
    print(f"Nombre d'optimizer states: {len(checkpoint['optimizer_states'])}")

    for idx, opt_state in enumerate(checkpoint['optimizer_states']):
        print(f"\nOptimizer {idx}:")
        print(f"  Keys: {list(opt_state.keys())}")

        if 'state' in opt_state:
            print(f"\n  État de l'optimizer (momentum, etc.):")
            print(f"    Nombre de param states: {len(opt_state['state'])}")
            # Afficher un exemple d'état
            if len(opt_state['state']) > 0:
                first_key = list(opt_state['state'].keys())[0]
                first_state = opt_state['state'][first_key]
                print(f"    Exemple état param {first_key}:")
                for k, v in first_state.items():
                    if torch.is_tensor(v):
                        print(f"      {k}: tensor shape {v.shape}, dtype {v.dtype}")
                    else:
                        print(f"      {k}: {v}")

        if 'param_groups' in opt_state:
            print(f"\n  Param Groups:")
            for i, pg in enumerate(opt_state['param_groups']):
                print(f"    Group {i}:")
                for k, v in pg.items():
                    if k == 'params':
                        print(f"      params: {len(v)} parameter indices")
                    elif k == 'lr':
                        print(f"      lr: {v:.10e}")
                    else:
                        print(f"      {k}: {v}")

print("\n4. DÉTAILS DU LR_SCHEDULERS:")
print("="*100)
if 'lr_schedulers' in checkpoint:
    print(f"Nombre de schedulers: {len(checkpoint['lr_schedulers'])}")

    for idx, sch_state in enumerate(checkpoint['lr_schedulers']):
        print(f"\nScheduler {idx}:")
        print(f"  Keys: {list(sch_state.keys())}")
        for k, v in sch_state.items():
            if isinstance(v, list) and len(v) <= 10:
                print(f"  {k}: {v}")
            elif isinstance(v, list):
                print(f"  {k}: list with {len(v)} items")
            else:
                print(f"  {k}: {v}")

print("\n5. DÉTAILS DU STATE_DICT (POIDS DU MODÈLE):")
print("="*100)
if 'state_dict' in checkpoint:
    print(f"Nombre de clés dans state_dict: {len(checkpoint['state_dict'])}")

    # Group by prefix
    prefixes = {}
    for key in checkpoint['state_dict'].keys():
        prefix = key.split('.')[0] if '.' in key else key
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(key)

    print(f"\nGROUPES DE POIDS:")
    for prefix, keys in sorted(prefixes.items()):
        print(f"  {prefix}: {len(keys)} parameters")
        if len(keys) <= 5:
            for k in keys:
                tensor = checkpoint['state_dict'][k]
                print(f"    {k}: shape {tensor.shape}, dtype {tensor.dtype}")
        else:
            # Afficher quelques exemples
            for k in keys[:3]:
                tensor = checkpoint['state_dict'][k]
                print(f"    {k}: shape {tensor.shape}, dtype {tensor.dtype}")
            print(f"    ... ({len(keys) - 3} more)")

print("\n6. LOOPS (ÉTAT DES BOUCLES D'ENTRAÎNEMENT):")
print("="*100)
if 'loops' in checkpoint:
    def print_nested_dict(d, indent=0):
        for k, v in d.items():
            if isinstance(v, dict):
                print("  " * indent + f"{k}:")
                print_nested_dict(v, indent + 1)
            else:
                print("  " * indent + f"{k}: {v}")

    print_nested_dict(checkpoint['loops'])

print("\n7. CALLBACKS:")
print("="*100)
if 'callbacks' in checkpoint:
    print(f"Type: {type(checkpoint['callbacks'])}")
    if isinstance(checkpoint['callbacks'], dict):
        for k, v in checkpoint['callbacks'].items():
            print(f"  {k}:")
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"    {v}")

print("\n8. HYPER_PARAMETERS:")
print("="*100)
if 'hyper_parameters' in checkpoint:
    print(f"Keys dans hyper_parameters: {list(checkpoint['hyper_parameters'].keys())}")
    if 'config' in checkpoint['hyper_parameters']:
        config = checkpoint['hyper_parameters']['config']
        print(f"\nConfig keys:")
        for k in config.keys():
            print(f"  {k}")

print("\n9. ANALYSE CRITIQUE - POURQUOI L'OPTIMIZER NE SE CHARGE PAS?")
print("="*100)

if 'optimizer_states' in checkpoint and len(checkpoint['optimizer_states']) > 0:
    opt_state = checkpoint['optimizer_states'][0]

    print("\nVÉRIFICATIONS:")

    # Check 1: Number of param groups
    n_groups = len(opt_state['param_groups'])
    print(f"1. Nombre de param_groups dans checkpoint: {n_groups}")
    print(f"   → Le nouveau optimizer DOIT avoir exactement {n_groups} param groups")

    # Check 2: Number of params in each group
    print(f"\n2. Nombre de paramètres par groupe:")
    for i, pg in enumerate(opt_state['param_groups']):
        n_params = len(pg['params'])
        print(f"   Group {i}: {n_params} params")

    # Check 3: LRs
    print(f"\n3. Learning rates dans le checkpoint:")
    for i, pg in enumerate(opt_state['param_groups']):
        print(f"   Group {i}: lr={pg['lr']:.10e}")

    # Check 4: Optimizer state keys
    print(f"\n4. États de l'optimizer (momentum, variance):")
    print(f"   Nombre d'états sauvegardés: {len(opt_state['state'])}")
    if len(opt_state['state']) > 0:
        first_key = list(opt_state['state'].keys())[0]
        print(f"   Clés dans chaque état: {list(opt_state['state'][first_key].keys())}")

    print(f"\n5. DIAGNOSTIC:")
    print(f"   Pour que PyTorch Lightning charge l'optimizer correctement:")
    print(f"   - Le nouvel optimizer DOIT avoir {n_groups} param_groups")
    print(f"   - Chaque groupe doit avoir le MÊME nombre de paramètres")
    print(f"   - Les paramètres doivent être dans le MÊME ORDRE")
    print(f"   - Si UNE SEULE de ces conditions échoue → silent fail, LRs reset")

print("\n" + "="*100)
print("FIN DE L'ANALYSE")
print("="*100)
