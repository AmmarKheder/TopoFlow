#!/usr/bin/env python3
"""
Script de diagnostic pour comprendre pourquoi le resume donne train_loss=3.840
au lieu de ~0.35
"""

import torch
import yaml
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule

print("="*80)
print("🔍 DIAGNOSTIC CHECKPOINT RESUME")
print("="*80)

# Load config
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config

ckpt_path = config['model']['checkpoint_path']
print(f"\n📂 Checkpoint: {ckpt_path}")

# 1. Load checkpoint directly
print("\n" + "="*80)
print("1️⃣ CHARGEMENT DU CHECKPOINT (torch.load)")
print("="*80)

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print(f"\n✅ Checkpoint chargé avec succès")
print(f"\n📊 Clés principales:")
for key in ckpt.keys():
    print(f"  - {key}")

print(f"\n📈 Informations d'entraînement:")
print(f"  - Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"  - Global step: {ckpt.get('global_step', 'N/A')}")
print(f"  - PyTorch Lightning version: {ckpt.get('pytorch-lightning_version', 'N/A')}")

# 2. Check optimizer state
print("\n" + "="*80)
print("2️⃣ ÉTAT DE L'OPTIMIZER")
print("="*80)

if 'optimizer_states' in ckpt and len(ckpt['optimizer_states']) > 0:
    opt_state = ckpt['optimizer_states'][0]
    print(f"✅ Optimizer state trouvé")

    # Check param groups (learning rate, etc.)
    if 'param_groups' in opt_state:
        for i, pg in enumerate(opt_state['param_groups']):
            print(f"\n  Param group {i}:")
            print(f"    - lr: {pg.get('lr', 'N/A')}")
            print(f"    - weight_decay: {pg.get('weight_decay', 'N/A')}")
            print(f"    - betas: {pg.get('betas', 'N/A')}")

    # Check state (momentum buffers, etc.)
    if 'state' in opt_state:
        num_params_with_state = len(opt_state['state'])
        print(f"\n  ✅ Nombre de paramètres avec état: {num_params_with_state}")

        # Sample one parameter state
        if num_params_with_state > 0:
            sample_key = list(opt_state['state'].keys())[0]
            sample_state = opt_state['state'][sample_key]
            print(f"  📊 État d'un paramètre (sample):")
            for k, v in sample_state.items():
                if isinstance(v, torch.Tensor):
                    print(f"    - {k}: tensor shape {v.shape}, mean={v.float().mean():.6f}")
                else:
                    print(f"    - {k}: {v}")
else:
    print("❌ PAS D'OPTIMIZER STATE - C'EST LE PROBLÈME!")

# 3. Check scheduler state
print("\n" + "="*80)
print("3️⃣ ÉTAT DU SCHEDULER")
print("="*80)

if 'lr_schedulers' in ckpt and len(ckpt['lr_schedulers']) > 0:
    sched_state = ckpt['lr_schedulers'][0]
    print(f"✅ Scheduler state trouvé")
    print(f"  - Clés: {list(sched_state.keys())}")

    if 'last_epoch' in sched_state:
        print(f"  - last_epoch: {sched_state['last_epoch']}")
    if '_step_count' in sched_state:
        print(f"  - step_count: {sched_state['_step_count']}")
    if '_last_lr' in sched_state:
        print(f"  - last_lr: {sched_state['_last_lr']}")
else:
    print("❌ PAS DE SCHEDULER STATE")

# 4. Check model state dict
print("\n" + "="*80)
print("4️⃣ STATE_DICT DU MODÈLE")
print("="*80)

if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
    print(f"✅ Model state_dict trouvé")
    print(f"  - Nombre de clés: {len(state_dict)}")

    # Sample some weights
    sample_keys = list(state_dict.keys())[:5]
    print(f"\n  📊 Exemple de poids (5 premiers):")
    for key in sample_keys:
        tensor = state_dict[key]
        if isinstance(tensor, torch.Tensor):
            print(f"    - {key}: shape={tensor.shape}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")
else:
    print("❌ PAS DE STATE_DICT")

# 5. Load model with Lightning and compare
print("\n" + "="*80)
print("5️⃣ CHARGEMENT AVEC LIGHTNING")
print("="*80)

print("\nCréation d'un nouveau modèle...")
model_new = MultiPollutantLightningModule(config=config)

print("\nChargement du checkpoint avec Lightning...")
try:
    model_loaded = MultiPollutantLightningModule.load_from_checkpoint(
        ckpt_path,
        config=config,
        strict=False
    )
    print("✅ Modèle chargé avec Lightning")

    # Compare some weights
    print("\n  📊 Comparaison des poids:")
    sample_param_name = list(model_loaded.state_dict().keys())[0]

    new_param = model_new.state_dict()[sample_param_name]
    loaded_param = model_loaded.state_dict()[sample_param_name]

    print(f"    Paramètre: {sample_param_name}")
    print(f"    - Nouveau (random init): mean={new_param.float().mean():.6f}, std={new_param.float().std():.6f}")
    print(f"    - Chargé (checkpoint): mean={loaded_param.float().mean():.6f}, std={loaded_param.float().std():.6f}")

    if torch.allclose(new_param, loaded_param):
        print("    ❌ ATTENTION: Les poids sont identiques! Le checkpoint ne s'est pas chargé!")
    else:
        print("    ✅ Les poids sont différents (bon signe)")

except Exception as e:
    print(f"❌ Erreur lors du chargement avec Lightning: {e}")
    import traceback
    traceback.print_exc()

# 6. Check hyperparameters
print("\n" + "="*80)
print("6️⃣ HYPERPARAMÈTRES DANS LE CHECKPOINT")
print("="*80)

if 'hyper_parameters' in ckpt:
    hparams = ckpt['hyper_parameters']
    print("✅ Hyperparamètres trouvés:")

    # Check critical hyperparameters
    if 'config' in hparams:
        saved_config = hparams['config']

        # Compare learning rate
        saved_lr = saved_config.get('train', {}).get('learning_rate')
        current_lr = config.get('train', {}).get('learning_rate')

        print(f"\n  📊 Learning Rate:")
        print(f"    - Checkpoint: {saved_lr}")
        print(f"    - Config actuelle: {current_lr}")
        if saved_lr != current_lr:
            print(f"    ⚠️  ATTENTION: LR différent!")

        # Compare batch size
        saved_bs = saved_config.get('train', {}).get('batch_size')
        current_bs = config.get('train', {}).get('batch_size')

        print(f"\n  📊 Batch Size:")
        print(f"    - Checkpoint: {saved_bs}")
        print(f"    - Config actuelle: {current_bs}")
        if saved_bs != current_bs:
            print(f"    ⚠️  ATTENTION: Batch size différent!")

        # Compare accumulate_grad_batches
        saved_acc = saved_config.get('train', {}).get('accumulate_grad_batches')
        current_acc = config.get('train', {}).get('accumulate_grad_batches')

        print(f"\n  📊 Accumulate Grad Batches:")
        print(f"    - Checkpoint: {saved_acc}")
        print(f"    - Config actuelle: {current_acc}")
        if saved_acc != current_acc:
            print(f"    ⚠️  ATTENTION: accumulate_grad_batches différent!")

print("\n" + "="*80)
print("✅ DIAGNOSTIC TERMINÉ")
print("="*80)
