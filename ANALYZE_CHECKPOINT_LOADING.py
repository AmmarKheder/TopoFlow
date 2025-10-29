#!/usr/bin/env python3
"""
Analyse pourquoi le checkpoint 0.3557 ne reprend pas correctement.
"""
import torch
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.model_multipollutants import MultiPollutantLightningModule

print("="*100)
print("ANALYSE DU CHECKPOINT 0.3557")
print("="*100)

# Config
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config

ckpt_path = config['model']['checkpoint_path']
print(f"\nCheckpoint: {ckpt_path}\n")

# Load checkpoint raw
print("📦 Loading checkpoint dict...")
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"\n📊 Checkpoint keys: {list(ckpt.keys())}")
print(f"\n📊 Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"📊 Global step: {ckpt.get('global_step', 'N/A')}")

# Analyze optimizer states
if 'optimizer_states' in ckpt and len(ckpt['optimizer_states']) > 0:
    opt_state = ckpt['optimizer_states'][0]
    print(f"\n📊 Optimizer state keys: {list(opt_state.keys())}")

    if 'param_groups' in opt_state:
        print(f"\n📊 CHECKPOINT PARAM GROUPS:")
        for i, pg in enumerate(opt_state['param_groups']):
            print(f"  Group {i}:")
            print(f"    LR: {pg.get('lr', 'N/A'):.6e}")
            print(f"    Weight decay: {pg.get('weight_decay', 'N/A')}")
            print(f"    Num params: {len(pg.get('params', []))}")
            if 'name' in pg:
                print(f"    Name: {pg['name']}")

    if 'state' in opt_state:
        print(f"\n📊 Optimizer state dict has {len(opt_state['state'])} parameter states")
        # Check a few states
        for i, (param_id, state) in enumerate(opt_state['state'].items()):
            if i < 3:
                print(f"  Param {param_id}: step={state.get('step', 'N/A')}, "
                      f"exp_avg shape={state.get('exp_avg', torch.tensor([])).shape}")

# Analyze LR scheduler
if 'lr_schedulers' in ckpt and len(ckpt['lr_schedulers']) > 0:
    sch_state = ckpt['lr_schedulers'][0]
    print(f"\n📊 LR SCHEDULER STATE:")
    print(f"  last_epoch: {sch_state.get('last_epoch', 'N/A')}")
    print(f"  base_lrs: {sch_state.get('base_lrs', 'N/A')}")
    print(f"  _last_lr: {sch_state.get('_last_lr', 'N/A')}")

# Now load model using Lightning
print("\n" + "="*100)
print("🔥 LOADING MODEL VIA LIGHTNING")
print("="*100)

model = MultiPollutantLightningModule.load_from_checkpoint(
    ckpt_path,
    config=config,
    strict=False
)

print("\n✅ Model loaded")
print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters())}")
print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Check a few weights
print("\n📊 Sample model weights:")
for name, param in list(model.named_parameters())[:5]:
    print(f"  {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")

print("\n" + "="*100)
print("🔧 PROBLÈME POTENTIEL:")
print("="*100)
print("""
Le checkpoint a été sauvegardé avec une certaine configuration d'optimizer,
mais le code actuel crée 4 param_groups différents (vit_blocks, wind_embedding, head, others).

Si le checkpoint original avait UNE SEULE param_group ou une config différente,
alors le mapping des optimizer states échoue et les moments Adam sont perdus.

SOLUTION:
1. Vérifier combien de param_groups avait le checkpoint original
2. Si différent, il faut soit:
   a) Charger avec la MÊME config d'optimizer que l'original
   b) Ou ignorer l'optimizer state et juste charger les poids (mais LR ne sera pas optimal)
""")

print("="*100)
