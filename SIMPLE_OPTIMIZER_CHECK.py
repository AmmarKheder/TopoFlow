"""
TEST SIMPLE: Comparer les LRs attendues vs réelles après chargement
"""
import torch

ckpt_path = 'logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt'

print("="*80)
print("ANALYSE RAPIDE: Optimizer LRs dans le checkpoint")
print("="*80)

# Load checkpoint
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print(f"\nCheckpoint info:")
print(f"  epoch: {checkpoint['epoch']}")
print(f"  global_step: {checkpoint['global_step']}")

# Optimizer state
opt_state = checkpoint['optimizer_states'][0]
print(f"\nOptimizer state:")
print(f"  Number of param_groups: {len(opt_state['param_groups'])}")

for i, pg in enumerate(opt_state['param_groups']):
    num_params = len(pg['params'])
    lr = pg['lr']
    print(f"    Group {i}: {num_params} params, lr={lr:.6e}")

# LR Scheduler
sch_state = checkpoint['lr_schedulers'][0]
print(f"\nLR Scheduler state:")
print(f"  last_epoch: {sch_state['last_epoch']}")
print(f"  base_lrs: {sch_state['base_lrs']}")
print(f"  current_lrs (_last_lr): {sch_state['_last_lr']}")

print("\n" + "="*80)
print("COMPARAISON AVEC LES LOGS DU JOB 13775567:")
print("="*80)
print("\nLogs du job montrent (au démarrage):")
print("  vit_blocks: LR=1.50e-05 (72 param groups)")
print("  wind_embedding: LR=3.00e-04 (2 param groups)")
print("  head: LR=7.50e-05 (6 param groups)")
print("  others: LR=1.50e-04 (11 param groups)")

print("\nCheckpoint devrait avoir (epoch 3, CosineAnnealing décroissant):")
for i, lr in enumerate(sch_state['_last_lr']):
    print(f"  Group {i}: LR={lr:.6e}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)

job_lrs = [1.50e-05, 3.00e-04, 7.50e-05, 1.50e-04]
ckpt_lrs = sch_state['_last_lr']

print(f"\nJob LRs:        {[f'{lr:.2e}' for lr in job_lrs]}")
print(f"Checkpoint LRs: {[f'{lr:.2e}' for lr in ckpt_lrs]}")

if job_lrs != ckpt_lrs:
    print("\n❌ LRs DON'T MATCH!")
    print("   L'optimizer n'est PAS chargé depuis le checkpoint.")
    print("   Le job utilise les LRs de base (config), pas ceux du scheduler!")
    print("\n   CAUSE PROBABLE:")
    print("   1. PyTorch Lightning ne charge pas l'optimizer state en DDP")
    print("   2. Ou il y a un problème de compatibilité des param_groups")
    print("   3. Ou le checkpoint n'est pas vraiment chargé (silent fail)")
else:
    print("\n✅ LRs MATCH! Optimizer state loaded successfully.")

print("="*80)
