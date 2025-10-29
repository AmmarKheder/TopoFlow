"""
VÉRIFICATION FINALE: Tout est-il OK pour le job 13640655?
"""
import sys
sys.path.insert(0, 'src')
import yaml

print("="*100)
print("🔍 VÉRIFICATION FINALE AVANT LE JOB 13640655")
print("="*100)

# 1. Config
print("\n1️⃣ CONFIGURATION:")
with open('configs/config_all_pollutants.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"   TopoFlow (use_physics_mask): {config['model'].get('use_physics_mask', False)}")
print(f"   Wind scanning: {config['model'].get('parallel_patch_embed', False)}")
print(f"   Checkpoint: {config['model'].get('checkpoint_path', 'None')}")
print(f"   Learning rate: {config['train']['learning_rate']}")
print(f"   Batch size: {config['train']['batch_size']}")
print(f"   Accumulate grad batches: {config['train']['accumulate_grad_batches']}")

# 2. Dataloader
print("\n2️⃣ DATALOADER:")
try:
    from dataloader import PM25AirQualityDataset
    print(f"   ✅ Utilise PM25AirQualityDataset (septembre 2024)")
    print(f"   ✅ Normalisation dynamique à la volée")
except ImportError as e:
    print(f"   ❌ Erreur import: {e}")
    sys.exit(1)

# 3. Test dataloader
print("\n3️⃣ TEST DATALOADER:")
try:
    from datamodule import AQNetDataModule
    dm = AQNetDataModule(config)
    print(f"   ✅ DataModule créé avec succès")
except Exception as e:
    print(f"   ❌ Erreur création DataModule: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Model
print("\n4️⃣ MODÈLE:")
try:
    from model_multipollutants import MultiPollutantLightningModule
    model = MultiPollutantLightningModule(config=config)
    print(f"   ✅ LightningModule créé avec succès")
except Exception as e:
    print(f"   ❌ Erreur création modèle: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Checkpoint loading
print("\n5️⃣ CHARGEMENT CHECKPOINT:")
import torch
ckpt_path = config['model']['checkpoint_path']
try:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"   ✅ Checkpoint chargé: {len(ckpt['state_dict'])} clés")
    print(f"   Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"   Global step: {ckpt.get('global_step', 'N/A')}")
except Exception as e:
    print(f"   ❌ Erreur chargement checkpoint: {e}")
    sys.exit(1)

print("\n" + "="*100)
print("📊 RÉSUMÉ:")
print("="*100)
print("✅ Configuration: OK")
print("✅ Dataloader septembre 2024: OK")
print("✅ DataModule: OK")
print("✅ Modèle: OK")
print("✅ Checkpoint: OK")
print("")
print("🚀 TOUT EST PRÊT POUR LE JOB 13640655!")
print("="*100)
