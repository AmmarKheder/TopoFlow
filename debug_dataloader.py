#!/usr/bin/env python3
"""
Test simple pour diagnostiquer le problème du dataloader
"""
import sys
sys.path.append('src')
sys.path.append('.')

from src.dataloader_zarr_optimized import create_dataset
import yaml

print("🔍 DIAGNOSTIC DU DATALOADER")
print("="*50)

# Charger la config
print("📋 Chargement de la configuration...")
with open('configs/config_all_pollutants.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"🎯 Polluants cibles configurés: {config['data']['target_variables']}")
print(f"📊 Variables d'entrée: {len(config['data']['variables'])}")

# Créer le dataset
print("\n📁 Création du dataset...")
try:
    dataset = create_dataset(config, mode='train')
    print(f"✅ Dataset créé: {len(dataset)} échantillons")
    
    # Tester un échantillon
    print("\n🧪 Test d'un échantillon...")
    sample = dataset[0]
    
    input_tensor, target_tensor, lead_time, variables = sample
    
    print(f"📥 Input shape: {input_tensor.shape}")
    print(f"🎯 Target shape: {target_tensor.shape}")
    print(f"⏱️  Lead time: {lead_time}")
    print(f"📋 Variables: {len(variables)}")
    
    print("\n" + "="*50)
    if target_tensor.shape[0] == 1:
        print("❌ PROBLÈME CONFIRMÉ!")
        print("   Le dataloader ne charge qu'UN seul polluant cible")
        print("   au lieu des 6 polluants configurés!")
    else:
        print("✅ Pas de problème:")
        print(f"   Le dataloader charge {target_tensor.shape[0]} polluants")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
