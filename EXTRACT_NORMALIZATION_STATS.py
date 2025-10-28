#!/usr/bin/env python3
"""
Extraire les stats de normalisation utilisées lors de l'entraînement du checkpoint 0.3557
et les sauvegarder pour les réutiliser
"""
import torch
import pickle
import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2')

from src.config_manager import ConfigManager
from src.datamodule_fixed import AQNetDataModule

print("="*100)
print("EXTRACTION DES STATS DE NORMALISATION")
print("="*100)

# Config (même config que lors de l'entraînement original)
config_path = "/scratch/project_462000640/ammar/aq_net2/configs/config_all_pollutants.yaml"
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config
config['data']['num_workers'] = 0

print(f"\nTraining years: {config['data']['train_years']}")
print(f"Variables: {config['data']['variables']}")
print(f"Normalize: {config['data']['normalize']}")

# Créer le dataset d'entraînement pour calculer les stats
print("\n📊 Creating training dataset to compute stats...")
data_module = AQNetDataModule(config)
data_module.setup('fit')

# Extraire les stats du train dataset
if hasattr(data_module.train_dataset, 'stats'):
    stats = data_module.train_dataset.stats
    print("\n✅ Stats extracted from train dataset!")
    print(f"\n📊 Stats keys: {list(stats.keys())}")

    # Afficher quelques stats
    print("\n📊 Sample stats (first 3 variables):")
    for i, var in enumerate(config['data']['variables'][:3]):
        if var in stats:
            print(f"  {var}:")
            print(f"    mean: {stats[var]['mean']:.6f}")
            print(f"    std:  {stats[var]['std']:.6f}")

    # Sauvegarder les stats
    stats_file = "data_processed/normalization_stats_version_47.pkl"
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)

    print(f"\n✅ Stats sauvegardées dans: {stats_file}")

    # Vérifier qu'on peut les recharger
    with open(stats_file, 'rb') as f:
        loaded_stats = pickle.load(f)

    print(f"✅ Stats rechargées avec succès!")
    print(f"   {len(loaded_stats)} variables")

    print("\n" + "="*100)
    print("🎉 SUCCÈS!")
    print("="*100)
    print(f"\nLes stats ont été extraites et sauvegardées.")
    print(f"Elles peuvent maintenant être utilisées lors du resume pour avoir")
    print(f"EXACTEMENT les mêmes valeurs normalisées qu'à l'époque de l'entraînement.")
    print("\n📝 Fichier créé: {stats_file}")
    print("="*100)

else:
    print("\n❌ Pas de stats trouvées dans le train dataset!")
    print("   Vérifier que normalize=True dans la config.")
