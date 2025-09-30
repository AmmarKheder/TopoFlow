#!/usr/bin/env python3
"""
Génère des indices fixes pour l'évaluation 
1000 échantillons par horizon pour cohérence entre modèles
"""

import sys
sys.path.append('src')
import numpy as np
import json
import random
from dataloader import create_dataset
import yaml

def generate_fixed_indices():
    print('🎯 GÉNÉRATION INDICES FIXES POUR ÉVALUATION')
    print('='*50)
    
    # Charger config
    with open('configs/config_all_pollutants.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Créer dataset complet
    dataset = create_dataset(config, mode='test')  # 2018
    forecast_hours = config["data"]["forecast_hours"]
    
    print(f'📊 Dataset total: {len(dataset):,} échantillons')
    print(f'📋 Horizons: {forecast_hours} heures')
    
    # Grouper les indices par horizon
    indices_by_horizon = {h: [] for h in forecast_hours}
    
    for idx, (ds_idx, t, forecast_h) in enumerate(dataset.valid_indices):
        indices_by_horizon[forecast_h].append(idx)
    
    print(f'\n📈 Échantillons par horizon:')
    for h in forecast_hours:
        print(f'   {h}h: {len(indices_by_horizon[h]):,} échantillons')
    
    # Sélectionner 1000 indices aléatoires par horizon
    random.seed(42)  # Seed fixe pour reproductibilité
    fixed_indices = {}
    
    for h in forecast_hours:
        available = indices_by_horizon[h]
        if len(available) >= 1000:
            selected = random.sample(available, 1000)
        else:
            selected = available
            print(f'⚠️  Seulement {len(available)} échantillons disponibles pour {h}h')
        
        fixed_indices[h] = sorted(selected)
        print(f'✅ {h}h: {len(selected)} indices sélectionnés')
    
    # Sauvegarder
    output_file = 'data_processed/fixed_eval_indices.json'
    with open(output_file, 'w') as f:
        json.dump(fixed_indices, f, indent=2)
    
    print(f'\n💾 Indices sauvegardés dans: {output_file}')
    
    # Statistiques finales
    total_samples = sum(len(indices) for indices in fixed_indices.values())
    print(f'📊 Total échantillons fixes: {total_samples:,}')
    
    return fixed_indices

if __name__ == '__main__':
    generate_fixed_indices()
