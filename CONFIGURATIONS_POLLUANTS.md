=== RÉSUMÉ DES CONFIGURATIONS CRÉÉES ===

# Configurations pour la prédiction des polluants atmosphériques

## Configuration actuelle
- **Fichier**: `configs/config.yaml`
- **Target**: PM2.5 (particules fines)
- **Description**: Configuration de base pour la prédiction des PM2.5

## Configurations individuelles par polluant

### 1. PM10 (Particules grossières)
- **Fichier**: `configs/config_pm10.yaml`
- **Target**: `pm10`
- **Usage**: `python main.py --config configs/config_pm10.yaml`

### 2. SO2 (Dioxyde de soufre)
- **Fichier**: `configs/config_so2.yaml` 
- **Target**: `so2`
- **Usage**: `python main.py --config configs/config_so2.yaml`

### 3. NO2 (Dioxyde d'azote)
- **Fichier**: `configs/config_no2.yaml`
- **Target**: `no2`
- **Usage**: `python main.py --config configs/config_no2.yaml`

### 4. CO (Monoxyde de carbone)
- **Fichier**: `configs/config_co.yaml`
- **Target**: `co`
- **Usage**: `python main.py --config configs/config_co.yaml`

### 5. O3 (Ozone)
- **Fichier**: `configs/config_o3.yaml`
- **Target**: `o3`
- **Usage**: `python main.py --config configs/config_o3.yaml`

## Configuration multi-polluants

### 6. Tous les polluants simultanément
- **Fichier**: `configs/config_all_pollutants.yaml`
- **Targets**: `['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']`
- **Usage**: `python main.py --config configs/config_all_pollutants.yaml`
- **Avantage**: Prédiction simultanée des 6 polluants avec un seul modèle

## Variables disponibles dans le dataset

### Variables météorologiques (input)
- `u`, `v` : composantes du vent
- `temp` : température
- `rh` : humidité relative  
- `psfc` : pression de surface

### Variables de polluants (input + targets)
- `pm25` : particules fines (≤ 2.5 μm)
- `pm10` : particules grossières (≤ 10 μm)
- `so2` : dioxyde de soufre
- `no2` : dioxyde d'azote
- `co` : monoxyde de carbone
- `o3` : ozone

### Variables de coordonnées
- `lat2d`, `lon2d` : latitude et longitude

## Recommandations d'usage

1. **Pour une prédiction spécialisée** : Utilisez les configurations individuelles
2. **Pour une analyse comparative** : Utilisez la configuration multi-polluants
3. **Pour des ressources limitées** : Commencez par PM2.5 ou PM10 (polluants les plus critiques)

