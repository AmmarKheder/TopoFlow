"""Test CRITIQUE: Déterminer quel wind scanner cache est compatible avec le checkpoint.

Méthode: Tester les deux caches et voir lequel donne la loss la plus proche de 0.356
"""
import torch
import yaml
import sys
import shutil
import os
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantModel

print("="*70)
print("TEST CRITIQUE: QUEL CACHE EST COMPATIBLE AVEC LE CHECKPOINT?")
print("="*70)

# Load config
with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

checkpoint_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"

# Paths to the two caches
cache1_path = "/scratch/project_462000640/ammar/aq_net2/wind_scanner_cache.pkl"
cache2_path = "/scratch/project_462000640/ammar/aq_net2/data_processed/wind_cache_64x128.pkl"

print(f"\n📁 Caches à tester:")
print(f"   Cache 1: {cache1_path}")
print(f"   Cache 2: {cache2_path}")

# Create synthetic but realistic data
print(f"\n🔧 Création de données synthétiques...")
batch_size = 4
variables = config['data']['variables']
target_variables = config['data']['target_variables']
H, W = config['model']['img_size']

# Create realistic synthetic data
x = torch.randn(batch_size, len(variables), H, W)
for i, var_name in enumerate(variables):
    if var_name == 'u':
        x[:, i, :, :] = torch.randn(batch_size, H, W) * 5
    elif var_name == 'v':
        x[:, i, :, :] = torch.randn(batch_size, H, W) * 5
    elif var_name == 'temp':
        x[:, i, :, :] = 280 + torch.randn(batch_size, H, W) * 10
    elif var_name == 'elevation':
        x[:, i, :, :] = torch.abs(torch.randn(batch_size, H, W) * 500)
    elif var_name in ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']:
        x[:, i, :, :] = torch.abs(torch.randn(batch_size, H, W) * 30)

# Create target (same as input for this test)
y = x[:, [variables.index(v) for v in target_variables], :, :]

lead_times = torch.zeros(batch_size, 1)

print(f"   Input shape: {x.shape}")
print(f"   Target shape: {y.shape}")

# Test function
def test_with_cache(cache_path, cache_name):
    print(f"\n{'='*70}")
    print(f"TEST AVEC {cache_name}")
    print(f"{'='*70}")

    # Backup current cache and use test cache
    temp_cache = "/tmp/wind_scanner_cache_temp.pkl"
    if os.path.exists(temp_cache):
        os.remove(temp_cache)
    shutil.copy(cache1_path, temp_cache)  # Backup

    try:
        # Replace with test cache
        shutil.copy(cache_path, cache1_path)

        # Create fresh model (will load this cache)
        print(f"\n1. Création du modèle...")
        # Need to reimport to force cache reload
        import importlib
        import model_multipollutants
        importlib.reload(model_multipollutants)

        model = model_multipollutants.MultiPollutantModel(config)

        # Load checkpoint
        print(f"2. Chargement du checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']

        # Apply prefix fix
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]
                fixed_state_dict[new_key] = value
            else:
                fixed_state_dict[key] = value

        result = model.load_state_dict(fixed_state_dict, strict=False)
        print(f"   Loaded: {len(state_dict)} params, Missing: {len(result.missing_keys)}")

        # Fix elevation_alpha
        for name, param in model.named_parameters():
            if 'elevation_alpha' in name:
                param.data.fill_(0.0)

        # Forward pass
        print(f"3. Forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(x, lead_times, variables, target_variables)

            # Compute loss
            loss = torch.nn.functional.mse_loss(output, y)

            print(f"\n📊 RÉSULTATS:")
            print(f"   Output shape: {output.shape}")
            print(f"   MSE Loss: {loss.item():.6f}")

            # Check for NaN/Inf
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()

            if has_nan:
                print(f"   ❌ Output contient NaN!")
            if has_inf:
                print(f"   ❌ Output contient Inf!")

            # Output stats
            print(f"\n   Output stats:")
            print(f"   - Mean: {output.mean().item():.4f}")
            print(f"   - Std: {output.std().item():.4f}")
            print(f"   - Min: {output.min().item():.4f}")
            print(f"   - Max: {output.max().item():.4f}")

        return loss.item(), has_nan, has_inf

    finally:
        # Restore original cache
        shutil.copy(temp_cache, cache1_path)
        os.remove(temp_cache)

# Run tests
results = {}

print(f"\n" + "="*70)
print(f"DÉBUT DES TESTS")
print(f"="*70)

try:
    loss1, nan1, inf1 = test_with_cache(cache1_path, "CACHE 1 (wind_scanner_cache.pkl)")
    results['cache1'] = {'loss': loss1, 'nan': nan1, 'inf': inf1}
except Exception as e:
    print(f"\n❌ Erreur avec Cache 1: {e}")
    import traceback
    traceback.print_exc()
    results['cache1'] = {'loss': float('inf'), 'nan': True, 'inf': True, 'error': str(e)}

try:
    loss2, nan2, inf2 = test_with_cache(cache2_path, "CACHE 2 (wind_cache_64x128.pkl)")
    results['cache2'] = {'loss': loss2, 'nan': nan2, 'inf': inf2}
except Exception as e:
    print(f"\n❌ Erreur avec Cache 2: {e}")
    import traceback
    traceback.print_exc()
    results['cache2'] = {'loss': float('inf'), 'nan': True, 'inf': True, 'error': str(e)}

# Compare results
print(f"\n" + "="*70)
print(f"COMPARAISON FINALE")
print(f"="*70)

print(f"\nCache 1 (wind_scanner_cache.pkl):")
if 'error' in results['cache1']:
    print(f"   ❌ ERREUR: {results['cache1']['error']}")
else:
    print(f"   Loss: {results['cache1']['loss']:.6f}")
    print(f"   NaN: {'❌ Oui' if results['cache1']['nan'] else '✅ Non'}")
    print(f"   Inf: {'❌ Oui' if results['cache1']['inf'] else '✅ Non'}")

print(f"\nCache 2 (wind_cache_64x128.pkl):")
if 'error' in results['cache2']:
    print(f"   ❌ ERREUR: {results['cache2']['error']}")
else:
    print(f"   Loss: {results['cache2']['loss']:.6f}")
    print(f"   NaN: {'❌ Oui' if results['cache2']['nan'] else '✅ Non'}")
    print(f"   Inf: {'❌ Oui' if results['cache2']['inf'] else '✅ Non'}")

print(f"\n" + "="*70)
print(f"RECOMMANDATION:")
print(f"="*70)

# Determine which is better
if 'error' not in results['cache1'] and 'error' not in results['cache2']:
    loss1 = results['cache1']['loss']
    loss2 = results['cache2']['loss']

    if loss1 < loss2:
        print(f"\n🏆 CACHE 1 est MEILLEUR (loss plus faible)")
        print(f"   ➡️ Utiliser: wind_scanner_cache.pkl")
    elif loss2 < loss1:
        print(f"\n🏆 CACHE 2 est MEILLEUR (loss plus faible)")
        print(f"   ➡️ Utiliser: wind_cache_64x128.pkl")
        print(f"\n⚠️  ACTION REQUISE:")
        print(f"   Mettre à jour le path du cache dans le code:")
        print(f"   /scratch/project_462000640/ammar/aq_net2/src/climax_core/parallelpatchembed_wind.py")
        print(f"   Ligne 96: cache_path = '.../data_processed/wind_cache_64x128.pkl'")
    else:
        print(f"\n🤔 Les deux caches donnent la MÊME loss")
        print(f"   ➡️ Ils sont peut-être équivalents")
else:
    print(f"\n⚠️  Impossible de déterminer - au moins un cache a causé une erreur")

print(f"\n⚠️  NOTE IMPORTANTE:")
print(f"   Ce test utilise des données SYNTHÉTIQUES.")
print(f"   La loss absolue n'est pas comparable à la val_loss du checkpoint (0.356).")
print(f"   MAIS: Le cache compatible devrait donner une loss RELATIVE plus faible.")

print(f"\n" + "="*70)
