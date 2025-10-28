"""Tracer d'où vient le préfixe 'model.' - analyse en profondeur."""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

# Force clean import
for module in list(sys.modules.keys()):
    if 'climax' in module or 'model_multi' in module:
        del sys.modules[module]

from model_multipollutants import MultiPollutantModel

print("="*70)
print("TRAÇAGE EN PROFONDEUR: D'OÙ VIENT LE PRÉFIXE 'model.'?")
print("="*70)

# Load config
with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

# Create model
print("\n1. Création du modèle MultiPollutantLightningModule...")
model = MultiPollutantModel(config)

print("\n2. Structure du modèle:")
print(f"   Type: {type(model)}")
print(f"   Class name: {model.__class__.__name__}")

# Check attributes
print("\n3. Attributs du modèle:")
for attr_name in dir(model):
    if not attr_name.startswith('_'):
        attr = getattr(model, attr_name)
        if isinstance(attr, torch.nn.Module):
            print(f"   model.{attr_name}: {type(attr).__name__}")

print("\n4. Sous-modules (named_children):")
for name, module in model.named_children():
    print(f"   {name}: {type(module).__name__}")

print("\n5. Tous les paramètres (premiers 20):")
for i, (name, param) in enumerate(model.named_parameters()):
    if i < 20:
        print(f"   {name}: {param.shape}")
    else:
        print(f"   ... ({len(list(model.named_parameters()))} total)")
        break

print("\n6. Vérifions si 'model' est un attribut:")
if hasattr(model, 'model'):
    print(f"   ✅ model.model existe!")
    print(f"   Type: {type(model.model)}")
    print(f"   Class: {model.model.__class__.__name__}")

    # Check if model.model has climax
    if hasattr(model.model, 'climax'):
        print(f"   ✅ model.model.climax existe!")
        print(f"   Type: {type(model.model.climax)}")
else:
    print(f"   ❌ model.model N'EXISTE PAS")

if hasattr(model, 'climax'):
    print(f"   ✅ model.climax existe!")
    print(f"   Type: {type(model.climax)}")
else:
    print(f"   ❌ model.climax N'EXISTE PAS")

# Load checkpoint and check structure
print(f"\n{'='*70}")
print("7. ANALYSE DU CHECKPOINT:")
print(f"{'='*70}")

checkpoint_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"\nClés du checkpoint:")
for key in checkpoint.keys():
    print(f"   {key}")

print(f"\nPremières clés dans state_dict (20):")
state_dict = checkpoint['state_dict']
for i, key in enumerate(sorted(state_dict.keys())):
    if i < 20:
        print(f"   {key}: {state_dict[key].shape}")
    else:
        print(f"   ... ({len(state_dict)} total)")
        break

# Analyze the prefix structure
print(f"\n{'='*70}")
print("8. ANALYSE DU PRÉFIXE:")
print(f"{'='*70}")

prefixes = {}
for key in state_dict.keys():
    parts = key.split('.')
    prefix = parts[0] if len(parts) > 0 else 'NO_PREFIX'
    if prefix not in prefixes:
        prefixes[prefix] = 0
    prefixes[prefix] += 1

print(f"\nPréfixes trouvés:")
for prefix, count in sorted(prefixes.items()):
    print(f"   '{prefix}': {count} paramètres")

# Check if checkpoint has hyper_parameters
if 'hyper_parameters' in checkpoint:
    print(f"\n9. HYPER_PARAMETERS du checkpoint:")
    hp = checkpoint['hyper_parameters']
    if 'config' in hp:
        config_ckpt = hp['config']
        if 'model' in config_ckpt:
            print(f"   Config model:")
            for k, v in config_ckpt['model'].items():
                print(f"      {k}: {v}")

# Try to understand the wrapping
print(f"\n{'='*70}")
print("10. COMPRÉHENSION DU WRAPPING:")
print(f"{'='*70}")

print(f"\nLe checkpoint a été sauvegardé par PyTorch Lightning.")
print(f"PyTorch Lightning enregistre les paramètres comme suit:")
print(f"   - Si le module principal s'appelle 'model', alors: model.xxx")
print(f"   - Si pas de wrapping, alors: xxx")

print(f"\nDans notre cas:")
print(f"   Checkpoint: TOUTES les clés commencent par 'model.'")
print(f"   Modèle actuel: AUCUNE clé ne commence par 'model.'")

print(f"\n❓ QUESTION: Le modèle qui a créé le checkpoint avait-il un attribut 'model'?")
print(f"\nRéponse probable: OUI!")
print(f"   class SomeModule(pl.LightningModule):")
print(f"       def __init__(self):")
print(f"           self.model = MultiPollutantPredictionNet(...)")
print(f"")
print(f"   Alors PyTorch Lightning sauvegarde: 'model.climax.xxx'")

print(f"\n❓ Et le modèle actuel?")
print(f"\nRéponse: Le modèle actuel HÉRITE DIRECTEMENT!")
print(f"   class MultiPollutantLightningModule(pl.LightningModule):")
print(f"       def __init__(self):")
print(f"           self.climax = ClimaX(...)")
print(f"")
print(f"   Alors PyTorch Lightning sauvegarde: 'climax.xxx'")

print(f"\n{'='*70}")
print("CONCLUSION:")
print(f"{'='*70}")
print(f"\n⚠️  INCOMPATIBILITÉ DE STRUCTURE!")
print(f"")
print(f"   Checkpoint (version_47, sept 2024):")
print(f"      Module avec self.model = MultiPollutantPredictionNet()")
print(f"      → Clés: model.climax.xxx")
print(f"")
print(f"   Modèle actuel (oct 2025):")
print(f"      Module avec self.climax = ClimaX() directement")
print(f"      → Clés: climax.xxx")
print(f"")
print(f"✅ SOLUTION: Strip 'model.' du checkpoint AVANT de charger!")
print("="*70)
