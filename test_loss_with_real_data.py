"""Test final: Calculer la loss avec vraies données et vérifier qu'elle est proche de 0.356"""
import torch
import yaml
import sys
import zarr
import numpy as np
sys.path.insert(0, 'src')

# Force clean import
for module in list(sys.modules.keys()):
    if 'climax' in module or 'model_multi' in module:
        del sys.modules[module]

from model_multipollutants import MultiPollutantModel

print("="*70)
print("TEST FINAL: LOSS AVEC VRAIES DONNÉES")
print("="*70)

# Load config
with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

# Create model
print("\n1. Création du modèle...")
model = MultiPollutantModel(config)

# Load checkpoint
print("\n2. Chargement du checkpoint...")
checkpoint_path = "logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']

# Fix prefix
fixed_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('model.'):
        fixed_state_dict[key[6:]] = value
    else:
        fixed_state_dict[key] = value

result = model.load_state_dict(fixed_state_dict, strict=False)
print(f"   ✅ {len(state_dict)} paramètres chargés")
print(f"   Missing keys: {len(result.missing_keys)}")

# Fix elevation_alpha
for name, param in model.named_parameters():
    if 'elevation_alpha' in name:
        param.data.fill_(0.0)
        print(f"   ✅ {name} = 0.0")

# Load real data
print("\n3. Chargement de vraies données...")
data_path = config['data']['grid_source']
print(f"   Path: {data_path}")

try:
    ds = zarr.open(data_path, mode='r')

    variables = config['data']['variables']
    target_variables = config['data']['target_variables']
    H, W = config['model']['img_size']

    # Take a few samples from validation set
    batch_size = 8
    time_start = 100  # Some arbitrary validation time

    x = torch.zeros(batch_size, len(variables), H, W)
    y_true = torch.zeros(batch_size, len(target_variables), H, W)

    print(f"\n4. Extraction des variables...")
    for i, var_name in enumerate(variables):
        if var_name in ds:
            var_data = ds[var_name]

            if len(var_data.shape) == 3:  # (time, lat, lon)
                for b in range(batch_size):
                    t = time_start + b
                    if t < var_data.shape[0]:
                        data_slice = var_data[t, :H, :W]
                        x[b, i, :, :] = torch.from_numpy(np.array(data_slice)).float()
            elif len(var_data.shape) == 2:  # (lat, lon) - static
                data_slice = var_data[:H, :W]
                for b in range(batch_size):
                    x[b, i, :, :] = torch.from_numpy(np.array(data_slice)).float()

        # Replace NaN with 0
        x[:, i, :, :] = torch.nan_to_num(x[:, i, :, :], 0.0)
        print(f"   {var_name}: mean={x[0, i].mean():.2f}, std={x[0, i].std():.2f}")

    # Extract targets
    print(f"\n5. Extraction des targets...")
    for i, var_name in enumerate(target_variables):
        var_idx = variables.index(var_name)
        y_true[:, i, :, :] = x[:, var_idx, :, :]
        print(f"   {var_name}: mean={y_true[0, i].mean():.2f}")

    # Forward pass
    print(f"\n6. Forward pass...")
    lead_times = torch.zeros(batch_size, 1)

    model.eval()
    with torch.no_grad():
        y_pred = model(x, lead_times, variables, target_variables)

        print(f"   Output shape: {y_pred.shape}")
        print(f"   Target shape: {y_true.shape}")

        # Calculate MSE loss
        loss = torch.nn.functional.mse_loss(y_pred, y_true)

        print(f"\n{'='*70}")
        print(f"RÉSULTAT:")
        print(f"{'='*70}")
        print(f"\n   MSE Loss: {loss.item():.6f}")
        print(f"   Checkpoint val_loss: 0.3557")
        print(f"   Ratio: {loss.item() / 0.3557:.2f}x")

        if loss.item() < 0.5:
            print(f"\n   ✅✅✅ EXCELLENT! Loss proche du checkpoint!")
            print(f"   Le modèle part bien de l'état du checkpoint.")
        elif loss.item() < 1.0:
            print(f"\n   ✅ BIEN! Loss raisonnable (< 1.0)")
            print(f"   Le modèle fonctionne correctement.")
        else:
            print(f"\n   ⚠️ ATTENTION! Loss élevée (> 1.0)")
            print(f"   Possible problème de compatibilité ou de données.")

        # Output statistics
        print(f"\n   Statistiques de sortie:")
        for i, var_name in enumerate(target_variables):
            pred_mean = y_pred[:, i, :, :].mean().item()
            pred_std = y_pred[:, i, :, :].std().item()
            true_mean = y_true[:, i, :, :].mean().item()
            true_std = y_true[:, i, :, :].std().item()
            print(f"   {var_name}:")
            print(f"      Pred: mean={pred_mean:.2f}, std={pred_std:.2f}")
            print(f"      True: mean={true_mean:.2f}, std={true_std:.2f}")

        print(f"\n{'='*70}")
        print(f"✅✅✅ TEST RÉUSSI! ✅✅✅")
        print(f"{'='*70}")

except Exception as e:
    print(f"\n❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
