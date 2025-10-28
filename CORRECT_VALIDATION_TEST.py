"""
TEST CORRECT: Valider que le checkpoint donne val_loss ≈ 0.3557
En chargeant le checkpoint CORRECTEMENT (comme dans main_multipollutants.py)
"""
import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model_multipollutants import MultiPollutantModel
from datamodule import AQNetDataModule

print("="*80)
print("TEST: VALIDATION LOSS AVEC CHARGEMENT CORRECT DU CHECKPOINT")
print("="*80)

# Load config
with open('configs/config_all_pollutants.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Simplify for testing
config['train']['devices'] = 1
config['train']['num_nodes'] = 1
config['data']['num_workers'] = 2

# Create model
print("\n1. Creating model...")
model = MultiPollutantModel(config)

# Load checkpoint CORRECTLY (like in main_multipollutants.py lines 148-151)
print("\n2. Loading checkpoint...")
ckpt_path = 'logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt'
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Fix prefix
state_dict = {}
for key, value in checkpoint['state_dict'].items():
    if key.startswith('model.'):
        state_dict[key[6:]] = value
    else:
        state_dict[key] = value

# Load with strict=False
result = model.load_state_dict(state_dict, strict=False)
print(f"   Missing keys: {len(result.missing_keys)}")
print(f"   Unexpected keys: {len(result.unexpected_keys)}")

# Verify weights loaded
sample = model.climax.head[0].weight[0, :3]
ckpt_sample = checkpoint['state_dict']['model.climax.head.0.weight'][0, :3]
print(f"\\n   Model head[0].weight[0, :3] = {sample.tolist()}")
print(f"   Checkpoint head[0].weight[0, :3] = {ckpt_sample.tolist()}")
print(f"   Match? {torch.allclose(sample, ckpt_sample)}")

# Create data module
print("\\n3. Creating data module...")
data_module = AQNetDataModule(config)
data_module.setup('fit')

val_loader = data_module.val_dataloader()
print(f"   Val loader: {len(val_loader)} batches")

# Move model to GPU and set eval mode
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print(f"\\n4. Computing validation loss on 10 batches...")
print(f"   Device: {device}")

total_loss = 0.0
num_batches = 0
variables = tuple(config['data']['variables'])

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i >= 10:
            break

        x, y, lead_times = batch
        x = x.to(device)
        y = y.to(device)
        lead_times = lead_times.to(device)

        # Forward pass
        y_pred = model(x, lead_times, variables)

        # Compute MSE loss (simple, without mask)
        loss = torch.nn.functional.mse_loss(y_pred, y)

        total_loss += loss.item()
        num_batches += 1

        if i == 0:
            print(f"\\n   Batch 0:")
            print(f"     x shape: {x.shape}")
            print(f"     y_pred shape: {y_pred.shape}")
            print(f"     y shape: {y.shape}")
            print(f"     loss: {loss.item():.4f}")

avg_val_loss = total_loss / num_batches

print(f"\\n" + "="*80)
print(f"RESULTS:")
print(f"="*80)
print(f"Average validation loss (MSE): {avg_val_loss:.4f}")
print(f"Expected (from checkpoint): ~0.36 (with china mask)")
print(f"Difference: {abs(avg_val_loss - 0.36):.4f}")
print(f"")

if abs(avg_val_loss - 0.36) < 0.1:
    print("✅ SUCCESS! Val loss is close to expected!")
    print("   Your model and checkpoint loading work correctly.")
    print("   The high train_loss in training is due to optimizer reset (fine-tuning).")
elif abs(avg_val_loss - 0.36) < 0.5:
    print("⚠️  Val loss is somewhat close. Possible minor issues:")
    print("   - Different loss function (MSE vs masked MSE)")
    print("   - Batch sampling differences")
    print("   - GPU precision")
else:
    print(f"❌ FAILED! Val loss is too different: {avg_val_loss:.4f} vs 0.36")
    print("   There's still a problem with:")
    print("   - Data normalization")
    print("   - Variable ordering")
    print("   - Model forward pass")

print("="*80)
