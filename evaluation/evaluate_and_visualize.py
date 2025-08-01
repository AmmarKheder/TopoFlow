import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from some_module import CustomDataset, load_model, predict

# Set paths
data_path = "data/"
model_path = "model/checkpoint.pth"
output_dir = "results/"
os.makedirs(output_dir, exist_ok=True)

# Set parameters
batch_size = 1
forecast_horizons = [24, 72, 120, 168]  # hours
resolution = [128, 256]

# Load model
model = load_model(model_path)

# Define dataset and dataloader
val_dataset = CustomDataset(data_path, resolution)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize results holders
rmse_per_horizon = {h: [] for h in forecast_horizons}

# Evaluation loop
model.eval()
with torch.no_grad():
    for X, y_true in val_loader:
        y_pred = predict(model, X, resolution)

        # Compute RMSE per forecast horizon
        for h in forecast_horizons:
            pred_h = y_pred[:h]
            true_h = y_true[:h]
            rmse = np.sqrt(np.mean((pred_h - true_h) ** 2))
            rmse_per_horizon[h].append(rmse)

# Average RMSE across samples
avg_rmse = {h: np.mean(rmse_per_horizon[h]) for h in forecast_horizons}

# Plot RMSE
plt.figure(figsize=(10, 6))
horizons_days = [h // 24 for h in forecast_horizons]
plt.plot(horizons_days, list(avg_rmse.values()), marker='o')
plt.title('RMSE per Forecast Horizon')
plt.xlabel('Horizon (days)')
plt.ylabel('RMSE')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'rmse_per_horizon.png'))
plt.close()

print('Evaluation complete. Results saved to', output_dir)
