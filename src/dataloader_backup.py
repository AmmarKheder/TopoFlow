from scipy.ndimage import zoom
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from datetime import datetime
import glob
from typing import List, Tuple

NORM_STATS = {
    "u": (0.0, 10.0), "v": (0.0, 10.0), "temp": (273.15, 30.0),
    "rh": (50.0, 30.0), "psfc": (101325.0, 1000.0), "pm10": (50.0, 25.0),
    "so2": (5.0, 5.0), "no2": (20.0, 15.0), "co": (200.0, 100.0),
    "o3": (40.0, 20.0), "pm25": (25.0, 15.0)
}

class CAQRADataset(Dataset):
    def __init__(self, data_path: str, years: List[int], variables: List[str], 
                 target_variables: List[str], time_history: int, time_future: int, 
                 normalize: bool, target_resolution: Tuple[int, int]):
        self.data_path = data_path
        self.years = years
        self.variables = variables
        self.target_variables = target_variables
        self.time_history = time_history
        self.time_future = time_future
        self.normalize = normalize
        self.target_resolution = target_resolution
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        all_files = []
        for year in self.years:
            pattern = os.path.join(self.data_path, f"{year}*", "*.nc")
            all_files.extend(sorted(glob.glob(pattern)))
        
        timestamps = []
        for f in all_files:
            basename = os.path.basename(f)[:-3]  # Remove .nc
            if basename.startswith("CN-Reanalysis"):
                date_part = basename[len("CN-Reanalysis"):]
                timestamps.append(datetime.strptime(date_part, "%Y%m%d%H"))
        
        for i in range(len(timestamps) - self.time_history - self.time_future + 1):
            input_ts = timestamps[i : i + self.time_history]
            target_ts = timestamps[i + self.time_history + self.time_future - 1]
            sequences.append({'input_times': input_ts, 'target_time': target_ts})
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        last_input = seq['input_times'][-1]
        target_dt = seq['target_time']
        
        # Load input
        try:
            data = self._load_file(last_input)
            input_data = [self._get_var_data(data, var) for var in self.variables]
            input_tensor = torch.from_numpy(np.stack(input_data, axis=0)).float()
        except:
            input_tensor = torch.zeros((len(self.variables), *self.target_resolution))
        
        # Load target
        try:
            data = self._load_file(target_dt)
            target_data = [self._get_var_data(data, var) for var in self.target_variables]
            target_tensor = torch.from_numpy(np.stack(target_data, axis=0)).float().squeeze(0)
        except:
            target_tensor = torch.zeros(self.target_resolution)
        
        lead_time = (target_dt - last_input).total_seconds() / 3600.0
        return input_tensor, target_tensor, lead_time

    def _load_file(self, dt):
        year_month = dt.strftime('%Y%m')
        filename = f"CN-Reanalysis{dt.strftime('%Y%m%d%H')}.nc"
        path = os.path.join(self.data_path, year_month, filename)
        return xr.open_dataset(path)

    def _get_var_data(self, data, var):
        if var in data:
            var_data = data[var].values
            # Remove time dimension if present: (1, H, W) -> (H, W)
            if len(var_data.shape) == 3 and var_data.shape[0] == 1:
                var_data = var_data[0]
            # Resize to target resolution
            from scipy.ndimage import zoom
            if var_data.shape != self.target_resolution:
                scale_factors = [self.target_resolution[i] / var_data.shape[i] for i in range(2)]
                var_data = zoom(var_data, scale_factors, order=1)

            if self.normalize and var in NORM_STATS:
                mean, std = NORM_STATS[var]
                var_data = (var_data - mean) / std
            return var_data
        return np.zeros(self.target_resolution)
