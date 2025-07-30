from scipy.ndimage import zoom
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from datetime import datetime, timedelta
import glob
from typing import List, Tuple
import random

NORM_STATS = {
    "u": (0.0, 10.0), "v": (0.0, 10.0), "temp": (273.15, 30.0),
    "rh": (50.0, 30.0), "psfc": (101325.0, 1000.0), "pm10": (50.0, 25.0),
    "so2": (5.0, 5.0), "no2": (20.0, 15.0), "co": (200.0, 100.0),
    "o3": (40.0, 20.0), "pm25": (25.0, 15.0)
}

class CAQRADataset(Dataset):
    """Multi-horizon optimized dataset - single timestep input, variable lead times"""
    
    def __init__(self, data_path: str, years: List[int], variables: List[str], 
                 target_variables: List[str], lead_times_hours: List[int], 
                 normalize: bool, target_resolution: Tuple[int, int]):
        self.data_path = data_path
        self.years = years
        self.variables = variables
        self.target_variables = target_variables
        self.lead_times_hours = lead_times_hours
        self.normalize = normalize
        self.target_resolution = target_resolution
        
        print(f"🚀 Multi-horizon dataset initialization:")
        print(f"   Lead times: {lead_times_hours} hours")
        print(f"   Years: {years}")
        print(f"   Variables: {len(variables)}")
        
        # Create all valid input timestamps
        self.input_timestamps = self._create_input_timestamps()
        print(f"   📊 Total samples: {len(self.input_timestamps)}")

    def _create_input_timestamps(self):
        """Create list of all valid input timestamps"""
        all_files = []
        for year in self.years:
            pattern = os.path.join(self.data_path, f"{year}*", "*.nc")
            all_files.extend(sorted(glob.glob(pattern)))
        
        # Extract timestamps from filenames
        timestamps = []
        for f in all_files:
            basename = os.path.basename(f)[:-3]  # Remove .nc
            if basename.startswith("CN-Reanalysis"):
                date_part = basename[len("CN-Reanalysis"):]
                timestamps.append(datetime.strptime(date_part, "%Y%m%d%H"))
        
        # Filter: keep only timestamps where ALL lead times have valid targets
        valid_timestamps = []
        max_lead_time = max(self.lead_times_hours)
        
        for ts in timestamps:
            # Check if all required target files exist
            all_targets_exist = True
            for lead_hours in self.lead_times_hours:
                target_time = ts + timedelta(hours=lead_hours)
                if not self._file_exists(target_time):
                    all_targets_exist = False
                    break
            
            if all_targets_exist:
                valid_timestamps.append(ts)
        
        return valid_timestamps

    def _file_exists(self, dt):
        """Check if file exists for given datetime"""
        year_month = dt.strftime('%Y%m')
        filename = f"CN-Reanalysis{dt.strftime('%Y%m%d%H')}.nc"
        path = os.path.join(self.data_path, year_month, filename)
        return os.path.exists(path)

    def __len__(self):
        # Each input timestamp × number of lead times = total samples
        return len(self.input_timestamps) * len(self.lead_times_hours)

    def __getitem__(self, idx: int):
        """
        Returns: (input_tensor, target_tensor, lead_time_hours)
        - input_tensor: [C, H, W] - single timestep
        - target_tensor: [H, W] - target at lead_time
        - lead_time_hours: float - actual lead time in hours
        """
        # Decode which input timestamp and which lead time
        timestamp_idx = idx // len(self.lead_times_hours)
        lead_time_idx = idx % len(self.lead_times_hours)
        
        input_time = self.input_timestamps[timestamp_idx]
        lead_time_hours = self.lead_times_hours[lead_time_idx]
        target_time = input_time + timedelta(hours=lead_time_hours)
        
        # Load input data (single timestep)
        try:
            input_data = self._load_file(input_time)
            input_tensors = []
            for var in self.variables:
                var_data = self._get_var_data(input_data, var)
                input_tensors.append(var_data)
            input_tensor = torch.from_numpy(np.stack(input_tensors, axis=0)).float()
        except Exception as e:
            # Fallback to zeros if loading fails
            input_tensor = torch.zeros((len(self.variables), *self.target_resolution))
        
        # Load target data
        try:
            target_data = self._load_file(target_time)
            target_tensors = []
            for var in self.target_variables:
                var_data = self._get_var_data(target_data, var)
                target_tensors.append(var_data)
            target_tensor = torch.from_numpy(np.stack(target_tensors, axis=0)).float()
            # Remove channel dimension if single target variable
            if len(self.target_variables) == 1:
                target_tensor = target_tensor.squeeze(0)
        except Exception as e:
            # Fallback to zeros if loading fails
            if len(self.target_variables) == 1:
                target_tensor = torch.zeros(self.target_resolution)
            else:
                target_tensor = torch.zeros((len(self.target_variables), *self.target_resolution))
        
        return input_tensor, target_tensor, float(lead_time_hours)

    def _load_file(self, dt):
        """Load NetCDF file for given datetime"""
        year_month = dt.strftime('%Y%m')
        filename = f"CN-Reanalysis{dt.strftime('%Y%m%d%H')}.nc"
        path = os.path.join(self.data_path, year_month, filename)
        return xr.open_dataset(path)

    def _get_var_data(self, data, var):
        """Extract and preprocess variable data"""
        if var in data:
            var_data = data[var].values
            # Remove time dimension if present: (1, H, W) -> (H, W)
            if len(var_data.shape) == 3 and var_data.shape[0] == 1:
                var_data = var_data[0]
            
            # Resize to target resolution
            if var_data.shape != self.target_resolution:
                scale_factors = [self.target_resolution[i] / var_data.shape[i] for i in range(2)]
                var_data = zoom(var_data, scale_factors, order=1)

            # Normalize
            if self.normalize and var in NORM_STATS:
                mean, std = NORM_STATS[var]
                var_data = (var_data - mean) / std
            
            return var_data
        else:
            return np.zeros(self.target_resolution)

    def get_sample_for_lead_time(self, lead_time_hours: int, max_samples: int = 100):
        """Get samples for a specific lead time (useful for validation)"""
        if lead_time_hours not in self.lead_times_hours:
            raise ValueError(f"Lead time {lead_time_hours}h not in configured lead times: {self.lead_times_hours}")
        
        lead_time_idx = self.lead_times_hours.index(lead_time_hours)
        samples = []
        
        for i in range(min(max_samples, len(self.input_timestamps))):
            idx = i * len(self.lead_times_hours) + lead_time_idx
            samples.append(self.__getitem__(idx))
        
        return samples
