import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
import pytorch_lightning as pl
from pathlib import Path


NORM_STATS = {
    "u": (0.0, 10.0),
    "v": (0.0, 10.0),
    "temp": (273.15, 30.0),
    "rh": (50.0, 30.0),
    "psfc": (101325.0, 1000.0),
    "pm10": (50.0, 25.0),
    "so2": (5.0, 5.0),
    "no2": (20.0, 15.0),
    "co": (200.0, 100.0),
    "o3": (40.0, 20.0),
    "pm25": (25.0, 15.0),
    "lat2d": (32.0, 12.0),
    "lon2d": (106.0, 16.0),
    "elevation": (1039.13, 1931.40),
    "population": (13381.20, 67986.97),
}


class TopoFlowDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        variables: list,
        target_variables: list,
        years: list,
        forecast_hours: list = [12, 24, 48, 96],
        time_step: int = 1,
        normalize: bool = True,
        target_resolution: tuple = (128, 256),
    ):
        self.data_path = Path(data_path)
        self.variables = tuple(variables)
        self.target_variables = target_variables
        self.years = years
        self.forecast_hours = forecast_hours
        self.time_step = time_step
        self.normalize = normalize
        self.target_h, self.target_w = target_resolution

        self.datasets = []
        self.valid_indices = []
        self._load_data()
        self._prepare_indices()

    def _load_data(self):
        for year in self.years:
            zarr_path = self.data_path / f"data_{year}_china_masked.zarr"
            if zarr_path.exists():
                ds = xr.open_zarr(zarr_path, consolidated=True)
                self.datasets.append(ds)

                if len(self.datasets) == 1:
                    self._setup_coords(ds)

    def _setup_coords(self, ds):
        sample_var = next((v for v in self.variables if v in ds.data_vars), None)
        if sample_var:
            sample = ds[sample_var].isel(time=0)
            self.orig_h, self.orig_w = sample.shape

        if "lat2d" in ds.coords and "lon2d" in ds.coords:
            lat_1d = ds.coords["lat2d"].values
            lon_1d = ds.coords["lon2d"].values

            lat_idx = np.linspace(0, len(lat_1d) - 1, self.target_h)
            lon_idx = np.linspace(0, len(lon_1d) - 1, self.target_w)

            lat_1d = np.interp(lat_idx, np.arange(len(lat_1d)), lat_1d)
            lon_1d = np.interp(lon_idx, np.arange(len(lon_1d)), lon_1d)

            self.lon_grid, self.lat_grid = np.meshgrid(lon_1d, lat_1d)
            self.lat_grid = self.lat_grid.astype(np.float32)
            self.lon_grid = self.lon_grid.astype(np.float32)
        else:
            self.lat_grid = None
            self.lon_grid = None

    def _prepare_indices(self):
        for ds_idx, ds in enumerate(self.datasets):
            max_forecast = max(self.forecast_hours)
            max_t = len(ds.time) - max_forecast - 1

            for t in range(0, max_t, self.time_step):
                for fh in self.forecast_hours:
                    self.valid_indices.append((ds_idx, t, fh))

    def _normalize(self, data, var_name):
        if not self.normalize or var_name not in NORM_STATS:
            return data
        mean, std = NORM_STATS[var_name]
        return (data - mean) / std

    def _downsample(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        data = data.unsqueeze(0).unsqueeze(0)
        data = interpolate(data, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False)
        return data.squeeze(0).squeeze(0)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        ds_idx, t, forecast_h = self.valid_indices[idx]
        ds = self.datasets[ds_idx]

        inputs = []
        for var in self.variables:
            if var in ds.data_vars:
                if "time" in ds[var].dims:
                    data = ds[var].isel(time=t).values
                else:
                    data = ds[var].values
                data = self._normalize(data, var)
                data = self._downsample(data).numpy()
                inputs.append(data)
            elif var == "lat2d" and self.lat_grid is not None:
                inputs.append(self._normalize(self.lat_grid, "lat2d"))
            elif var == "lon2d" and self.lon_grid is not None:
                inputs.append(self._normalize(self.lon_grid, "lon2d"))

        targets = []
        target_t = t + forecast_h
        for var in self.target_variables:
            data = ds[var].isel(time=target_t).values
            data = self._normalize(data, var)
            data = self._downsample(data)
            targets.append(data)

        x = torch.from_numpy(np.stack(inputs, axis=0)).float()
        y = torch.stack(targets, dim=0).float()
        lead_time = torch.tensor(forecast_h, dtype=torch.float32)

        return x, y, lead_time


class TopoFlowDataModule(pl.LightningDataModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.data_cfg = config["data"]
        self.train_cfg = config["train"]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TopoFlowDataset(
                data_path=self.data_cfg["data_path"],
                variables=self.data_cfg["variables"],
                target_variables=self.data_cfg["target_variables"],
                years=self.data_cfg["train_years"],
                forecast_hours=self.data_cfg["forecast_hours"],
                time_step=self.data_cfg.get("time_step", 1),
                normalize=self.data_cfg.get("normalize", True),
                target_resolution=tuple(self.data_cfg.get("target_resolution", [128, 256])),
            )
            self.val_dataset = TopoFlowDataset(
                data_path=self.data_cfg["data_path"],
                variables=self.data_cfg["variables"],
                target_variables=self.data_cfg["target_variables"],
                years=self.data_cfg["val_years"],
                forecast_hours=self.data_cfg["forecast_hours"],
                time_step=self.data_cfg.get("time_step", 1),
                normalize=self.data_cfg.get("normalize", True),
                target_resolution=tuple(self.data_cfg.get("target_resolution", [128, 256])),
            )

        if stage == "test" or stage is None:
            self.test_dataset = TopoFlowDataset(
                data_path=self.data_cfg["data_path"],
                variables=self.data_cfg["variables"],
                target_variables=self.data_cfg["target_variables"],
                years=self.data_cfg["test_years"],
                forecast_hours=self.data_cfg["forecast_hours"],
                time_step=self.data_cfg.get("time_step", 1),
                normalize=self.data_cfg.get("normalize", True),
                target_resolution=tuple(self.data_cfg.get("target_resolution", [128, 256])),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
            num_workers=self.data_cfg.get("num_workers", 4),
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_cfg.get("val_batch_size", self.train_cfg["batch_size"]),
            shuffle=False,
            num_workers=self.data_cfg.get("num_workers", 4),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.train_cfg.get("val_batch_size", self.train_cfg["batch_size"]),
            shuffle=False,
            num_workers=self.data_cfg.get("num_workers", 4),
            pin_memory=True,
        )
