import yaml
from typing import Dict, Any, List

class ConfigManager:
    """Multi-horizon configuration management"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self._validate_config()
        self._compute_temporal_parameters()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _validate_config(self):
        required_keys = [
            ('data', 'time_step'),
            ('data', 'forecast_hours'),
            ('train', 'epochs'),
            ('train', 'batch_size')
        ]
        
        for section, key in required_keys:
            if section not in self.config or key not in self.config[section]:
                raise ValueError(f"Missing required config parameter: {section}.{key}")
        
        forecast_hours = self.config['data']['forecast_hours']
        if not isinstance(forecast_hours, list):
            raise ValueError("forecast_hours must be a list of integers")
    
    def _compute_temporal_parameters(self):
        data_config = self.config['data']
        time_step = data_config['time_step']
        forecast_hours = data_config['forecast_hours']
        self.lead_times_hours = forecast_hours  # Already in hours
        data_config['lead_times_hours'] = self.lead_times_hours
        
        print("Multi-horizon temporal configuration:")
        print(f"  Time step: {time_step} hour(s)")
        print(f"  Forecast horizons: {forecast_hours} hours")
    
    def get_data_config(self) -> Dict[str, Any]:
        return self.config['data']
    
    def get_train_config(self) -> Dict[str, Any]:
        return self.config['train']
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.config['model']
    
    def get_system_config(self) -> Dict[str, Any]:
        return self.config['system']
    
    def print_summary(self):
        print("\n=== AQ_Net2 Multi-Horizon Configuration Summary ===")
        train_config = self.get_train_config()
        print("Training:")
        print(f"  Epochs: {train_config.get('epochs')}")
        print(f"  Batch size: {train_config.get('batch_size')}")
        print(f"  Learning rate: {train_config.get('learning_rate')}")
        print(f"  Patience: {train_config.get('patience')}")
        
        data_config = self.get_data_config()
        print("\nData:")
        print(f"  Train years: {data_config.get('train_years')}")
        print(f"  Val years: {data_config.get('val_years')}")
        print(f"  Test years: {data_config.get('test_years')}")
        print(f"  Variables: {len(data_config.get('variables', []))} ({', '.join(data_config.get('variables', []))})")
        
        print("\nMulti-horizon setup:")
        print(f"  Time step: {data_config.get('time_step')} hour(s)")
        print(f"  Forecast horizons: {data_config.get('forecast_hours')} days")
        print(f"  Lead times: {data_config.get('lead_times_hours')} hours")
        
        print("=" * 52)

    # Propriétés pour compatibilité avec le main.py existant
    @property
    def data_path(self):
        return self.config['data']['data_path']
    
    @property
    def train_years(self):
        return self.config['data']['train_years']
    
    @property
    def val_years(self):
        return self.config['data']['val_years']
    
    @property
    def test_years(self):
        return self.config['data']['test_years']
    
    @property
    def input_variables(self):
        return self.config['data']['variables']
    
    @property
    def target_variables(self):
        return self.config['data']['target_variables']
    
    @property
    def normalize(self):
        return self.config['data']['normalize']
    
    @property
    def target_resolution(self):
        return tuple(self.config['data']['target_resolution'])
    
    @property
    def batch_size(self):
        return self.config['train']['batch_size']
    
    @property
    def num_epochs(self):
        return self.config['train']['epochs']
    
    @property
    def learning_rate(self):
        return self.config['train']['learning_rate']
    
    @property
    def patch_size(self):
        return self.config['model']['patch_size']
    
    @property
    def embed_dim(self):
        return self.config['model']['embed_dim']
    
    @property
    def depth(self):
        return self.config['model']['depth']
    
    @property
    def decoder_depth(self):
        return self.config['model']['decoder_depth']
    
    @property
    def num_heads(self):
        return self.config['model']['num_heads']
