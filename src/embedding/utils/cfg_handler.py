import yaml
from pathlib import Path

class train_config:
    def __init__(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                self.cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    def get_model_name(self) -> str:
        """Retrieve model name from configuration."""
        model_name = self.cfg.get("model_name", "")
        if not model_name:
            raise ValueError("Model name not found in the config file.")
        return model_name
    
    def get_hyperparams(self) -> dict:
        """Retrieve hyperparameters dictionary."""
        train_hyperparams = self.cfg.get("hyperparameters", {})
        if not train_hyperparams:
            raise ValueError("Hyperparameters not found in the config file.")
        return train_hyperparams
    
    def get_hyperparam(self, key: str, default=None):
        """Hyperparameter getter for configuration values."""
        return self.cfg["hyperparameters"].get(key, default)
    
    def get_trdata_cfg(self, key: str, default=None):
        """Training data processing getter for configuration values."""
        return self.cfg["data"].get(key, default)
    
    def get_entire_cfg(self) -> dict:
        """Retrieve the entire configuration dictionary."""
        return self.cfg

    def hp(self, key: str, default=None):
        """Hyperparameter getter for configuration values."""
        return self.get_hyperparam(key, default)


    
class data_config:
    def __init__(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                self.cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    def __getitem__(self, key: str):
        """Allow dictionary-like access to configuration values."""
        return self.cfg["data_processing"][key]
    
    def get_file_label_map(self) -> list[tuple[str, int]]:
        """Retrieve list of (file_path, class_label) tuples from the config."""
        samples = self.cfg.get("data_processing", {}).get("samples", [])
        if not samples:
            raise ValueError("Samples configuration not found in the config file.")
        
        file_label_tuples = []
        for cls_label, sample_dict in enumerate(samples):
            fnames = list(sample_dict.values())[0]
            if fnames is not None:
                for fname in fnames:
                    file_label_tuples.append((fname, cls_label))
        return file_label_tuples
    
    def get_name_label_map(self) -> list[tuple[str, int]]:
        """Retrieve list of (sample_name, class_label) tuples from the config."""
        samples = self.cfg.get("data_processing", {}).get("samples", [])
        if not samples:
            raise ValueError("Samples configuration not found in the config file.")

        file_name_dict = {}
        for cls_label, sample_dict in enumerate(samples):
            name = list(sample_dict.keys())[0]
            file_name_dict[name] = cls_label
        return file_name_dict
    
    def get_label_name_map(self) -> dict[int, str]:
        """Retrieve dictionary mapping class_label to sample_name from the config."""
        samples = self.cfg.get("data_processing", {}).get("samples", [])
        if not samples:
            raise ValueError("Samples configuration not found in the config file.")

        label_name_dict = {}
        for cls_label, sample_dict in enumerate(samples):
            name = list(sample_dict.keys())[0]
            label_name_dict[cls_label] = name
        return label_name_dict

    def get_ds_name(self) -> str:
        """Retrieve dataset name from configuration."""
        ds_name = self.cfg.get("ds_name", "")
        if not ds_name:
            raise ValueError("Dataset name not found in the config file.")
        return ds_name    
    
    def get(self, key: str, default=None):
        """Data processing getter for configuration values."""
        return self.cfg["data_processing"].get(key, default)
    
    def get_entire_cfg(self) -> dict:
        """Retrieve the entire configuration dictionary."""
        return self.cfg

def join_remote(prefix: str, p) -> str:
    """Join a redirector like root://host with a local/posix path safely."""
    p_str = p.as_posix() if isinstance(p, Path) else str(p)
    sep = '//' if p_str.startswith('/') else '/'
    return f"{prefix.rstrip('/')}{sep}{p_str.lstrip('/')}"