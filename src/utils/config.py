import os
import yaml

def load_config(config_path=None):
    """
    Loads configuration from a YAML file.
    By default, resolves and loads config/default.yaml.
    """
    if config_path is None:
        # Try multiple locations to find config/default.yaml
        possible_paths = [
            "config/default.yaml",
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "default.yaml"),
            os.path.join(os.path.dirname(__file__), "config", "default.yaml"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
                
    if config_path is None or not os.path.exists(config_path):
        raise FileNotFoundError("Configuration file default.yaml could not be located.")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config
