import torch.optim as optim
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("optimizer_builder")

def build_optimizer(model, lr=0.001, weight_decay=1e-5, optimizer_type=None):
    """
    Builds the optimizer (Adam, AdamW, SGD) based on configuration.
    """
    try:
        config = load_config()
        if optimizer_type is None:
            optimizer_type = config.get("training", {}).get("optimizer", "adamw")
        lr = float(config.get("training", {}).get("learning_rate", lr))
        weight_decay = float(config.get("training", {}).get("weight_decay", weight_decay))
    except Exception as e:
        logger.warning(f"Could not load config in build_optimizer, using defaults: {e}")
        if optimizer_type is None:
            optimizer_type = "adamw"
            
    optimizer_type = optimizer_type.lower()
    logger.info(f"Building optimizer '{optimizer_type}' with lr={lr}, weight_decay={weight_decay}")
    
    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer type '{optimizer_type}' not supported.")
