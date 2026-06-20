import torch.optim as optim
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("scheduler_builder")

def build_scheduler(optimizer, epochs=50, scheduler_type=None):
    """
    Builds the learning rate scheduler based on configuration.
    """
    try:
        config = load_config()
        if scheduler_type is None:
            scheduler_type = config.get("training", {}).get("scheduler", "cosine")
        epochs = config.get("training", {}).get("epochs", epochs)
    except Exception as e:
        logger.warning(f"Could not load config in build_scheduler, using defaults: {e}")
        if scheduler_type is None:
            scheduler_type = "cosine"
            
    scheduler_type = scheduler_type.lower()
    logger.info(f"Building scheduler '{scheduler_type}' for {epochs} epochs")
    
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        step_size = max(1, epochs // 3)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' not supported.")
