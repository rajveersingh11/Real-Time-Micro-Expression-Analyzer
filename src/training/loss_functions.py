import torch.nn as nn
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("loss_functions")

def build_loss(weight=None, label_smoothing=0.0):
    """
    Builds the CrossEntropyLoss function with optional label smoothing.
    """
    try:
        config = load_config()
        label_smoothing = config.get("training", {}).get("label_smoothing", label_smoothing)
    except Exception as e:
        logger.warning(f"Could not load config in build_loss, using default: {e}")
        
    logger.info(f"Building CrossEntropyLoss with label_smoothing={label_smoothing}")
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
