import torch
from .cnn_emotion_model import EmotionCNN
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("model_factory")

def build_model(model_name=None, num_classes=None, pretrained=True, device=None, freeze_backbone=None):
    """
    Factory function to build and return the configured model.
    Loads configurations from default.yaml if available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    try:
        config = load_config()
        if model_name is None:
            model_name = config.get("model", {}).get("architecture", "resnet18")
        if num_classes is None:
            num_classes = config.get("model", {}).get("num_classes", 7)
        if freeze_backbone is None:
            freeze_backbone = config.get("model", {}).get("freeze_backbone", False)
        dropout_rate = config.get("model", {}).get("dropout", 0.5)
    except Exception as e:
        logger.warning(f"Could not load config in build_model, using defaults: {e}")
        if model_name is None:
            model_name = "resnet18"
        if num_classes is None:
            num_classes = 7
        if freeze_backbone is None:
            freeze_backbone = False
        dropout_rate = 0.5
        
    model = EmotionCNN(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        backbone_name=model_name,
        freeze_backbone=freeze_backbone
    )
    
    model = model.to(device)
    logger.info(f"Model '{model_name}' built and moved to {device}")
    
    return model

if __name__ == "__main__":
    model = build_model("resnet18")
    dummy_input = torch.randn(2, 3, 224, 224).to(next(model.parameters()).device)
    output = model(dummy_input)
    logger.info(f"Factory test - Output shape: {output.shape}")
