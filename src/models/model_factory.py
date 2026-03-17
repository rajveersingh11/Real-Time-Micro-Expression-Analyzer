import torch
from .cnn_emotion_model import EmotionCNN

def build_model(model_name="resnet18", num_classes=7, pretrained=True, device=None):
    """
    Factory function to build and return the requested model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name.lower() == "resnet18":
        model = EmotionCNN(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported by factory.")
    
    model = model.to(device)
    print(f"Model '{model_name}' built and moved to {device}")
    
    return model

if __name__ == "__main__":
    # Test factory
    model = build_model("resnet18")
    dummy_input = torch.randn(2, 3, 224, 224).to(next(model.parameters()).device)
    output = model(dummy_input)
    print(f"Factory test - Output shape: {output.shape}")
