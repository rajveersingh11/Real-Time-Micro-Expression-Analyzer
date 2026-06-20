import torch
import torch.nn as nn
from torchvision import models
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("cnn_emotion_model")

class EmotionCNN(nn.Module):
    """
    CNN model for emotion classification with configurable backbones.
    """
    def __init__(self, num_classes=7, pretrained=True, dropout_rate=0.5, backbone_name="resnet18", freeze_backbone=False):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        
        if self.backbone_name == "resnet18":
            if pretrained:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                self.backbone = models.resnet18(weights=None)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif self.backbone_name == "efficientnet_b0":
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif self.backbone_name == "mobilenet_v3_small":
            if pretrained:
                self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            else:
                self.backbone = models.mobilenet_v3_small(weights=None)
            # MobileNetV3 small classifier has multiple layers; extract in_features from first linear
            num_ftrs = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            
        elif self.backbone_name == "mobilenet_v3_large":
            if pretrained:
                self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            else:
                self.backbone = models.mobilenet_v3_large(weights=None)
            num_ftrs = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Backbone {backbone_name} not supported.")
            
        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Froze all layers in '{self.backbone_name}' backbone.")
                
        # Add Dropout and a new Linear layer for classification
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
        
        # Proper initialization for the classification layer
        nn.init.xavier_uniform_(self.classifier[1].weight)
        if self.classifier[1].bias is not None:
            nn.init.constant_(self.classifier[1].bias, 0)
            
        logger.info(f"Built EmotionCNN with backbone={self.backbone_name}, pretrained={pretrained}, freeze_backbone={freeze_backbone}")

    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        # Classify features
        logits = self.classifier(features)
        return logits

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=7, backbone_name="efficientnet_b0").to(device)
    
    # Dummy input [batch_size, channels, height, width]
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    logger.info(f"Device: {device}")
    logger.info(f"Input shape: {dummy_input.shape}")
    logger.info(f"Output shape: {output.shape}")
    
    expected_shape = torch.Size([1, 7])
    if output.shape == expected_shape:
        logger.info("Model test successful!")
    else:
        logger.error(f"Model test failed. Expected {expected_shape}, got {output.shape}")
