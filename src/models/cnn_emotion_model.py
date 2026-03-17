import torch
import torch.nn as nn
from torchvision import models

class EmotionCNN(nn.Module):
    """
    CNN model for emotion classification using ResNet18 as a backbone.
    """
    def __init__(self, num_classes=7, pretrained=True, dropout_rate=0.5):
        super(EmotionCNN, self).__init__()
        
        # Load ResNet18 backbone
        if pretrained:
            # Using updated weights parameter as per recent torchvision versions
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.backbone = models.resnet18(weights=None)
            
        # Get the number of input features for the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        
        # Replace the final fully connected layer with custom layers
        # Remove the original fc layer
        self.backbone.fc = nn.Identity()
        
        # Add Dropout and a new Linear layer for classification
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
        
        # Proper initialization for the classification layer
        nn.init.xavier_uniform_(self.classifier[1].weight)
        if self.classifier[1].bias is not None:
            nn.init.constant_(self.classifier[1].bias, 0)

    def forward(self, x):
        # Extract features using ResNet backbone
        features = self.backbone(x)
        # Classify features
        logits = self.classifier(features)
        return logits

if __name__ == "__main__":
    # Test snippet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=7).to(device)
    
    # Dummy input [batch_size, channels, height, width]
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"Device: {device}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    expected_shape = torch.Size([1, 7])
    if output.shape == expected_shape:
        print("Model test successful!")
    else:
        print(f"Model test failed. Expected {expected_shape}, got {output.shape}")
