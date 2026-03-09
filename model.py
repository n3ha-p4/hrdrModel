"""
PyTorch Model Architecture for HR vs DR Detection
Uses EfficientNet as backbone with custom classifier
"""

import torch
import torch.nn as nn
import torchvision.models as models


class RetinopathyClassifier(nn.Module):
    """
    CNN model for classifying Hypertensive Retinopathy vs Diabetic Retinopathy
    Based on EfficientNet-B0 with custom classification head
    """
    
    def __init__(self, num_classes=3, model_name='efficientnet_b0', pretrained=True):
        super(RetinopathyClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained backbone
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            # Remove the original classifier
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            # Remove the original FC layer
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'vit_b16':
            self.backbone = models.vision_transformer.vit_b_16(pretrained=pretrained)
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.num_features = num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Get backbone features (useful for GradCAM)"""
        return self.backbone(x)


class ResNet50Classifier(nn.Module):
    """
    ResNet50 based classifier with custom head
    """
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet50Classifier, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        
        # Replace final FC layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features from layer4 (before FC)"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def create_model(num_classes=3, model_name='efficientnet_b0', pretrained=True, device='cpu'):
    """
    Factory function to create a model
    
    Args:
        num_classes: Number of output classes
        model_name: Name of the model architecture
        pretrained: Whether to use pretrained weights
        device: Device to put model on
    
    Returns:
        Model on specified device
    """
    if model_name in ['efficientnet_b0', 'resnet50', 'vit_b16']:
        model = RetinopathyClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_model(num_classes=3, model_name='efficientnet_b0', device=device)
    print(f"Model created successfully!")
    print(f"Model: {model}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(device)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
