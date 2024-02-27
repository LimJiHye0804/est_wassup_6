import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights, resnet101 , ResNet101_Weights
from torchvision.models import vit_b_16

class ViTForEmotionClassification(nn.Module):
    def __init__(self, num_classes=7):
        super(ViTForEmotionClassification, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)  # Load a pre-trained ViT model
        hidden_dim = self.vit.hidden_dim  # Assuming hidden_dim is accessible and correctly set
        
        # Check if representation_size is used and adjust accordingly
        if self.vit.representation_size is None:
            # Directly replace the 'head' if no pre-logits layer is used
            self.vit.heads.head = nn.Linear(hidden_dim, num_classes)
        else:
            # Replace the 'head' after the pre-logits layer if representation_size is used
            representation_size = self.vit.representation_size  # Assuming this is correctly set
            self.vit.heads.head = nn.Linear(representation_size, num_classes)

    def forward(self, x):
        # Forward pass through the modified Vision Transformer
        return self.vit(x)
    
    
class ResNet50(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet50, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(2048,num_classes)
        
    def forward(self,x):
        return self.resnet(x)
    

class ResNet101(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet101, self).__init__()
        self.resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(2048,num_classes)
        
    def forward(self,x):
        return self.resnet(x)
    
    
    
class ViTMultitask(nn.Module):
    def __init__(self, num_classes, num_bbox, img_width=224.0, img_height=224.0):
        super(ViTMultitask, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.img_width = img_width
        self.img_height = img_height
        
        # Replace the classifier in ViT
        hidden_dim = self.vit.heads.head.in_features  
        self.vit.heads.head = nn.Identity()  
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Bounding box regression head
        self.linear = nn.Linear(hidden_dim, hidden_dim//2)
        self.prelu = nn.PReLU()  # Corrected: instantiate PReLU
        self.bbox_regressor = nn.Linear(hidden_dim//2, num_bbox)  # 4 outputs for (min_X, min_Y, max_X, max_Y)

    def forward(self, x):
        features = self.vit(x)
        
        # Classification
        class_logits = self.classifier(features)
        
        # Bounding Box Regression
        bbox_raw = self.linear(features)
        bbox_raw = self.prelu(bbox_raw)  # Corrected: apply PReLU correctly
        bbox_raw = torch.sigmoid(self.bbox_regressor(bbox_raw))
        
        bbox = torch.stack([
            bbox_raw[:, 0] * self.img_width,
            bbox_raw[:, 1] * self.img_height,
            bbox_raw[:, 2] * self.img_width,
            bbox_raw[:, 3] * self.img_height
        ], dim=1)
        
        return class_logits, bbox
    
class ResNetMultitask(nn.Module):
    def __init__(self, num_classes=7, num_bbox=4, img_width=256.0, img_height=256.0):
        super(ResNetMultitask, self).__init__()
        self.resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        #self.vit = vit_b_16(pretrained=True)
        self.img_width = img_width
        self.img_height = img_height
        #self.fc = nn.Identity()
        # Replace the classifier in ViT
        hidden_dim = 1000
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Bounding box regression head
        self.linear = nn.Linear(hidden_dim, hidden_dim//2)
        self.prelu = nn.PReLU()  # Corrected: instantiate PReLU
        self.bbox_regressor = nn.Linear(hidden_dim//2, num_bbox)  # 4 outputs for (min_X, min_Y, max_X, max_Y)

    def forward(self, x):
        features = self.resnet(x)
        
        # Classification
        class_logits = self.classifier(features)
        
        # Bounding Box Regression
        bbox_raw = self.linear(features)
        bbox_raw = self.prelu(bbox_raw)  # Corrected: apply PReLU correctly
        bbox_raw = torch.sigmoid(self.bbox_regressor(bbox_raw))
        
        bbox = torch.stack([
            bbox_raw[:, 0] * self.img_width,
            bbox_raw[:, 1] * self.img_height,
            bbox_raw[:, 2] * self.img_width,
            bbox_raw[:, 3] * self.img_height
        ], dim=1)
        
        return class_logits, bbox