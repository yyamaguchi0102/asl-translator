import torch
import torch.nn as nn
import torchvision.models as models

class ASLClassifier(nn.Module):
    def __init__(self, num_classes=26, pretrained=True):
        """
        Initialize the ASL classifier using a pretrained ResNet model
        
        Args:
            num_classes (int): Number of classes (26 for ASL alphabet)
            pretrained (bool): Whether to use pretrained weights
        """
        super(ASLClassifier, self).__init__()
        
        # Load pretrained ResNet model
        self.model = models.resnet50(pretrained=pretrained)
        
        # Freeze the feature extraction layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Modify the final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def unfreeze_layers(self, num_layers=0):
        """
        Unfreeze the last n layers for fine-tuning
        
        Args:
            num_layers (int): Number of layers to unfreeze
        """
        if num_layers == 0:
            return
            
        # Get all layers
        layers = list(self.model.children())
        
        # Unfreeze the last n layers
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

def create_model(num_classes=26, pretrained=True):
    """
    Factory function to create the model
    
    Args:
        num_classes (int): Number of classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        ASLClassifier: The initialized model
    """
    model = ASLClassifier(num_classes=num_classes, pretrained=pretrained)
    return model

def load_model(model_path, num_classes=26):
    """
    Load a trained model from a checkpoint
    
    Args:
        model_path (str): Path to the model checkpoint
        num_classes (int): Number of classes
        
    Returns:
        ASLClassifier: The loaded model
    """
    model = create_model(num_classes=num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model 