import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_B0_Weights

class SpatialModel(nn.Module):
    def __init__(self):
        super(SpatialModel, self).__init__()
        # Load EfficientNet-B0 pretrained on ImageNet
        self.base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Replace classifier head for binary classification (real/fake)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_features, 1)

    def forward(self, x):
        prob = torch.sigmoid(self.base_model(x))
        return prob
