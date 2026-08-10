import torch
import torch.nn as nn
from torchvision import models


class BitumenRegressor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 3)

    def forward(self, x):
        # Output order is always [Water, Solids, Bitumen]; raw linear outputs, no activation.
        return self.backbone(x)

    @classmethod
    def from_pretrained(cls, path, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Skip ImageNet download — checkpoint weights replace the backbone entirely.
        model = cls(pretrained=False)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def save(self, path):
        torch.save(self.state_dict(), path)


# Input:  (batch, 3, 224, 224)
# Output: (batch, 3)  [Water, Solids, Bitumen]
