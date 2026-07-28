"""
CNN architecture definition.

Defines the PyTorch nn.Module(s) implementing the convolutional neural
network used to classify/grade bitumen sample images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

#: Default set of bitumen performance grades predicted by the model, in
#: the same order as the model's output logits.
DEFAULT_GRADE_LABELS = [
    "PG 52-28",
    "PG 58-28",
    "PG 64-22",
    "PG 70-22",
    "PG 76-16",
]


class BitumenCNN(nn.Module):
    """CNN for classifying bitumen samples into performance grades.

    Wraps a torchvision ResNet-18 backbone (optionally initialized with
    ImageNet-pretrained weights) with its final fully-connected layer
    replaced so the model outputs ``num_classes`` logits, one per bitumen
    grade.
    """

    def __init__(self, num_classes: int = len(DEFAULT_GRADE_LABELS), pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass, returning raw class logits of shape (N, num_classes)."""
        return self.backbone(x)

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        num_classes: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "BitumenCNN":
        """Instantiate a BitumenCNN and load weights from a saved checkpoint.

        Args:
            path: Path to a ``.pt`` file, either a checkpoint dict written by
                ``app.utils.model_io.save_model`` (containing "state_dict"
                and "num_classes") or a raw ``state_dict``.
            num_classes: Number of output classes for the model architecture.
                Used when the checkpoint does not itself store this value.
                Defaults to ``len(DEFAULT_GRADE_LABELS)`` if not provided.
            device: Optional device to map the loaded weights onto.

        Returns:
            A ``BitumenCNN`` instance loaded with the checkpoint's weights,
            moved to ``device`` (if given) and set to eval() mode.
        """
        map_location = torch.device(device) if device is not None else torch.device("cpu")
        checkpoint = torch.load(path, map_location=map_location)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            num_classes = checkpoint.get("num_classes", num_classes) or len(DEFAULT_GRADE_LABELS)
        else:
            state_dict = checkpoint
            num_classes = num_classes or len(DEFAULT_GRADE_LABELS)

        model = cls(num_classes=num_classes, pretrained=False)
        model.load_state_dict(state_dict)
        model.eval()

        if device is not None:
            model.to(device)

        return model
