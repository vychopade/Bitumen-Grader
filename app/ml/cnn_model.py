"""Froth-image regressors aligned with Prince & Prasad transfer-learning findings.

Default is a compact CNN trained from scratch. ImageNet VGG16 / ResNet50
variants are optional second-stage candidates (mainly for Solids). Over-
parameterised 3-layer batch-normalised heads are not offered: they collapse
into negative transfer on this domain.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torchvision import models

# Shared across train / val / inference for new models (paper Table 2).
IMAGE_SIZE = 256
NUM_OUTPUTS = 3  # Water, Solids, Bitumen

ARCHITECTURES = ("baseline", "resnet50", "vgg16", "resnet18")
TRAINABLE_ARCHITECTURES = ("baseline", "resnet50", "vgg16")
HEAD_TYPES = ("native", "c2")

ARCHITECTURE_LABELS = {
    "baseline": "Baseline CNN (recommended)",
    "resnet50": "ResNet50 (ImageNet transfer)",
    "vgg16": "VGG16 (ImageNet transfer)",
    "resnet18": "ResNet18 (legacy)",
}


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CompactFrothCNN(nn.Module):
    """Compact texture CNN: five conv stages + global average pool → 256-d features.

    Designed for froth surfaces (repetitive texture, weak object structure)
    rather than ImageNet object semantics.
    """

    feature_dim = 256

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),  # 256 → 128
            _conv_block(32, 64),  # 128 → 64
            _conv_block(64, 128),  # 64 → 32
            _conv_block(128, 256),  # 32 → 16
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(x), 1)


def infer_architecture(state_dict: dict) -> str:
    """Guess architecture from checkpoint keys when metadata is missing."""
    keys = list(state_dict.keys())
    if any(key == "backbone.fc.weight" or key == "backbone.fc.bias" for key in keys):
        return "resnet18"
    if any(key.startswith("backbone.layer") for key in keys):
        return "resnet50"
    if any(key.startswith("backbone.features") for key in keys):
        return "baseline"
    if any(key.startswith("backbone.0.") for key in keys):
        return "vgg16"
    return "resnet18"


def infer_head(state_dict: dict) -> str:
    """Native is a single Linear (`head.weight`); C2 uses `head.0.weight`."""
    if "head.0.weight" in state_dict:
        return "c2"
    return "native"


def _make_head(in_features: int, head_type: str, num_outputs: int = NUM_OUTPUTS) -> nn.Module:
    """Native linear head, or a lightweight 2-layer FC head (C2).

    C2 is the only custom head evaluated as competitive in the paper.
    3-layer batch-normalised heads (C3BN) are intentionally omitted.
    """
    if head_type not in HEAD_TYPES:
        raise ValueError(f"head must be one of {HEAD_TYPES}, got {head_type!r}")
    if head_type == "c2":
        hidden = 256 if in_features >= 256 else 128
        return nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_outputs),
        )
    return nn.Linear(in_features, num_outputs)


class BitumenRegressor(nn.Module):
    """Predicts [Water, Solids, Bitumen] from a froth RGB image.

    Parameters
    ----------
    architecture:
        ``baseline`` (default), ``resnet50``, ``vgg16``, or ``resnet18`` (legacy).
    pretrained:
        ImageNet initialisation. Ignored for ``baseline`` (always from scratch).
        Default False: ImageNet transfer is not the operational default.
    head:
        ``native`` (single linear) or ``c2`` (2-layer FC).
    """

    def __init__(
        self,
        architecture: str = "baseline",
        pretrained: bool = False,
        head: str = "native",
        num_outputs: int = NUM_OUTPUTS,
    ):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")
        if head not in HEAD_TYPES:
            raise ValueError(f"head must be one of {HEAD_TYPES}, got {head!r}")

        self.architecture = architecture
        self.head_type = head
        self.pretrained = bool(pretrained) and architecture != "baseline"
        self.num_outputs = num_outputs
        # Legacy ResNet-18 checkpoints store the linear layer as backbone.fc.
        self._legacy_combined = architecture == "resnet18"

        if architecture == "baseline":
            self.backbone = CompactFrothCNN()
            self.head = _make_head(CompactFrothCNN.feature_dim, head, num_outputs)
        elif architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if self.pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, num_outputs)
            self.backbone = backbone
            self.head = None
        elif architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.head = _make_head(in_features, head, num_outputs)
        else:  # vgg16
            weights = models.VGG16_Weights.DEFAULT if self.pretrained else None
            vgg = models.vgg16(weights=weights)
            self.backbone = nn.Sequential(
                vgg.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.head = _make_head(512, head, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always [Water, Solids, Bitumen]; raw linear outputs, no activation.
        features = self.backbone(x)
        if self._legacy_combined:
            return features
        return self.head(features)

    def head_parameters(self) -> Iterable[nn.Parameter]:
        """Just the regression head."""
        if self._legacy_combined:
            return self.backbone.fc.parameters()
        return self.head.parameters()

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        """Everything except the regression head."""
        if self._legacy_combined:
            head_ids = {id(parameter) for parameter in self.backbone.fc.parameters()}
            return (parameter for parameter in self.backbone.parameters() if id(parameter) not in head_ids)
        return self.backbone.parameters()

    def freeze_backbone(self) -> None:
        for parameter in self.backbone_parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone_parameters():
            parameter.requires_grad = True

    def config_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "head": self.head_type,
            "pretrained": self.pretrained,
            "image_size": IMAGE_SIZE if self.architecture != "resnet18" else 224,
        }

    @classmethod
    def from_checkpoint(cls, path, metadata=None, device=None):
        """Load weights, reconstructing architecture from metadata when present.

        Checkpoints saved before this overhaul have no architecture field and
        are treated as ResNet-18 with a native head.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metadata = metadata or {}
        state_dict = torch.load(path, map_location=device)
        architecture = metadata.get("architecture") or infer_architecture(state_dict)
        head = metadata.get("head") or infer_head(state_dict)
        if architecture not in ARCHITECTURES:
            architecture = infer_architecture(state_dict)
        if head not in HEAD_TYPES:
            head = infer_head(state_dict)

        model = cls(architecture=architecture, pretrained=False, head=head)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)
