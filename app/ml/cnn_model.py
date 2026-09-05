"""CNNs that turn a froth photo into water, solids, and bitumen. The default is a small network trained from scratch. ResNet50 and VGG16 are optional if you want to try ImageNet transfer, mostly for solids. We do not offer a deep batch-norm head because those fell apart on this data."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
from torchvision import models

from app.ml.recipe import IMAGE_SIZE

NUM_OUTPUTS = 3  # three grades: water, solids, bitumen


def select_torch_device() -> torch.device:
    """Picks a torch device. Tries CUDA, then Apple Metal, then CPU. You get a torch.device back."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    backends_mps = getattr(torch.backends, "mps", None)
    is_available = (
        getattr(backends_mps, "is_available", None)
        if backends_mps is not None
        else None
    )
    try:
        if callable(is_available) and is_available():
            return torch.device("mps")
    except (RuntimeError, AttributeError):
        pass
    return torch.device("cpu")


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
        nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CompactFrothCNN(nn.Module):
    """Small CNN for froth texture: five conv stages then global average pool down to 256 numbers. Built for repetitive bubble texture, not ImageNet-style objects."""

    feature_dim = 256

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),  # 256 px down to 128
            _conv_block(32, 64),  # 128 down to 64
            _conv_block(64, 128),  # 64 down to 32
            _conv_block(128, 256),  # 32 down to 16
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(x), 1)


def infer_architecture(state_dict: dict) -> str:
    """Guesses which architecture a checkpoint used when the json forgot to say. Pass the state dict. You get a string like baseline or resnet50."""
    keys = list(state_dict.keys())
    if any(
        key == "backbone.fc.weight" or key == "backbone.fc.bias"
        for key in keys
    ):
        return "resnet18"
    if any(key.startswith("backbone.layer") for key in keys):
        return "resnet50"
    if any(key.startswith("backbone.features") for key in keys):
        return "baseline"
    if any(key.startswith("backbone.0.") for key in keys):
        return "vgg16"
    return "resnet18"


def infer_head(state_dict: dict) -> str:
    """Guesses the head type from checkpoint keys. Native is a single Linear named head.weight. C2 stores the first layer as head.0.weight. Pass the state dict."""
    if "head.0.weight" in state_dict:
        return "c2"
    return "native"


def _make_head(
    in_features: int, head_type: str, num_outputs: int = NUM_OUTPUTS
) -> nn.Module:
    """Builds the regression head. Native is one linear layer. C2 is a small two-layer head, the only extra head that actually helped in the paper. Pass feature size and head type. You get an nn.Module."""
    if head_type not in HEAD_TYPES:
        raise ValueError(
            f"head must be one of {HEAD_TYPES}, got {head_type!r}"
        )
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
    """Predicts water, solids, and bitumen from one RGB froth photo. Pass architecture (baseline, resnet50, vgg16, or old resnet18), whether to start from ImageNet weights, and native or c2 for the head. pretrained is ignored on baseline because that one always trains from scratch."""

    def __init__(
        self,
        architecture: str = "baseline",
        pretrained: bool = False,
        head: str = "native",
        num_outputs: int = NUM_OUTPUTS,
    ):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {ARCHITECTURES}, got "
                f"{architecture!r}"
            )
        if head not in HEAD_TYPES:
            raise ValueError(f"head must be one of {HEAD_TYPES}, got {head!r}")

        self.architecture = architecture
        self.head_type = head
        self.pretrained = bool(pretrained) and architecture != "baseline"
        self.num_outputs = num_outputs
        # Old ResNet-18 files stored the last linear layer as backbone.fc, not a separate head.
        self._legacy_combined = architecture == "resnet18"

        if architecture == "baseline":
            self.backbone = CompactFrothCNN()
            self.head = _make_head(
                CompactFrothCNN.feature_dim, head, num_outputs
            )
        elif architecture == "resnet18":
            weights = (
                models.ResNet18_Weights.DEFAULT if self.pretrained else None
            )
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, num_outputs)
            self.backbone = backbone
            self.head = None
        elif architecture == "resnet50":
            weights = (
                models.ResNet50_Weights.DEFAULT if self.pretrained else None
            )
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.head = _make_head(in_features, head, num_outputs)
        else:  # remaining architecture is vgg16
            weights = models.VGG16_Weights.DEFAULT if self.pretrained else None
            vgg = models.vgg16(weights=weights)
            self.backbone = nn.Sequential(
                vgg.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.head = _make_head(512, head, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Order is always water, solids, bitumen. Raw linear numbers, no softmax.
        features = self.backbone(x)
        if self._legacy_combined:
            return features
        return self.head(features)

    def head_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters for just the regression head, used when we freeze the backbone."""
        if self._legacy_combined:
            return self.backbone.fc.parameters()
        return self.head.parameters()

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters for everything except the regression head."""
        if self._legacy_combined:
            head_ids = {
                id(parameter) for parameter in self.backbone.fc.parameters()
            }
            return (
                parameter
                for parameter in self.backbone.parameters()
                if id(parameter) not in head_ids
            )
        return self.backbone.parameters()

    def freeze_backbone(self) -> None:
        for parameter in self.backbone_parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone_parameters():
            parameter.requires_grad = True

    def _output_linear(self) -> Optional[nn.Linear]:
        """Finds the last linear layer that actually emits the three grades."""
        if self._legacy_combined:
            layer = getattr(self.backbone, "fc", None)
            return layer if isinstance(layer, nn.Linear) else None
        module = self.head
        if isinstance(module, nn.Linear):
            return module
        if isinstance(module, nn.Sequential):
            for child in reversed(list(module.children())):
                if isinstance(child, nn.Linear):
                    return child
        return None

    def init_output_bias(self, output_stats: dict) -> None:
        """Biases the last layer to the training-set means so epoch 1 predicts the average instead of zero. A fresh head outputs about 0, and water around 70 percent would already give a terrible MAE before any learning. Pass the output_stats dict from the dataset."""
        layer = self._output_linear()
        if layer is None or layer.bias is None:
            return
        means = []
        for name in ("Water", "Solids", "Bitumen"):
            stats = output_stats.get(name) or {}
            means.append(float(stats.get("mean", 0.0)))
        bias = torch.tensor(
            means, dtype=layer.bias.dtype, device=layer.bias.device
        )
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.copy_(bias)

    def config_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "head": self.head_type,
            "pretrained": self.pretrained,
            "image_size": IMAGE_SIZE
            if self.architecture != "resnet18"
            else 224,
        }

    @classmethod
    def from_checkpoint(cls, path, metadata=None, device=None):
        """Loads weights from a .pt file. Pass the path, optional metadata, and optional device. If the json has no architecture field we treat it as old ResNet-18. You get a model already on the device and in eval mode."""
        if device is None:
            device = select_torch_device()
        metadata = metadata or {}
        state_dict = torch.load(path, map_location=device)
        architecture = metadata.get("architecture") or infer_architecture(
            state_dict
        )
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
