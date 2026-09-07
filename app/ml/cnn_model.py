"""CNNs that turn a froth photo into water, solids, and bitumen.

The default is a frozen ImageNet ResNet18 with a small linear head. There is
also a from-scratch CNN, plus ResNet50 and VGG16 if you want to try those.
A deep batch-norm head is not offered because it collapsed on this data.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
from torchvision import models

from app.constants import OUTPUT_NAMES
from app.ml.recipe import image_size_for_architecture

NUM_OUTPUTS = 3  # three grades: water, solids, bitumen


def select_torch_device() -> torch.device:
    """Pick CUDA, then Apple Metal, then CPU."""
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


ARCHITECTURES = ("resnet18_tap", "baseline", "resnet50", "vgg16", "resnet18")
TRAINABLE_ARCHITECTURES = ("resnet18_tap", "baseline", "resnet50", "vgg16")
HEAD_TYPES = ("native", "c2", "norm")

# Defaults that worked best for bitumen on this dataset.
DEFAULT_ARCHITECTURE = "resnet18_tap"
DEFAULT_HEAD = "norm"

ARCHITECTURE_LABELS = {
    "resnet18_tap": "ResNet18 transfer",
    "baseline": "Baseline CNN (from scratch)",
    "resnet50": "ResNet50 (ImageNet transfer)",
    "vgg16": "VGG16 (ImageNet transfer)",
    "resnet18": "ResNet18 (legacy)",
}

# Dihedral group of the square: four rotations, each with and without a flip.
# Froth has no "up", so all eight views of a photo are equally valid.
TTA_VIEWS = 8


def dihedral_view(x: torch.Tensor, view: int) -> torch.Tensor:
    """One of the eight square symmetries of a [.., H, W] tensor."""
    if view & 4:
        x = torch.flip(x, [-1])
    turns = view & 3
    return torch.rot90(x, turns, dims=[-2, -1]) if turns else x


def _norm_layer(num_channels: int, use_groupnorm: bool) -> nn.Module:
    if use_groupnorm:
        groups = 8 if num_channels >= 8 else 1
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    return nn.BatchNorm2d(num_channels)


def _conv_block(
    in_channels: int, out_channels: int, use_groupnorm: bool = True
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        ),
        _norm_layer(out_channels, use_groupnorm),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CompactFrothCNN(nn.Module):
    """Small CNN trained from scratch: five conv stages, then global average pool to 256 numbers."""

    feature_dim = 256

    def __init__(self, use_groupnorm: bool = True) -> None:
        super().__init__()
        self.use_groupnorm = bool(use_groupnorm)
        self.features = nn.Sequential(
            _conv_block(3, 32, self.use_groupnorm),  # 256 px down to 128
            _conv_block(32, 64, self.use_groupnorm),  # 128 down to 64
            _conv_block(64, 128, self.use_groupnorm),  # 64 down to 32
            _conv_block(128, 256, self.use_groupnorm),  # 32 down to 16
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            _norm_layer(256, self.use_groupnorm),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(x), 1)


class FrothTransferTrunk(nn.Module):
    """ImageNet ResNet18 used as a feature extractor for froth photos.

    Only the last stage (layer4) is pooled, giving 512 features. BatchNorm
    stays on its ImageNet running stats instead of updating them from our
    small batches. The trunk is frozen by default so we only train the head.
    """

    TAPS = (4,)
    WIDTHS = {1: 64, 2: 128, 3: 256, 4: 512}

    def __init__(self, pretrained: bool = True, bn_eval: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList(
            [net.layer1, net.layer2, net.layer3, net.layer4]
        )
        self.bn_eval = bool(bn_eval)
        self.feature_dim = sum(self.WIDTHS[i] for i in self.TAPS)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.bn_eval:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        taps = []
        for index, stage in enumerate(self.stages, 1):
            hidden = stage(hidden)
            if index in self.TAPS:
                taps.append(hidden.mean(dim=(2, 3)))
        return torch.cat(taps, dim=1)


def infer_architecture(state_dict: dict) -> str:
    """Guess the architecture when the json forgot to say. Returns something like baseline or resnet50."""
    keys = list(state_dict.keys())
    if any(key.startswith("backbone.stages.") for key in keys):
        return "resnet18_tap"
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


def infer_num_outputs(state_dict: dict) -> int:
    """How many grades the last linear layer emits, usually 1 or 3."""
    for key in (
        "head.weight",
        "head.3.weight",
        "backbone.fc.weight",
    ):
        weight = state_dict.get(key)
        if weight is not None and getattr(weight, "ndim", 0) == 2:
            return int(weight.shape[0])
    return NUM_OUTPUTS


def infer_head(state_dict: dict) -> str:
    """Guess the head from checkpoint keys: native is one Linear, c2 starts with a Linear at head.0, and norm starts with the feature standardiser."""
    if "head.0.mean" in state_dict and "head.0.std" in state_dict:
        return "norm"
    if "head.0.weight" in state_dict:
        return "c2"
    return "native"


class FeatureStandardizer(nn.Module):
    """Z-score each pooled backbone channel using stats from the training set.

    ResNet channels come out at very different scales, so a linear layer on the
    raw vector mostly listens to a few loud channels. These mean/std values are
    fitted once on the training photos and stored as buffers in the .pt file.
    Grading reloads them; it should not recompute them from the photos being
    graded, because those are usually a handful of shots of one pan.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("std", torch.ones(num_features))
        # Lets training assert it fitted, and grading assert it loaded.
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> None:
        """Set the statistics from a [N, D] block of training-split features."""
        self.mean.copy_(features.mean(dim=0).to(self.mean.dtype))
        self.std.copy_(
            features.std(dim=0).clamp_min(1e-6).to(self.std.dtype)
        )
        self.fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def find_feature_standardizer(module: nn.Module):
    """The FeatureStandardizer inside a model, or None if the head has none."""
    for child in module.modules():
        if isinstance(child, FeatureStandardizer):
            return child
    return None


def _make_head(
    in_features: int, head_type: str, num_outputs: int = NUM_OUTPUTS
) -> nn.Module:
    """Build the regression head. Native is one linear layer. C2 is a two-layer ReLU head. Norm standardises the features first."""
    if head_type not in HEAD_TYPES:
        raise ValueError(
            f"head must be one of {HEAD_TYPES}, got {head_type!r}"
        )
    if head_type == "norm":
        # Standardise the channels, then LayerNorm, then one linear layer.
        # A hidden layer on top of this overfit on our data.
        head = nn.Sequential(
            FeatureStandardizer(in_features),
            nn.LayerNorm(in_features),
            nn.Dropout(0.2),
            nn.Linear(in_features, num_outputs),
        )
        with torch.no_grad():
            # Start at zero so a z-scored target begins at the training mean.
            head[3].weight.zero_()
            head[3].bias.zero_()
        return head
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
    """Predict water, solids, and bitumen from one RGB froth photo. pretrained is ignored on baseline because that one always trains from scratch."""

    def __init__(
        self,
        architecture: str = "baseline",
        pretrained: bool = False,
        head: str = "native",
        num_outputs: int = NUM_OUTPUTS,
        use_groupnorm: bool = True,
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
        self.use_groupnorm = bool(use_groupnorm) and architecture == "baseline"
        # Extra dropout on pooled features. The norm head already has its own,
        # so this is turned off there to avoid dropping twice.
        self.feature_dropout = nn.Dropout(0.0 if head == "norm" else 0.2)
        # Old ResNet-18 files stored the last linear layer as backbone.fc, not a separate head.
        self._legacy_combined = architecture == "resnet18"

        if architecture == "resnet18_tap":
            self.backbone = FrothTransferTrunk(pretrained=self.pretrained)
            self.head = _make_head(
                self.backbone.feature_dim, head, num_outputs
            )
        elif architecture == "baseline":
            self.backbone = CompactFrothCNN(use_groupnorm=self.use_groupnorm)
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
        return self.head(self.feature_dropout(features))

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
        """Bias the last layer to the training-set means so epoch 1 predicts the average instead of zero."""
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
            "image_size": image_size_for_architecture(self.architecture)
            if self.architecture != "resnet18"
            else 224,
            "preserve_aspect_ratio": True,
            "num_outputs": self.num_outputs,
            "use_groupnorm": self.use_groupnorm,
            # Average the eight square symmetries at grade time for the
            # transfer model. Older checkpoints omit this and stay single-view.
            "tta_views": TTA_VIEWS if self.architecture == "resnet18_tap" else 1,
        }

    @classmethod
    def from_checkpoint(cls, path, metadata=None, device=None):
        """Load weights from a .pt file. If the json has no architecture field we treat it as old ResNet-18."""
        if device is None:
            device = select_torch_device()
        metadata = metadata or {}
        state_dict = torch.load(path, map_location=device)
        if is_triple_payload(state_dict):
            raise ValueError(
                "This file is a triple-head bundle. Load it with "
                "TripleBitumenRegressor.from_checkpoint."
            )
        architecture = metadata.get("architecture") or infer_architecture(
            state_dict
        )
        head = metadata.get("head") or infer_head(state_dict)
        if architecture not in ARCHITECTURES:
            architecture = infer_architecture(state_dict)
        if head not in HEAD_TYPES:
            head = infer_head(state_dict)
        num_outputs = int(
            metadata.get("num_outputs") or infer_num_outputs(state_dict)
        )
        use_groupnorm = True
        if architecture == "baseline":
            use_groupnorm = not any(
                key.endswith("running_mean")
                and key.startswith("backbone.features")
                for key in state_dict
            )

        model = cls(
            architecture=architecture,
            pretrained=False,
            head=head,
            num_outputs=num_outputs,
            use_groupnorm=use_groupnorm,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)


TRIPLE_HEADS_FORMAT = "triple_heads"


def is_triple_payload(payload) -> bool:
    """True when this checkpoint stores three single-output heads instead of one 3-wide linear layer."""
    if not isinstance(payload, dict):
        return False
    if payload.get("format") == TRIPLE_HEADS_FORMAT:
        return True
    heads = payload.get("heads")
    return isinstance(heads, dict) and all(name in heads for name in OUTPUT_NAMES)


class TripleBitumenRegressor(nn.Module):
    """Three single-output networks, one per grade. Forward stacks Water, Solids, then Bitumen into N by 3."""

    def __init__(self, heads: dict):
        super().__init__()
        missing = [name for name in OUTPUT_NAMES if name not in heads]
        if missing:
            raise ValueError(
                "heads must include Water, Solids, and Bitumen; "
                f"missing {missing}"
            )
        self.heads = nn.ModuleDict(
            {name: heads[name] for name in OUTPUT_NAMES}
        )

    def forward(self, x: torch.Tensor, residual_output: Optional[str] = None) -> torch.Tensor:
        columns = []
        for name in OUTPUT_NAMES:
            if residual_output == name:
                columns.append(
                    torch.zeros(
                        x.shape[0],
                        1,
                        device=x.device,
                        dtype=x.dtype,
                    )
                )
            else:
                columns.append(self.heads[name](x))
        return torch.cat(columns, dim=1)

    def config_dict(self) -> dict:
        cfg = self.heads[OUTPUT_NAMES[0]].config_dict()
        cfg["num_outputs"] = NUM_OUTPUTS
        cfg["model_format"] = TRIPLE_HEADS_FORMAT
        return cfg

    def save(self, path) -> None:
        torch.save(
            {
                "format": TRIPLE_HEADS_FORMAT,
                "heads": {
                    name: module.state_dict()
                    for name, module in self.heads.items()
                },
            },
            path,
        )

    @classmethod
    def heads_from_payload(cls, payload: dict, metadata=None, device=None):
        """Build three independent 1-output networks from a triple-head checkpoint dict."""
        if device is None:
            device = select_torch_device()
        metadata = metadata or {}
        if not is_triple_payload(payload):
            raise ValueError("Checkpoint is not a triple-head bundle.")
        heads_state = payload["heads"]
        water_state = heads_state["Water"]
        architecture = metadata.get("architecture") or infer_architecture(
            water_state
        )
        head = metadata.get("head") or infer_head(water_state)
        if architecture not in ARCHITECTURES:
            architecture = infer_architecture(water_state)
        if head not in HEAD_TYPES:
            head = infer_head(water_state)
        use_groupnorm = bool(metadata.get("use_groupnorm", True))
        if architecture == "baseline":
            use_groupnorm = not any(
                key.endswith("running_mean")
                and key.startswith("backbone.features")
                for key in water_state
            )
        built = {}
        for name in OUTPUT_NAMES:
            model = BitumenRegressor(
                architecture=architecture,
                pretrained=False,
                head=head,
                num_outputs=1,
                use_groupnorm=use_groupnorm,
            )
            model.load_state_dict(heads_state[name])
            model.to(device)
            model.eval()
            built[name] = model
        return built

    @classmethod
    def from_payload(cls, payload: dict, metadata=None, device=None):
        if device is None:
            device = select_torch_device()
        bundle = cls(cls.heads_from_payload(payload, metadata, device))
        bundle.to(device)
        bundle.eval()
        return bundle

    @classmethod
    def from_checkpoint(cls, path, metadata=None, device=None):
        if device is None:
            device = select_torch_device()
        payload = torch.load(path, map_location=device)
        return cls.from_payload(payload, metadata, device)

    @classmethod
    def heads_from_checkpoint(cls, path, metadata=None, device=None):
        """Load the three heads without wrapping them, so training can continue from a saved run."""
        if device is None:
            device = select_torch_device()
        payload = torch.load(path, map_location=device)
        return cls.heads_from_payload(payload, metadata, device)
