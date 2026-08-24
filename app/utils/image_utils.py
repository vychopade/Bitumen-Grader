"""Shared image resize / normalize helpers for train and inference."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from app.ml.cnn_model import IMAGE_SIZE as DEFAULT_IMAGE_SIZE

# 256×256×3 for new models. Legacy ResNet-18 checkpoints used 224.
IMAGE_SIZE = DEFAULT_IMAGE_SIZE
LEGACY_IMAGE_SIZE = 224
_INTERPOLATION = InterpolationMode.BILINEAR

# Same ImageNet mean/std used for transfer backbones; also applied to the
# baseline so train/val/inference share one preprocessing pipeline.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_rgb_image(source: Union[str, Path, Image.Image]) -> Image.Image:
    """Load a photo as RGB, honouring EXIF orientation when present."""
    if isinstance(source, Image.Image):
        image = source
        try:
            transposed = ImageOps.exif_transpose(image)
        except (OSError, ValueError, SyntaxError):
            transposed = None
        if transposed is not None:
            image = transposed
        return image.convert("RGB")

    with Image.open(source) as opened:
        opened.load()
        try:
            transposed = ImageOps.exif_transpose(opened)
        except (OSError, ValueError, SyntaxError):
            transposed = None
        image = transposed if transposed is not None else opened
        return image.convert("RGB")


def cap_long_edge(image: Image.Image, max_edge: int) -> Image.Image:
    """Shrink huge camera files before the square resize (keeps aspect ratio).

    Final ``image_size`` × ``image_size`` conversion still happens in the
    transform pipeline. Skipping this step would decode 12MP photos every epoch.
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge or max_edge <= 0:
        return image
    scale = max_edge / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def prepare_image(
    source: Union[str, Path, Image.Image],
    image_size: int = IMAGE_SIZE,
) -> Image.Image:
    """RGB image ready for train/eval transforms: EXIF-corrected, long-edge capped."""
    image = load_rgb_image(source)
    return cap_long_edge(image, max(int(image_size) * 2, 512))


def _square_resize(image_size: int) -> transforms.Resize:
    """Force every photo to the model's square input, any source resolution."""
    return transforms.Resize((image_size, image_size), interpolation=_INTERPOLATION)


def build_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Resize to the study input (256×256) with only orientation flips.

    Froth signal lives in colour, texture, and bubble packing, so colour jitter
    and random zoom are omitted — they erase the cues the model needs.
    """
    return transforms.Compose(
        [
            _square_resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size: int = IMAGE_SIZE, *, legacy_crop: bool = False) -> transforms.Compose:
    """Validation / inference transforms.

    New models resize to ``image_size`` × ``image_size`` (same square as training).
    Legacy ResNet-18 checkpoints used Resize(256) + CenterCrop(224).
    """
    if legacy_crop:
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=_INTERPOLATION),
                transforms.CenterCrop(LEGACY_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            _square_resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def is_legacy_resnet18(metadata: Optional[dict]) -> bool:
    """True for checkpoints saved before the architecture field existed."""
    metadata = metadata or {}
    architecture = metadata.get("architecture", "resnet18")
    image_size = int(metadata.get("image_size", LEGACY_IMAGE_SIZE))
    return architecture == "resnet18" and image_size == LEGACY_IMAGE_SIZE


def image_size_from_metadata(metadata: Optional[dict]) -> int:
    metadata = metadata or {}
    if "image_size" in metadata:
        return int(metadata["image_size"])
    if metadata.get("architecture", "resnet18") == "resnet18":
        return LEGACY_IMAGE_SIZE
    return IMAGE_SIZE
