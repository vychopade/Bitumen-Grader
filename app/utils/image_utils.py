"""Shared image resize / normalize helpers for train and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from app.ml.recipe import IMAGE_SIZE

# 256×256×3 for new models (paper Table 2). Legacy ResNet-18 checkpoints used
# 224.
LEGACY_IMAGE_SIZE = 224
_INTERPOLATION = InterpolationMode.BILINEAR

# Same ImageNet mean/std used for transfer backbones; also applied to the
# baseline so train/val/inference share one preprocessing pipeline.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _as_rgb_oriented(image: Image.Image) -> Image.Image:
    try:
        transposed = ImageOps.exif_transpose(image)
    except (OSError, ValueError, SyntaxError):
        transposed = None
    if transposed is not None:
        image = transposed
    return image.convert("RGB")


def load_rgb_image(
    source: Union[str, Path, Image.Image],
    *,
    max_decode_edge: Optional[int] = None,
) -> Image.Image:
    """Load a photo as RGB, honouring EXIF orientation when present.

    ``max_decode_edge`` asks JPEG (and similar) decoders for a smaller bitmap
    so 12MP camera files are not fully expanded just to grade at 256×256.
    Preview callers omit it and get the original pixels.
    """
    if isinstance(source, Image.Image):
        image = _as_rgb_oriented(source)
        if max_decode_edge:
            image = cap_long_edge(image, max_decode_edge)
        return image

    with Image.open(source) as opened:
        if max_decode_edge and max_decode_edge > 0:
            try:
                opened.draft(
                    "RGB", (int(max_decode_edge), int(max_decode_edge))
                )
            except (OSError, ValueError, SyntaxError):
                pass
        opened.load()
        image = _as_rgb_oriented(opened)
        if max_decode_edge:
            image = cap_long_edge(image, max_decode_edge)
        return image


def cap_long_edge(image: Image.Image, max_edge: int) -> Image.Image:
    """Shrink huge camera files before the square resize (keeps aspect ratio).

    Final ``image_size`` × ``image_size`` conversion still happens in
    ``standardize_to_model_size``. Skipping this step would decode 12MP photos
    every epoch.
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge or max_edge <= 0:
        return image
    scale = max_edge / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def standardize_to_model_size(
    image: Image.Image, image_size: int = IMAGE_SIZE
) -> Image.Image:
    """RGB square of ``image_size``×``image_size`` (paper Table 2).

    Phone, crop, and thumbnail photos all become the same canvas before
    training or grading. Non-square sources are stretched to fit.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    target = (int(image_size), int(image_size))
    if image.size == target:
        return image
    return image.resize(target, Image.Resampling.BILINEAR)


def prepare_image(
    source: Union[str, Path, Image.Image],
    image_size: int = IMAGE_SIZE,
    *,
    square: bool = True,
) -> Image.Image:
    """RGB image ready for train/eval transforms: EXIF-corrected, size-capped.

    When ``square`` is True (new models), the photo is also resized to
    ``image_size``×``image_size`` here so every tensor starts from the same
    geometry. Legacy ResNet-18 eval keeps ``square=False`` so CenterCrop(224)
    can run on a 256 short-edge image.
    """
    max_edge = max(int(image_size) * 2, 512)
    image = load_rgb_image(source, max_decode_edge=max_edge)
    if square:
        image = standardize_to_model_size(image, image_size)
    return image


def _square_resize(image_size: int) -> transforms.Resize:
    """Force every photo to the model's square input, any source resolution."""
    return transforms.Resize(
        (image_size, image_size), interpolation=_INTERPOLATION
    )


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


def build_eval_transforms(
    image_size: int = IMAGE_SIZE, *, legacy_crop: bool = False
) -> transforms.Compose:
    """Validation / inference transforms.

    New models resize to ``image_size`` × ``image_size`` (same square as
    training).
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
