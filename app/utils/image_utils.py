"""Resize and normalize photos the same way for training and grading."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from app.ml.recipe import IMAGE_SIZE

# New models take 256 by 256. Old ResNet-18 checkpoints were trained at 224.
LEGACY_IMAGE_SIZE = 224
_INTERPOLATION = InterpolationMode.BILINEAR

# ImageNet mean and std. Transfer backbones expect this, and we use the same
# numbers on the baseline so train, val, and grading all see the same pixels.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Pad colour matches ImageNet mean so letterbox bars are ~0 after Normalize.
_LETTERBOX_FILL = tuple(int(round(channel * 255.0)) for channel in IMAGENET_MEAN)


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
    """Open a photo as RGB and honour EXIF rotation. If you set max_decode_edge, big JPEGs are downsampled while decoding so we do not expand a 12 megapixel file just to grade at 256 pixels."""
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
    """Shrink huge camera files before the model-size resize, keeping the aspect ratio. Skipping this would decode 12 megapixel photos every epoch."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge or max_edge <= 0:
        return image
    scale = max_edge / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def _letterbox_to_square(
    image: Image.Image, image_size: int, fill=_LETTERBOX_FILL
) -> Image.Image:
    """Fits the photo inside image_size by image_size without stretching, then pads the leftover bars."""
    target = int(image_size)
    width, height = image.size
    if width <= 0 or height <= 0:
        return Image.new("RGB", (target, target), fill)
    scale = target / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    if (new_width, new_height) != (width, height):
        image = image.resize(
            (new_width, new_height), Image.Resampling.BILINEAR
        )
    if image.size == (target, target):
        return image
    canvas = Image.new("RGB", (target, target), fill)
    left = (target - image.size[0]) // 2
    top = (target - image.size[1]) // 2
    canvas.paste(image, (left, top))
    return canvas


def standardize_to_model_size(
    image: Image.Image,
    image_size: int = IMAGE_SIZE,
    *,
    preserve_aspect: bool = True,
) -> Image.Image:
    """Put the photo on an image_size by image_size canvas. By default the original aspect ratio is kept and empty bars are padded with ImageNet-mean grey. preserve_aspect=False stretches, which is what older checkpoints were trained with."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    target = (int(image_size), int(image_size))
    if image.size == target:
        return image
    if not preserve_aspect:
        return image.resize(target, Image.Resampling.BILINEAR)
    return _letterbox_to_square(image, image_size)


def prepare_image(
    source: Union[str, Path, Image.Image],
    image_size: int = IMAGE_SIZE,
    *,
    square: bool = True,
    preserve_aspect: bool = True,
) -> Image.Image:
    """Get a photo ready for train or eval: EXIF-corrected and size-capped. If square is True we resize to that size here, keeping aspect ratio unless preserve_aspect is False. Old ResNet-18 eval leaves square False so CenterCrop can still run on a 256 short-edge image."""
    max_edge = max(int(image_size) * 2, 512)
    image = load_rgb_image(source, max_decode_edge=max_edge)
    if square:
        image = standardize_to_model_size(
            image, image_size, preserve_aspect=preserve_aspect
        )
    return image


class _LetterboxToSquare:
    """Torchvision-compatible transform that letterboxes a PIL image to a square."""

    def __init__(self, image_size: int):
        self.image_size = int(image_size)

    def __call__(self, image: Image.Image) -> Image.Image:
        return standardize_to_model_size(
            image, self.image_size, preserve_aspect=True
        )


def _square_resize(image_size: int) -> transforms.Resize:
    """Builds a torchvision Resize that stretches every photo to the model's square input. Used only for older checkpoints that were trained that way."""
    return transforms.Resize(
        (image_size, image_size), interpolation=_INTERPOLATION
    )


def build_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Training transforms: letterbox to a square, a small rotate/shift, then left-right and up-down flips. Colour jitter and random zoom wipe out froth texture so we skip them."""
    return transforms.Compose(
        [
            _LetterboxToSquare(image_size),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.04, 0.04),
                interpolation=_INTERPOLATION,
                fill=_LETTERBOX_FILL,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(
    image_size: int = IMAGE_SIZE,
    *,
    legacy_crop: bool = False,
    preserve_aspect: bool = True,
) -> transforms.Compose:
    """Validation and grading transforms. New models letterbox to a square so aspect ratio is kept. Old ResNet-18 checkpoints used Resize 256 then CenterCrop 224 when legacy_crop is True. preserve_aspect=False stretches, matching older non-ResNet checkpoints."""
    if legacy_crop:
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=_INTERPOLATION),
                transforms.CenterCrop(LEGACY_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    resize = (
        _LetterboxToSquare(image_size)
        if preserve_aspect
        else _square_resize(image_size)
    )
    return transforms.Compose(
        [
            resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def is_legacy_resnet18(metadata: Optional[dict]) -> bool:
    """True when this checkpoint is an old ResNet-18 saved at 224 pixels, before we stored an architecture field."""
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
