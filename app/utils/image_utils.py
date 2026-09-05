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
    """Opens a photo as RGB and respects EXIF rotation when it is there. Pass a path or a PIL image. If you set max_decode_edge, big JPEGs are downsampled while decoding so we do not expand a 12 megapixel file just to grade at 256 pixels."""
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
    """Shrinks huge camera files before the square resize, keeping the aspect ratio. Pass the image and a max edge length. You get a smaller PIL image back. Skipping this would decode 12 megapixel photos every epoch."""
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
    """Stretches the photo to a square of image_size by image_size so phone shots, crops, and thumbnails all hit the same canvas. Pass a PIL image and the target size. You get an RGB square back."""
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
    """Gets a photo ready for the train or eval transforms: EXIF-corrected and size-capped. Pass a path or image and the model size. If square is True we resize to that size here. Old ResNet-18 eval leaves square False so CenterCrop can still run on a 256 short-edge image."""
    max_edge = max(int(image_size) * 2, 512)
    image = load_rgb_image(source, max_decode_edge=max_edge)
    if square:
        image = standardize_to_model_size(image, image_size)
    return image


def _square_resize(image_size: int) -> transforms.Resize:
    """Builds a torchvision Resize that forces every photo to the model's square input, whatever resolution it started at."""
    return transforms.Resize(
        (image_size, image_size), interpolation=_INTERPOLATION
    )


def build_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Training transforms: square resize plus left-right and up-down flips only. Colour jitter and random zoom wipe out froth texture so we skip them. Pass the image size. You get a Compose you can call on a PIL image."""
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
    """Validation and grading transforms. New models just resize to a square. Old ResNet-18 checkpoints used Resize 256 then CenterCrop 224, which you get when legacy_crop is True. Pass the image size. You get a Compose."""
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
    """True when this checkpoint is an old ResNet-18 saved at 224 pixels, before we stored an architecture field. Pass the metadata dict."""
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
