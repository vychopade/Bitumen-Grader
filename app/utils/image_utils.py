"""
Image preprocessing helpers.

Provides shared image loading and preprocessing utilities (resizing,
normalization, tensor conversion, augmentation, etc.) used by both the
training pipeline and the inference/prediction pipeline.
"""
from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms

#: Spatial resolution expected by BitumenCNN's ResNet-18 backbone.
IMAGE_SIZE = 224

#: Standard ImageNet normalization statistics, matching the data the
#: pretrained ResNet-18 backbone was originally trained on.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

_INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess_for_training(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL image into an augmented, normalized training tensor.

    Applies a random resized crop, random horizontal flip, and color jitter
    (for data augmentation) before converting to a tensor and normalizing
    with ImageNet statistics.

    Args:
        pil_image: Input image.

    Returns:
        A ``(3, IMAGE_SIZE, IMAGE_SIZE)`` float tensor.
    """
    return _TRAIN_TRANSFORM(pil_image.convert("RGB"))


def preprocess_for_inference(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL image into a normalized inference tensor (no augmentation).

    Resizes the image to ``(IMAGE_SIZE, IMAGE_SIZE)`` and normalizes it with
    ImageNet statistics, with no random augmentation applied.

    Args:
        pil_image: Input image.

    Returns:
        A ``(3, IMAGE_SIZE, IMAGE_SIZE)`` float tensor.
    """
    return _INFERENCE_TRANSFORM(pil_image.convert("RGB"))
