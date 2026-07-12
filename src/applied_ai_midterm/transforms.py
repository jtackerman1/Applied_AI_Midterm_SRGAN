"""Classifier and SRGAN image transformations and display helpers."""

from __future__ import annotations

import random
from collections.abc import Callable

from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SRGAN_MIN = -1.0
SRGAN_MAX = 1.0


def classifier_transform(
    *, training: bool, image_size: int = 128
) -> Callable[[Image.Image], Tensor]:
    """Build ImageNet-normalized classifier preprocessing with train-only augmentation."""
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.80, 1.0),
                    ratio=(0.90, 1.10),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12, interpolation=InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def denormalize_imagenet(tensor: Tensor) -> Tensor:
    """Reverse ImageNet normalization and clamp the result to ``[0, 1]``."""
    mean = tensor.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


def normalize_srgan(tensor: Tensor) -> Tensor:
    """Map an image tensor from ``[0, 1]`` to the SRGAN range ``[-1, 1]``."""
    return tensor.mul(2.0).sub(1.0)


def denormalize_srgan(tensor: Tensor) -> Tensor:
    """Map an SRGAN tensor from ``[-1, 1]`` back to display range ``[0, 1]``."""
    return tensor.add(1.0).div(2.0).clamp(0.0, 1.0)


class SRGANPairTransform:
    """Create corresponding 32x32 LR and 128x128 HR tensors in ``[-1, 1]``."""

    def __init__(
        self,
        *,
        training: bool,
        low_resolution_size: int = 32,
        high_resolution_size: int = 128,
    ) -> None:
        if high_resolution_size != low_resolution_size * 4:
            raise ValueError("high_resolution_size must be four times low_resolution_size.")
        self.training = training
        self.low_resolution_size = low_resolution_size
        self.high_resolution_size = high_resolution_size
        self.color_jitter = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05
        )

    def _prepare_high_resolution(self, image: Image.Image) -> Image.Image:
        if not self.training:
            return functional.resize(
                image,
                [self.high_resolution_size, self.high_resolution_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

        top, left, height, width = transforms.RandomResizedCrop.get_params(
            image, scale=(0.80, 1.0), ratio=(0.90, 1.10)
        )
        image = functional.resized_crop(
            image,
            top,
            left,
            height,
            width,
            [self.high_resolution_size, self.high_resolution_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        if random.random() < 0.5:
            image = functional.hflip(image)
        angle = transforms.RandomRotation.get_params([-12.0, 12.0])
        image = functional.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        return self.color_jitter(image)

    def __call__(self, image: Image.Image) -> tuple[Tensor, Tensor]:
        """Return an LR/HR pair derived from the same augmented HR image."""
        high_resolution_image = self._prepare_high_resolution(image.convert("RGB"))
        low_resolution_image = functional.resize(
            high_resolution_image,
            [self.low_resolution_size, self.low_resolution_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        low_resolution = normalize_srgan(functional.to_tensor(low_resolution_image))
        high_resolution = normalize_srgan(functional.to_tensor(high_resolution_image))
        return low_resolution, high_resolution
