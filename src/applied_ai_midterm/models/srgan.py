"""Residual SRGAN generator, discriminator, and composite generator loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision.models import VGG19_Weights, vgg19


class ResidualBlock(nn.Module):
    """Two-convolution residual block with batch normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Add the learned residual to the input features."""
        return inputs + self.block(inputs)


class UpsampleBlock(nn.Sequential):
    """Pixel-shuffle block that doubles spatial resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )


class Generator(nn.Module):
    """Residual generator mapping 32x32 RGB tensors to 128x128 in ``[-1, 1]``."""

    def __init__(self, *, channels: int = 64, residual_blocks: int = 8) -> None:
        super().__init__()
        if residual_blocks <= 0:
            raise ValueError("residual_blocks must be greater than zero.")
        self.initial = nn.Sequential(nn.Conv2d(3, channels, 9, padding=4), nn.PReLU())
        self.residual_trunk = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(residual_blocks))
        )
        self.post_residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.upsampling = nn.Sequential(UpsampleBlock(channels), UpsampleBlock(channels))
        self.output = nn.Sequential(nn.Conv2d(channels, 3, 9, padding=4), nn.Tanh())

    def forward(self, inputs: Tensor) -> Tensor:
        """Generate a four-times-upscaled RGB tensor."""
        initial_features = self.initial(inputs)
        residual_features = self.post_residual(self.residual_trunk(initial_features))
        return self.output(self.upsampling(initial_features + residual_features))


def _discriminator_block(
    input_channels: int,
    output_channels: int,
    *,
    stride: int,
    batch_normalization: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1)
    ]
    if batch_normalization:
        layers.append(nn.BatchNorm2d(output_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    """Progressively deeper discriminator returning one real/fake logit."""

    def __init__(self, *, base_channels: int = 64) -> None:
        super().__init__()
        specifications = (
            (3, base_channels, 1, False),
            (base_channels, base_channels, 2, True),
            (base_channels, base_channels * 2, 1, True),
            (base_channels * 2, base_channels * 2, 2, True),
            (base_channels * 2, base_channels * 4, 1, True),
            (base_channels * 4, base_channels * 4, 2, True),
            (base_channels * 4, base_channels * 8, 1, True),
            (base_channels * 8, base_channels * 8, 2, True),
        )
        self.features = nn.Sequential(
            *(
                _discriminator_block(
                    source,
                    target,
                    stride=stride,
                    batch_normalization=normalization,
                )
                for source, target, stride, normalization in specifications
            )
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base_channels * 8, 1)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return one unbounded real/fake logit per image."""
        return self.classifier(self.features(inputs))


class VGGPerceptualFeatures(nn.Module):
    """Frozen VGG19 features for optional perceptual loss."""

    def __init__(
        self,
        *,
        weights: VGG19_Weights | None = VGG19_Weights.DEFAULT,
        feature_layer: int = 36,
    ) -> None:
        super().__init__()
        self.features = vgg19(weights=weights).features[:feature_layer].eval()
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> VGGPerceptualFeatures:
        """Keep frozen VGG features in evaluation mode."""
        super().train(False)
        return self

    def forward(self, inputs: Tensor) -> Tensor:
        """Convert ``[-1, 1]`` tensors to ImageNet-normalized VGG features."""
        unit_range = inputs.add(1.0).div(2.0)
        mean = inputs.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        standard_deviation = inputs.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return self.features((unit_range - mean) / standard_deviation)


@dataclass(frozen=True, slots=True)
class GeneratorLossValues:
    """Total generator loss and its detached component values."""

    total: Tensor
    pixel: Tensor
    adversarial: Tensor
    perceptual: Tensor


class GeneratorLoss(nn.Module):
    """Combine pixel/content, adversarial, and optional perceptual losses."""

    def __init__(
        self,
        *,
        adversarial_weight: float = 1e-3,
        perceptual_weight: float = 6e-3,
        perceptual_features: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_features = perceptual_features
        self.pixel_criterion = nn.L1Loss()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()

    def forward(
        self, generated: Tensor, target: Tensor, generated_logits: Tensor
    ) -> GeneratorLossValues:
        """Calculate weighted generator loss from discriminator logits."""
        pixel = self.pixel_criterion(generated, target)
        adversarial = self.adversarial_criterion(
            generated_logits, torch.ones_like(generated_logits)
        )
        perceptual = generated.new_zeros(())
        if self.perceptual_features is not None:
            generated_features = self.perceptual_features(generated)
            with torch.no_grad():
                target_features = self.perceptual_features(target)
            perceptual = self.pixel_criterion(generated_features, target_features)
        total = pixel + self.adversarial_weight * adversarial
        total = total + self.perceptual_weight * perceptual
        return GeneratorLossValues(
            total=total,
            pixel=pixel.detach(),
            adversarial=adversarial.detach(),
            perceptual=perceptual.detach(),
        )
