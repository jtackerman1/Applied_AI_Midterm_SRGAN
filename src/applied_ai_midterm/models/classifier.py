"""MobileNetV2 transfer-learning classifier for Model A and Model B."""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_mobilenet_v2_classifier(
    *,
    weights: MobileNet_V2_Weights | None = MobileNet_V2_Weights.DEFAULT,
    freeze_features: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Build MobileNetV2 with one binary logit and optional frozen features."""
    model = mobilenet_v2(weights=weights, dropout=dropout)
    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    return model

