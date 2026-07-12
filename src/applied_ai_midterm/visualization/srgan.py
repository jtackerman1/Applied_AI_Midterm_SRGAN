"""SRGAN low-resolution, bicubic, generated, and real-target comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import nn

from applied_ai_midterm.transforms import SRGANPairTransform, denormalize_srgan


@torch.inference_mode()
def plot_srgan_comparisons(
    generator: nn.Module,
    frame: pd.DataFrame,
    device: torch.device,
    *,
    output_path: Path | None = None,
    max_samples: int = 4,
    low_resolution_size: int = 32,
    high_resolution_size: int = 128,
) -> Figure:
    """Plot LR, bicubic, SRGAN, and real HR columns for source images."""
    if frame.empty:
        raise ValueError("Comparison frame cannot be empty.")
    if max_samples <= 0:
        raise ValueError("max_samples must be greater than zero.")
    transform = SRGANPairTransform(
        training=False,
        low_resolution_size=low_resolution_size,
        high_resolution_size=high_resolution_size,
    )
    sample_frame = frame.head(max_samples)
    figure, axes = plt.subplots(len(sample_frame), 4, figsize=(14, 3.5 * len(sample_frame)))
    axes = axes.reshape(len(sample_frame), 4)
    generator.to(device)
    generator.eval()
    for row_axes, row in zip(axes, sample_frame.itertuples(), strict=True):
        with Image.open(row.filepath) as image:
            low_resolution, high_resolution = transform(image.convert("RGB"))
        generated = generator(low_resolution.unsqueeze(0).to(device)).squeeze(0).cpu()
        bicubic = torch.nn.functional.interpolate(
            low_resolution.unsqueeze(0),
            size=(high_resolution_size, high_resolution_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        panels = (
            ("LR 32×32", low_resolution),
            ("Bicubic 128×128", bicubic),
            ("SRGAN 128×128", generated),
            ("Real HR 128×128", high_resolution),
        )
        for axis, (title, tensor) in zip(row_axes, panels, strict=True):
            display_image = denormalize_srgan(tensor).permute(1, 2, 0).numpy()
            axis.imshow(display_image)
            axis.set_title(f"{title}\n{row.class_name}")
            axis.axis("off")
    figure.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight", dpi=150)
    return figure
