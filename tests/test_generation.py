"""Tests for best-generator selection and training-only image generation."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn

from applied_ai_midterm.generation import MANIFEST_COLUMNS, generate_training_images
from applied_ai_midterm.training import (
    best_srgan_generator_checkpoint,
    load_best_generator,
)
from applied_ai_midterm.visualization import plot_srgan_comparisons


class TinyGenerator(nn.Module):
    """Small 4x generator for inference tests."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(3, 3, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        upscaled = torch.nn.functional.interpolate(inputs, scale_factor=4, mode="nearest")
        return torch.tanh(self.convolution(upscaled))


def _save_generator_checkpoint(
    path: Path, generator: nn.Module, *, epoch: int, loss: float
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "generator_state": generator.state_dict(),
            "generator_selection_loss": loss,
            "history": {"generator_total": [loss]},
        },
        path,
    )


def _training_frame(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for index, (class_name, label) in enumerate((("apples", 0), ("oranges", 1))):
        source_path = tmp_path / "sources" / class_name / f"source_{index}.png"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 45), (index * 150, 60, 100)).save(source_path)
        rows.append(
            {"filepath": str(source_path), "class_name": class_name, "label": label}
        )
    return pd.DataFrame(rows)


def test_best_generator_uses_lowest_loss_and_prefers_later_tie(tmp_path: Path) -> None:
    generator = TinyGenerator()
    _save_generator_checkpoint(tmp_path / "srgan_epoch_005.pt", generator, epoch=5, loss=0.4)
    _save_generator_checkpoint(tmp_path / "srgan_epoch_010.pt", generator, epoch=10, loss=0.2)
    _save_generator_checkpoint(tmp_path / "srgan_epoch_015.pt", generator, epoch=15, loss=0.2)

    selected = best_srgan_generator_checkpoint(tmp_path)
    loaded_path, checkpoint = load_best_generator(
        generator, tmp_path, torch.device("cpu")
    )

    assert selected == tmp_path / "srgan_epoch_015.pt"
    assert loaded_path == selected
    assert checkpoint["epoch"] == 15
    assert not generator.training


def test_generation_preserves_classes_and_complete_source_manifest(tmp_path: Path) -> None:
    frame = _training_frame(tmp_path)
    generator = TinyGenerator().eval()
    checkpoint_path = tmp_path / "srgan_epoch_150.pt"
    _save_generator_checkpoint(checkpoint_path, generator, epoch=150, loss=0.1)
    output_dir = tmp_path / "generated"
    manifest_path = output_dir / "manifest.csv"

    manifest = generate_training_images(
        generator,
        frame,
        output_dir,
        manifest_path,
        checkpoint_path,
        torch.device("cpu"),
    )

    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert len(manifest) == len(frame)
    assert set(manifest["source_filepath"]) == {
        str(Path(path).resolve()) for path in frame["filepath"]
    }
    assert set(manifest["class_name"]) == {"apples", "oranges"}
    assert all(Path(path).is_file() for path in manifest["generated_filepath"])
    assert all(
        Path(row.generated_filepath).parent.name == row.class_name
        for row in manifest.itertuples()
    )
    pd.testing.assert_frame_equal(manifest, pd.read_csv(manifest_path))


def test_comparison_plot_contains_lr_bicubic_srgan_and_real_hr(tmp_path: Path) -> None:
    frame = _training_frame(tmp_path)
    output_path = tmp_path / "comparison.png"

    figure = plot_srgan_comparisons(
        TinyGenerator().eval(),
        frame,
        torch.device("cpu"),
        output_path=output_path,
        max_samples=2,
    )
    titles = [axis.get_title() for axis in figure.axes]

    assert output_path.is_file()
    assert len(figure.axes) == 8
    assert any("LR 32×32" in title for title in titles)
    assert any("Bicubic 128×128" in title for title in titles)
    assert any("SRGAN 128×128" in title for title in titles)
    assert any("Real HR 128×128" in title for title in titles)

