"""Generate class-preserving Model B training images and a source manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.transforms.functional import to_pil_image

from applied_ai_midterm.transforms import SRGANPairTransform, denormalize_srgan

MANIFEST_COLUMNS = [
    "generated_filepath",
    "source_filepath",
    "class_name",
    "label",
    "generator_checkpoint",
]
REQUIRED_TRAIN_COLUMNS = {"filepath", "class_name", "label"}


def _validate_training_frame(training_frame: pd.DataFrame) -> None:
    missing_columns = REQUIRED_TRAIN_COLUMNS - set(training_frame.columns)
    if missing_columns:
        raise ValueError(f"Training frame is missing columns: {sorted(missing_columns)}")
    if training_frame.empty:
        raise ValueError("Training frame cannot be empty.")
    if training_frame["filepath"].duplicated().any():
        raise ValueError("Training frame contains duplicate source filepaths.")
    for class_name in training_frame["class_name"].unique():
        if Path(str(class_name)).name != str(class_name):
            raise ValueError(f"Invalid class name for output directory: {class_name!r}")


def load_generated_training_manifest(
    manifest_path: Path, expected_training_frame: pd.DataFrame
) -> pd.DataFrame:
    """Validate a one-to-one generated manifest and return a classifier-ready frame."""
    _validate_training_frame(expected_training_frame)
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Generated training manifest not found: {manifest_path.resolve()}")
    manifest = pd.read_csv(manifest_path)
    if list(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError(
            f"Invalid generated manifest columns: expected {MANIFEST_COLUMNS}, "
            f"found {list(manifest.columns)}."
        )
    if manifest["source_filepath"].duplicated().any():
        raise ValueError("Generated manifest contains duplicate source mappings.")
    expected = expected_training_frame.copy()
    expected["source_filepath"] = expected["filepath"].map(lambda value: str(Path(value).resolve()))
    expected_records = {
        (row.source_filepath, str(row.class_name), int(row.label))
        for row in expected.itertuples()
    }
    manifest_records = {
        (str(Path(row.source_filepath).resolve()), str(row.class_name), int(row.label))
        for row in manifest.itertuples()
    }
    if manifest_records != expected_records or len(manifest) != len(expected):
        raise ValueError(
            "Generated manifest must map exactly every persisted training source with "
            "matching class and label."
        )
    missing_generated = [
        path for path in manifest["generated_filepath"] if not Path(path).is_file()
    ]
    if missing_generated:
        raise FileNotFoundError(
            f"Generated manifest references missing image: {missing_generated[0]}"
        )
    classifier_frame = manifest[["generated_filepath", "class_name", "label"]].rename(
        columns={"generated_filepath": "filepath"}
    )
    return classifier_frame.sort_values("filepath").reset_index(drop=True)


@torch.inference_mode()
def generate_training_images(
    generator: nn.Module,
    training_frame: pd.DataFrame,
    output_dir: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    *,
    low_resolution_size: int = 32,
    high_resolution_size: int = 128,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Generate images from training rows only and persist a complete source manifest."""
    _validate_training_frame(training_frame)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    checkpoint_path = Path(checkpoint_path)
    transform = SRGANPairTransform(
        training=False,
        low_resolution_size=low_resolution_size,
        high_resolution_size=high_resolution_size,
    )
    generator.to(device)
    generator.eval()
    records: list[dict[str, str | int]] = []
    ordered_frame = training_frame.sort_values("filepath").reset_index(drop=True)
    for index, row in ordered_frame.iterrows():
        source_path = Path(row["filepath"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Training source image not found: {source_path}")
        class_directory = output_dir / str(row["class_name"])
        class_directory.mkdir(parents=True, exist_ok=True)
        generated_path = class_directory / f"{index:06d}_{source_path.stem}.png"
        if generated_path.exists() and not overwrite:
            raise FileExistsError(
                f"Generated image already exists: {generated_path}. "
                "Use overwrite=True to replace it."
            )
        with Image.open(source_path) as image:
            low_resolution, _ = transform(image.convert("RGB"))
        generated = generator(low_resolution.unsqueeze(0).to(device)).squeeze(0).cpu()
        to_pil_image(denormalize_srgan(generated)).save(generated_path)
        records.append(
            {
                "generated_filepath": str(generated_path.resolve()),
                "source_filepath": str(source_path.resolve()),
                "class_name": str(row["class_name"]),
                "label": int(row["label"]),
                "generator_checkpoint": str(checkpoint_path.resolve()),
            }
        )
    manifest = pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)
    if len(manifest) != len(training_frame):
        raise RuntimeError("Generated manifest does not map every training source image.")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest
