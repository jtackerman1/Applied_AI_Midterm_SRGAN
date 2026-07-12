"""Dataset discovery and persistent stratified split preparation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import Dataset

VALID_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
SPLIT_COLUMNS = ["filepath", "class_name", "label"]


@dataclass(frozen=True, slots=True)
class PreparedSplits:
    """Persisted training and testing split tables."""

    train: pd.DataFrame
    test: pd.DataFrame


def resolve_class_root(raw_data_dir: Path) -> Path:
    """Resolve either ``data/raw/<classes>`` or ``data/raw/train/<classes>``."""
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {raw_data_dir.resolve()}. "
            "Create it with exactly two class folders."
        )

    train_directory = raw_data_dir / "train"
    if train_directory.is_dir():
        # Some downloaded datasets include vendor-defined validation/test siblings.
        # The project deliberately ignores them and creates its own single 70/30 split.
        return train_directory
    return raw_data_dir


def discover_images(raw_data_dir: Path) -> pd.DataFrame:
    """Discover supported images recursively beneath exactly two class folders."""
    class_root = resolve_class_root(raw_data_dir)
    class_directories = sorted(path for path in class_root.iterdir() if path.is_dir())
    if len(class_directories) != 2:
        names = [path.name for path in class_directories]
        raise ValueError(
            f"Expected exactly two class folders under {class_root.resolve()}, "
            f"found {len(class_directories)}: {names}."
        )

    records: list[dict[str, str | int]] = []
    for label, class_directory in enumerate(class_directories):
        image_paths = sorted(
            path.resolve()
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(
                f"Class folder contains no JPG, JPEG, PNG, or WEBP images: "
                f"{class_directory.resolve()}"
            )
        records.extend(
            {
                "filepath": str(image_path),
                "class_name": class_directory.name,
                "label": label,
            }
            for image_path in image_paths
        )
    return pd.DataFrame.from_records(records, columns=SPLIT_COLUMNS)


def _validate_split_frame(frame: pd.DataFrame, csv_path: Path) -> None:
    if list(frame.columns) != SPLIT_COLUMNS:
        raise ValueError(
            f"Invalid split CSV columns in {csv_path}: expected {SPLIT_COLUMNS}, "
            f"found {list(frame.columns)}."
        )
    if frame.empty:
        raise ValueError(f"Split CSV is empty: {csv_path}")
    if frame["class_name"].nunique() != 2 or frame["label"].nunique() != 2:
        raise ValueError(f"Split CSV must contain exactly two classes and labels: {csv_path}")
    missing_paths = [path for path in frame["filepath"] if not Path(path).is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"Split CSV {csv_path} references missing image: {missing_paths[0]}"
        )


def load_splits(splits_dir: Path) -> PreparedSplits:
    """Load and validate the previously persisted train/test split."""
    splits_dir = Path(splits_dir)
    train_path = splits_dir / "train.csv"
    test_path = splits_dir / "test.csv"
    if train_path.exists() != test_path.exists():
        raise FileNotFoundError(
            f"Incomplete persisted split in {splits_dir.resolve()}; both train.csv and "
            "test.csv are required."
        )
    if not train_path.is_file():
        raise FileNotFoundError(f"Persisted split files not found in: {splits_dir.resolve()}")

    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    _validate_split_frame(train_frame, train_path)
    _validate_split_frame(test_frame, test_path)
    overlap = set(train_frame["filepath"]) & set(test_frame["filepath"])
    if overlap:
        raise ValueError(f"Persisted train/test splits overlap at: {next(iter(overlap))}")
    return PreparedSplits(train=train_frame, test=test_frame)


def prepare_splits(
    raw_data_dir: Path,
    splits_dir: Path,
    *,
    train_ratio: float = 0.70,
    random_seed: int = 42,
) -> PreparedSplits:
    """Create the split once, or reuse the existing split without reshuffling."""
    splits_dir = Path(splits_dir)
    train_path = splits_dir / "train.csv"
    test_path = splits_dir / "test.csv"
    if train_path.exists() or test_path.exists():
        return load_splits(splits_dir)
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    discovered = discover_images(raw_data_dir)
    try:
        train_frame, test_frame = train_test_split(
            discovered,
            train_size=train_ratio,
            random_state=random_seed,
            stratify=discovered["label"],
        )
    except ValueError as error:
        raise ValueError(
            "Unable to create a stratified split. Each class needs enough images for both "
            f"training and testing. Original error: {error}"
        ) from error

    train_frame = train_frame.sort_values("filepath").reset_index(drop=True)
    test_frame = test_frame.sort_values("filepath").reset_index(drop=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(train_path, index=False, columns=SPLIT_COLUMNS)
    test_frame.to_csv(test_path, index=False, columns=SPLIT_COLUMNS)
    return PreparedSplits(train=train_frame, test=test_frame)


class ImagePathDataset(Dataset[tuple[Tensor, int]]):
    """Load RGB images from a split table and apply a tensor transform."""

    def __init__(self, frame: pd.DataFrame, transform: object) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(row["filepath"]) as image:
            tensor = self.transform(image.convert("RGB"))  # type: ignore[operator]
        return tensor, int(row["label"])


class SRGANPathDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load RGB images and return corresponding low/high-resolution tensor pairs."""

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Callable[[Image.Image], tuple[Tensor, Tensor]],
    ) -> None:
        if frame.empty:
            raise ValueError("SRGAN dataset frame cannot be empty.")
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        with Image.open(self.frame.iloc[index]["filepath"]) as image:
            return self.transform(image.convert("RGB"))


class SRGANClassifierDataset(Dataset[tuple[Tensor, int]]):
    """Generate classifier inputs on-the-fly from reserved source examples."""

    def __init__(
        self,
        frame: pd.DataFrame,
        generator: nn.Module,
        device: torch.device,
        *,
        low_resolution_size: int = 32,
        high_resolution_size: int = 128,
    ) -> None:
        if frame.empty:
            raise ValueError("SRGAN classifier dataset frame cannot be empty.")
        self.frame = frame.reset_index(drop=True).copy()
        self.generator = generator.to(device).eval()
        self.device = device
        from applied_ai_midterm.transforms import SRGANPairTransform

        self.pair_transform = SRGANPairTransform(
            training=False,
            low_resolution_size=low_resolution_size,
            high_resolution_size=high_resolution_size,
        )

    def __len__(self) -> int:
        return len(self.frame)

    @torch.inference_mode()
    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        from applied_ai_midterm.transforms import denormalize_srgan, normalize_imagenet_tensor

        row = self.frame.iloc[index]
        with Image.open(row["filepath"]) as image:
            low_resolution, _ = self.pair_transform(image.convert("RGB"))
        generated = self.generator(low_resolution.unsqueeze(0).to(self.device)).squeeze(0)
        classifier_tensor = normalize_imagenet_tensor(denormalize_srgan(generated).cpu())
        return classifier_tensor, int(row["label"])
