"""Tests for discovery, splitting, and image transformations."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

from applied_ai_midterm.data import SPLIT_COLUMNS, discover_images, prepare_splits
from applied_ai_midterm.transforms import (
    SRGANPairTransform,
    classifier_transform,
    denormalize_imagenet,
    denormalize_srgan,
)


def _create_dataset(root: Path, *, nested_train: bool = False) -> Path:
    raw_directory = root / "raw"
    class_root = raw_directory / "train" if nested_train else raw_directory
    extensions = ("jpg", "jpeg", "png", "webp", "JPG")
    for class_index, class_name in enumerate(("apples", "oranges")):
        nested_directory = class_root / class_name / "nested"
        nested_directory.mkdir(parents=True)
        for image_index in range(10):
            extension = extensions[image_index % len(extensions)]
            color = (class_index * 180, image_index * 10, 80)
            Image.new("RGB", (48 + image_index, 52), color).save(
                nested_directory / f"image_{image_index}.{extension}"
            )
        (nested_directory / "ignored.txt").write_text("not an image", encoding="utf-8")
    return raw_directory


@pytest.mark.parametrize("nested_train", [False, True])
def test_discover_images_supports_both_layouts(tmp_path: Path, nested_train: bool) -> None:
    raw_directory = _create_dataset(tmp_path, nested_train=nested_train)

    discovered = discover_images(raw_directory)

    assert list(discovered.columns) == SPLIT_COLUMNS
    assert len(discovered) == 20
    assert discovered.groupby("class_name").size().to_dict() == {"apples": 10, "oranges": 10}
    assert discovered.groupby("class_name")["label"].first().to_dict() == {
        "apples": 0,
        "oranges": 1,
    }


def test_discovery_rejects_non_binary_dataset(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    for class_name in ("one", "two", "three"):
        class_directory = raw_directory / class_name
        class_directory.mkdir(parents=True)
        Image.new("RGB", (16, 16)).save(class_directory / "sample.png")

    with pytest.raises(ValueError, match="exactly two class folders"):
        discover_images(raw_directory)


def test_split_is_stratified_persisted_and_reused(tmp_path: Path) -> None:
    raw_directory = _create_dataset(tmp_path)
    splits_directory = tmp_path / "splits"

    first = prepare_splits(raw_directory, splits_directory)
    original_train_csv = (splits_directory / "train.csv").read_text(encoding="utf-8")
    original_test_csv = (splits_directory / "test.csv").read_text(encoding="utf-8")
    second = prepare_splits(raw_directory, splits_directory)

    assert len(first.train) == 14
    assert len(first.test) == 6
    assert first.train.groupby("label").size().to_dict() == {0: 7, 1: 7}
    assert first.test.groupby("label").size().to_dict() == {0: 3, 1: 3}
    assert set(first.train["filepath"]).isdisjoint(first.test["filepath"])
    assert list(pd.read_csv(splits_directory / "train.csv").columns) == SPLIT_COLUMNS
    assert (splits_directory / "train.csv").read_text(encoding="utf-8") == original_train_csv
    assert (splits_directory / "test.csv").read_text(encoding="utf-8") == original_test_csv
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.test, second.test)


def test_classifier_transforms_have_required_shape_and_inverse() -> None:
    image = Image.new("RGB", (180, 140), (120, 80, 200))
    evaluation_transform = classifier_transform(training=False)
    training_transform = classifier_transform(training=True)

    evaluation_tensor = evaluation_transform(image)
    training_tensor = training_transform(image)
    displayed = denormalize_imagenet(evaluation_tensor)

    assert evaluation_tensor.shape == (3, 128, 128)
    assert training_tensor.shape == (3, 128, 128)
    assert 0.0 <= displayed.min() <= displayed.max() <= 1.0
    assert not any(
        isinstance(operation, transforms.RandomHorizontalFlip)
        for operation in evaluation_transform.transforms
    )
    assert any(
        isinstance(operation, transforms.RandomHorizontalFlip)
        for operation in training_transform.transforms
    )


def test_srgan_pair_shapes_ranges_and_inverse() -> None:
    image = Image.new("RGB", (180, 140), (25, 100, 220))
    low_resolution, high_resolution = SRGANPairTransform(training=False)(image)

    assert low_resolution.shape == (3, 32, 32)
    assert high_resolution.shape == (3, 128, 128)
    assert torch.all(low_resolution >= -1.0) and torch.all(low_resolution <= 1.0)
    assert torch.all(high_resolution >= -1.0) and torch.all(high_resolution <= 1.0)
    displayed = denormalize_srgan(high_resolution)
    assert 0.0 <= displayed.min() <= displayed.max() <= 1.0
