"""Tests for Model B input handling and final comparison tables."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn

from applied_ai_midterm.data import SRGANClassifierDataset
from applied_ai_midterm.evaluation import BinaryMetrics, build_model_comparison_table
from applied_ai_midterm.generation import (
    MANIFEST_COLUMNS,
    load_generated_training_manifest,
)


class TinyGenerator(nn.Module):
    """Small 4x inference generator for dataset tests."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(inputs, scale_factor=4, mode="nearest")


def _source_frame(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for index, (class_name, label) in enumerate((("apples", 0), ("oranges", 1))):
        path = tmp_path / "source" / class_name / f"{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 40), (index * 100, 50, 150)).save(path)
        rows.append({"filepath": str(path), "class_name": class_name, "label": label})
    return pd.DataFrame(rows)


def test_manifest_loader_returns_classifier_frame_for_exact_training_sources(
    tmp_path: Path,
) -> None:
    source_frame = _source_frame(tmp_path)
    records = []
    for row in source_frame.itertuples():
        generated_path = tmp_path / "generated" / row.class_name / Path(row.filepath).name
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 128)).save(generated_path)
        records.append(
            {
                "generated_filepath": str(generated_path),
                "source_filepath": str(Path(row.filepath).resolve()),
                "class_name": row.class_name,
                "label": row.label,
                "generator_checkpoint": str(tmp_path / "generator.pt"),
            }
        )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(records, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)

    classifier_frame = load_generated_training_manifest(manifest_path, source_frame)

    assert list(classifier_frame.columns) == ["filepath", "class_name", "label"]
    assert len(classifier_frame) == len(source_frame)
    assert all(Path(path).is_file() for path in classifier_frame["filepath"])


def test_srgan_classifier_dataset_preserves_reserved_labels(tmp_path: Path) -> None:
    source_frame = _source_frame(tmp_path)
    dataset = SRGANClassifierDataset(
        source_frame, TinyGenerator(), torch.device("cpu")
    )

    tensor, label = dataset[1]

    assert tensor.shape == (3, 128, 128)
    assert label == 1
    assert len(dataset) == len(source_frame)


def test_final_comparison_table_has_required_measured_columns(tmp_path: Path) -> None:
    model_a = BinaryMetrics(accuracy=0.8, precision=0.75, recall=0.9, f1=0.82, roc_auc=0.88)
    model_b = BinaryMetrics(accuracy=0.85, precision=0.8, recall=0.95, f1=0.87, roc_auc=0.91)
    output_path = tmp_path / "comparison.csv"

    table = build_model_comparison_table(model_a, model_b, output_path=output_path)

    assert list(table.columns) == [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC AUC",
    ]
    assert table["Model"].tolist() == ["Model A", "Model B"]
    assert table["Accuracy"].tolist() == [0.8, 0.85]
    pd.testing.assert_frame_equal(table, pd.read_csv(output_path))

