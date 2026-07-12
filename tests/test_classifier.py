"""Tests for MobileNetV2 transfer learning, metrics, and checkpointing."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from applied_ai_midterm.evaluation import calculate_binary_metrics
from applied_ai_midterm.models import build_mobilenet_v2_classifier
from applied_ai_midterm.training import (
    create_train_validation_frames,
    fit_classifier,
    load_best_classifier,
)


def test_mobilenet_has_one_logit_without_downloading_weights() -> None:
    model = build_mobilenet_v2_classifier(weights=None, freeze_features=True)

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 128, 128))

    assert logits.shape == (2, 1)
    assert model.classifier[1].out_features == 1
    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_bce_with_logits_accepts_raw_single_logits() -> None:
    criterion = nn.BCEWithLogitsLoss()
    logits = torch.tensor([-2.0, 2.0], requires_grad=True)
    labels = torch.tensor([0.0, 1.0])

    loss = criterion(logits, labels)
    loss.backward()

    assert loss.item() > 0.0
    assert logits.grad is not None


def test_required_binary_metrics() -> None:
    metrics = calculate_binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0


def test_metrics_reject_single_class_roc_auc() -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        calculate_binary_metrics([0, 0], [0.1, 0.2])


def test_validation_split_uses_only_supplied_training_rows() -> None:
    frame = pd.DataFrame(
        {
            "filepath": [f"train_{index}.png" for index in range(20)],
            "class_name": ["a"] * 10 + ["b"] * 10,
            "label": [0] * 10 + [1] * 10,
        }
    )

    train_frame, validation_frame = create_train_validation_frames(frame)

    assert len(train_frame) == 16
    assert len(validation_frame) == 4
    assert set(train_frame["filepath"]).isdisjoint(validation_frame["filepath"])
    assert set(train_frame["filepath"]) | set(validation_frame["filepath"]) == set(
        frame["filepath"]
    )
    assert validation_frame.groupby("label").size().to_dict() == {0: 2, 1: 2}


def test_fit_saves_and_loads_best_validation_checkpoint(tmp_path: Path) -> None:
    images = torch.tensor([[[[-1.0]]], [[[1.0]]], [[[-0.5]]], [[[0.5]]]])
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(1, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "model_a_best.pt"

    history = fit_classifier(
        model,
        loader,
        loader,
        optimizer,
        checkpoint_path,
        epochs=2,
        device=torch.device("cpu"),
        config={"random_seed": 42},
    )
    checkpoint = load_best_classifier(model, checkpoint_path, torch.device("cpu"))

    assert len(history) == 2
    assert checkpoint_path.is_file()
    assert checkpoint["epoch"] in {1, 2}
    assert checkpoint["validation_loss"] == min(
        record["validation_loss"] for record in history
    )
    assert checkpoint["config"] == {"random_seed": 42}
    assert "model_state" in checkpoint
    assert "optimizer_state" in checkpoint
    assert "validation_metrics" in checkpoint

