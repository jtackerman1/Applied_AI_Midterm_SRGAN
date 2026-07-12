"""Training workflow and best-validation checkpointing for binary classifiers."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.optim import Optimizer

from applied_ai_midterm.evaluation import evaluate_classifier

ClassifierBatch = tuple[Tensor, Tensor]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_train_validation_frames(
    training_frame: pd.DataFrame,
    *,
    validation_ratio: float = 0.20,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify persisted training rows into training and validation subsets."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if "label" not in training_frame.columns:
        raise ValueError("training_frame must contain a label column.")
    try:
        train_frame, validation_frame = train_test_split(
            training_frame,
            test_size=validation_ratio,
            random_state=random_seed,
            stratify=training_frame["label"],
        )
    except ValueError as error:
        raise ValueError(
            "Unable to create a stratified validation split from training rows. "
            f"Original error: {error}"
        ) from error
    return (
        train_frame.sort_values("filepath").reset_index(drop=True),
        validation_frame.sort_values("filepath").reset_index(drop=True),
    )


def train_classifier_epoch(
    model: nn.Module,
    loader: Iterable[ClassifierBatch],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch using raw logits with BCEWithLogitsLoss."""
    model.train()
    total_loss = 0.0
    example_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.float32).view(-1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        example_count += len(labels)
    if example_count == 0:
        raise ValueError("Cannot train with an empty data loader.")
    return total_loss / example_count


def fit_classifier(
    model: nn.Module,
    train_loader: Iterable[ClassifierBatch],
    validation_loader: Iterable[ClassifierBatch],
    optimizer: Optimizer,
    checkpoint_path: Path,
    *,
    epochs: int,
    device: torch.device,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Train with BCE logits loss and checkpoint the lowest validation loss."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    criterion = nn.BCEWithLogitsLoss()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    for epoch_index in range(epochs):
        training_loss = train_classifier_epoch(
            model, train_loader, optimizer, criterion, device
        )
        validation_loss, validation_metrics, _, _ = evaluate_classifier(
            model, validation_loader, criterion, device
        )
        epoch_record: dict[str, Any] = {
            "epoch": epoch_index + 1,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics.as_dict(),
        }
        history.append(epoch_record)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "epoch": epoch_index + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_metrics": validation_metrics.as_dict(),
                    "history": history.copy(),
                    "config": config or {},
                },
                checkpoint_path,
            )
    return history


def load_best_classifier(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> dict[str, Any]:
    """Load the saved best-validation model state onto the selected device."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path.resolve()}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return checkpoint
