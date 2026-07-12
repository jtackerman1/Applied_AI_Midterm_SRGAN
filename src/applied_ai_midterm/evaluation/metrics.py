"""Binary classifier prediction collection and metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor, nn

ClassifierBatch = tuple[Tensor, Tensor]


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Required scalar metrics for binary classifier comparison."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def as_dict(self) -> dict[str, float]:
        """Return metrics in a display- and checkpoint-friendly mapping."""
        return asdict(self)


def calculate_binary_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Calculate binary metrics from labels and sigmoid probabilities."""
    label_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if label_array.ndim != 1 or probability_array.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional.")
    if len(label_array) != len(probability_array) or not len(label_array):
        raise ValueError("labels and probabilities must have the same non-zero length.")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probabilities must be within [0, 1].")
    if np.unique(label_array).size != 2:
        raise ValueError("ROC AUC requires both binary classes to be present.")

    predictions = (probability_array >= threshold).astype(np.int64)
    return BinaryMetrics(
        accuracy=float(accuracy_score(label_array, predictions)),
        precision=float(precision_score(label_array, predictions, zero_division=0)),
        recall=float(recall_score(label_array, predictions, zero_division=0)),
        f1=float(f1_score(label_array, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(label_array, probability_array)),
    )


@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    loader: Iterable[ClassifierBatch],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, BinaryMetrics, np.ndarray, np.ndarray]:
    """Evaluate loss and metrics, applying sigmoid only to produce probabilities."""
    model.eval()
    losses: list[float] = []
    labels: list[int] = []
    probabilities: list[float] = []
    example_count = 0
    for images, batch_labels in loader:
        images = images.to(device)
        batch_labels = batch_labels.to(device=device, dtype=torch.float32).view(-1)
        logits = model(images).view(-1)
        loss = criterion(logits, batch_labels)
        losses.append(loss.item() * len(batch_labels))
        example_count += len(batch_labels)
        labels.extend(batch_labels.to(dtype=torch.int64).cpu().tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    if example_count == 0:
        raise ValueError("Cannot evaluate an empty data loader.")
    label_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    metrics = calculate_binary_metrics(label_array, probability_array)
    return sum(losses) / example_count, metrics, label_array, probability_array

