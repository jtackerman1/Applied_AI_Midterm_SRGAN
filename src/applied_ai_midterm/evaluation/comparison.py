"""Final measured comparison table for Models A and B."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from applied_ai_midterm.evaluation.metrics import BinaryMetrics


def build_model_comparison_table(
    model_a_metrics: BinaryMetrics,
    model_b_metrics: BinaryMetrics,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build and optionally save the required measured Model A/B table."""
    table = pd.DataFrame(
        [
            {"Model": "Model A", **_display_metrics(model_a_metrics)},
            {"Model": "Model B", **_display_metrics(model_b_metrics)},
        ],
        columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"],
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
    return table


def _display_metrics(metrics: BinaryMetrics) -> dict[str, float]:
    return {
        "Accuracy": metrics.accuracy,
        "Precision": metrics.precision,
        "Recall": metrics.recall,
        "F1": metrics.f1,
        "ROC AUC": metrics.roc_auc,
    }
