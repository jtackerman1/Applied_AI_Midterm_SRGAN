"""Evaluation metrics and reporting utilities."""

from applied_ai_midterm.evaluation.comparison import build_model_comparison_table
from applied_ai_midterm.evaluation.metrics import (
    BinaryMetrics,
    calculate_binary_metrics,
    evaluate_classifier,
)

__all__ = [
    "BinaryMetrics",
    "build_model_comparison_table",
    "calculate_binary_metrics",
    "evaluate_classifier",
]

