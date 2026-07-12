"""Load and validate project configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml"


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Validated settings shared by the planned experiments."""

    random_seed: int
    train_ratio: float
    test_ratio: float
    low_resolution_size: int
    high_resolution_size: int
    classifier_batch_size: int
    srgan_batch_size: int
    classifier_epochs: int
    srgan_epochs: int
    checkpoint_interval: int
    num_workers: int

    def __post_init__(self) -> None:
        if abs(self.train_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("train_ratio and test_ratio must sum to 1.0.")
        if not 0.0 < self.train_ratio < 1.0 or not 0.0 < self.test_ratio < 1.0:
            raise ValueError("train_ratio and test_ratio must each be between 0 and 1.")
        if self.high_resolution_size != self.low_resolution_size * 4:
            raise ValueError("high_resolution_size must be four times low_resolution_size.")

        positive_fields = (
            "classifier_batch_size",
            "srgan_batch_size",
            "classifier_epochs",
            "srgan_epochs",
            "checkpoint_interval",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Read a YAML file and return validated project settings."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config: Any = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    expected_keys = {field.name for field in fields(ProjectConfig)}
    supplied_keys = set(raw_config)
    missing_keys = expected_keys - supplied_keys
    unknown_keys = supplied_keys - expected_keys
    if missing_keys:
        raise ValueError(f"Configuration is missing required keys: {sorted(missing_keys)}")
    if unknown_keys:
        raise ValueError(f"Configuration contains unknown keys: {sorted(unknown_keys)}")

    try:
        return ProjectConfig(**raw_config)
    except TypeError as error:
        raise ValueError(f"Configuration has invalid value types: {error}") from error

