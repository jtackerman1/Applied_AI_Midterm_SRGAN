"""Tests for YAML project configuration loading."""

from pathlib import Path

import pytest

from applied_ai_midterm.config import ProjectConfig, load_config


def test_load_repository_config() -> None:
    config = load_config(Path("configs") / "config.yaml")

    assert isinstance(config, ProjectConfig)
    assert config.random_seed == 42
    assert config.train_ratio == 0.70
    assert config.test_ratio == 0.30
    assert config.low_resolution_size == 32
    assert config.high_resolution_size == 128
    assert config.classifier_batch_size == 32
    assert config.srgan_batch_size == 16
    assert config.classifier_epochs == 20
    assert config.srgan_epochs == 150
    assert config.checkpoint_interval == 5
    assert config.num_workers == 2


def test_load_config_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_rejects_missing_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "incomplete.yaml"
    config_path.write_text("random_seed: 42\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        load_config(config_path)


def test_project_config_rejects_invalid_resolution_scale() -> None:
    with pytest.raises(ValueError, match="four times"):
        ProjectConfig(
            random_seed=42,
            train_ratio=0.70,
            test_ratio=0.30,
            low_resolution_size=32,
            high_resolution_size=64,
            classifier_batch_size=32,
            srgan_batch_size=16,
            classifier_epochs=20,
            srgan_epochs=150,
            checkpoint_interval=5,
            num_workers=2,
        )

