"""Create or reuse the project's persistent stratified dataset split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python scripts/prepare_data.py`` from a fresh, non-installed checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from applied_ai_midterm.config import load_config  # noqa: E402
from applied_ai_midterm.data import prepare_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line paths for dataset preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    return parser.parse_args()


def main() -> None:
    """Prepare the split and print a compact class summary."""
    args = parse_args()
    config = load_config(args.config)
    splits = prepare_splits(
        args.raw_data_dir,
        args.splits_dir,
        train_ratio=config.train_ratio,
        random_seed=config.random_seed,
    )
    print(f"Training examples: {len(splits.train)}")
    print(splits.train.groupby(["class_name", "label"]).size().to_string())
    print(f"Testing examples: {len(splits.test)}")
    print(splits.test.groupby(["class_name", "label"]).size().to_string())
    print(f"Splits saved in: {args.splits_dir.resolve()}")


if __name__ == "__main__":
    main()
