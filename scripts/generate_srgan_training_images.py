"""Generate class-preserving Model B images from persisted training rows only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from applied_ai_midterm.config import load_config  # noqa: E402
from applied_ai_midterm.data import load_splits  # noqa: E402
from applied_ai_midterm.generation import generate_training_images  # noqa: E402
from applied_ai_midterm.models import Generator  # noqa: E402
from applied_ai_midterm.training import load_best_generator, select_device  # noqa: E402
from applied_ai_midterm.visualization import plot_srgan_comparisons  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse generation paths and overwrite behavior."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/srgan"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/generated/manifest.csv")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load the best generator and create training-only generated images."""
    args = parse_args()
    config = load_config(args.config)
    splits = load_splits(args.splits_dir)
    device = select_device()
    generator = Generator()
    checkpoint_path, checkpoint = load_best_generator(
        generator, args.checkpoint_dir, device
    )
    manifest = generate_training_images(
        generator,
        splits.train,
        args.output_dir,
        args.manifest,
        checkpoint_path,
        device,
        low_resolution_size=config.low_resolution_size,
        high_resolution_size=config.high_resolution_size,
        overwrite=args.overwrite,
    )
    plot_srgan_comparisons(
        generator,
        splits.train,
        device,
        output_path=Path("artifacts/srgan_generation_comparison.png"),
        low_resolution_size=config.low_resolution_size,
        high_resolution_size=config.high_resolution_size,
    )
    print(f"Selected generator epoch: {checkpoint['epoch']} ({checkpoint_path})")
    print(f"Generated {len(manifest)} training images; manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
