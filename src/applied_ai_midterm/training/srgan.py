"""SRGAN optimization, complete checkpointing, and latest-checkpoint resume."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.utils import save_image

from applied_ai_midterm.models import GeneratorLoss

SRGANBatch = tuple[Tensor, Tensor]
History = dict[str, list[float]]
CHECKPOINT_PATTERN = re.compile(r"srgan_epoch_(\d+)\.pt$")
PRETRAIN_CHECKPOINT_PATTERN = re.compile(r"generator_pretrain_epoch_(\d+)\.pt$")


def empty_srgan_history() -> History:
    """Create the loss history fields stored in every checkpoint."""
    return {
        "generator_total": [],
        "generator_pixel": [],
        "generator_adversarial": [],
        "generator_perceptual": [],
        "discriminator": [],
    }


@torch.inference_mode()
def save_srgan_samples(
    generator: nn.Module,
    low_resolution: Tensor,
    high_resolution: Tensor,
    output_path: Path,
    device: torch.device,
    *,
    max_images: int = 4,
) -> None:
    """Save rows of bicubic, generated, and real images for visual monitoring."""
    if max_images <= 0:
        raise ValueError("max_images must be greater than zero.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    was_training = generator.training
    generator.eval()
    low_resolution = low_resolution[:max_images].to(device)
    high_resolution = high_resolution[:max_images].to(device)
    generated = generator(low_resolution)
    bicubic = torch.nn.functional.interpolate(
        low_resolution,
        size=high_resolution.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )
    comparison_rows = torch.cat((bicubic, generated, high_resolution), dim=3)
    comparison_rows = comparison_rows.add(1.0).div(2.0).clamp(0.0, 1.0).cpu()
    save_image(comparison_rows, output_path, nrow=1)
    generator.train(was_training)


def save_generator_pretrain_checkpoint(
    path: Path,
    *,
    epoch: int,
    generator: nn.Module,
    optimizer: Optimizer,
    history: list[float],
    config: dict[str, Any],
    scheduler: LRScheduler | None = None,
) -> None:
    """Save resumable generator reconstruction-pretraining state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator_state": generator.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "reconstruction_history": history,
            "random_seed": config.get("random_seed"),
            "config": config,
        },
        path,
    )


def latest_generator_pretrain_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the generator pretraining checkpoint with the greatest epoch."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("generator_pretrain_epoch_*.pt"):
        match = PRETRAIN_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def resume_generator_pretraining(
    checkpoint_dir: Path,
    *,
    generator: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    scheduler: LRScheduler | None = None,
) -> tuple[int, list[float], Path | None]:
    """Restore the latest generator pretraining state, or return fresh state."""
    latest_path = latest_generator_pretrain_checkpoint(checkpoint_dir)
    if latest_path is None:
        return 0, [], None
    checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    generator.to(device)
    return int(checkpoint["epoch"]), checkpoint["reconstruction_history"], latest_path


def pretrain_generator(
    generator: nn.Module,
    loader: Iterable[SRGANBatch],
    optimizer: Optimizer,
    checkpoint_dir: Path,
    *,
    total_epochs: int,
    start_epoch: int,
    device: torch.device,
    config: dict[str, Any],
    history: list[float] | None = None,
    checkpoint_interval: int = 5,
    scheduler: LRScheduler | None = None,
    sample_tensors: tuple[Tensor, Tensor] | None = None,
    sample_dir: Path | None = None,
    sample_interval: int = 5,
) -> list[float]:
    """Pretrain the generator using only L1 reconstruction loss."""
    if total_epochs <= 0:
        raise ValueError("Generator pretraining total_epochs must be greater than zero.")
    if checkpoint_interval <= 0 or sample_interval <= 0:
        raise ValueError("Checkpoint and sample intervals must be greater than zero.")
    if not 0 <= start_epoch <= total_epochs:
        raise ValueError("start_epoch must be between zero and total_epochs.")
    history = history if history is not None else []
    generator.to(device)
    reconstruction_criterion = nn.L1Loss()
    for epoch_index in range(start_epoch, total_epochs):
        generator.train()
        total_loss = 0.0
        batch_count = 0
        for low_resolution, high_resolution in loader:
            low_resolution = low_resolution.to(device)
            high_resolution = high_resolution.to(device)
            optimizer.zero_grad(set_to_none=True)
            generated = generator(low_resolution)
            loss = reconstruction_criterion(generated, high_resolution)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        if batch_count == 0:
            raise ValueError("Cannot pretrain the generator with an empty data loader.")
        history.append(total_loss / batch_count)
        if scheduler is not None:
            scheduler.step()
        completed_epoch = epoch_index + 1
        if completed_epoch % checkpoint_interval == 0:
            save_generator_pretrain_checkpoint(
                Path(checkpoint_dir) / f"generator_pretrain_epoch_{completed_epoch:03d}.pt",
                epoch=completed_epoch,
                generator=generator,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                config=config,
            )
        if (
            sample_tensors is not None
            and sample_dir is not None
            and completed_epoch % sample_interval == 0
        ):
            save_srgan_samples(
                generator,
                sample_tensors[0],
                sample_tensors[1],
                Path(sample_dir) / f"pretrain_epoch_{completed_epoch:03d}.png",
                device,
            )
    return history


def save_srgan_checkpoint(
    path: Path,
    *,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    history: History,
    config: dict[str, Any],
    generator_scheduler: LRScheduler | None = None,
    discriminator_scheduler: LRScheduler | None = None,
) -> None:
    """Save all model, optimizer, scheduler, history, seed, and configuration state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "generator_optimizer_state": generator_optimizer.state_dict(),
            "discriminator_optimizer_state": discriminator_optimizer.state_dict(),
            "generator_scheduler_state": (
                generator_scheduler.state_dict() if generator_scheduler is not None else None
            ),
            "discriminator_scheduler_state": (
                discriminator_scheduler.state_dict()
                if discriminator_scheduler is not None
                else None
            ),
            "history": history,
            "generator_selection_loss": (
                history["generator_total"][-1] if history["generator_total"] else None
            ),
            "random_seed": config.get("random_seed"),
            "config": config,
        },
        path,
    )


def latest_srgan_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the checkpoint with the greatest encoded epoch, if one exists."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("srgan_epoch_*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def best_srgan_generator_checkpoint(checkpoint_dir: Path) -> Path:
    """Select the checkpoint with the lowest recorded total generator loss."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"SRGAN checkpoint directory not found: {checkpoint_dir.resolve()}")
    candidates: list[tuple[float, int, Path]] = []
    for path in checkpoint_dir.glob("srgan_epoch_*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        selection_loss = checkpoint.get("generator_selection_loss")
        if selection_loss is None:
            losses = checkpoint.get("history", {}).get("generator_total", [])
            selection_loss = losses[-1] if losses else None
        if selection_loss is not None:
            candidates.append((float(selection_loss), -int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(
            f"No SRGAN checkpoints with generator loss history found in: "
            f"{checkpoint_dir.resolve()}"
        )
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def load_best_generator(
    generator: nn.Module, checkpoint_dir: Path, device: torch.device
) -> tuple[Path, dict[str, Any]]:
    """Load the best recorded adversarial generator state for inference."""
    checkpoint_path = best_srgan_generator_checkpoint(checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state"])
    generator.to(device)
    generator.eval()
    return checkpoint_path, checkpoint


def load_srgan_checkpoint(
    path: Path,
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    device: torch.device,
    generator_scheduler: LRScheduler | None = None,
    discriminator_scheduler: LRScheduler | None = None,
) -> dict[str, Any]:
    """Restore every available state from an SRGAN checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SRGAN checkpoint not found: {path.resolve()}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state"])
    discriminator.load_state_dict(checkpoint["discriminator_state"])
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state"])
    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state"])
    generator_state = checkpoint.get("generator_scheduler_state")
    discriminator_state = checkpoint.get("discriminator_scheduler_state")
    if generator_scheduler is not None and generator_state is not None:
        generator_scheduler.load_state_dict(generator_state)
    if discriminator_scheduler is not None and discriminator_state is not None:
        discriminator_scheduler.load_state_dict(discriminator_state)
    generator.to(device)
    discriminator.to(device)
    return checkpoint


def resume_latest_srgan(
    checkpoint_dir: Path,
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    device: torch.device,
    generator_scheduler: LRScheduler | None = None,
    discriminator_scheduler: LRScheduler | None = None,
) -> tuple[int, History, Path | None]:
    """Resume the latest checkpoint or return a fresh epoch/history state."""
    latest_path = latest_srgan_checkpoint(checkpoint_dir)
    if latest_path is None:
        return 0, empty_srgan_history(), None
    checkpoint = load_srgan_checkpoint(
        latest_path,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        device=device,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
    )
    return int(checkpoint["epoch"]), checkpoint["history"], latest_path


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def train_srgan(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: Iterable[SRGANBatch],
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    generator_loss: GeneratorLoss,
    checkpoint_dir: Path,
    *,
    total_epochs: int,
    start_epoch: int,
    device: torch.device,
    config: dict[str, Any],
    history: History | None = None,
    checkpoint_interval: int = 5,
    generator_scheduler: LRScheduler | None = None,
    discriminator_scheduler: LRScheduler | None = None,
    sample_tensors: tuple[Tensor, Tensor] | None = None,
    sample_dir: Path | None = None,
    sample_interval: int = 5,
) -> History:
    """Train SRGAN to at least 150 total epochs and checkpoint every five epochs."""
    if total_epochs < 150:
        raise ValueError("SRGAN total_epochs must be at least 150.")
    if checkpoint_interval != 5:
        raise ValueError("SRGAN checkpoint_interval must be exactly 5 epochs.")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be greater than zero.")
    if not 0 <= start_epoch <= total_epochs:
        raise ValueError("start_epoch must be between zero and total_epochs.")
    history = history or empty_srgan_history()
    generator.to(device)
    discriminator.to(device)
    generator_loss.to(device)
    discriminator_criterion = nn.BCEWithLogitsLoss()

    for epoch_index in range(start_epoch, total_epochs):
        generator.train()
        discriminator.train()
        totals = {key: 0.0 for key in history}
        batch_count = 0
        for low_resolution, high_resolution in loader:
            low_resolution = low_resolution.to(device)
            high_resolution = high_resolution.to(device)

            _set_requires_grad(discriminator, True)
            discriminator.train()
            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                detached_generated = generator(low_resolution)
            real_logits = discriminator(high_resolution)
            generated_logits = discriminator(detached_generated)
            discriminator_value = 0.5 * (
                discriminator_criterion(real_logits, torch.ones_like(real_logits))
                + discriminator_criterion(
                    generated_logits, torch.zeros_like(generated_logits)
                )
            )
            discriminator_value.backward()
            discriminator_optimizer.step()

            _set_requires_grad(discriminator, False)
            discriminator.eval()
            generator_optimizer.zero_grad(set_to_none=True)
            generated = generator(low_resolution)
            loss_values = generator_loss(
                generated, high_resolution, discriminator(generated)
            )
            loss_values.total.backward()
            generator_optimizer.step()

            totals["generator_total"] += loss_values.total.item()
            totals["generator_pixel"] += loss_values.pixel.item()
            totals["generator_adversarial"] += loss_values.adversarial.item()
            totals["generator_perceptual"] += loss_values.perceptual.item()
            totals["discriminator"] += discriminator_value.item()
            batch_count += 1
        if batch_count == 0:
            raise ValueError("Cannot train SRGAN with an empty data loader.")
        for key in history:
            history[key].append(totals[key] / batch_count)
        if generator_scheduler is not None:
            generator_scheduler.step()
        if discriminator_scheduler is not None:
            discriminator_scheduler.step()

        completed_epoch = epoch_index + 1
        if completed_epoch % checkpoint_interval == 0:
            save_srgan_checkpoint(
                Path(checkpoint_dir) / f"srgan_epoch_{completed_epoch:03d}.pt",
                epoch=completed_epoch,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                generator_scheduler=generator_scheduler,
                discriminator_scheduler=discriminator_scheduler,
                history=history,
                config=config,
            )
        if (
            sample_tensors is not None
            and sample_dir is not None
            and completed_epoch % sample_interval == 0
        ):
            save_srgan_samples(
                generator,
                sample_tensors[0],
                sample_tensors[1],
                Path(sample_dir) / f"adversarial_epoch_{completed_epoch:03d}.png",
                device,
            )
    _set_requires_grad(discriminator, True)
    return history
