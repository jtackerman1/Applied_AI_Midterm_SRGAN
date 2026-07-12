"""Synthetic tests for SRGAN architecture, losses, checkpoints, and resume."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from applied_ai_midterm.models import Discriminator, Generator, GeneratorLoss
from applied_ai_midterm.training import (
    empty_srgan_history,
    latest_generator_pretrain_checkpoint,
    latest_srgan_checkpoint,
    pretrain_generator,
    resume_generator_pretraining,
    resume_latest_srgan,
    save_srgan_checkpoint,
    train_srgan,
)


class TinyGenerator(nn.Module):
    """Small differentiable 4x generator used only for checkpoint tests."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(3, 3, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        upscaled = torch.nn.functional.interpolate(inputs, scale_factor=4, mode="nearest")
        return torch.tanh(self.convolution(upscaled))


class TinyDiscriminator(nn.Module):
    """Small discriminator used only for checkpoint tests."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(3, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.convolution(inputs).mean(dim=(2, 3))


def test_generator_performs_four_times_upscaling() -> None:
    generator = Generator(channels=8, residual_blocks=1).eval()

    with torch.no_grad():
        generated = generator(torch.randn(2, 3, 32, 32))

    assert generated.shape == (2, 3, 128, 128)
    assert torch.all(generated >= -1.0) and torch.all(generated <= 1.0)


def test_discriminator_returns_one_unbounded_logit() -> None:
    discriminator = Discriminator(base_channels=8).eval()

    with torch.no_grad():
        logits = discriminator(torch.randn(2, 3, 128, 128))

    assert logits.shape == (2, 1)


def test_generator_loss_includes_optional_perceptual_component() -> None:
    perceptual_features = nn.AvgPool2d(2)
    criterion = GeneratorLoss(perceptual_features=perceptual_features)
    generated = torch.zeros(1, 3, 8, 8, requires_grad=True)
    target = torch.ones(1, 3, 8, 8)
    logits = torch.zeros(1, 1, requires_grad=True)

    values = criterion(generated, target, logits)
    values.total.backward()

    assert values.total.ndim == 0
    assert values.pixel.item() > 0.0
    assert values.adversarial.item() > 0.0
    assert values.perceptual.item() > 0.0
    assert generated.grad is not None


def test_checkpoint_contains_complete_state_and_resume_uses_latest(tmp_path: Path) -> None:
    generator = TinyGenerator()
    discriminator = TinyDiscriminator()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    generator_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=1)
    discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
        discriminator_optimizer, step_size=1
    )
    history = empty_srgan_history()
    for epoch in (5, 10):
        save_srgan_checkpoint(
            tmp_path / f"srgan_epoch_{epoch:03d}.pt",
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator_scheduler=generator_scheduler,
            discriminator_scheduler=discriminator_scheduler,
            history=history,
            config={"random_seed": 42, "srgan_epochs": 150},
        )

    start_epoch, restored_history, latest_path = resume_latest_srgan(
        tmp_path,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        device=torch.device("cpu"),
    )
    checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)

    assert latest_srgan_checkpoint(tmp_path) == tmp_path / "srgan_epoch_010.pt"
    assert start_epoch == 10
    assert restored_history == history
    assert checkpoint["random_seed"] == 42
    assert checkpoint["generator_scheduler_state"] is not None
    assert checkpoint["discriminator_scheduler_state"] is not None
    assert {
        "generator_state",
        "discriminator_state",
        "generator_optimizer_state",
        "discriminator_optimizer_state",
        "history",
        "config",
    }.issubset(checkpoint)


def test_training_rejects_fewer_than_150_total_epochs(tmp_path: Path) -> None:
    generator = TinyGenerator()
    discriminator = TinyDiscriminator()
    generator_optimizer = torch.optim.Adam(generator.parameters())
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters())

    try:
        train_srgan(
            generator,
            discriminator,
            [],
            generator_optimizer,
            discriminator_optimizer,
            GeneratorLoss(),
            tmp_path,
            total_epochs=149,
            start_epoch=0,
            device=torch.device("cpu"),
            config={"random_seed": 42},
        )
    except ValueError as error:
        assert "at least 150" in str(error)
    else:
        raise AssertionError("Expected fewer than 150 SRGAN epochs to be rejected.")


def test_generator_pretraining_saves_reconstruction_checkpoint_and_sample(
    tmp_path: Path,
) -> None:
    generator = TinyGenerator()
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    low_resolution = torch.randn(2, 3, 4, 4).clamp(-1, 1)
    high_resolution = torch.randn(2, 3, 16, 16).clamp(-1, 1)
    loader = DataLoader(TensorDataset(low_resolution, high_resolution), batch_size=2)
    checkpoint_dir = tmp_path / "pretrain_checkpoints"
    sample_dir = tmp_path / "samples"

    history = pretrain_generator(
        generator,
        loader,
        optimizer,
        checkpoint_dir,
        total_epochs=5,
        start_epoch=4,
        device=torch.device("cpu"),
        config={"random_seed": 42},
        sample_tensors=(low_resolution, high_resolution),
        sample_dir=sample_dir,
    )
    start_epoch, restored_history, latest_path = resume_generator_pretraining(
        checkpoint_dir,
        generator=generator,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert len(history) == 1
    assert latest_generator_pretrain_checkpoint(checkpoint_dir) == (
        checkpoint_dir / "generator_pretrain_epoch_005.pt"
    )
    assert latest_path == checkpoint_dir / "generator_pretrain_epoch_005.pt"
    assert start_epoch == 5
    assert restored_history == history
    assert (sample_dir / "pretrain_epoch_005.png").is_file()


def test_resumed_epoch_150_creates_five_epoch_checkpoint(tmp_path: Path) -> None:
    generator = TinyGenerator()
    discriminator = TinyDiscriminator()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    low_resolution = torch.randn(2, 3, 4, 4)
    high_resolution = torch.randn(2, 3, 16, 16)
    loader = DataLoader(
        TensorDataset(low_resolution, high_resolution),
        batch_size=2,
    )

    history = train_srgan(
        generator,
        discriminator,
        loader,
        generator_optimizer,
        discriminator_optimizer,
        GeneratorLoss(),
        tmp_path,
        total_epochs=150,
        start_epoch=149,
        device=torch.device("cpu"),
        config={"random_seed": 42, "srgan_epochs": 150},
        sample_tensors=(low_resolution, high_resolution),
        sample_dir=tmp_path / "samples",
    )

    assert (tmp_path / "srgan_epoch_150.pt").is_file()
    assert (tmp_path / "samples" / "adversarial_epoch_150.png").is_file()
    assert len(history["generator_total"]) == 1
