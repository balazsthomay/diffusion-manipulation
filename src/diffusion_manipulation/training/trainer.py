"""Training loop for diffusion policy."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_manipulation.config import TrainConfig
from diffusion_manipulation.data.normalizer import LinearNormalizer
from diffusion_manipulation.policy.diffusion_policy import DiffusionUnetPolicy
from diffusion_manipulation.training.ema import EMAModel


class Trainer:
    """Training loop with checkpointing, EMA, LR scheduling, and optional wandb."""

    def __init__(
        self,
        policy: DiffusionUnetPolicy,
        dataloader: DataLoader,
        normalizer: LinearNormalizer,
        config: TrainConfig,
        device: torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.dataloader = dataloader
        self.normalizer = normalizer
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move policy to device and set normalizer
        self.policy.to(self.device)
        self.policy.set_normalizer(normalizer)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # LR scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * len(dataloader),
        )

        # EMA
        self.ema = EMAModel(self.policy, decay=config.ema_decay)

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        self.wandb_run = None
        if config.use_wandb:
            import wandb
            self.wandb_run = wandb.init(
                project=config.wandb_project,
                config={
                    "num_epochs": config.num_epochs,
                    "batch_size": config.batch_size,
                    "lr": config.lr,
                    "ema_decay": config.ema_decay,
                    "seed": config.seed,
                },
            )

        self.global_step = 0
        self.epoch = 0

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            Dict with training metrics history.
        """
        history: dict[str, list[float]] = {"loss": [], "lr": []}

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch()
            history["loss"].append(epoch_loss)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if (epoch + 1) % self.config.log_interval == 0:
                print(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                    f"Loss: {epoch_loss:.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }, step=self.global_step)

        # Save final checkpoint
        self.save_checkpoint("checkpoint_final.pt")

        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f)

        return history

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.policy.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            loss = self.policy.compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.gradient_clip_norm,
                )

            self.optimizer.step()
            self.lr_scheduler.step()
            self.ema.update(self.policy)

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, filename: str) -> Path:
        """Save training checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "normalizer_state_dict": self.normalizer.state_dict(),
        }
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.ema.load_state_dict(checkpoint["ema_state_dict"])
        self.normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
