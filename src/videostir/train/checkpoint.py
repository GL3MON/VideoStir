"""Checkpointing utilities for VideoStir training.

This module enables resumable training by caching model states
and optimizer/scheduler states to disk.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class TrainingCheckpointManager:
    """Manages checkpointing for VideoStir training.

    The checkpoint manager enables resumable training by:
    1. Saving model weights (PEFT format for LoRA)
    2. Saving optimizer and scheduler states
    3. Tracking training progress (global step, epoch)
    4. Supporting resume from checkpoint

    Example:
        >>> manager = TrainingCheckpointManager("output_dir")
        >>> if manager.has_checkpoint("latest"):
        ...     trainer = load_trainer_from_checkpoint(manager, "latest")
        ...     trainer.train()
    """

    def __init__(
        self,
        output_dir: str,
        checkpoint_dir: str = "checkpoints",
    ):
        """Initialize the checkpoint manager.

        Args:
            output_dir: Base output directory for training.
            checkpoint_dir: Subdirectory for checkpoint files.
        """
        self.output_dir = os.path.abspath(output_dir)
        self.checkpoint_dir = os.path.join(self.output_dir, checkpoint_dir)

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_checkpoint_path(self, name: str) -> str:
        """Get the path for a checkpoint directory."""
        return os.path.join(self.checkpoint_dir, name)

    def has_checkpoint(self, name: str = "latest") -> bool:
        """Check if a checkpoint exists.

        Args:
            name: Name of the checkpoint (e.g., 'latest', 'checkpoint-1500').

        Returns:
            True if the checkpoint exists.
        """
        checkpoint_path = self._get_checkpoint_path(name)
        if name == "latest":
            # Check for latest symlink or file
            latest_path = os.path.join(self.checkpoint_dir, "latest")
            return os.path.exists(latest_path) or os.path.exists(checkpoint_path)
        return os.path.exists(checkpoint_path)

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
        """
        latest_path = os.path.join(self.checkpoint_dir, "latest")
        if os.path.islink(latest_path):
            return os.readlink(latest_path)
        if os.path.exists(latest_path) and os.path.isdir(latest_path):
            return latest_path
        # No latest checkpoint
        return None

    def save_checkpoint(
        self,
        trainer,
        step: Optional[int] = None,
        name: Optional[str] = None,
    ) -> str:
        """Save model checkpoint.

        Args:
            trainer: The SFTTrainer instance.
            step: Step number (uses trainer.state.global_step if None).
            name: Checkpoint name (uses step if None).

        Returns:
            Path to saved checkpoint.
        """
        if step is None:
            step = getattr(trainer.state, "global_step", 0)

        if name is None:
            name = f"checkpoint-{step}"

        checkpoint_path = self._get_checkpoint_path(name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save trainer state
        trainer.save_model(checkpoint_path)

        # Save optimizer and scheduler states
        if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            torch.save(
                trainer.optimizer.state_dict(),
                os.path.join(checkpoint_path, "optimizer.pt"),
            )

        if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler is not None:
            torch.save(
                trainer.lr_scheduler.state_dict(),
                os.path.join(checkpoint_path, "scheduler.pt"),
            )

        # Save training args
        torch.save(
            trainer.args,
            os.path.join(checkpoint_path, "training_args.bin"),
        )

        # Update latest symlink
        latest_path = os.path.join(self.checkpoint_dir, "latest")
        if os.path.islink(latest_path) or os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(checkpoint_path, latest_path)

        # Save checkpoint metadata
        metadata = {
            "step": step,
            "saved_at": _get_timestamp(),
            "checkpoint_dir": checkpoint_path,
        }
        metadata_path = os.path.join(self.checkpoint_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return checkpoint_path

    def load_checkpoint(
        self,
        trainer,
        checkpoint_path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ) -> int:
        """Load model from checkpoint.

        Args:
            trainer: The SFTTrainer instance.
            checkpoint_path: Path to checkpoint directory.
            load_optimizer: Whether to load optimizer state.
            load_scheduler: Whether to load scheduler state.

        Returns:
            Step number to resume from.
        """
        # Load model
        trainer.model.load_adapter(checkpoint_path, is_trainable=True)
        trainer.model.config = torch.load(
            os.path.join(checkpoint_path, "config.pth"),
            map_location=trainer.model.device,
        )

        step = 0

        # Load optimizer state
        if load_optimizer:
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            if os.path.exists(optimizer_path):
                trainer.optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=trainer.model.device)
                )
                step = max(step, trainer.state.global_step if hasattr(trainer, 'state') else 0)

        # Load scheduler state
        if load_scheduler:
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
            if os.path.exists(scheduler_path):
                trainer.lr_scheduler.load_state_dict(
                    torch.load(scheduler_path, map_location=trainer.model.device)
                )

        return step

    def save_trainer_state(
        self,
        trainer,
        name: Optional[str] = None,
    ) -> str:
        """Save trainer state (without model weights).

        Args:
            trainer: The SFTTrainer instance.
            name: State name.

        Returns:
            Path to saved state.
        """
        if name is None:
            name = "trainer_state"

        state_path = self._get_checkpoint_path(name)
        os.makedirs(state_path, exist_ok=True)

        # Save trainer state only
        trainer.save_state()

        # Save optimizer and scheduler states separately
        if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            torch.save(
                trainer.optimizer.state_dict(),
                os.path.join(state_path, "optimizer.pt"),
            )

        if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler is not None:
            torch.save(
                trainer.lr_scheduler.state_dict(),
                os.path.join(state_path, "scheduler.pt"),
            )

        return state_path

    def cleanup_old_checkpoints(
        self,
        max_checkpoints: int = 2,
    ) -> None:
        """Remove old checkpoints to save disk space.

        Args:
            max_checkpoints: Maximum number of checkpoints to keep.
        """
        checkpoints = []
        for item in os.listdir(self.checkpoint_dir):
            item_path = os.path.join(self.checkpoint_dir, item)
            if os.path.isdir(item_path) and item != "latest":
                checkpoints.append(item_path)

        # Sort by modification time (newest first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)

        # Remove old checkpoints
        for old_checkpoint in checkpoints[max_checkpoints:]:
            shutil.rmtree(old_checkpoint, ignore_errors=True)
            print(f"Cleaned up old checkpoint: {old_checkpoint}")


def _get_timestamp() -> str:
    """Get current timestamp as ISO format string."""
    import datetime
    return datetime.datetime.now().isoformat()