"""Training module for VideoStir.

This module provides tools for fine-tuning Qwen2.5-VL models using
knowledge distillation and LoRA fine-tuning.

Quick Start:
    >>> from videostir.train import TrainingConfig, train_black_box
    >>> config = TrainingConfig(
    ...     job_type="mmkd_black_box",
    ...     student_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    ...     teacher_model_id="Qwen/Qwen2.5-VL-72B-Instruct",
    ...     labeled_path="./train.json",
    ...     output_dir="./result/",
    ... )
    >>> result = train_black_box(config)

API:
    - TrainingConfig: Configure the training pipeline
    - train_black_box(): Train with black-box knowledge distillation
    - train_white_box(): Train with white-box knowledge distillation
    - train_lora(): Train with LoRA fine-tuning only
    - resume_training(): Resume training from checkpoint

Examples:
    # Black-box distillation (student learns from teacher outputs)
    >>> from videostir.train import train_black_box
    >>> result = train_black_box(config)

    # White-box distillation (intermediate feature distillation)
    >>> from videostir.train import train_white_box
    >>> config.logits_path = "./train_logits.json"
    >>> result = train_white_box(config)

    # Resume from checkpoint
    >>> from videostir.train import resume_training
    >>> resume_training(config, checkpoint_path="./result/checkpoint-1500")
"""

from __future__ import annotations

from .config import TrainingConfig, TrainingResult
from .checkpoint import TrainingCheckpointManager

# Lazy imports for training functions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models.distiller import DistillSFTTrainer
    from .models.data import MMDataset

__version__ = "0.1.0"
__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "TrainingCheckpointManager",
    "train_black_box",
    "train_white_box",
    "train_lora",
    "resume_training",
]


def train_black_box(config: TrainingConfig) -> TrainingResult:
    """Train with black-box knowledge distillation.

    In black-box distillation, the student model learns from the teacher's
    output probabilities only, without access to intermediate features.

    Args:
        config: TrainingConfig with all parameters.

    Returns:
        TrainingResult with training metadata.
    """
    from .models.distiller import create_black_box_trainer

    trainer = create_black_box_trainer(config)
    trainer.train()
    return _create_training_result(trainer, config)


def train_white_box(config: TrainingConfig) -> TrainingResult:
    """Train with white-box knowledge distillation.

    In white-box distillation, the student model learns from both
    the teacher's outputs AND intermediate feature representations.

    Args:
        config: TrainingConfig with all parameters. Must have logits_path set.

    Returns:
        TrainingResult with training metadata.
    """
    from .models.distiller import create_white_box_trainer

    trainer = create_white_box_trainer(config)
    trainer.train()
    return _create_training_result(trainer, config)


def train_lora(config: TrainingConfig) -> TrainingResult:
    """Train with LoRA fine-tuning only.

    This performs standard LoRA fine-tuning without knowledge distillation.

    Args:
        config: TrainingConfig with all parameters.

    Returns:
        TrainingResult with training metadata.
    """
    from .models.distiller import create_lora_trainer

    trainer = create_lora_trainer(config)
    trainer.train()
    return _create_training_result(trainer, config)


def resume_training(config: TrainingConfig, checkpoint_path: str) -> TrainingResult:
    """Resume training from a checkpoint.

    Args:
        config: TrainingConfig with base parameters.
        checkpoint_path: Path to checkpoint directory to resume from.

    Returns:
        TrainingResult with training metadata.
    """
    from .models.distiller import create_trainer_from_checkpoint

    trainer = create_trainer_from_checkpoint(config, checkpoint_path)
    trainer.train()
    return _create_training_result(trainer, config)


def _create_training_result(trainer, config: TrainingConfig) -> TrainingResult:
    """Create TrainingResult from trainer state."""
    import os

    # Get training stats from trainer state if available
    training_steps = getattr(trainer.state, "global_step", 0)
    final_loss = getattr(trainer.state, "total_loss", 0.0)

    # Get metrics from trainer
    metrics = getattr(trainer, "metrics", {})

    return TrainingResult(
        output_dir=config.output_dir,
        training_steps=training_steps,
        final_loss=final_loss,
        metrics=metrics,
    )