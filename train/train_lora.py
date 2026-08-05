#!/usr/bin/env python
"""Training script for VideoStir.

This script provides the main entry point for training VideoStir models
using knowledge distillation and LoRA fine-tuning.

Usage:
    # Black-box distillation (student learns from teacher outputs)
    python train_lora.py --config mmkd_black_box_lora_single.json

    # Resume from checkpoint
    python train_lora.py --config config.json --resume_from checkpoint-1500

    # With custom output directory
    python train_lora.py --config config.json --output_dir ./my_result/
"""

import argparse
import json
import os
import sys
from typing import Optional

# Add VideoStir to path
videostir_dir = "/hfcache/harissh/VideoStir"
if videostir_dir not in sys.path:
    sys.path.insert(0, videostir_dir)

from train.config import TrainingConfig
from train import train_black_box, train_white_box, train_lora, resume_training


def load_config(config_path: str) -> TrainingConfig:
    """Load training config from JSON file.

    Args:
        config_path: Path to JSON config file.

    Returns:
        TrainingConfig instance.
    """
    with open(config_path, "r") as f:
        data = json.load(f)
    return TrainingConfig.from_dict(data)


def find_latest_checkpoint(output_dir: str, checkpoint_dir: str = "checkpoints") -> Optional[str]:
    """Find the latest checkpoint in output directory.

    Args:
        output_dir: Base output directory.
        checkpoint_dir: Checkpoints subdirectory.

    Returns:
        Path to latest checkpoint, or None.
    """
    checkpoint_path = os.path.join(output_dir, checkpoint_dir, "latest")
    if os.path.islink(checkpoint_path):
        return os.readlink(checkpoint_path)
    if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
        return checkpoint_path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Train VideoStir model with knowledge distillation or LoRA"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (overrides checkpoint in config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--job_type",
        type=str,
        default=None,
        help="Override job type (mmkd_black_box, mmkd_white_box, lora_only)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.job_type:
        config.job_type = args.job_type

    # Handle resume
    resume_path = args.resume_from
    if resume_path is None and config.resume_from_checkpoint:
        resume_path = config.resume_from_checkpoint

    if resume_path:
        # Resume from checkpoint
        print(f"Resuming training from: {resume_path}")
        result = resume_training(config, resume_path)
        print(f"Training resumed from step {result.training_steps}")
    else:
        # Determine training mode from config
        job_type = config.job_type.lower()

        if "mmkd_white_box" in job_type:
            if not config.logits_path:
                raise ValueError(
                    "White-box distillation requires logits_path in config"
                )
            print("Starting white-box knowledge distillation...")
            result = train_white_box(config)
        elif "mmkd_black_box" in job_type or "lora_only" in job_type:
            print("Starting black-box distillation or LoRA training...")
            result = train_lora(config)
        else:
            raise ValueError(f"Unknown job type: {job_type}")

    print(f"\nTraining complete!")
    print(f"Output directory: {result.output_dir}")
    print(f"Training steps: {result.training_steps}")
    print(f"Final loss: {result.final_loss:.4f}")

    # Save training summary
    summary_path = os.path.join(result.output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()