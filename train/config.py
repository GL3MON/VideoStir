"""Configuration classes for VideoStir training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingConfig:
    """Configuration for VideoStir training.

    This dataclass encapsulates all parameters needed to run the
    video training pipeline with knowledge distillation and LoRA.

    Args:
        job_type: Type of training ("mmkd_black_box", "mmkd_white_box", "lora_only").
        instruction_path: Path to instruction JSON file (optional).
        labeled_path: Path to labeled training data JSON file.
        logits_path: Path to teacher model logits JSONL file (for white-box KD).
        seed: Random seed for reproducibility (default: 42).

    Model Settings:
        teacher_model_id: Hugging Face model ID for teacher model.
        student_model_id: Hugging Face model ID for student model.

    Training Settings:
        output_dir: Directory to save training outputs.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Steps to accumulate gradients.
        max_length: Maximum sequence length.
        save_steps: Number of steps between saves.
        logging_steps: Number of steps between logging.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for regularization.
        warmup_ratio: Ratio of warmup steps.
        lr_scheduler_type: Scheduler type (cosine, linear, etc.).
        bf16: Use bfloat16 precision.

    LoRA Settings:
        lora_enable: Enable LoRA fine-tuning.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout rate.
        lora_bias: LoRA bias setting.
        lora_target_modules: Modules to apply LoRA to.

    Distillation Settings:
        kd_ratio: Knowledge distillation loss ratio.
        max_seq_length: Maximum sequence length for distillation.
        distillation_type: Type of distillation ("forward_kld", "reverse_kld").

    Checkpoint Settings:
        enable_checkpoint: Enable checkpointing for resumable training.
        checkpoint_dir: Directory for checkpoint files.
        resume_from_checkpoint: Path to resume from checkpoint.
        save_total_limit: Maximum number of checkpoints to keep.

    Example:
        >>> from videostir import TrainingConfig, train_black_box
        >>> config = TrainingConfig(
        ...     job_type="mmkd_black_box",
        ...     student_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        ...     teacher_model_id="Qwen/Qwen2.5-VL-72B-Instruct",
        ...     labeled_path="./train.json",
        ...     output_dir="./result/",
        ... )
        >>> train_black_box(config)
    """

    # Job type
    job_type: str = "mmkd_black_box"

    # Dataset settings
    instruction_path: Optional[str] = None
    labeled_path: str = "./train.json"
    logits_path: Optional[str] = None
    seed: int = 42

    # Model settings
    teacher_model_id: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    student_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Training settings (SFTConfig compatible)
    output_dir: str = "./result/"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_length: int = 512
    save_steps: int = 1500
    logging_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    bf16: bool = True

    # LoRA settings
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Distillation settings
    kd_ratio: float = 0.1
    max_seq_length: int = 512
    distillation_type: str = "forward_kld"

    # Checkpoint settings
    enable_checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 2

    @classmethod
    def from_json(cls, json_path: str) -> TrainingConfig:
        """Load config from JSON file.

        Args:
            json_path: Path to JSON configuration file.

        Returns:
            TrainingConfig instance.
        """
        import json
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingConfig:
        """Load config from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            TrainingConfig instance.
        """
        # Handle nested structures
        dataset = data.get("dataset", {})
        models = data.get("models", {})
        training = data.get("training", {})
        lora = data.get("lora", {})
        distillation = data.get("distillation", {})

        return cls(
            job_type=data.get("job_type", "mmkd_black_box"),
            instruction_path=dataset.get("instruction_path"),
            labeled_path=dataset.get("labeled_path", "./train.json"),
            logits_path=dataset.get("logits_path"),
            seed=dataset.get("seed", 42),
            teacher_model_id=models.get("teacher", "Qwen/Qwen2.5-VL-72B-Instruct"),
            student_model_id=models.get("student", "Qwen/Qwen2.5-VL-3B-Instruct"),
            output_dir=training.get("output_dir", "./result/"),
            num_train_epochs=training.get("num_train_epochs", 1),
            per_device_train_batch_size=training.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=training.get("gradient_accumulation_steps", 8),
            max_length=training.get("max_length", 512),
            save_steps=training.get("save_steps", 1500),
            logging_steps=training.get("logging_steps", 1),
            learning_rate=training.get("learning_rate", 2e-5),
            weight_decay=training.get("weight_decay", 0.05),
            warmup_ratio=training.get("warmup_ratio", 0.05),
            lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
            bf16=training.get("bf16", True),
            lora_enable=lora.get("enable", True),
            lora_r=lora.get("r", 16),
            lora_alpha=lora.get("alpha", 32),
            lora_dropout=lora.get("dropout", 0.05),
            lora_bias=lora.get("bias", "none"),
            lora_target_modules=lora.get("target_modules", ["q_proj", "v_proj"]),
            kd_ratio=distillation.get("kd_ratio", 0.1),
            max_seq_length=distillation.get("max_seq_length", 512),
            distillation_type=distillation.get("distillation_type", "forward_kld"),
            enable_checkpoint=data.get("enable_checkpoint", True),
            checkpoint_dir=data.get("checkpoint_dir", "checkpoints"),
            resume_from_checkpoint=data.get("resume_from_checkpoint"),
            save_total_limit=data.get("save_total_limit", 2),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "job_type": self.job_type,
            "dataset": {
                "instruction_path": self.instruction_path,
                "labeled_path": self.labeled_path,
                "logits_path": self.logits_path,
                "seed": self.seed,
            },
            "models": {
                "teacher": self.teacher_model_id,
                "student": self.student_model_id,
            },
            "training": {
                "output_dir": self.output_dir,
                "num_train_epochs": self.num_train_epochs,
                "per_device_train_batch_size": self.per_device_train_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_length": self.max_length,
                "save_steps": self.save_steps,
                "logging_steps": self.logging_steps,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_ratio": self.warmup_ratio,
                "lr_scheduler_type": self.lr_scheduler_type,
                "bf16": self.bf16,
            },
            "lora": {
                "enable": self.lora_enable,
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "bias": self.lora_bias,
                "target_modules": self.lora_target_modules,
            },
            "distillation": {
                "kd_ratio": self.kd_ratio,
                "max_seq_length": self.max_seq_length,
                "distillation_type": self.distillation_type,
            },
            "enable_checkpoint": self.enable_checkpoint,
            "checkpoint_dir": self.checkpoint_dir,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "save_total_limit": self.save_total_limit,
        }

    def to_json(self, json_path: str) -> None:
        """Save config to JSON file.

        Args:
            json_path: Path to save JSON configuration.
        """
        import json
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TrainingResult:
    """Result from running the VideoStir training pipeline.

    Args:
        output_dir: Directory where model was saved.
        training_steps: Number of training steps completed.
        final_loss: Final training loss.
        metrics: Training metrics.
    """

    output_dir: str
    training_steps: int
    final_loss: float
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "output_dir": self.output_dir,
            "training_steps": self.training_steps,
            "final_loss": self.final_loss,
            "metrics": self.metrics,
        }