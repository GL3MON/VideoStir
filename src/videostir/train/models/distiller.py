"""Distillation training utilities for VideoStir.

This module provides specialized trainers for knowledge distillation
and LoRA fine-tuning.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import jsonlines
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
from transformers.trainer_utils import speed_metrics
from trl import SFTConfig, SFTTrainer

from .data import MMDataset, MultimodalCollator, load_training_data


class DistillSFTTrainer(SFTTrainer):
    """SFTTrainer with knowledge distillation support.

    This trainer extends SFTTrainer to support both black-box and
    white-box knowledge distillation from a teacher model.

    Args:
        logits_dir: Directory containing teacher model logits.
        teacher_vocab_size: Vocabulary size of teacher model.
        kd_ratio: Knowledge distillation loss ratio.
        max_seq_length: Maximum sequence length.
        distillation_type: Type of distillation ("forward_kld" or "reverse_kld").
        **kwargs: Additional arguments for SFTTrainer.
    """

    def __init__(
        self,
        logits_dir: Optional[str] = None,
        teacher_vocab_size: Optional[int] = None,
        kd_ratio: float = 0.5,
        max_seq_length: int = 1024,
        distillation_type: str = "forward_kld",
        **kwargs,
    ):
        """Initialize the distillation trainer."""
        super().__init__(**kwargs)
        self.logits_dir = logits_dir
        self.teacher_vocab_size = teacher_vocab_size
        self.kd_ratio = kd_ratio
        self.max_seq_length = max_seq_length
        self.distillation_type = distillation_type
        self.teacher_logits: List[Dict[int, float]] = []

        # Load teacher logits if provided
        if self.logits_dir and os.path.exists(self.logits_dir):
            self._load_teacher_logits()

    def _load_teacher_logits(self) -> None:
        """Load teacher logits from JSONL file."""
        with jsonlines.open(self.logits_dir) as reader:
            for obj in reader:
                self.teacher_logits.append(obj)

    def _load_teacher_logits_for_batch(
        self,
        batch_size: int,
        step: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Load teacher logits for current batch.

        Args:
            batch_size: Batch size.
            step: Current global step.
            device: Target device.

        Returns:
            Teacher logits tensor.
        """
        start_idx = step * batch_size
        end_idx = start_idx + batch_size
        loaded_data = self.teacher_logits[start_idx:end_idx]

        # Handle case where we've exhausted logits
        if not loaded_data:
            return None

        arr = np.zeros(
            (batch_size, self.max_seq_length, self.teacher_vocab_size),
            dtype=np.float32,
        )
        for i, logit_dict in enumerate(loaded_data):
            for j, (key, value) in enumerate(logit_dict.items()):
                arr[i, j, int(key)] = value

        logits_tensor = torch.tensor(arr, dtype=torch.bfloat16, device=device)
        return self._shift_tensor_right(logits_tensor)

    def _shift_tensor_right(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Shift tensor right to align with labels.

        Args:
            inputs: Input tensor to shift.

        Returns:
            Shifted tensor.
        """
        batch_size, seqlen, vocab_size = inputs.shape
        device = inputs.device

        # Find shift distances based on labels (if available)
        if hasattr(self, 'state') and self.state is not None:
            labels_ne = torch.ones(batch_size, seqlen, device=device)
        else:
            labels_ne = torch.ones(batch_size, seqlen, device=device)

        shift_distances = torch.argmax(labels_ne.int(), dim=1)
        idx = torch.arange(seqlen, device=device).unsqueeze(0).expand(batch_size, seqlen)
        shifted_idx = idx - shift_distances.unsqueeze(1)
        mask = shifted_idx >= 0
        shifted_idx = shifted_idx.clamp(min=0)

        inputs_flat = inputs.view(batch_size, seqlen, vocab_size)
        shifted_idx = shifted_idx.unsqueeze(2).expand(-1, -1, vocab_size)
        gathered = torch.gather(inputs_flat, 1, shifted_idx)
        mask = mask.unsqueeze(2).expand(-1, -1, vocab_size)
        return torch.where(mask, gathered, torch.full_like(gathered, 0.0))

    def _compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute distillation loss.

        Args:
            student_logits: Student model logits.
            teacher_logits: Teacher model logits.
            labels: Ground truth labels.

        Returns:
            Distillation loss.
        """
        student_logits = student_logits[:, : self.max_seq_length, :]
        teacher_probs = teacher_logits[
            :, : student_logits.size(1), : student_logits.size(-1)
        ]

        # Create mask for valid positions
        mask = (labels != -100).float() if labels is not None else torch.ones_like(
            student_logits[:, :, 0]
        )

        if self.distillation_type == "forward_kld":
            # Forward KLD: student learns from teacher
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction="none",
                log_target=False,
            ).sum(dim=-1) / torch.sum(mask.view(-1), dim=0)

        elif self.distillation_type == "reverse_kld":
            # Reverse KLD: teacher provides certainty to student
            loss = F.kl_div(
                torch.log(teacher_probs.clamp(min=1e-10)),
                F.softmax(student_logits, dim=-1),
                reduction="none",
                log_target=False,
            ).sum(dim=-1) / torch.sum(mask.view(-1), dim=0)

        else:
            raise ValueError(
                f"Unsupported distillation type: {self.distillation_type}. "
                "Use 'forward_kld' or 'reverse_kld'"
            )

        return (loss * mask).sum() / mask.sum()

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple]:
        """Compute combined loss (LM + distillation).

        Args:
            model: The model to train.
            inputs: Input batch.
            return_outputs: Whether to return model outputs.
            num_items_in_batch: Number of items in batch.

        Returns:
            Loss or (loss, outputs).
        """
        outputs = model(**inputs)
        lm_loss = outputs.loss

        # Compute distillation loss if teacher logits available
        if hasattr(self, "logits_dir") and self.logits_dir:
            batch_size = inputs["input_ids"].size(0)
            device = model.device

            teacher_logits = self._load_teacher_logits_for_batch(
                batch_size=batch_size,
                step=self.state.global_step if hasattr(self, "state") else 0,
                device=device,
            )

            if teacher_logits is not None:
                distil_loss = self._compute_distillation_loss(
                    student_logits=outputs.logits,
                    teacher_logits=teacher_logits,
                    labels=inputs.get("labels", None),
                )
                total_loss = (1 - self.kd_ratio) * lm_loss + self.kd_ratio * distil_loss
            else:
                total_loss = lm_loss
        else:
            total_loss = lm_loss

        return (total_loss, outputs) if return_outputs else total_loss


def create_black_box_trainer(config: Any) -> SFTTrainer:
    """Create SFTTrainer for black-box distillation.

    Args:
        config: TrainingConfig or path to config file.

    Returns:
        Configured SFTTrainer.
    """
    from ..config import TrainingConfig

    if isinstance(config, str):
        config = TrainingConfig.from_json(config)
    elif isinstance(config, dict):
        config = TrainingConfig.from_dict(config)

    # Load model and processor
    student_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.student_model_id,
        trust_remote_code=True,
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(config.student_model_id)

    # Apply LoRA if enabled
    if config.lora_enable:
        print(">>> Enabling LoRA fine-tuning ...")
        if hasattr(student_model, "is_loaded_in_8bit") or hasattr(
            student_model, "is_loaded_in_4bit"
        ):
            student_model = prepare_model_for_kbit_training(student_model)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type="CAUSAL_LM",
        )
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()

    # Create data collator
    collator = MultimodalCollator(processor, max_length=config.max_length)

    # Create training arguments
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Load data
    data = load_training_data(
        labeled_path=config.labeled_path,
        instruction_path=config.instruction_path,
    )
    dataset = MMDataset(data, processor, config.max_length)

    # Create trainer
    trainer = SFTTrainer(
        model=student_model,
        data_collator=collator,
        processing_class=processor.tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    return trainer


def create_white_box_trainer(config: Any) -> DistillSFTTrainer:
    """Create DistillSFTTrainer for white-box distillation.

    Args:
        config: TrainingConfig or path to config file.

    Returns:
        Configured DistillSFTTrainer.
    """
    from ..config import TrainingConfig

    if isinstance(config, str):
        config = TrainingConfig.from_json(config)
    elif isinstance(config, dict):
        config = TrainingConfig.from_dict(config)

    # Validate logits path
    if not config.logits_path or not os.path.exists(config.logits_path):
        raise ValueError(
            "White-box distillation requires logits_path to point to teacher logits"
        )

    # Load model and processor
    student_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.student_model_id,
        trust_remote_code=True,
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(config.student_model_id)

    # Get teacher vocab size
    teacher_config_path = os.path.join(config.teacher_model_id, "config.json")
    if os.path.exists(teacher_config_path):
        with open(teacher_config_path, "r") as f:
            teacher_config = json.load(f)
        teacher_vocab_size = teacher_config.get("vocab_size", 152064)
    else:
        teacher_vocab_size = 152064  # Default Qwen2.5 vocab size

    # Apply LoRA if enabled
    if config.lora_enable:
        print(">>> Enabling LoRA fine-tuning ...")
        if hasattr(student_model, "is_loaded_in_8bit") or hasattr(
            student_model, "is_loaded_in_4bit"
        ):
            student_model = prepare_model_for_kbit_training(student_model)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type="CAUSAL_LM",
        )
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()

    # Create data collator
    collator = MultimodalCollator(processor, max_length=config.max_length)

    # Create training arguments
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Load data
    data = load_training_data(
        labeled_path=config.labeled_path,
        instruction_path=config.instruction_path,
    )
    dataset = MMDataset(data, processor, config.max_length)

    # Create trainer
    trainer = DistillSFTTrainer(
        logits_dir=config.logits_path,
        teacher_vocab_size=teacher_vocab_size,
        kd_ratio=config.kd_ratio,
        max_seq_length=config.max_seq_length,
        distillation_type=config.distillation_type,
        data_collator=collator,
        model=student_model,
        processing_class=processor.tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    return trainer


def create_lora_trainer(config: Any) -> SFTTrainer:
    """Create SFTTrainer for LoRA fine-tuning only.

    Args:
        config: TrainingConfig or path to config file.

    Returns:
        Configured SFTTrainer.
    """
    from ..config import TrainingConfig

    if isinstance(config, str):
        config = TrainingConfig.from_json(config)
    elif isinstance(config, dict):
        config = TrainingConfig.from_dict(config)

    # Load model and processor
    student_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.student_model_id,
        trust_remote_code=True,
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(config.student_model_id)

    # Apply LoRA
    print(">>> Enabling LoRA fine-tuning ...")
    if hasattr(student_model, "is_loaded_in_8bit") or hasattr(
        student_model, "is_loaded_in_4bit"
    ):
        student_model = prepare_model_for_kbit_training(student_model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # Create data collator
    collator = MultimodalCollator(processor, max_length=config.max_length)

    # Create training arguments
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Load data
    data = load_training_data(
        labeled_path=config.labeled_path,
        instruction_path=config.instruction_path,
    )
    dataset = MMDataset(data, processor, config.max_length)

    # Create trainer
    trainer = SFTTrainer(
        model=student_model,
        data_collator=collator,
        processing_class=processor.tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    return trainer


def create_trainer_from_checkpoint(
    config: Any,
    checkpoint_path: str,
) -> SFTTrainer:
    """Create trainer and resume from checkpoint.

    Args:
        config: TrainingConfig or path to config file.
        checkpoint_path: Path to checkpoint directory.

    Returns:
        Configured trainer ready to resume training.
    """
    from ..config import TrainingConfig

    if isinstance(config, str):
        config = TrainingConfig.from_json(config)
    elif isinstance(config, dict):
        config = TrainingConfig.from_dict(config)

    # Determine training mode from config
    if config.logits_path:
        trainer = create_white_box_trainer(config)
    elif config.lora_enable:
        trainer = create_lora_trainer(config)
    else:
        trainer = create_black_box_trainer(config)

    # Load checkpoint
    trainer.model.load_adapter(checkpoint_path, is_trainable=True)

    # Load trainer state
    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)
        trainer.state.global_step = trainer_state.get("global_step", 0)

    return trainer