# Advanced Training Guide

This guide covers advanced VideoStir training features and customization options.

## Configuration Options

### Complete TrainingConfig Reference

```python
from videostir.train import TrainingConfig

config = TrainingConfig(
    # Job type
    job_type="mmkd_black_box",  # "mmkd_black_box", "mmkd_white_box", "lora_only"
    
    # Dataset settings
    instruction_path=None,       # Optional instruction data
    labeled_path="./train.json", # Labeled training data
    logits_path=None,            # Teacher logits (white-box only)
    seed=42,
    
    # Model settings
    teacher_model_id="Qwen/Qwen2.5-VL-72B-Instruct",
    student_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    
    # Training settings (SFTConfig compatible)
    output_dir="./result/",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_length=512,
    save_steps=1500,
    logging_steps=1,
    learning_rate=2e-5,
    weight_decay=0.05,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    bf16=True,
    
    # LoRA settings
    lora_enable=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_bias="none",
    lora_target_modules=["q_proj", "v_proj"],
    
    # Distillation settings
    kd_ratio=0.1,
    max_seq_length=512,
    distillation_type="forward_kld",  # "forward_kld", "reverse_kld"
    
    # Checkpoint settings
    enable_checkpoint=True,
    checkpoint_dir="checkpoints",
    resume_from_checkpoint=None,
    save_total_limit=2,
)
```

## Customizing Training Behavior

### Adjusting Learning Rate Schedule

```python
from videostir.train import TrainingConfig, train_lora

config = TrainingConfig(
    job_type="lora_only",
    labeled_path="./train.json",
    output_dir="./result/",
    learning_rate=1e-4,        # Higher LR for faster convergence
    warmup_ratio=0.1,          # Longer warmup
    lr_scheduler_type="cosine_with_restarts",  # Cosine with restarts
)

result = train_lora(config)
```

### Modifying LoRA Configuration

```python
config = TrainingConfig(
    job_type="lora_only",
    labeled_path="./train.json",
    output_dir="./result/",
    lora_enable=True,
    lora_r=64,                 # Higher rank for more capacity
    lora_alpha=64,             # Scale with rank
    lora_dropout=0.1,          # More dropout for regularization
    lora_target_modules=[
        "q_proj", "v_proj",
        "k_proj", "o_proj",    # Add more modules
        "gate_proj", "up_proj", "down_proj",  # MLP modules
    ],
)
```

### White-Box Distillation Parameters

```python
config = TrainingConfig(
    job_type="mmkd_white_box",
    labeled_path="./train.json",
    logits_path="./train_logits.json",
    output_dir="./result/",
    kd_ratio=0.3,              # Higher distillation weight
    distillation_type="reverse_kld",  # Teacher certainty to student
    max_seq_length=1024,       # Longer sequences
)
```

## Checkpoint Management

### Per-Stage Checkpoint Control

```python
config = TrainingConfig(
    job_type="lora_only",
    output_dir="./result/",
    enable_checkpoint=True,
    save_total_limit=3,  # Keep only 3 checkpoints
)

# Access checkpoint manager
from videostir.train import TrainingCheckpointManager

checkpoint_manager = TrainingCheckpointManager("./result/")
checkpoint_manager.cleanup_old_checkpoints(max_checkpoints=3)
```

### Manual Checkpoint Control

```python
from videostir.train import TrainingCheckpointManager, train_lora
from transformers import SFTTrainer

# Custom training loop with checkpoint control
config = TrainingConfig.from_json("config.json")

# Create trainer
trainer = train_lora(config)

# Manual checkpoint save
checkpoint_manager = TrainingCheckpointManager(config.output_dir)
checkpoint_path = checkpoint_manager.save_checkpoint(trainer, step=trainer.state.global_step)
print(f"Saved checkpoint to: {checkpoint_path}")

# Save state only (no model weights)
checkpoint_manager.save_trainer_state(trainer)
```

## Multi-GPU Training

### Using DeepSpeed

Create a DeepSpeed config file (`ds_config.json`):

```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 1e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 1e8,
    "contiguous_gradients": true
  },
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "train_batch_size": 16,
  "gradient_accumulation_steps": 8
}
```

Run with DeepSpeed:

```bash
deepspeed --num_gpus=4 train_lora.py --config config.json
```

### Data Parallel Training

```python
import torch.distributed as dist
from videostir.train import TrainingConfig, train_lora

# Initialize distributed training
dist.init_process_group("nccl")

config = TrainingConfig.from_json("config.json")

# Adjust batch size for multiple GPUs
config.per_device_train_batch_size = 4  # Per GPU
config.gradient_accumulation_steps = 8

result = train_lora(config)
dist.destroy_process_group()
```

## Debugging and Validation

### Logging Custom Metrics

```python
from videostir.train import TrainingConfig
from transformers import SFTConfig

config = TrainingConfig.from_json("config.json")

# Access training args for customization
training_args = SFTConfig(
    output_dir=config.output_dir,
    logging_steps=config.logging_steps,
    # Add custom metrics
    report_to=["tensorboard", "wandb"],  # MLflow, wandb
)
```

### Validate Checkpoint

```python
from videostir.train import TrainingCheckpointManager

checkpoint_manager = TrainingCheckpointManager("./result/")

# Check available checkpoints
print("Available checkpoints:")
for item in checkpoint_manager.list_checkpoints():
    print(f"  - {item['name']}: {item['path']}")

# Verify latest checkpoint
latest = checkpoint_manager.get_latest_checkpoint()
if latest:
    print(f"Latest checkpoint: {latest}")
```

## Performance Optimization

### Faster Training for Large Datasets

```python
config = TrainingConfig(
    job_type="lora_only",
    labeled_path="./train.json",
    output_dir="./result/",
    max_length=256,            # Shorter sequences
    per_device_train_batch_size=8,  # Larger batch
    gradient_accumulation_steps=4,
    logging_steps=10,          # Less frequent logging
    save_steps=500,            # Less frequent saves
)
```

### Memory-Efficient Training

```python
config = TrainingConfig(
    job_type="lora_only",
    labeled_path="./train.json",
    output_dir="./result/",
    lora_r=8,                  # Smaller LoRA rank
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    max_length=512,
)

# Also enable gradient checkpointing in training args
training_args = SFTConfig(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

## Integration with Other Tools

### Weights & Biases

```python
from videostir.train import TrainingConfig
from transformers import SFTConfig
import wandb

# Initialize wandb
wandb.init(project="videostir-training")

config = TrainingConfig.from_json("config.json")

# Update training args
training_args = SFTConfig(
    output_dir=config.output_dir,
    report_to=["wandb"],
    logging_steps=config.logging_steps,
)

# Train with logging
result = train_lora(config)

wandb.finish()
```

### MLflow

```python
from videostir.train import TrainingConfig
from transformers import SFTConfig
import mlflow

mlflow.set_experiment("videostir-training")

with mlflow.start_run():
    config = TrainingConfig.from_json("config.json")
    
    # Log config
    mlflow.log_params(config.to_dict())
    
    result = train_lora(config)
    
    # Log metrics
    mlflow.log_metric("training_steps", result.training_steps)
    mlflow.log_metric("final_loss", result.final_loss)
```

## Customizing Data Loading

### Custom Dataset

```python
from videostir.train.models.data import MMDataset
from transformers import Qwen2_5_VLProcessor

# Custom data loading
def load_custom_data():
    # Your data loading logic
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video"},
                {"type": "image", "image": "path/to/frame.jpg"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The video shows..."}],
        },
    ]

processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
dataset = MMDataset(load_custom_data(), processor)
```

### Custom Collator

```python
from videostir.train.models.data import MultimodalCollator
from transformers import Qwen2_5_VLProcessor

class CustomCollator(MultimodalCollator):
    def __call__(self, examples):
        batch = super().__call__(examples)
        # Add custom preprocessing
        batch["custom_mask"] = self._create_custom_mask(batch)
        return batch

    def _create_custom_mask(self, batch):
        # Custom mask logic
        return torch.ones_like(batch["input_ids"])

collator = CustomCollator(processor)
```