# Getting Started with VideoStir Training

This guide will help you install and run training for VideoStir models.

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (recommended for faster training)
- At least 24GB VRAM for training Qwen2.5-VL-3B

### Install Dependencies

```bash
cd VideoStir/train

# Install training dependencies
pip install -r requirements_train.txt
```

Required packages include:
- `transformers` - Hugging Face model library
- `trl` - Transformers Reinforcement Learning (SFTTrainer)
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `datasets` - Hugging Face datasets
- `qwen-vl-utils` - Qwen Vision-Language model utilities
- `deepspeed` - Multi-GPU training (optional)

### Download Models and Checkpoints

1. **Student Model** (Qwen2.5-VL-3B-Instruct):
   - Automatically downloaded on first use by Hugging Face

2. **Teacher Model** (Qwen2.5-VL-72B-Instruct):
   - Automatically downloaded on first use by Hugging Face
   - Alternatively, download from Hugging Face Hub

3. **Pretrained Weights** (Intent-Relevance Scorer):
   - Download from: [Google Drive Link](https://drive.google.com/file/d/1CUC1i7zstZktWDp30Pts4s8E7QULDgns/view?usp=drive_link)
   - Extract to `./result/` directory

## Quick Start

### Training with JSON Config

Create a config JSON file:

```json
{
  "job_type": "mmkd_black_box",
  "dataset": {
    "labeled_path": "./train.json",
    "logits_path": "./train_logits.json"
  },
  "models": {
    "teacher": "Qwen/Qwen2.5-VL-72B-Instruct",
    "student": "Qwen/Qwen2.5-VL-3B-Instruct"
  },
  "training": {
    "output_dir": "./result/",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5
  },
  "lora": {
    "enable": true,
    "r": 16,
    "alpha": 32
  }
}
```

Run training:

```bash
python train_lora.py --config mmkd_black_box_lora_single.json
```

### Programmatic Training

```python
from videostir.train import TrainingConfig, train_black_box

# Create config
config = TrainingConfig(
    job_type="mmkd_black_box",
    student_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    teacher_model_id="Qwen/Qwen2.5-VL-72B-Instruct",
    labeled_path="./train.json",
    output_dir="./result/",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    lora_enable=True,
    lora_r=16,
    lora_alpha=32,
)

# Train
result = train_black_box(config)
print(f"Training completed: {result.training_steps} steps")
```

## Training Modes

### 1. Black-Box Distillation

Student learns from teacher outputs only:

```python
config = TrainingConfig(
    job_type="mmkd_black_box",
    student_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    teacher_model_id="Qwen/Qwen2.5-VL-72B-Instruct",
    labeled_path="./train.json",
    output_dir="./result/",
)

result = train_black_box(config)
```

### 2. White-Box Distillation

Student learns from teacher outputs AND intermediate features:

```python
config = TrainingConfig(
    job_type="mmkd_white_box",
    student_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    teacher_model_id="Qwen/Qwen2.5-VL-72B-Instruct",
    labeled_path="./train.json",
    logits_path="./train_logits.json",  # Teacher logits required
    kd_ratio=0.1,  # Distillation loss weight
    distillation_type="forward_kld",
    output_dir="./result/",
)

result = train_white_box(config)
```

### 3. LoRA Fine-Tuning Only

Standard LoRA without distillation:

```python
config = TrainingConfig(
    job_type="lora_only",
    student_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    labeled_path="./train.json",
    output_dir="./result/",
    lora_enable=True,
)

result = train_lora(config)
```

## Checkpointing and Resume

Checkpoints are saved automatically to `output_dir/checkpoints/`:

```bash
# Resume from latest checkpoint
python train_lora.py --config config.json --resume_from latest

# Resume from specific checkpoint
python train_lora.py --config config.json --resume_from result/checkpoints/checkpoint-1500
```

Or programmatically:

```python
from videostir.train import TrainingConfig, resume_training

config = TrainingConfig.from_json("config.json")
result = resume_training(config, "result/checkpoints/checkpoint-1500")
```

## Configuration Options

### Complete TrainingConfig

```python
config = TrainingConfig(
    # Job type
    job_type="mmkd_black_box",  # or "mmkd_white_box", "lora_only"
    
    # Dataset
    labeled_path="./train.json",
    logits_path="./train_logits.json",  # For white-box
    seed=42,
    
    # Models
    teacher_model_id="Qwen/Qwen2.5-VL-72B-Instruct",
    student_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    
    # Training
    output_dir="./result/",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_length=512,
    learning_rate=2e-5,
    weight_decay=0.05,
    
    # LoRA
    lora_enable=True,
    lora_r=16,
    lora_alpha=32,
    
    # Distillation
    kd_ratio=0.1,
    distillation_type="forward_kld",
    
    # Checkpointing
    enable_checkpoint=True,
    checkpoint_dir="checkpoints",
)
```

## Monitoring Training

Training logs are saved to `output_dir/`:

| File | Description |
|------|-------------|
| `training_summary.json` | Summary of completed training |
| `checkpoint-XXX/` | Model checkpoints |
| `logs/` | Training logs |

### TensorBoard

```bash
tensorboard --logdir ./result/
```

## Troubleshooting

### Out of Memory

Reduce batch size:

```python
config = TrainingConfig(
    per_device_train_batch_size=1,  # Smaller batch
    gradient_accumulation_steps=16,  # Increase accum steps
)
```

Enable gradient checkpointing:

```python
training_args = SFTConfig(
    gradient_checkpointing=True,
)
```

### Invalid Checkpoint

Clear checkpoints and restart:

```python
import shutil
shutil.rmtree("result/checkpoints", ignore_errors=True)
```