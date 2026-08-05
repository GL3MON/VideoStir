# [ACL 2026] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG

![Framework](/Figure/framework.png)

**Paper**: [ArXiv Link](https://arxiv.org/pdf/2604.05418)

VideoStir is a structured and intent-aware long-video RAG framework that:
- Structures videos as spatio-temporal graphs at clip level
- Performs multi-hop retrieval to aggregate evidence across distant events
- Uses MLLM-backed intent-relevance scoring for frame retrieval

## Quick Start

### Installation

```bash
cd VideoStir/inference
pip install -r requirements_inference.txt
```

### Basic Usage

```python
from videostir import simple_rag

# Run retrieval on a video
frames = simple_rag(
    video_path="my_video.mp4",
    query="Show me the opening scene with the car"
)

# Print results
for frame in frames[:5]:
    print(f"Frame {frame['rank']}: {frame['output_path']} (score: {frame['score']:.2f})")
```


### Advanced Usage

```python
from videostir import PipelineConfig, run_pipeline

# Configure with custom parameters
config = PipelineConfig(
    video_path="my_video.mp4",
    query="What happens at the end?",
    output_dir="results",
    top_frames=64,        # fewer frames = faster
    frame_interval=10,    # finer sampling
)

result = run_pipeline(config)
print(f"Found {len(result.reranked_frames)} relevant frames")
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (recommended)

### Install Dependencies

```bash
cd VideoStir/inference
pip install -r requirements_inference.txt
```

### Download Checkpoints

1. **Intent Analysis Model**: Automatically downloaded by Hugging Face on first use
2. **Frame Reranker**: Download from [Google Drive](https://drive.google.com/file/d/1CUC1i7zstZktWDp30Pts4s8E7QULDgns/view?usp=drive_link)
   - Extract to `./result/` directory

## Full Pipeline Examples

### Single Video

```python
from videostir import simple_rag

# Simple one-line usage
frames = simple_rag("video.mp4", "Show me the car scene")

# With custom options
frames = simple_rag(
    "video.mp4",
    "What happens at the end?",
    output_dir="./my_results",
    top_frames=32  # Reduce for speed
)
```

### Batch Processing

Create a JSON configuration file:

```json
[
  {
    "id": "video1",
    "video_path": "videos/cat_video.mp4",
    "query": "What does the cat do?"
  },
  {
    "id": "video2",
    "video_path": "videos/dog_video.mp4",
    "query": "Where is the dog running?"
  }
]
```

Run batch processing:

```python
from videostir import run_batch_from_config

results = run_batch_from_config(
    config_path="batch_config.json",
    output_root="batch_results"
)
```

### Command Line Interface

```bash
# Single video
python -m inference \
    --video video.mp4 \
    --query "What happens?" \
    --output results/

# With options
python -m inference \
    --video video.mp4 \
    --query "What happens?" \
    --output results/ \
    --top-frames 64 \
    --frame-interval 20

# Batch mode
python -m inference \
    --batch-config samples.json \
    --output results/
```

## Pipeline Architecture

VideoStir processes videos through these stages:

1. **Segmentation**: Video is split into segments using K-means clustering
2. **Graph Construction**: Spatiotemporal graph built from segment features
3. **Intent Analysis**: Query analyzed to determine retrieval strategy
4. **Multi-hop Retrieval**: Top-k segments retrieved with spatial/temporal expansion
5. **Frame Reranking**: Frames scored and ranked by relevance

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting_started.md) | Installation and first run |
| [Basic Usage](docs/basic_usage.md) | Common workflows and examples |
| [Advanced Usage](docs/advanced_usage.md) | Customization and advanced features |

## Output Format

The pipeline produces several files:

| File | Description |
|------|-------------|
| `rerank_results.json` | Final ranked frames with scores |
| `time_focus_frames.json` | Time-focused frames (if applicable) |
| `retrieval_plan.json` | Detailed retrieval process info |

### Result Structure

```json
{
  "rank": 1,
  "score": 4.5,
  "output_path": "output/t00h01m30s_seg0001_frame00123.jpg",
  "timestamp": 90.5,
  "segment_index": 1
}
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `simple_rag(video_path, query, output_dir)` | One-line video retrieval |
| `run_pipeline(config)` | Full pipeline with custom config |
| `run_batch_from_config(config_path, output_root)` | Batch processing |

### Configuration

| Class | Description |
|-------|-------------|
| `PipelineConfig` | Pipeline parameters |
| `PipelineResult` | Pipeline output |

### Models

| Class | Description |
|-------|-------------|
| `IntentAnalyzer` | Query intent analysis |
| `FrameReranker` | Frame scoring and reranking |
| `VideoEmbedder` | Visual feature computation |

## Training

For training the intent-relevance scorer:

```bash
cd ../train
pip install -r requirements_train.txt
python train_lora.py --config mmkd_black_box_lora_single.json
```

See [Training Guide](../train/README.md) for details.

## Citation

```bibtex
@article{fu2026videostir,
  title={VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG},
  author={Fu, Honghao and Xu, Miao and Wang, Yiwei and Zhang, Dailing and Liu, Jun and Cai, Yujun},
  journal={arXiv preprint arXiv:2604.05418},
  year={2026}
}
```

## License

This project is licensed under the Apache 2.0 License. See LICENSE for details.