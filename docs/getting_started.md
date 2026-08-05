# Getting Started with VideoStir

This guide will help you install and run your first video retrieval with VideoStir.

## Checkpointing and Resumable Pipelines

VideoStir supports checkpointing to enable resumable pipeline execution. This is useful for long-running pipelines that may need to be interrupted and resumed later.

### Enabling Checkpointing

By default, checkpointing is **enabled**. The pipeline will save intermediate results to the `checkpoints/` and `cache/` subdirectories in your output directory.

```python
from videostir import PipelineConfig, run_pipeline

config = PipelineConfig(
    video_path="my_video.mp4",
    query="Show me the opening scene",
    output_dir="results",
    enable_checkpoint=True,  # Default: True
)
```

### Disabling Checkpointing

To disable checkpointing (e.g., for quick tests):

```python
config = PipelineConfig(
    video_path="my_video.mp4",
    query="test",
    output_dir="results",
    enable_checkpoint=False,
)
```

### CLI Usage with Checkpointing

```bash
# Enable checkpointing (default)
python -m inference run --video video.mp4 --query "query"

# Disable checkpointing
python -m inference run --video video.mp4 --query "query" --disable-checkpoint

# Custom checkpoint directory
python -m inference run --video video.mp4 --query "query" \
    --checkpoint-dir my_checkpoints \
    --cache-dir my_cache
```

### Resume from Checkpoint

To resume a pipeline from the last completed stage:

```bash
# Check the checkpoints first
ls results/checkpoints/

# Resume - will skip completed stages
python test_medical_retrieval.py resume
```

Or programmatically:

```python
from inference.checkpoint import CheckpointManager

checkpoint_manager = CheckpointManager("results")

# Check which stages are available
print(checkpoint_manager.list_checkpoints())

# Load from checkpoint
if checkpoint_manager.has_checkpoint("segments"):
    segments = checkpoint_manager.load_segments()
if checkpoint_manager.has_checkpoint("segment_features"):
    segment_features = checkpoint_manager.load_segment_features()
```

### Available Checkpoint Stages

| Stage | Description | File |
|-------|-------------|------|
| `segments` | Video segmentation results | `checkpoints/segments.json` |
| `segment_features` | Visual embeddings for each segment | `cache/segment_features.pt` |
| `intent_analysis` | Query intent analysis | `checkpoints/intent_analysis.json` |
| `graph` | Spatiotemporal graph structure | `checkpoints/graph.json` |
| `retrieval` | Top-k segment retrieval results | `checkpoints/retrieval.json` |
| `rerank_results` | Final reranked frames | `cache/rerank_results.pt` |
| `final_results` | Complete pipeline results | `cache/final_results.pt` |

### Clearing Checkpoints

To clear all checkpoints and start fresh:

```python
from inference.checkpoint import CheckpointManager

checkpoint_manager = CheckpointManager("results")
checkpoint_manager.clear_all()  # Deletes all checkpoints
```

## Installation

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (recommended for faster inference)

### Install Dependencies

```bash
cd VideoStir/inference

# Install inference dependencies
pip install -r requirements_inference.txt
```

Required packages include:
- `transformers` - Hugging Face model library
- `qwen-vl-utils` - Qwen Vision-Language model utilities
- `networkx` - Graph operations
- `sentence-transformers` - Text embeddings
- `opencv-python` - Video processing
- `decord` - Video frame extraction
- `peft` - LoRA model adapters
- `scikit-learn` - Clustering
- `ruptures` - Change point detection
- `bitsandbytes` - Quantized model support

### Download Models and Checkpoints

1. **Intent Analysis Model** (Qwen2.5-VL-7B-Instruct):
   - Automatically downloaded on first use by Hugging Face

2. **Frame Reranker** (Qwen2.5-VL-3B-Instruct + LoRA adapter):
   - Download from: [Google Drive Link](https://drive.google.com/file/d/1CUC1i7zstZktWDp30Pts4s8E7QULDgns/view?usp=drive_link)
   - Extract to `./result/` directory

## First Run

### Simple Video Retrieval

```python
from videostir import simple_rag

# Run retrieval on a video
frames = simple_rag(
    video_path="my_video.mp4",
    query="Show me the opening scene with the car"
)

# Print top 5 results
for frame in frames[:5]:
    print(f"Frame {frame['rank']}: {frame['output_path']} (score: {frame['score']:.2f})")
```

### Using the Full Pipeline

For more control over the retrieval process:

```python
from videostir import PipelineConfig, run_pipeline

# Configure the pipeline
config = PipelineConfig(
    video_path="my_video.mp4",
    query="What happens at the end?",
    output_dir="results",
    top_frames=64,          # Reduce for faster processing
    frame_interval=10,      # Sample every 10th frame
)

# Run the pipeline
result = run_pipeline(config)

# Access results
print(f"Found {len(result.reranked_frames)} relevant frames")
for frame in result.reranked_frames[:5]:
    print(f"  {frame['rank']}: {frame['output_path']} (score: {frame['score']:.2f})")

# Time-focused frames (if applicable)
if result.time_focus_frames:
    print(f"Time-focused frames: {len(result.time_focus_frames)}")
```

## Understanding Output

The pipeline creates several output files:

| File | Description |
|------|-------------|
| `rerank_results.json` | Final ranked frames with scores |
| `time_focus_frames.json` | Time-focused frames (if time search enabled) |
| `retrieval_plan.json` | Detailed retrieval process information |

### Output Format

**rerank_results.json** contains:
```json
{
  "rank": 1,
  "score": 4.5,
  "output_path": "output/t00h01m30s_seg0001_frame00123.jpg",
  "timestamp": 90.5,
  "segment_index": 1,
  "frame_in_segment": 123
}
```

## Common Issues

### Out of Memory Error

Reduce memory usage by adjusting parameters:

```python
config = PipelineConfig(
    video_path="my_video.mp4",
    query="test",
    output_dir="results",
    top_frames=32,          # Fewer frames to return
    frame_interval=60,      # Sample less frequently
    n_clusters=5,           # Fewer segments
    top_k=1,                # Retrieve fewer segments
)
```

### Short Video Processing

Videos under 240 seconds automatically use "short video mode" which skips graph retrieval:

```python
config = PipelineConfig(
    video_path="short_video.mp4",
    query="test",
    output_dir="results",
    short_video_threshold=60.0,  # Treat videos <60s as short
)
```

## Next Steps

- See [Basic Usage](basic_usage.md) for common workflows
- See [Advanced Usage](advanced_usage.md) for customization
- Try the example notebook: `VideoStir_example.ipynb`