# Basic Usage Guide

This guide covers common VideoStir workflows with step-by-step examples.

## Quick Reference

| Task | Code |
|------|------|
| Simple retrieval | `simple_rag("video.mp4", "query")` |
| Full pipeline | `run_pipeline(PipelineConfig(...))` |
| Batch processing | `run_batch_from_config("config.json", "output/")` |

## 1. Single Video Retrieval

### Basic Example

```python
from videostir.inference import simple_rag

# Run retrieval
frames = simple_rag(
    video_path="path/to/video.mp4",
    query="Show me the person wearing a red shirt"
)

# Process results
for frame in frames[:10]:
    print(f"Score: {frame['score']:.2f} | Path: {frame['output_path']}")
```

### With Custom Output Directory

```python
from videostir.inference import simple_rag

frames = simple_rag(
    video_path="my_video.mp4",
    query="What happens at the end?",
    output_dir="./my_results",
    top_frames=64
)
```

## 2. Query Types

### Time-Focused Queries

VideoStir automatically detects time-focused queries:

```python
from videostir.inference import simple_rag

# Queries about beginning/end will trigger time-focused retrieval
frames = simple_rag(
    video_path="video.mp4",
    query="What happens in the opening scene?"
)

# Time-focused frames will be available
print(f"Time-focused frames: {len(result.time_focus_frames)}")
```

### Subtitle-Based Queries

Subtitle queries trigger enhanced subtitle search:

```python
from videostir.inference import simple_rag

# Queries mentioning subtitles will enable subtitle search
frames = simple_rag(
    video_path="video.mp4",
    query="When the character says 'hello world', what do they look like?"
)

# Subtitle analysis is available
print(f"Subtitle text: {result.retrieval_plan.get('subtitle_analysis', {}).get('subtitle_text')}")
```

## 3. Batch Processing

Process multiple videos from a JSON configuration:

### Create Batch Configuration

```json
[
  {
    "id": "video1",
    "video_path": "videos/cat_video.mp4",
    "query": "What does the cat do?",
    "subtitle_path": "subtitles/cat_video_en.json"
  },
  {
    "id": "video2",
    "video_path": "videos/dog_video.mp4",
    "query": "Where is the dog running?",
    "subtitle_path": "subtitles/dog_video_en.json"
  }
]
```

### Run Batch Processing

```python
from videostir.inference import run_batch_from_config

results = run_batch_from_config(
    config_path="batch_config.json",
    output_root="batch_results",
    video_root="videos",
    subtitle_root="subtitles",
    skip_existing=True  # Skip already-processed videos
)

# Results are saved per-video
# batch_results/video1/...
# batch_results/video2/...
```

### Access Batch Results

```python
# Results dictionary
for video_id, result in results.items():
    print(f"Video {video_id}: {len(result['reranked_frames'])} frames")

# Or load batch summary
import json
with open("batch_results/batch_results.json") as f:
    summary = json.load(f)
```

## 4. Working with Results

### Save Results to File

```python
import json
from videostir.inference import run_pipeline, PipelineConfig

config = PipelineConfig("video.mp4", "query", "output/")
result = run_pipeline(config)

# Save as JSON
with open("results.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)
```

### Load and Use Saved Results

```python
import json

# Load saved results
with open("results.json") as f:
    data = json.load(f)

# Access reranked frames
frames = data["reranked_frames"]
for frame in frames[:5]:
    print(f"Rank {frame['rank']}: {frame['output_path']}")
```

### Get Top-K Frames

```python
from videostir.inference import simple_rag

frames = simple_rag("video.mp4", "query", top_frames=10)

# Get top 5 frames
top_frames = frames[:5]

# Filter by minimum score
high_confidence = [f for f in frames if f['score'] >= 4.0]
```

## 5. Video Metadata

Access video information from the result:

```python
from videostir.inference import run_pipeline, PipelineConfig

config = PipelineConfig("video.mp4", "query", "output/")
result = run_pipeline(config)

# Video metadata
meta = result.video_metadata
print(f"FPS: {meta['fps']}")
print(f"Duration: {meta['duration_sec']} seconds")
print(f"Total frames: {meta['total_frames']}")

# Intent analysis
intent = result.intent
print(f"Subtitle search: {intent['subtitle_search']}")
print(f"Time search: {intent['time_search']}")
print(f"Reason: {intent['reason']}")
```

## 6. Short Video Mode

Automatically handled for videos under 240 seconds. For shorter videos:

```python
from videostir.inference import PipelineConfig, run_pipeline

# Force short video mode threshold
config = PipelineConfig(
    video_path="short.mp4",
    query="query",
    output_dir="output",
    short_video_threshold=60.0  # Treat videos <60s as short
)

result = run_pipeline(config)
# Graph retrieval is skipped, direct frame sampling used instead
```

## 7. Checkpointing and Resumable Pipelines

VideoStir supports checkpointing to enable resumable pipeline execution:

```python
from videostir.inference import PipelineConfig, run_pipeline

# Checkpointing is enabled by default
config = PipelineConfig(
    video_path="video.mp4",
    query="query",
    output_dir="results",
    enable_checkpoint=True,  # Default
)

result = run_pipeline(config)

# Later, to resume:
from videostir.inference.checkpoint import CheckpointManager
checkpoint_manager = CheckpointManager("results")

# Check available checkpoints
print(checkpoint_manager.list_checkpoints())

# Load from checkpoint if pipeline was interrupted
if checkpoint_manager.has_checkpoint("segment_features"):
    segment_features = checkpoint_manager.load_segment_features()
```

## 8. Command Line Interface

The CLI provides an alternative to Python scripting:

```bash
# Basic usage
python -m videostir.inference \
    --video my_video.mp4 \
    --query "Show me the car" \
    --output results/

# With options
python -m videostir.inference \
    --video video.mp4 \
    --query "What happens?" \
    --output results/ \
    --top-frames 64 \
    --frame-interval 20

# Batch mode
python -m videostir.inference \
    --batch-config samples.json \
    --output results/
```

See `python -m videostir.inference --help` for all options.