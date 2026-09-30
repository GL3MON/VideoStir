# Advanced Usage Guide

This guide covers advanced VideoStir features and customization options.

## Checkpointing Configuration

### Per-Stage Checkpoint Control

You can control which stages are checkpointed individually:

```python
from videostir.inference import PipelineConfig

config = PipelineConfig(
    video_path="video.mp4",
    query="query",
    output_dir="results",
    
    # Enable/disable checkpointing
    enable_checkpoint=True,           # Master switch for checkpointing
    
    # Per-stage control (default: True for all)
    checkpoint_segments=True,         # Save segment information
    checkpoint_embeddings=True,       # Save visual embeddings
    checkpoint_graph=True,            # Save graph structure
    checkpoint_retrieval=True,        # Save retrieval results
    checkpoint_intent=True,           # Save intent analysis
    checkpoint_rerank=True,           # Save rerank results
)
```

### Custom Checkpoint Directories

```python
from videostir.inference import PipelineConfig

config = PipelineConfig(
    video_path="video.mp4",
    query="query",
    output_dir="results",
    checkpoint_dir="my_checkpoints",  # Custom checkpoint directory
    cache_dir="my_cache",             # Custom cache directory
)
```

### Working with CheckpointManager

```python
from videostir.inference.checkpoint import CheckpointManager

# Initialize
checkpoint_manager = CheckpointManager(
    output_dir="results",
    checkpoint_dir="checkpoints",
    cache_dir="cache"
)

# Check if a stage completed
if checkpoint_manager.has_checkpoint("segments"):
    print("Segments already computed")

# Save custom data
checkpoint_manager.save("my_data", {"key": "value"})

# Load custom data
data = checkpoint_manager.load("my_data")

# Delete specific checkpoint
checkpoint_manager.delete_checkpoint("segments")

# Clear all checkpoints
checkpoint_manager.clear_all()

# List all checkpoints
checkpoints = checkpoint_manager.list_checkpoints()
for cp in checkpoints:
    print(f"{cp['name']}: {cp['path']}")
```

## Configuration Options

### Complete PipelineConfig Reference

```python
from videostir.inference import PipelineConfig

config = PipelineConfig(
    # Required
    video_path="video.mp4",
    query="What happens in this video?",
    output_dir="results",
    
    # Frame sampling
    frame_interval=30,              # Sample every N frames for segmentation
    embedding_frame_interval=10,    # Sample every N frames for embeddings
    rerank_frame_interval=5,        # Sample every N frames for reranking
    time_sampling_interval=10,      # Sample every N frames for time focus
    
    # Segmentation
    n_clusters=10,                  # Number of K-means clusters
    min_segment_sec=0.4,            # Minimum segment duration (seconds)
    
    # Retrieval
    top_k=3,                        # Top-k segments to retrieve
    spatial_k=3,                    # Spatial neighbors to expand
    attribute_top_k=3,              # Subtitle matches to consider
    temporal_weight=1.0,            # Weight for temporal edges
    
    # Reranking
    top_frames=128,                 # Max frames to return
    min_frames_per_clip=6,          # Min frames kept per clip
    
    # Time-focused retrieval
    time_focus_ratio=0.05,          # Video ratio for time search
    time_range_padding=1.0,         # Padding for time windows
    time_min_window=2.0,            # Minimum time window (seconds)
    
    # Advanced
    short_video_threshold=240.0,    # Short video threshold (seconds)
    subtitle_json=None,             # Optional subtitle JSON path
    correct_choice=None,            # For evaluation (ground truth)
    
    # Model settings
    intent_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    reranker_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    reranker_adapter_dir="./result",
)
```

## Customizing Retrieval Behavior

### Adjusting Top-K Retrieval

```python
from videostir.inference import PipelineConfig, run_pipeline

# Retrieve more segments for better coverage
config = PipelineConfig(
    video_path="video.mp4",
    query="find relevant scenes",
    output_dir="output",
    top_k=10,       # Retrieve top 10 segments instead of 3
    spatial_k=5,    # Expand to 5 spatial neighbors each
)

result = run_pipeline(config)
```

### Controlling Subtitle Search

```python
from videostir.inference import PipelineConfig, run_pipeline

# Control subtitle expansion
config = PipelineConfig(
    video_path="video.mp4",
    query="Find scenes with 'hello world'",
    output_dir="output",
    subtitle_neighbor_hops=5,  # Expand to 5 temporal hops for subtitle matches
    attribute_top_k=10,        # Consider top 10 subtitle matches
)

result = run_pipeline(config)
```

### Time-Focused Search Control

```python
from videostir.inference import PipelineConfig, run_pipeline

# Adjust time-focused search parameters
config = PipelineConfig(
    video_path="video.mp4",
    query="What happens in the first 30 seconds?",
    output_dir="output",
    time_focus_ratio=0.1,        # Look at first 10% of video
    time_range_padding=2.0,      # Add 2 seconds padding
    time_min_window=5.0,         # Minimum 5 second window
)

result = run_pipeline(config)
```

## Intent Analysis Control

### Analyze Query Intent Separately

```python
from videostir.inference import IntentAnalyzer

analyzer = IntentAnalyzer()

# Analyze query
result = analyzer.analyze("What happens at the beginning?")

print(f"Subtitle search: {result['subtitle_search']}")
print(f"Time search: {result['time_search']}")
print(f"Reason: {result['reason']}")

# Analyze time focus
time_result = analyzer.analyze_time_focus("Show me the ending")
print(f"Time mode: {time_result['mode']}")
print(f"Start: {time_result['start_time_sec']}, End: {time_result['end_time_sec']}")

# Extract subtitles
subtitle_result = analyzer.extract_subtitles("When the phrase 'hello' appears")
print(f"Subtitle: {subtitle_result['subtitle_text']}")
print(f"Cleaned query: {subtitle_result['cleaned_query']}")

# Cleanup
analyzer.unload()
```

### Batch Intent Analysis

```python
from videostir.inference import IntentAnalyzer

queries = [
    "What happens at the start?",
    "Find scenes with dialogue",
    "Show me the ending",
]

analyzer = IntentAnalyzer(keep_model_loaded=True)  # Keep model loaded for speed

for query in queries:
    intent = analyzer.analyze(query, keep_model_loaded=True)
    print(f"Query: {query}")
    print(f"  Intent: subtitle={intent['subtitle_search']}, time={intent['time_search']}")

analyzer.unload()  # Cleanup when done
```

## Reranking Control

### Adjusting Reranking Parameters

```python
from videostir.inference import PipelineConfig, run_pipeline

# Custom reranking settings
config = PipelineConfig(
    video_path="video.mp4",
    query="Find relevant frames",
    output_dir="output",
    rerank_frame_interval=10,   # Sample every 10 frames for scoring
    top_frames=64,              # Return top 64 frames
    min_frames_per_clip=4,      # Keep at least 4 frames per clip
)

result = run_pipeline(config)
```

### Using FrameReranker Directly

```python
from videostir.inference import FrameReranker
import json

# Load precomputed segments
with open("segments.json") as f:
    segments = json.load(f)

reranker = FrameReranker()

# Rerank segments
frames = reranker.rerank(
    segment_infos=segments,
    query="What is happening?",
    frame_interval=5,
    top_frames=128,
    output_dir="reranked_frames",
)

for frame in frames[:5]:
    print(f"Frame {frame['rank']}: {frame['output_path']} (score: {frame['score']:.2f})")

reranker.unload()
```

## Video Segmentation Control

### Custom Segmentation

```python
from videostir.inference.preprocessing import VideoSegmenter

segmenter = VideoSegmenter(
    frame_interval=30,    # Extract features every 30 frames
    n_clusters=15,        # Create 15 clusters
    min_segment_sec=2.0,  # Minimum 2 second segments
    output_dir="custom_segments",
)

# Segment video
segments = segmenter.segment_video("video.mp4")

# Work with segments
for seg in segments:
    print(f"Segment {seg['segment_index']}: "
          f"{seg['start_sec']:.2f}s - {seg['end_sec']:.2f}s")
```

## Graph-Based Retrieval

### Building and Querying Graph

```python
from videostir.inference.models import VideoEmbedder
from videostir.inference.retrieval import SpatiotemporalGraphBuilder, VideoRetriever

# Compute features
embedder = VideoEmbedder()
segment_infos = [{"path": "segment1.mp4"}, {"path": "segment2.mp4"}]
features = embedder.compute_video_features(segment_infos, frame_interval=10)

# Build graph
builder = SpatiotemporalGraphBuilder(temporal_weight=1.0)
graph = builder.build(features)

# Query graph
retriever = VideoRetriever()
results = retriever.retrieve(graph, "query", top_k=5, spatial_k=3)

# Access results
for result in results:
    print(f"Node {result['node_id']}: similarity = {result['similarity']:.3f}")
```

## Advanced CLI Usage

### Customizing the CLI

```bash
# Run with custom parameters
python -m videostir.inference \
    --video video.mp4 \
    --query "What happens?" \
    --output results/ \
    --frame-interval 20 \
    --clusters 20 \
    --top-k 5 \
    --top-frames 64

# Batch with custom parameters
python -m videostir.inference \
    --batch-config samples.json \
    --output results/ \
    --top-frames 32 \
    --force  # Process even existing results
```

## Error Handling

### Graceful Error Handling

```python
from videostir.inference import PipelineConfig, run_pipeline
import traceback

try:
    config = PipelineConfig(
        video_path="video.mp4",
        query="query",
        output_dir="output"
    )
    result = run_pipeline(config)
except Exception as e:
    print(f"Pipeline failed: {e}")
    print(traceback.format_exc())
```

### Skipping Failed Videos in Batch

```python
from videostir.inference import run_batch_from_config

results = run_batch_from_config(
    config_path="batch.json",
    output_root="results",
    skip_existing=True,  # Skip already processed
)

# Check for failures
for video_id, result in results.items():
    if result.get("skipped"):
        print(f"Skipped {video_id}: {result.get('error', 'unknown')}")
```

## Performance Optimization

### Faster Processing for Large Videos

```python
from videostir.inference import PipelineConfig, run_pipeline

# Optimize for speed (less accurate but faster)
config = PipelineConfig(
    video_path="large_video.mp4",
    query="query",
    output_dir="output",
    frame_interval=60,        # Less frequent sampling
    embedding_frame_interval=30,
    n_clusters=5,             # Fewer segments
    top_k=1,                  # Retrieve fewer segments
    top_frames=32,            # Fewer frames
    time_sampling_interval=20,
)
```

### Memory-Efficient Processing

```python
from videostir.inference import PipelineConfig, run_pipeline

# Optimize for memory (slower but less VRAM)
config = PipelineConfig(
    video_path="video.mp4",
    query="query",
    output_dir="output",
    top_frames=16,            # Very few frames
    min_frames_per_clip=2,    # Minimal per clip
)
```

## Integration with Other Libraries

### Integrating with LLaVA

```python
from videostir.inference import simple_rag
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images

# Get frames from VideoStir
frames = simple_rag("video.mp4", "query", top_frames=16)

# Use frames with LLaVA
model_path = "lmms-lab/LLaVA-Video-7B-Qwen2"
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, None, "llava_qwen", torch_dtype="bfloat16", device_map="auto"
)

# Build prompt with frames
frame_paths = [f['output_path'] for f in frames[:8]]  # Use top 8 frames
prompt = "Analyze these frames and answer the question."
```

### Custom Video Processing Pipeline

```python
from videostir.inference.preprocessing import VideoSegmenter
from videostir.inference.models import VideoEmbedder
from videostir.inference.retrieval import SpatiotemporalGraphBuilder, VideoRetriever
import cv2

# Custom frame extraction
def custom_frame_extraction(video_path, interval):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

# Custom segment creation
def create_segments_from_frames(frames, fps):
    segments = []
    for i, frame in enumerate(frames):
        segments.append({
            "path": f"frame_{i}.jpg",
            "frame_index": i,
            "timestamp": i / fps,
        })
    return segments
```