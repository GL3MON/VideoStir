# VideoStir Pipeline Documentation

## Overview

VideoStir is a structured and intent-aware RAG (Retrieval-Augmented Generation) framework for long-video understanding. It processes videos to retrieve relevant frames based on natural language queries using a multi-stage pipeline with visual embeddings, spatiotemporal graphs, and MLLM-based reranking.

### Key Features

- **Intent-aware retrieval**: Detects subtitle search, time-focused search, or visual search
- **Spatiotemporal graph**: Combines visual similarity with temporal context
- **Multi-hop retrieval**: Semantic + spatial + temporal expansion
- **Frame reranking**: Qwen2.5-VL + LoRA for relevance scoring
- **Checkpointing**: Resumable pipelines for long videos
- **Subtitle fusion**: Finds frames matching quoted subtitle text
- **Subtitle-aware answer generation**: Passes subtitle frames + extracted text to downstream VLM for verification
- **Smart frame limiting**: Reduces frames to stay within MLLM context limits (128 → 25 max)

---

## Quick Start

### Run the Full Pipeline

```bash
cd /hfcache/harissh/VideoStir

python -m inference run \
  --video artifacts/test_medical_video.mp4 \
  --query "Show me the medical procedure steps" \
  --output artifacts/output \
  --subtitle-json artifacts/test_medical_video.json
```

### Interactive Answer Generation (with Qwen2.5-VL-3B)

```bash
python -m inference answer \
  --video artifacts/test_medical_video.mp4 \
  --output artifacts/output \
  --top-frames 5
```

Then type queries and get MLLM answers in a loop.

---

## Pipeline Architecture

### Stage 0: INPUT & SETUP

**Inputs:**
- Video file path (e.g., `artifacts/test_medical_video.mp4`)
- User query (e.g., `"Who is the patient and what is her date of birth?"`)
- Optional: Subtitle JSON file path

**Output Directory Structure:**
```
output/
└── video_name/
    ├── segments/          # Extracted segment clips
    ├── time_focus_frames/ # Frames from time-focused regions
    ├── checkpoint/        # Checkpoint files
    ├── cache/             # Cached embeddings and features
    ├── logs/              # Pipeline logs
    ├── rerank_results.json
    ├── retrieval_plan.json
    ├── time_focus_results.json
    └── subtitle_frames/   # Frames from subtitle-matched segments (if applicable)
```

---

### Stage 1: VIDEO SEGMENTATION

**Input:** Video file  
**Output:** List of segment dictionaries

**Segment Structure:**
```json
{
  "segment_index": 0,
  "path": "path/to/segment.mp4",
  "start_frame": 0,
  "end_frame": 750,
  "start_sec": 0.0,
  "end_sec": 30.0,
  "fps": 25,
  "duration_sec": 30.0
}
```

**Algorithm:**
1. Sample frames at configurable interval (default: 30 frames)
2. Extract visual embeddings using PE-Core-G14-448 (CLIP ViT-G/14)
3. Cluster frames using K-means (default: 10 clusters)
4. Identify segment boundaries at cluster transitions
5. Merge short segments below minimum duration (default: 0.4s)
6. Export segments as separate video files

**Configuration:**
- `frame_interval`: Sample every N frames (default: 30)
- `n_clusters`: Number of K-means clusters (default: 10)
- `min_segment_sec`: Minimum segment duration (default: 0.4s)

**Key Files:**
- `inference/preprocessing/video.py`: `VideoSegmenter` class

---

### Stage 1b: SUBTITLE ATTACHMENT (Optional)

**Triggered when:** `--subtitle-json` is provided

**Process:**
1. Load subtitle entries from JSON file
2. Match subtitle timestamps to segment time ranges
3. Attach subtitle text to corresponding segments

**Result:**
- Segments get `subtitle_text` field with the text appearing in that time range
- Used later for subtitle-based retrieval when `subtitle_search=True`

**Key Files:**
- `inference/utils.py`: `_attach_subtitles()` function

---

### Stage 2: VISUAL EMBEDDINGS COMPUTATION

**Input:** Segments from Stage 1  
**Output:** Segment features with visual embeddings

**Algorithm:**
1. For each segment, sample frames at interval (default: 10)
2. Extract CLIP ViT-G/14 visual features
3. Use video encoding: process in chunks (default: 16 frames)
4. Average embeddings across all frames in segment
5. Normalize embeddings for cosine similarity

**Embedding Details:**
- Model: PE-Core-G14-448 (CLIP ViT-G/14)
- Output dimension: 1024
- Preprocessing: Resize to 448x448, normalize

**Configuration:**
- `embedding_frame_interval`: Sample every N frames (default: 10)
- `chunk_size`: Frames processed per chunk (default: 16)

**Key Files:**
- `inference/models/embedding.py`: `VideoEmbedder` class

---

### Stage 3: SPATIOTEMPORAL GRAPH BUILDING

**Input:** Segments with embeddings  
**Output:** NetworkX graph

**Graph Structure:**
```
Nodes = Video segments (with visual features)
Edges:
  - Temporal: Connects consecutive segments (i to i+1)
  - Spatial: Connects non-consecutive segments based on visual similarity
```

**Edge Weights:**
- Temporal: Fixed weight (default: 1.0)
- Spatial: Cosine similarity of embeddings

**Algorithm:**
1. Add all segments as nodes with their features
2. Add temporal edges between consecutive segments
3. Compute pairwise cosine similarity between all segment features
4. Add spatial edges for non-consecutive pairs above threshold

**Configuration:**
- `temporal_weight`: Weight for temporal edges (default: 1.0)

**Key Files:**
- `inference/retrieval/graph.py`: `SpatiotemporalGraphBuilder` class

---

### Stage 4: QUERY INTENT ANALYSIS

**Input:** User query  
**Output:** Intent analysis dictionary

**Intent Structure:**
```json
{
  "subtitle_search": true/false,
  "time_search": true/false,
  "reason": "Explanation of analysis",
  "raw_response": "Raw model response"
}
```

**Intent Types:**

| Intent | Trigger | Action |
|--------|---------|--------|
| `subtitle_search` | Query mentions quoted text or specific subtitles | Extract subtitle text and search for matching frames |
| `time_search` | Query mentions time position (beginning, end, specific time) | Sample frames from relevant time range |
| `none` | General visual query | Use visual similarity only |

**Subtitles Extraction (if subtitle_search=True):**
- Uses Qwen2.5-VL to extract quoted/paraphrased subtitle text
- Rewrites query without subtitle references
- Falls back to original query if no subtitle text found

**Time Focus Analysis (if time_search=True):**
- Determines temporal focus mode: 'start', 'end', 'range', or 'none'
- Extracts start/end time if specified
- Samples frames from the relevant time range

**Configuration:**
- `intent_model_id`: Model to use (default: Qwen/Qwen2.5-VL-7B-Instruct)

**Example:**
```
Query: "After the subtitle 'hello my name is Dr Gil'"
→ subtitle_search: true
→ subtitle_query_text: "hello my name is Dr Gil"
→ cleaned_query: "Who is Dr Gil?"

Query: "What happens at the end?"
→ time_search: true
→ mode: "end"
→ start_sec: 180.0, end_sec: 240.0 (last 20% of 4-min video)
```

**Key Files:**
- `inference/models/intent_model.py`: `IntentAnalyzer` class

---

### Stage 5: MULTI-HOP RETRIEVAL

**Input:** Graph, query intent, vision query  
**Output:** Selected segments for reranking

**Algorithm:**

**Hop 1 - Semantic Search:**
1. Encode query to text embedding using CLIP
2. Compute cosine similarity between query and all segment features
3. Sort segments by similarity
4. Select top-k visually similar segments (default: 3)

**Hop 2 - Spatial Expansion:**
1. For each top segment, find spatial neighbors (non-consecutive)
2. Sort neighbors by edge weight (similarity)
3. Add top spatial neighbors to selection (default: 3)

**Hop 3 - Temporal Expansion:**
1. For each selected segment, add all temporal neighbors
2. This provides full temporal context around relevant moments

**Subtitle Fusion (if subtitle_search=True):**
1. Search graph for nodes matching subtitle query
2. Match by exact substring or semantic similarity
3. Merge subtitle results with semantic results
4. Add subtitle similarity scores to all segments

**Result Structure:**
```json
{
  "node_id": 15,
  "path": "video.mp4",
  "start_frame": 375,
  "end_frame": 500,
  "similarity": 0.78,
  "subtitle_similarity": 0.45,
  "time_similarity": 0.0,
  "combined_score": 1.23
}
```

**Configuration:**
- `top_k`: Top K segments to select (default: 3)
- `spatial_k`: Spatial neighbors to expand (default: 3)
- `attribute_top_k`: Subtitle matches to consider (default: 3)
- `subtitle_neighbor_hops`: Temporal hops for subtitle expansion (default: 3)

**Key Files:**
- `inference/retrieval/semantic.py`: `VideoRetriever` class

---

### Stage 6: FRAME RERANKING

**Input:** Selected segments, query  
**Output:** Final ranked frames

**Algorithm:**

1. **Sample Frames:**
   - For each selected segment, sample frames at interval
   - Default: every 5 frames (adjusts for target FPS if specified)

2. **Score Frames (Qwen2.5-VL-3B + LoRA):**
   - For each frame, present query and ask for relevance score (1-5)
   - Prompt: "Rate how relevant this frame is for answering the question..."
   - Options: 1=irrelevant, 2=slightly, 3=moderately, 4=mostly, 5=highly

3. **Mark Subtitle Frames:**
   - If subtitle search was enabled, frames from subtitle-matched segments are marked with `is_subtitle_frame: true`
   - Segments with `subtitle_similarity >= 0.5` are considered subtitle matches

4. **Group and Sort:**
   - Group frames by segment
   - Sort within each segment by score

5. **Select Top Frames:**
   - Ensure minimum frames per segment (default: 6)
   - Select remaining slots from top-scoring frames
   - Sort final list by timestamp

6. **Output:**
   - Copy selected frames to output directory
   - Name format: `t00h00m10s_400ms_seg0001_frame00140.jpg`

**Scoring Prompt:**
```
Given the image, which is a frame from a video, rate how relevant 
this frame is for answering the question: '{query}'.

Output only one number from 1 to 5, where:
1 = completely irrelevant
2 = slightly relevant
3 = moderately relevant
4 = mostly relevant
5 = highly relevant (decisive evidence)
```

**Configuration:**
- `rerank_frame_interval`: Sample every N frames (default: 5)
- `batch_size`: Frames per batch (default: 64)
- `top_frames`: Max frames to return (default: 128)
- `min_frames_per_clip`: Min frames per segment (default: 6)
- `target_sample_fps`: Target FPS for sampling (optional)

**Key Files:**
- `inference/models/reranker.py`: `FrameReranker` class

**Subtitle Frame Marking:**
- Segments with `subtitle_similarity >= 0.5` are marked
- Matching frames get `is_subtitle_frame: true` field
- Used by downstream VLM for subtitle verification

**Performance:**
- Batch processing: 64 frames at once
- Typical time: 0.2-0.3s per frame
- Total reranking time: ~15-30s for 100 frames

---

### Stage 7: TIME-FOCUSED CONTEXT (Optional)

**Triggered when:** `time_search=True` in intent analysis

**Process:**
1. Sample frames from specified time range
2. Mark frames with `segment_index: -1`
3. Pass to MLLM as separate labeled context

**MLLM Prompt Integration:**
```
[Main frames appear here...]

============================================================
TIME-FOCUSED CONTEXT:
============================================================
The following 5 frames were identified as being from the time range
relevant to your query (e.g., specific timestamps, start/end of video).
These frames provide temporal context that may help with answering questions
about WHEN events occur in the video.
============================================================
```

---

### Stage 8: ANSWER GENERATION (Optional)

**Input:** Reranked frames, optional time_focus_frames, subtitle_frames, subtitle_text, query  
**Output:** Natural language answer

**Model:** Qwen2.5-VL-3B-Instruct

**Process:**
1. Load MLLM model
2. Extract subtitle frames (frames with `is_subtitle_frame: true`)
3. Build messages with frames interleaved with text
4. Send to MLLM for generation
5. Decode and return answer

**Subtitle-Aware Input:**
- **Main frames (top 16):** Frames sorted by score, then timestamp
- **Time focus frames (top 5):** Frames from time-focused regions
- **Subtitle frames (top 4):** Frames from subtitle-matched segments
- **Subtitle text:** Extracted subtitle text for verification

**MLLM Prompt Integration:**
```
[Main frames appear here...]

============================================================
TIME-FOCUSED CONTEXT:
============================================================
The following 5 frames were identified as being from the time range
relevant to your query (e.g., specific timestamps, start/end of video).
These frames provide temporal context that may help with answering questions
about WHEN events occur in the video.
============================================================

============================================================
SUBTITLE-VERIFICATION CONTEXT:
============================================================
The extracted subtitle text from the query was:
"The patient's condition is stable"

The following 4 frames come from video segments that were identified
as potentially containing this subtitle text.
Verify whether the subtitle text matches what you see in these frames.
============================================================

Provide a concise, direct answer based on what you see in the frames.
```

**Frame Limits (to stay within 131k token context):**
| Frame Type | Max Count | Sorting |
|------------|-----------|---------|
| Main frames | 16 | By score (desc), then timestamp |
| Time focus frames | 5 | By timestamp |
| Subtitle frames | 4 | By timestamp |

**Key Files:**
- `inference/models/answer_generator.py`: `AnswerGenerator` class

**Debug Logging:**
Set `VIDEOSTIR_DEBUG=1` to log full LLM input text and image paths.

---

## Output Files

### rerank_results.json
Final ranked frames with metadata:
```json
[
  {
    "temp_path": "/tmp/frames_tmp_xxx/frame.jpg",
    "segment_index": 1,
    "segment_path": "video.mp4",
    "frame_in_segment": 140,
    "global_frame_index": 260,
    "timestamp": 10.4,
    "score": 2.57,
    "rank": 1,
    "output_path": "output/t00h00m10s_400ms_seg0001_frame00140.jpg",
    "timestamp_label": "00h00m10s_400ms",
    "is_subtitle_frame": true
  }
]
```

**New Fields:**
- `is_subtitle_frame`: true if frame comes from a subtitle-matched segment

### retrieval_plan.json
Detailed retrieval information:
```json
{
  "intent": {...},
  "selected_segments": [...],
  "query_variants": {
    "original": "...",
    "vision": "...",
    "subtitle": "..."
  },
  "subtitle_analysis": {...},
  "time_focus_analysis": {...},
  "video_metadata": {...}
}
```

### subtitle_frames.json
Only present when `subtitle_search=True`. Contains frames from subtitle-matched segments:
```json
[
  {
    "temp_path": "/tmp/frames_tmp_xxx/frame.jpg",
    "segment_index": 2,
    "segment_path": "video.mp4",
    "frame_in_segment": 140,
    "global_frame_index": 260,
    "timestamp": 10.4,
    "score": 3.25,
    "output_path": "output/t00h00m10s_400ms_seg0002_frame00140.jpg",
    "is_subtitle_frame": true
  }
]
```

### time_focus_results.json
Only present when time_search=True

### checkpoint/
Cached intermediate results for resume:
- `cache/segments.json`: Segmentation results
- `cache/segment_features.json`: Visual embeddings
- `cache/graph.json`: Graph structure
- `cache/intent_analysis.json`: Intent analysis
- `cache/retrieval.json`: Retrieval results
- `cache/rerank_results.json`: Reranked frames
- `cache/subtitle_frames.json`: Subtitle-matched frames (if applicable)

---

## Pipeline Timing

The pipeline includes built-in timing to identify bottlenecks.

**Typical Timing (medical video, 5-min, 25 FPS):**

| Stage | Time | Notes |
|-------|------|-------|
| Video Segmentation | ~5-10s | First run only, uses K-means clustering |
| Embeddings | ~10-20s | Extracts CLIP features for all segments |
| Graph Building | ~1-2s | Fast, O(n²) pairwise similarity |
| Intent Analysis | ~5-10s | Qwen2.5-VL-7B inference |
| Retrieval | ~1-2s | CLIP query encoding + similarity |
| Reranking | ~15-30s | **Main bottleneck**, 64-frame batch |

**Total Pipeline Time:** ~45-60s (first run)  
**With Checkpoint:** ~15-20s (subsequent runs with same query)

---

## Subtitle Handling Flow

1. **Load Subtitles:**
   ```python
   subtitle_entries = utils._load_subtitle_entries(config.subtitle_json)
   ```

2. **Attach to Segments:**
   ```python
   segment_infos = utils._attach_subtitles(segment_infos, subtitle_entries)
   ```
   - Matches subtitle timestamps to segment time ranges
   - Adds `subtitle_text` field to segment metadata

3. **Intent Determines Usage:**
   - If `subtitle_search=True`: Subtitle text used in `_merge_attribute_results`
   - If `subtitle_search=False`: Subtitles still attached but not used for retrieval

4. **Subtitle Extraction (if needed):**
   ```python
   subtitle_analysis = subtitle_analyzer.extract_subtitles(original_query)
   ```
   - Extracts quoted subtitle text from query
   - Rewrites query without subtitle references

**Important:** Subtitles are always loaded and attached to segments. They are only *used* for retrieval when the intent analyzer determines `subtitle_search=True`.

---

## Checkpoint System

### Checkpointed Stages

| Stage | Cache File | Meta File | Description |
|-------|------------|-----------|-------------|
| segments | `cache/segments.json` | `checkpoints/segments.meta.json` | Video segmentation results |
| segment_features | `cache/segment_features.json` | `checkpoints/segment_features.meta.json` | Visual embeddings |
| graph | `cache/graph.json` | `checkpoints/graph.meta.json` | Spatiotemporal graph |
| intent_analysis | `cache/intent_analysis.json` | `checkpoints/intent_analysis.meta.json` | Query intent analysis |
| retrieval | `cache/retrieval.json` | `checkpoints/retrieval.meta.json` | Top-k segment results |
| rerank_results | `cache/rerank_results.json` | `checkpoints/rerank_results.meta.json` | Final ranked frames |
| final_results | `cache/final_results.json` | `checkpoints/final_results.meta.json` | Complete results |

### Resume Behavior

| Situation | Behavior |
|-----------|----------|
| Same video + same query | Load all stages from checkpoint |
| Same video + different query | Re-run intent, retrieval, reranking |
| Different video | Full pipeline execution |

### Checkpoint Management

```python
# Disable checkpointing
python -m inference run --disable-checkpoint ...

# Custom checkpoint directories
python -m inference run --checkpoint-dir my_checkpoints --cache-dir my_cache ...
```

---

## Configuration Options

### Basic Options

| Option | Default | Description |
|--------|---------|-------------|
| `--video` | None | Path to input video (required) |
| `--query` | None | Text query for retrieval (required) |
| `--output` | ./output/ | Output directory |
| `--subtitle-json` | None | Path to subtitle JSON file |

### Segmentation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--frame-interval` | 30 | Frame sampling for segmentation |
| `--clusters` | 10 | Number of K-means clusters |
| `--min-segment-sec` | 1 | Minimum segment duration |

### Embedding Options

| Option | Default | Description |
|--------|---------|-------------|
| `--embed-frame-interval` | 10 | Frame sampling for embeddings |

### Retrieval Options

| Option | Default | Description |
|--------|---------|-------------|
| `--top-k` | 3 | Top segments to select |
| `--spatial-k` | 3 | Spatial neighbors to expand |
| `--attribute-top-k` | 3 | Subtitle matches to consider |
| `--subtitle-neighbor-hops` | 3 | Temporal hops for subtitle expansion |
| `--temporal-weight` | 1.0 | Weight for temporal edges |

### Reranking Options

| Option | Default | Description |
|--------|---------|-------------|
| `--rerank-frame-interval` | 5 | Frame sampling for reranking |
| `--top-frames` | 128 | Max frames to return |
| `--min-frames-per-clip` | 6 | Min frames per segment |
| `--batch-size` | 64 | Frames per batch (hardcoded in reranker) |

### Time-Focus Options

| Option | Default | Description |
|--------|---------|-------------|
| `--time-focus-ratio` | 0.04 | Ratio for start/end focus |
| `--time-sampling-interval` | 20 | Frame sampling interval |
| `--time-range-padding` | 1.0 | Padding for time ranges |
| `--time-min-window` | 2.0 | Minimum time window |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--intent-model-id` | Qwen/Qwen2.5-VL-7B-Instruct | Intent analysis model |
| `--reranker-model-id` | Qwen/Qwen2.5-VL-3B-Instruct | Reranking model |
| `--reranker-adapter-dir` | ./result | LoRA adapter path |

### Checkpoint Options

| Option | Default | Description |
|--------|---------|-------------|
| `--disable-checkpoint` | False | Skip checkpointing |
| `--checkpoint-dir` | checkpoints | Checkpoint directory |
| `--cache-dir` | cache | Cache directory |

### Batch Processing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--batch-config` | None | JSON config file |
| `--video-root` | ./videos | Video base directory |
| `--subtitle-root` | ./subtitles | Subtitle base directory |
| `--force` | False | Reprocess existing outputs |

---

## API Usage

```python
from inference import PipelineConfig, run_pipeline
from inference.models import AnswerGenerator

# Configure pipeline
config = PipelineConfig(
    video_path="video.mp4",
    query="Show me the opening",
    output_dir="./results",
    top_frames=64,
    subtitle_json="subtitles.json",
    enable_checkpoint=True,
)

# Run pipeline
result = run_pipeline(config)

# Access results
for frame in result.reranked_frames[:5]:
    print(f"Frame {frame['rank']}: {frame['output_path']} (score: {frame['score']:.2f})")

# Generate answer with MLLM
answer_gen = AnswerGenerator(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
answer = answer_gen.generate(
    frames=result.reranked_frames,
    query="What instrument is he using?",
    time_focus_frames=result.time_focus_frames,
    subtitle_frames=result.subtitle_frames,
    subtitle_text=result.retrieval_plan.get("subtitle", ""),
)
print(f"Answer: {answer}")
answer_gen.unload()
```

**Subtitle Frame Access:**
```python
# Extract subtitle frames from reranked results
subtitle_frames = [f for f in result.reranked_frames if f.get("is_subtitle_frame")]

# Or use the pre-computed subtitle_frames from the result
subtitle_frames = result.subtitle_frames
```

---

## Query Intent Examples

### Subtitle Search Examples

| Query | subtitle_search | Reason |
|-------|-----------------|--------|
| "After the subtitle 'hello my name is'" | true | Contains quoted subtitle text |
| "When the phrase 'medical procedure' appears" | true | Mentions specific phrase |
| "Show me what happens at 1:30" | false | Time-focused, not subtitle |

### Time-Focus Search Examples

| Query | time_search | mode | Reason |
|-------|-------------|------|--------|
| "What happens at the end?" | true | end | Mentions "end" |
| "Show me the opening" | true | start | Mentions "opening" |
| "What happens between 00:30 and 01:00?" | true | range | Specifies time range |

### Visual Search (Default)

| Query | subtitle_search | time_search | Reason |
|-------|-----------------|-------------|--------|
| "Show me the medical procedure steps" | false | false | General visual query |
| "Who is the patient?" | false | false | General question about content |

---

## Performance Optimization

### GPU Acceleration
- Use CUDA-compatible GPU for fastest inference
- Models automatically use GPU if available
- Enable torch.compile for vision models if desired

### Batch Processing
- Reranker processes frames in batches (default: 64)
- Increase batch size for larger GPU VRAM
- Larger batches reduce per-frame processing time

### Checkpointing Strategy
- First run: Full pipeline with all stages
- Subsequent runs: Load from checkpoint when query unchanged
- Use `--disable-checkpoint` for one-off runs

### Model Selection Tradeoffs

| Stage | Small (3B) | Large (7B) |
|-------|------------|------------|
| Intent Analysis | Fast (~5s) | Better reasoning (~10s) |
| Reranking | Good quality (~0.2s/frame) | Best quality (~0.3s/frame) |
| Answer Gen | Fast, concise | More coherent |

### Short Video Mode
Videos under `--short-video-threshold` (default: 240s) skip graph retrieval and sample at 3 FPS directly.

---

## Troubleshooting

### Out of Memory
- Reduce `top_frames` (default: 128)
- Reduce `batch_size` in FrameReranker
- Use smaller model (3B instead of 7B)
- Enable checkpointing to free memory between stages

### Slow Inference
- Enable GPU (CUDA)
- Reduce `frame_interval` for fewer samples
- Use `--disable-checkpoint` to skip loading overhead
- Increase batch size for reranking

### No Results Returned
- Check video path exists
- Verify query is not empty
- Check segment count (video might be too short)
- Verify checkpoint directory is writable

### Subtitles Not Used
- Ensure `--subtitle-json` path is correct
- Intent must have `subtitle_search: true`
- Check retrieval_plan.json for intent analysis

### Checkpoint Issues
- Delete checkpoint directory and rerun
- Use `--disable-checkpoint` to bypass
- Check file permissions on output directory

---

## Multi-Hop Retrieval Explained

The retrieval stage performs 3 hops to find relevant content:

### Hop 1: Semantic Search
```
Query: "Show me the medical procedure"
↓
CLIP encodes query to embedding
↓
Compare to all segment embeddings
↓
Select top-3 most similar segments
```

### Hop 2: Spatial Expansion
```
Selected: Segment 5 (score: 0.85)
↓
Find segments temporally adjacent to 5
↓
Add segments 2, 3, 4, 6, 7, 8 (spatial neighbors)
↓
Now have 9 segments selected
```

### Hop 3: Temporal Expansion
```
For each selected segment, add all temporal neighbors
↓
Segment 2 → adds 0, 1
Segment 3 → adds 0, 1, 2, 4
Segment 4 → adds 2, 3, 5
...
↓
Final selection: All segments in relevant time range
```

This gives you rich context around the most relevant moments, not just the exact match.

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Video not found` | Incorrect video path | Verify video path exists |
| `No segments generated` | Video too short or sampling issue | Reduce frame_interval |
| `Out of memory` | GPU VRAM exceeded | Reduce batch_size, top_frames |
| `Intent analysis failed` | Model loading issue | Check model ID, GPU memory |
| `Checkpoint corrupted` | Previous run interrupted | Delete checkpoint directory |

---

## Future Enhancements

Potential improvements to the pipeline:

1. **More sophisticated segmentation**: Shot boundary detection
2. **Alternative embedding models**: DINOv2, SigLIP
3. **Enhanced subtitle matching**: Fuzzy matching for OCR errors
4. **Multi-query expansion**: Generate related queries for better recall
5. **Advanced reranking**: Cross-encoder re-scoring
6. **Distributed processing**: Process very long videos in parallel

---

## References

- Model: Qwen2.5-VL (Qwen Team)
- Embeddings: CLIP ViT-G/14 (OpenAI)
- Graph: NetworkX
- Video Processing: OpenCV, Decord