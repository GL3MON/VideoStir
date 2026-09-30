# VideoStir Usage Guide

## Quick Start

```bash
# Basic retrieval
python -m videostir.inference run --video /path/to/video.mp4 --query "What happens in the opening?"

# With subtitle JSON file
python -m videostir.inference run --video /path/to/video.mp4 --query "What is the date of birth?" --subtitle-json /path/to/subtitles.json

# Interactive answer mode (Qwen2.5-VL-3B)
python -m videostir.inference answer --video /path/to/video.mp4 --output ./output --subtitle-json /path/to/subtitles.json
```

## Commands

### 1. Single Video Retrieval (`run`)

```bash
python -m videostir.inference run [OPTIONS] --video <video_path> --query <query>
```

**Required:**
- `--video <path>` - Path to input video file
- `--query <text>` - Text query for retrieval

**Common Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--subtitle-json <path>` | None | Path to subtitle JSON file for subtitle-based retrieval |
| `--output <dir>` | ./output/ | Output directory for results |
| `--top-frames <n>` | 128 | Maximum frames to return |
| `--frame-interval <n>` | 30 | Frame sampling interval (seconds) |
| `--clusters <n>` | 10 | Number of clusters for video segmentation |
| `--top-k <n>` | 3 | Top-k segments for semantic retrieval |
| `--spatial-k <n>` | 3 | Spatial neighbors to include |

**Subtitle Search Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--attribute-top-k <n>` | 3 | Top-k subtitle-matched segments |
| `--subtitle-neighbor-hops <n>` | 2 | Temporal neighbors for subtitle expansion |

**Time Focus Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--time-focus-ratio <f>` | 0.05 | Ratio of video for time-focused retrieval |
| `--time-sampling-interval <n>` | 10 | Frame sampling for time focus |
| `--time-range-padding <f>` | 1.0 | Padding for time range |

**Performance Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--min-segment-sec <f>` | 1 | Minimum segment duration |
| `--embed-frame-interval <n>` | 10 | Frame interval for embedding |
| `--batch-size <n>` | 32 | Batch size for embedding |
| `--disable-checkpoint` | False | Disable checkpoint saving |
| `--enable-segment-compile` | False | Enable torch.compile for segmentation |
| `--enable-embed-compile` | False | Enable torch.compile for embeddings |

**Example:**
```bash
python -m videostir.inference run \
  --video ./videos/demo.mp4 \
  --query "When does the character speak about their birthday?" \
  --subtitle-json ./subtitles/demo_en.json \
  --output ./results/demo
```

### 2. Interactive Answer Mode (`answer`)

Launches an interactive shell where you can query the video:

```bash
python -m videostir.inference answer [OPTIONS] --video <video_path>
```

**Required:**
- `--video <path>` - Path to input video file

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--subtitle-json <path>` | None | Path to subtitle JSON file |
| `--output <dir>` | ./output/ | Output directory |
| `--top-frames <n>` | 5 | Number of frames for answer generation |
| `--model-id <id>` | Qwen/Qwen2.5-VL-3B-Instruct | MLLM model ID |
| `--device <dev>` | cuda | Device to run on (cuda/cpu) |
| `--frame-interval <n>` | 30 | Frame sampling interval |

**Example:**
```bash
python -m videostir.inference answer \
  --video ./videos/demo.mp4 \
  --subtitle-json ./subtitles/demo_en.json

# Then in interactive mode:
# Your query: What is the woman's birthdate?
# Your query: What happens at the end?
# Your query: quit
```

### 3. Batch Processing (`batch`)

Process multiple videos from a configuration file:

```bash
python -m videostir.inference batch --input <config.json> --output <output_dir>
```

**Required:**
- `--input <path>` - Path to JSON batch config file
- `--output <dir>` - Output directory

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--video-root <dir>` | None | Base directory for video paths |
| `--subtitle-root <dir>` | None | Base directory for subtitle paths |
| `--skip-existing` | True | Skip already processed videos |
| `--force` | False | Reprocess existing outputs |

**Batch Config Format:**
```json
[
  {
    "id": "video1",
    "video_path": "videos/demo1.mp4",
    "subtitle_path": "subtitles/demo1_en.json",
    "query_retrieval": "What happens in the opening?",
    "correct_choice": "A"
  },
  {
    "id": "video2",
    "video_path": "videos/demo2.mp4",
    "subtitle_path": "subtitles/demo2_en.json",
    "query_retrieval": "When does the event occur?"
  }
]
```

**Example:**
```bash
python -m videostir.inference batch \
  --input ./configs/batch_jobs.json \
  --output ./results/batch_output \
  --video-root ./videos \
  --subtitle-root ./subtitles
```

## Output Structure

```
output/<video_name>/
├── retrieval_plan.json      # Full retrieval metadata
├── rerank_results.json      # Scored frames with is_subtitle_frame flag
├── time_focus_results.json  # Time-focused frames (if applicable)
├── checkpoints/             # Checkpointed intermediate results
│   ├── segments.meta.json
│   ├── graph.meta.json
│   ├── intent_analysis.meta.json
│   └── ...
└── segments/                # Extracted segment videos
```

## Subtitle JSON Format

The subtitle JSON should have one of these formats:

**Format 1:**
```json
{
  "subtitles": [
    {"start": 0.0, "end": 3.5, "text": "[Music]"},
    {"start": 3.5, "end": 8.2, "text": "Hello, my name is Dr. Gil"}
  ]
}
```

**Format 2:**
```json
[
  {"start": 0.0, "end": 3.5, "text": "[Music]"},
  {"start": 3.5, "end": 8.2, "text": "Hello, my name is Dr. Gil"}
]
```

**Format 3 (with timestamps as strings):**
```json
{
  "segments": [
    {"start": "00:00.000", "end": "00:03.500", "text": "[Music]"},
    {"start": "00:03.500", "end": "00:08.200", "text": "Hello, my name is Dr. Gil"}
  ]
}
```

## Query Patterns

### Subtitle Search Queries
These queries benefit from subtitle-based retrieval:
- "What does the doctor say about the patient's condition?"
- "When does the character mention their birthday?"
- "What is the woman's birthdate?"

### Time-Focused Queries
These queries benefit from time-focused retrieval:
- "What happens in the opening?"
- "What is at the end of the video?"
- "Show me the scene at 1:30"

### Hybrid Queries
- "What happens after the introduction?"
- "What is shown at the beginning involving the doctor?"

## Troubleshooting

**No subtitle matches found:**
- Ensure subtitle JSON file exists and has correct format
- Check that `--subtitle-json` path is absolute or relative to current directory
- Verify the query is about dialogue/text content, not just visual elements

**Empty results:**
- Check video path is correct
- Verify subtitle text covers the time range of interest
- Try increasing `--top-k` or `--attribute-top-k`

**Memory issues:**
- Reduce `--batch-size`
- Increase `--frame-interval` for fewer frames
- Use `--short-video-threshold` to enable short-video mode for videos < 240s