# [ACL 2026] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG

![Framework](Figure/framework.png)

**Paper**: [ArXiv Link](https://arxiv.org/pdf/2604.05418)

VideoStir is a structured and intent-aware long-video RAG framework that:
- Structures videos as spatio-temporal graphs at clip level
- Performs multi-hop retrieval to aggregate evidence across distant events
- Uses MLLM-backed intent-relevance scoring for frame retrieval

## Quick Start

### Installation

```bash
pip install -e .
```

For GPU environments such as Kaggle, attach your videos as a Kaggle Dataset,
then install this repository in a notebook:

```python
!git clone https://github.com/VideoStir/VideoStir.git
%cd /kaggle/working/VideoStir
!pip install -e .
```

Use the mounted dataset path for the input video and write results under
`/kaggle/working`, which is available for notebook output and download:

```python
from videostir import simple_rag

frames = simple_rag(
    "/kaggle/input/<your-dataset>/video.mp4",
    "What happens during the procedure?",
    output_dir="/kaggle/working/videostir-results",
    top_frames=32,
)
```

The first run downloads the base models from Hugging Face. The reranker also
uses the LoRA adapter described under [Download Checkpoints](#download-checkpoints);
place it in `/kaggle/working/VideoStir/result` or update the adapter path in your
pipeline configuration. Enable Kaggle internet access for model downloads, or
cache the weights in a Kaggle Dataset.

### Upload Generated Videos to Google Drive

Install the Drive uploader dependencies with `pip install -e ".[gdrive]"` and
enable the Google Drive API in a Google Cloud project. For a personal Drive,
create an OAuth client of type **Desktop app**, download its client JSON, and
save that downloaded file as `.secrets/gauth.json`. Then authorize your Google
account once:

```bash
python scripts/upload_videos_to_gdrive.py --auth-only --oauth-port 8765
```

The script recognizes the OAuth client JSON, opens a Google sign-in page, and
saves the resulting user token separately as `.secrets/gauth-token.json`. It
uses that token on later local runs. In Kaggle, add the contents of
`.secrets/gauth-token.json` as a Kaggle Secret named
`GDRIVE_OAUTH_TOKEN_JSON`; add the destination folder ID as `GDRIVE_FOLDER_ID`.
Load both secrets without printing them:

When the script runs on a remote cloud server, forward port `8765` from your
laptop to the server before approving access. In VS Code Remote, use the
**Ports** panel and forward port `8765`. With SSH, run this on your laptop in a
second terminal, replacing the host with your SSH target:

```bash
ssh -N -L 8765:localhost:8765 your-cloud-host
```

Keep the forward active while completing Google sign-in; the browser’s
`localhost:8765` callback will then reach the cloud process.

For local runs, `.secrets/gauth.json` is the default OAuth client file and
`.secrets/gauth-token.json` is the token file. Override the credential file with
`--credentials /path/to/credentials.json` or
`GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json` if needed.

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["GDRIVE_OAUTH_TOKEN_JSON"] = secrets.get_secret("GDRIVE_OAUTH_TOKEN_JSON")
os.environ["GDRIVE_FOLDER_ID"] = secrets.get_secret("GDRIVE_FOLDER_ID")
```

Then upload from the notebook:

```python
!python scripts/upload_videos_to_gdrive.py /kaggle/working/videostir-results --dry-run
!python scripts/upload_videos_to_gdrive.py /kaggle/working/videostir-results
```

To find the folder ID, open that folder in Drive and copy the string after
`/folders/` in its URL. You can also upload locally by setting
`GOOGLE_APPLICATION_CREDENTIALS` to a service-account key file. Service accounts
have no personal storage quota, so use one with a Workspace Shared Drive where
it has write access. Keep OAuth tokens and service-account keys out of Git.

The uploader scans recursively, keeps the folder structure, and replaces a
same-name file in its destination folder when you rerun it. Remove `--dry-run`
only after confirming the listed source videos are the ones you want to upload.

### Basic Usage

```python
from videostir.inference import simple_rag

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
from videostir.inference import PipelineConfig, run_pipeline

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
pip install -e .
```

### Download Checkpoints

1. **Intent Analysis Model**: Automatically downloaded by Hugging Face on first use
2. **Frame Reranker**: Download from [Google Drive](https://drive.google.com/file/d/1CUC1i7zstZktWDp30Pts4s8E7QULDgns/view?usp=drive_link)
   - Extract to `./result/` directory

## Full Pipeline Examples

### Single Video

```python
from videostir.inference import simple_rag

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
from videostir.inference import run_batch_from_config

results = run_batch_from_config(
    config_path="batch_config.json",
    output_root="batch_results"
)
```

### Command Line Interface

```bash
# Single video
python -m videostir.inference run \
    --video video.mp4 \
    --query "What happens?" \
    --output results/

# With options
python -m videostir.inference run \
    --video video.mp4 \
    --query "What happens?" \
    --output results/ \
    --top-frames 64 \
    --frame-interval 20

# Batch mode
python -m videostir.inference batch \
    --input samples.json \
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
