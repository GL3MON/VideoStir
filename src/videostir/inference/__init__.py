"""VideoStir - A structured and intent-aware RAG framework for long-video understanding.

VideoStir provides tools for retrieving and reasoning over long videos using
spatiotemporal graphs and intent-aware retrieval.

Quick Start:
    >>> from videostir import simple_rag
    >>> frames = simple_rag("video.mp4", "Show me the opening scene")
    >>> for frame in frames[:5]:
    >>>     print(f"Frame {frame['rank']}: {frame['output_path']}")

API:
    - simple_rag(): One-line video retrieval
    - PipelineConfig: Configure the retrieval pipeline
    - run_pipeline(): Full pipeline with custom configuration
    - CheckpointManager: Manage checkpoints for resumable pipelines

Models:
    - IntentAnalyzer: Analyze queries to determine retrieval strategy
    - FrameReranker: Score and rerank frames by relevance
    - VideoEmbedder: Compute visual embeddings for video segments

Checkpointing:
    Enable restartable pipelines with CheckpointManager in PipelineConfig.
    Set enable_checkpoint=True to save intermediate results to disk.
"""

from __future__ import annotations

from .config import PipelineConfig, PipelineResult, BatchConfig
from .pipeline import run_pipeline, run_batch_from_config
from .models import IntentAnalyzer, FrameReranker, VideoEmbedder
from .checkpoint import CheckpointManager
from . import preprocessing, retrieval, models

__version__ = "0.1.0"
__all__ = [
    "simple_rag",
    "PipelineConfig",
    "PipelineResult",
    "BatchConfig",
    "run_pipeline",
    "run_batch_from_config",
    "CheckpointManager",
    "IntentAnalyzer",
    "FrameReranker",
    "VideoEmbedder",
    "preprocessing",
    "retrieval",
    "models",
]


def simple_rag(
    video_path: str,
    query: str,
    output_dir: str | None = None,
    top_frames: int = 128,
) -> list[dict]:
    """Run a simple video retrieval pipeline.

    This is a convenience function that:
    1. Creates default configuration
    2. Runs the full pipeline
    3. Returns ranked frames

    Args:
        video_path: Path to the input video file.
        query: The natural language query for retrieval.
        output_dir: Directory to save results. If None, creates 'output/' in current dir.
        top_frames: Maximum frames to return (default: 128).

    Returns:
        List of frame dictionaries sorted by relevance score, each containing:
            - rank: Position in the final ranking
            - output_path: Path to saved frame image
            - score: Relevance score (1-5)
            - timestamp: Time in video
            - segment_index: Which segment this frame came from

    Example:
        >>> frames = simple_rag("my_video.mp4", "What happens at the end?")
        >>> print(f"Found {len(frames)} relevant frames")
        >>> for frame in frames[:3]:
        >>>     print(f"  {frame['rank']}: {frame['output_path']} (score: {frame['score']:.2f})")
    """
    if output_dir is None:
        output_dir = "output"

    config = PipelineConfig(
        video_path=video_path,
        query=query,
        output_dir=output_dir,
        top_frames=top_frames,
    )

    result = run_pipeline(config)
    return result.reranked_frames