"""Configuration classes for VideoStir inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineConfig:
    """Configuration for the VideoStir pipeline.

    This dataclass encapsulates all parameters needed to run the
    video retrieval and reranking pipeline.

    Args:
        video_path: Path to the input video file.
        query: The natural language query for retrieval.
        output_dir: Directory to save results.
        frame_interval: Frame sampling interval (default: 30).
        n_clusters: Number of clusters for video segmentation (default: 10).
        min_segment_sec: Minimum segment duration in seconds (default: 0.4).
        embedding_frame_interval: Frame interval for embedding computation (default: 10).
        top_k: Number of top segments to retrieve (default: 3).
        spatial_k: Number of spatial neighbors to expand (default: 3).
        rerank_frame_interval: Frame interval for reranking (default: 5).
        top_frames: Maximum frames to return after reranking (default: 128).
        temporal_weight: Weight for temporal edges in graph (default: 1.0).
        subtitle_json: Optional path to subtitle JSON file.
        attribute_top_k: Number of subtitle matches to consider (default: 3).
        min_frames_per_clip: Minimum frames per clip in reranking (default: 6).
        subtitle_neighbor_hops: Temporal expansion for subtitle matches (default: 2).
        time_focus_ratio: Ratio of video for time-focused retrieval (default: 0.05).
        time_sampling_interval: Frame interval for time-focused sampling (default: 10).
        time_range_padding: Padding for time range windows (default: 1.0).
        time_min_window: Minimum time window size in seconds (default: 2.0).
        short_video_threshold: Duration below which graph retrieval is skipped (default: 240.0).
        correct_choice: Optional ground truth answer for evaluation.

    Checkpoint Settings:
        enable_checkpoint: Enable checkpointing for resumable pipelines (default: True).
        checkpoint_dir: Directory for checkpoint files (default: "checkpoints").
        cache_dir: Directory for cached embeddings (default: "cache").
        skip_completed_stages: Skip stages that have been completed (default: True).
    """

    video_path: str
    query: str
    output_dir: str

    # Frame sampling
    frame_interval: int = 30
    embedding_frame_interval: int = 10
    rerank_frame_interval: int = 5
    time_sampling_interval: int = 10

    # Segmentation
    n_clusters: int = 10
    min_segment_sec: float = 0.4
    compile_model: bool = False  # Use torch.compile for vision model during segmentation
    batch_size: int = 32  # Batch size for embedding extraction

    # Parallel processing
    num_gpus: int = 8  # Number of GPUs for parallel segment processing

    # Retrieval
    top_k: int = 3
    spatial_k: int = 3
    attribute_top_k: int = 3
    temporal_weight: float = 1.0
    subtitle_neighbor_hops: int = 2

    # Reranking
    top_frames: int = 128
    min_frames_per_clip: int = 6

    # Time-focused retrieval
    time_focus_ratio: float = 0.05
    time_range_padding: float = 1.0
    time_min_window: float = 2.0

    # Advanced
    short_video_threshold: float = 240.0
    subtitle_json: Optional[str] = None
    correct_choice: Any = None

    # Model settings
    intent_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    reranker_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    reranker_adapter_dir: str = "./result"
    reranker_compile_model: bool = False  # Use torch.compile for faster inference

    # Checkpoint settings
    enable_checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    cache_dir: str = "cache"
    skip_completed_stages: bool = True

    # Per-stage checkpoint control
    checkpoint_segments: bool = True
    checkpoint_embeddings: bool = True
    checkpoint_graph: bool = True
    checkpoint_retrieval: bool = True
    checkpoint_intent: bool = True
    checkpoint_rerank: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "video_path": self.video_path,
            "query": self.query,
            "output_dir": self.output_dir,
            "frame_interval": self.frame_interval,
            "n_clusters": self.n_clusters,
            "min_segment_sec": self.min_segment_sec,
            "embedding_frame_interval": self.embedding_frame_interval,
            "top_k": self.top_k,
            "spatial_k": self.spatial_k,
            "rerank_frame_interval": self.rerank_frame_interval,
            "top_frames": self.top_frames,
            "temporal_weight": self.temporal_weight,
            "subtitle_json": self.subtitle_json,
            "attribute_top_k": self.attribute_top_k,
            "min_frames_per_clip": self.min_frames_per_clip,
            "subtitle_neighbor_hops": self.subtitle_neighbor_hops,
            "time_focus_ratio": self.time_focus_ratio,
            "time_sampling_interval": self.time_sampling_interval,
            "time_range_padding": self.time_range_padding,
            "time_min_window": self.time_min_window,
            "short_video_threshold": self.short_video_threshold,
            "correct_choice": self.correct_choice,
        }


@dataclass
class PipelineResult:
    """Result from running the VideoStir pipeline.

    Args:
        reranked_frames: List of frames ranked by relevance to the query.
        time_focus_frames: List of frames from time-focused sampling.
        subtitle_frames: List of frames from subtitle-matched segments.
        intent: The query intent analysis results.
        retrieval_plan: Detailed information about the retrieval process.
        video_metadata: Metadata about the processed video.
    """

    reranked_frames: List[Dict[str, Any]]
    time_focus_frames: List[Dict[str, Any]]
    subtitle_frames: List[Dict[str, Any]]
    intent: Dict[str, Any]
    retrieval_plan: Dict[str, Any]
    video_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "reranked_frames": self.reranked_frames,
            "time_focus_frames": self.time_focus_frames,
            "subtitle_frames": self.subtitle_frames,
            "intent": self.intent,
            "retrieval_plan": self.retrieval_plan,
            "video_metadata": self.video_metadata,
        }


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Args:
        input_path: Path to input JSON file with video entries.
        output_root: Root directory for output results.
        video_root: Base directory for resolving relative video paths.
        subtitle_root: Base directory for resolving relative subtitle paths.
        skip_existing: If True, skip entries with existing output (default: True).
        **pipeline_kwargs: Additional arguments for PipelineConfig.
    """

    input_path: str
    output_root: str
    video_root: Optional[str] = None
    subtitle_root: Optional[str] = None
    skip_existing: bool = True
    pipeline_kwargs: Dict[str, Any] = field(default_factory=dict)