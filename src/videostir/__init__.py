"""VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG.

Quick Start:
    >>> from videostir import simple_rag
    >>> frames = simple_rag("video.mp4", "Show me the opening scene")

Sub-packages:
    - videostir.inference: Retrieval pipeline, models, and graph construction
    - videostir.train:     Knowledge distillation and LoRA fine-tuning
    - videostir.eval:      MedVidQA dataset evaluation framework
"""

from __future__ import annotations

from videostir.inference import (
    PipelineConfig,
    PipelineResult,
    BatchConfig,
    CheckpointManager,
    FrameReranker,
    IntentAnalyzer,
    VideoEmbedder,
    run_batch_from_config,
    run_pipeline,
    simple_rag,
)

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "BatchConfig",
    "CheckpointManager",
    "FrameReranker",
    "IntentAnalyzer",
    "VideoEmbedder",
    "run_batch_from_config",
    "run_pipeline",
    "simple_rag",
    "__version__",
]
