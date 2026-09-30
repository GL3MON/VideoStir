"""Preprocessing utilities for VideoStir.

This module provides utilities for video preprocessing:
- video: Video loading, splitting, and segment extraction
- framing: Frame extraction and sampling
"""

from .video import VideoSegmenter, extract_visual_embeddings, cluster_and_segment, export_segments
from .framing import FrameExtractor

__all__ = ["VideoSegmenter", "extract_visual_embeddings", "cluster_and_segment", "export_segments", "FrameExtractor"]