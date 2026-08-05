"""Backward compatibility module for VideoStir video segmentation.

This module provides the original function-based API for backward compatibility.
New code should use `from videostir import VideoSegmenter` instead.
"""

from inference.preprocessing.video import (
    VideoSegmenter,
    extract_visual_embeddings,
    cluster_and_segment,
    export_segments,
)

__all__ = [
    "VideoSegmenter",
    "extract_visual_embeddings",
    "cluster_and_segment",
    "export_segments",
]