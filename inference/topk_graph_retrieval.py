"""Backward compatibility module for VideoStir graph retrieval.

This module provides the original function-based API for backward compatibility.
New code should use `from videostir import VideoRetriever` instead.
"""

from inference.retrieval import (
    VideoRetriever,
    retrieve_topk_segments,
    SpatiotemporalGraphBuilder,
    build_spatiotemporal_graph,
)

__all__ = [
    "VideoRetriever",
    "retrieve_topk_segments",
    "SpatiotemporalGraphBuilder",
    "build_spatiotemporal_graph",
]