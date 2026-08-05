"""Backward compatibility module for VideoStir video feature computation.

This module provides the original function-based API for backward compatibility.
New code should use `from videostir import VideoEmbedder` instead.
"""

from inference.models.embedding import VideoEmbedder, compute_video_features

__all__ = ["VideoEmbedder", "compute_video_features"]