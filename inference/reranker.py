"""Backward compatibility module for VideoStir frame reranking.

This module provides the original function-based API for backward compatibility.
New code should use `from videostir import FrameReranker` instead.
"""

from inference.models.reranker import FrameReranker, rerank_segments

__all__ = ["FrameReranker", "rerank_segments"]