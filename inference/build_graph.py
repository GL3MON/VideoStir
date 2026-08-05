"""Backward compatibility module for VideoStir graph building.

This module provides the original function-based API for backward compatibility.
New code should use `from videostir import SpatiotemporalGraphBuilder` instead.
"""

from inference.retrieval import (
    SpatiotemporalGraphBuilder,
    build_spatiotemporal_graph,
)

__all__ = ["SpatiotemporalGraphBuilder", "build_spatiotemporal_graph"]