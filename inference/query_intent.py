"""Backward compatibility module for VideoStir intent analysis.

This module provides the original function-based API for backward compatibility.
New code should use `from videostir import IntentAnalyzer` instead.
"""

from inference.models.intent_model import (
    IntentAnalyzer,
    analyze_query_intent,
    analyze_time_focus,
    rewrite_query_and_extract_subtitles,
    unload_intent_model,
)

__all__ = [
    "IntentAnalyzer",
    "analyze_query_intent",
    "analyze_time_focus",
    "rewrite_query_and_extract_subtitles",
    "unload_intent_model",
]