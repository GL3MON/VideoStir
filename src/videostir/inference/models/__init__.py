"""Model management for VideoStir inference.

This module provides easy-to-use wrappers around the vision-language models
used in the VideoStir pipeline:
- IntentAnalyzer: Analyzes queries to determine retrieval strategy
- FrameReranker: Scores and reranks frames for relevance
- VideoEmbedder: Computes visual embeddings for video segments
- AnswerGenerator: Generates natural language answers using MLLM

Note: Actual model imports are lazy to avoid dependency issues during
module loading. Use `from videostir.inference import models` and access classes
directly.
"""


class IntentAnalyzer:
    """Analyzer for query intent (lazy-loaded)."""

    @staticmethod
    def _get_class():
        from .intent_model import IntentAnalyzer as _IA
        return _IA

    def __new__(cls, *args, **kwargs):
        return cls._get_class()(*args, **kwargs)


class FrameReranker:
    """Frame reranker (lazy-loaded)."""

    @staticmethod
    def _get_class():
        from .reranker import FrameReranker as _FR
        return _FR

    def __new__(cls, *args, **kwargs):
        return cls._get_class()(*args, **kwargs)


class VideoEmbedder:
    """Video embedder (lazy-loaded)."""

    @staticmethod
    def _get_class():
        from .embedding import VideoEmbedder as _VE
        return _VE

    def __new__(cls, *args, **kwargs):
        return cls._get_class()(*args, **kwargs)


class AnswerGenerator:
    """Answer generator using MLLM (lazy-loaded)."""

    @staticmethod
    def _get_class():
        from .answer_generator import AnswerGenerator as _AG
        return _AG

    def __new__(cls, *args, **kwargs):
        return cls._get_class()(*args, **kwargs)


__all__ = ["IntentAnalyzer", "FrameReranker", "VideoEmbedder", "AnswerGenerator"]