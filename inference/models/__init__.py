"""Model management for VideoStir inference.

This module provides easy-to-use wrappers around the vision-language models
used in the VideoStir pipeline:
- IntentAnalyzer: Analyzes queries to determine retrieval strategy
- FrameReranker: Scores and reranks frames for relevance
- VideoEmbedder: Computes visual embeddings for video segments
- AnswerGenerator: Generates natural language answers using MLLM

Note: Actual model imports are lazy to avoid dependency issues during
module loading. Use `from videostir import models` and access classes
directly.
"""


class LazyImportModule:
    """Lazily imports modules to avoid dependency issues at load time."""

    def __init__(self, module_name):
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            import importlib
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, name):
        return getattr(self._load(), name)


# Create lazy modules
import sys

_models_module = LazyImportModule(__name__)

# Expose classes via lazy import
class IntentAnalyzer:
    """Analyzer for query intent (lazy-loaded)."""

    @staticmethod
    def _get_class():
        from inference.models.intent_model import IntentAnalyzer as _IA
        return _IA

    def __new__(cls, *args, **kwargs):
        return cls._get_class()(*args, **kwargs)


class FrameReranker:
    """Frame reranker (lazy-loaded)."""

    @staticmethod
    def _get_class():
        from inference.models.reranker import FrameReranker as _FR
        return _FR

    def __new__(cls, *args, **kwargs):
        return cls._get_class()(*args, **kwargs)


class VideoEmbedder:
    """Video embedder (lazy-loaded)."""

    @staticmethod
    def _get_class():
        from inference.models.embedding import VideoEmbedder as _VE
        return _VE

    def __new__(cls, *args, **kwargs):
        return cls._get_class()(*args, **kwargs)


class AnswerGenerator:
    """Answer generator using MLLM (lazy-loaded)."""

    @staticmethod
    def _get_class():
        from inference.models.answer_generator import AnswerGenerator as _AG
        return _AG

    def __new__(cls, *args, **kwargs):
        return cls._get_class()(*args, **kwargs)


__all__ = ["IntentAnalyzer", "FrameReranker", "VideoEmbedder", "AnswerGenerator"]