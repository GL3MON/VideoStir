"""Checkpointing utilities for VideoStir pipeline.

This module enables restartable pipelines by caching intermediate results
to disk. This is useful for long-running pipelines that may need to be
interrupted and resumed.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .utils import _to_serializable


class CheckpointManager:
    """Manages checkpointing for the VideoStir pipeline.

    The checkpoint manager enables resumable pipeline execution by:
    1. Caching embeddings to disk after computation
    2. Saving intermediate results (segments, graph, retrieval results)
    3. Tracking pipeline progress with markers

    Example:
        >>> manager = CheckpointManager("output_dir")
        >>> if manager.has_checkpoint("segments"):
        ...     segments = manager.load("segments")
        ... else:
        ...     segments = segment_video(video_path)
        ...     manager.save("segments", segments)
    """

    def __init__(
        self,
        output_dir: str,
        checkpoint_dir: str = "checkpoints",
        cache_dir: str = "cache",
    ):
        """Initialize the checkpoint manager.

        Args:
            output_dir: Base output directory for the pipeline.
            checkpoint_dir: Subdirectory for checkpoint files.
            cache_dir: Subdirectory for cached embeddings and intermediate data.
        """
        self.output_dir = os.path.abspath(output_dir)
        self.checkpoint_dir = os.path.join(self.output_dir, checkpoint_dir)
        self.cache_dir = os.path.join(self.output_dir, cache_dir)

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, name: str, extension: str = ".pt") -> str:
        """Get the path for a cached file."""
        return os.path.join(self.cache_dir, f"{name}{extension}")

    def _get_checkpoint_path(self, name: str, extension: str = ".json") -> str:
        """Get the path for a checkpoint file."""
        return os.path.join(self.checkpoint_dir, f"{name}{extension}")

    def has_checkpoint(self, name: str) -> bool:
        """Check if a checkpoint exists.

        Args:
            name: Name of the checkpoint (e.g., 'segments', 'graph', 'embeddings').

        Returns:
            True if the checkpoint exists and has valid data (not just metadata).
        """
        checkpoint_path = self._get_checkpoint_path(name)
        # Check for actual data files in cache directory (not just metadata)
        cache_pt = self._get_cache_path(name, ".pt")
        cache_npy = self._get_cache_path(name, ".npy")
        cache_json = self._get_cache_path(name, ".json")
        return (
            os.path.exists(checkpoint_path)
            or os.path.exists(cache_pt)
            or os.path.exists(cache_npy)
            or os.path.exists(cache_json)
        )

    def save(
        self,
        name: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save data to checkpoint cache.

        Args:
            name: Name for this checkpoint.
            data: Data to save (tensor, array, dict, or list).
            metadata: Optional metadata to save alongside data.
        """
        # Save data
        if isinstance(data, torch.Tensor):
            torch.save(data, self._get_cache_path(name, ".pt"))
        elif isinstance(data, np.ndarray):
            np.save(self._get_cache_path(name, ".npy"), data)
        elif isinstance(data, (dict, list)):
            path = self._get_cache_path(name, ".json")
            with open(path, "w") as f:
                json.dump(_to_serializable(data), f)
        else:
            # Try to save as-is
            torch.save(data, self._get_cache_path(name, ".pt"))

        # Save metadata if provided
        if metadata is not None:
            metadata_path = self._get_checkpoint_path(name, ".meta.json")
            with open(metadata_path, "w") as f:
                json.dump(_to_serializable(metadata), f, indent=2)

    def load(self, name: str) -> Any:
        """Load data from checkpoint cache.

        Args:
            name: Name of the checkpoint to load.

        Returns:
            The loaded data.

        Raises:
            FileNotFoundError: If the checkpoint doesn't exist.
        """
        # Try JSON first
        json_path = self._get_cache_path(name, ".json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                return _to_tensor(data)

        # Try numpy array
        npy_path = self._get_cache_path(name, ".npy")
        if os.path.exists(npy_path):
            return np.load(npy_path)

        # Try torch tensor
        pt_path = self._get_cache_path(name, ".pt")
        if os.path.exists(pt_path):
            return torch.load(pt_path, map_location="cpu", weights_only=False)

        raise FileNotFoundError(f"Checkpoint not found: {name}")

    def load_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Load metadata for a checkpoint.

        Args:
            name: Name of the checkpoint.

        Returns:
            Metadata dictionary or None if no metadata exists.
        """
        metadata_path = self._get_checkpoint_path(name, ".meta.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    def save_segments(self, segments: List[Dict[str, Any]]) -> None:
        """Save segment information.

        Args:
            segments: List of segment dictionaries.
        """
        metadata = {
            "count": len(segments),
            "saved_at": _get_timestamp(),
        }
        self.save("segments", segments, metadata)

    def load_segments(self) -> List[Dict[str, Any]]:
        """Load saved segments.

        Returns:
            List of segment dictionaries.
        """
        return self.load("segments")

    def save_segment_features(self, segment_features: List[Dict[str, Any]]) -> None:
        """Save segment features with embeddings.

        Args:
            segment_features: List of segment dictionaries containing 'image_features'.
        """
        # Extract just the features for efficient storage
        features = torch.cat([f["image_features"] for f in segment_features], dim=0)
        metadata = {
            "count": len(segment_features),
            "feature_dim": features.shape[1] if features.ndim > 1 else 1,
            "saved_at": _get_timestamp(),
        }
        self.save("segment_features", segment_features, metadata)

    def load_segment_features(self) -> List[Dict[str, Any]]:
        """Load saved segment features.

        Returns:
            List of segment dictionaries with 'image_features'.
        """
        return self.load("segment_features")

    def save_graph(
        self,
        graph_info: Dict[str, Any],
    ) -> None:
        """Save graph information.

        Args:
            graph_info: Dictionary with graph metadata and node data.
        """
        metadata = {
            "num_nodes": graph_info.get("num_nodes", 0),
            "num_edges": graph_info.get("num_edges", 0),
            "saved_at": _get_timestamp(),
        }
        self.save("graph", graph_info, metadata)

    def load_graph(self) -> Dict[str, Any]:
        """Load saved graph information.

        Returns:
            Graph information dictionary.
        """
        return self.load("graph")

    def save_retrieval_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> None:
        """Save retrieval results.

        Args:
            results: List of retrieval result dictionaries.
            query: The query that was used for retrieval.
        """
        metadata = {
            "query": query,
            "num_results": len(results),
            "saved_at": _get_timestamp(),
        }
        self.save("retrieval", results, metadata)

    def load_retrieval_results(self) -> tuple[List[Dict[str, Any]], str]:
        """Load saved retrieval results.

        Returns:
            Tuple of (results, query).
        """
        results = self.load("retrieval")
        metadata = self.load_metadata("retrieval")
        return results, metadata["query"]

    def save_final_results(
        self,
        reranked_frames: List[Dict[str, Any]],
        time_focus_frames: List[Dict[str, Any]],
        intent: Dict[str, Any],
    ) -> None:
        """Save final pipeline results.

        Args:
            reranked_frames: Final ranked frames.
            time_focus_frames: Time-focused frames.
            intent: Query intent analysis.
        """
        metadata = {
            "num_reranked_frames": len(reranked_frames),
            "num_time_focus_frames": len(time_focus_frames),
            "intent": intent,
            "saved_at": _get_timestamp(),
        }
        self.save(
            "final_results",
            {
                "reranked_frames": reranked_frames,
                "time_focus_frames": time_focus_frames,
            },
            metadata,
        )

    def load_final_results(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Load final pipeline results.

        Returns:
            Tuple of (reranked_frames, time_focus_frames, intent).
        """
        data = self.load("final_results")
        metadata = self.load_metadata("final_results")
        return (
            data["reranked_frames"],
            data["time_focus_frames"],
            metadata["intent"],
        )

    def save_intent_analysis(
        self,
        intent: Dict[str, Any],
        query: str,
    ) -> None:
        """Save intent analysis results.

        Args:
            intent: Query intent analysis results.
            query: The query that was analyzed.
        """
        metadata = {
            "query": query,
            "has_subtitle_search": intent.get("subtitle_search", False),
            "has_time_search": intent.get("time_search", False),
            "saved_at": _get_timestamp(),
        }
        self.save("intent_analysis", intent, metadata)

    def load_intent_analysis(self) -> tuple[Dict[str, Any], str]:
        """Load saved intent analysis.

        Returns:
            Tuple of (intent, query).
        """
        intent = self.load("intent_analysis")
        metadata = self.load_metadata("intent_analysis")
        return intent, metadata["query"]

    def save_rerank_results(
        self,
        reranked_frames: List[Dict[str, Any]],
        time_focus_frames: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> None:
        """Save reranking results.

        Args:
            reranked_frames: Final ranked frames.
            time_focus_frames: Time-focused frames.
            query: The query that was used for reranking (optional).
        """
        metadata = {
            "num_reranked_frames": len(reranked_frames),
            "num_time_focus_frames": len(time_focus_frames),
            "saved_at": _get_timestamp(),
        }
        if query is not None:
            metadata["query"] = query
        self.save(
            "rerank_results",
            {
                "reranked_frames": reranked_frames,
                "time_focus_frames": time_focus_frames,
            },
            metadata,
        )

    def load_rerank_results(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
        """Load saved reranking results.

        Returns:
            Tuple of (reranked_frames, time_focus_frames, query).
        """
        data = self.load("rerank_results")
        metadata = self.load_metadata("rerank_results")
        query = metadata.get("query") if metadata else None
        return data["reranked_frames"], data["time_focus_frames"], query

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints.

        Returns:
            List of checkpoint information dictionaries.
        """
        checkpoints = []

        for filename in os.listdir(self.cache_dir):
            name, ext = os.path.splitext(filename)
            checkpoint = {
                "name": name,
                "path": os.path.join(self.cache_dir, filename),
                "extension": ext,
            }

            metadata = self.load_metadata(name)
            if metadata:
                checkpoint.update(metadata)

            checkpoints.append(checkpoint)

        return checkpoints

    

def _to_tensor(obj: Any) -> Any:
    """Convert loaded data back to tensors where appropriate."""
    if isinstance(obj, dict):
        return {key: _to_tensor(value) for key, value in obj.items()}
    if isinstance(obj, list):
        # Try to convert list to tensor, but only if all elements are numeric lists/numbers
        # Check if this looks like a tensor (nested numeric data)
        if len(obj) > 0 and isinstance(obj[0], (list, float, int)):
            try:
                # Check if all elements are numeric
                def _is_numeric_sequence(x):
                    if isinstance(x, (float, int)):
                        return True
                    if isinstance(x, list):
                        return all(_is_numeric_sequence(i) for i in x)
                    return False
                if _is_numeric_sequence(obj):
                    return torch.tensor(obj)
            except (TypeError, ValueError):
                pass
        return [_to_tensor(item) for item in obj]
    if isinstance(obj, (float, int)):
        return obj
    return obj


def _get_timestamp() -> str:
    """Get current timestamp as ISO format string."""
    import datetime
    return datetime.datetime.now().isoformat()