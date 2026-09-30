"""Utility functions for VideoStir inference."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


def ensure_core_on_sys_path() -> None:
    """Ensure the vendored ``core`` package (inference/core) is importable.

    The vendored perception_models code uses absolute imports
    (``import core.vision_encoder.pe``), so the inference package directory
    (the parent of the vendored ``core`` package) must be on ``sys.path``.
    """
    inference_dir = os.path.dirname(os.path.abspath(__file__))
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)


def _to_serializable(obj: Any) -> Any:
    """Recursively convert objects containing numpy/torch types for JSON dumping."""
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [_to_serializable(item) for item in obj]

    # Check for numpy scalar types (need .item())
    obj_type = type(obj)
    obj_module = getattr(obj_type, "__module__", None)
    if obj_module and "numpy" in obj_module:
        # Check if it's a numpy scalar (0-dim), not an array
        if hasattr(obj, "item") and not hasattr(obj, "shape"):
            return obj.item()
        # Handle numpy arrays
        if hasattr(obj, "tolist"):
            return obj.tolist()

    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.ndim == 0:
                return obj.item()
            return obj.detach().cpu().tolist()
    except ImportError:
        pass

    return obj


def _normalize_subtitle_time(value: Any) -> Optional[float]:
    """Parse subtitle time value to float seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ":" in text:
            parts = text.split(":")
            try:
                parts = [float(p) for p in parts]
            except ValueError:
                return None
            seconds = 0.0
            for part in parts:
                seconds = seconds * 60.0 + part
            return seconds
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _resolve_path(base_dir: Optional[str], candidate: Optional[str]) -> Optional[str]:
    """Resolve a path relative to base_dir if not absolute."""
    if not candidate:
        return None
    candidate = candidate.strip()
    if not candidate:
        return None
    if os.path.isabs(candidate):
        return candidate
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    if base_dir:
        return os.path.abspath(os.path.join(base_dir, candidate))
    return None


def _probe_video_metadata(video_path: str) -> Dict[str, float]:
    """Probe video metadata (fps, total_frames, duration)."""
    try:
        import cv2
    except ImportError:
        return {"fps": 0.0, "total_frames": 0.0, "duration": 0.0}

    metadata = {"fps": 0.0, "total_frames": 0.0, "duration": 0.0}
    if not video_path:
        return metadata

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return metadata

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        total_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0.0
        duration = total_frames / fps if fps > 0.0 else 0.0

        metadata.update({
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
        })
        return metadata
    finally:
        cap.release()


def _load_subtitle_entries(path: Optional[str]) -> List[Dict[str, Any]]:
    """Load subtitle entries from JSON file."""
    if not path:
        return []
    if not os.path.exists(path):
        print(f"[pipeline] Subtitle file not found, skipping: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        candidates = raw.get("subtitles") or raw.get("segments") or []
    else:
        candidates = raw

    entries: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        start = _normalize_subtitle_time(item.get("start") or item.get("start_time"))
        end = _normalize_subtitle_time(item.get("end") or item.get("end_time"))
        text = item.get("text") or item.get("content") or item.get("line") or ""
        if start is None or end is None:
            continue
        entries.append({"start": float(start), "end": float(end), "text": str(text)})

    entries.sort(key=lambda x: x["start"])
    return entries


def _attach_subtitles(
    segment_infos: Iterable[Dict[str, Any]],
    subtitle_entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach subtitle text to segments based on time overlap."""
    _NEAR_ZERO_DURATION = 1e-3
    subtitle_entries = list(subtitle_entries)
    enriched_segments: List[Dict[str, Any]] = []

    for info in segment_infos:
        start = float(info.get("start_sec", 0.0) or 0.0)
        end = float(info.get("end_sec", start) or start)
        matched_texts: List[str] = []

        for entry in subtitle_entries:
            entry_start = float(entry.get("start", 0.0) or 0.0)
            entry_end = float(entry.get("end", entry_start) or entry_start)

            duration = entry_end - entry_start
            if duration <= _NEAR_ZERO_DURATION:
                if entry_start < start:
                    continue
                if entry_start > end:
                    break
            else:
                if entry_end < start:
                    continue
                if entry_start > end:
                    break
            matched_texts.append(entry["text"])

        enriched = dict(info)
        enriched["subtitle_text"] = " ".join(t for t in matched_texts if t).strip()
        enriched_segments.append(enriched)

    return enriched_segments


def _combine_segment_scores(info: Dict[str, Any]) -> float:
    """Compute combined score from individual similarity scores."""
    vision_score = float(info.get("similarity") or 0.0)
    subtitle_score = float(info.get("subtitle_similarity") or 0.0)
    time_score = float(info.get("time_similarity") or 0.0)
    combined = vision_score + subtitle_score + time_score
    info["combined_score"] = combined
    return combined


def _save_json(data: Any, path: str, ensure_ascii: bool = False) -> None:
    """Save data to JSON file with proper serialization."""
    serializable = _to_serializable(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=ensure_ascii, indent=2)


def _load_json(path: str) -> Any:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)