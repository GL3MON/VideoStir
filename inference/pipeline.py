"""Main pipeline for VideoStir video retrieval and reranking.

This module orchestrates the complete VideoStir pipeline:
1. Video preprocessing and segmentation
2. Spatiotemporal graph construction
3. Intent-aware query analysis
4. Multi-hop retrieval
5. Frame reranking and selection

Example:
    >>> from videostir import PipelineConfig, run_pipeline
    >>> config = PipelineConfig("video.mp4", "Show me the opening", "output/")
    >>> result = run_pipeline(config)
    >>> for frame in result.reranked_frames[:5]:
    >>>     print(f"Frame {frame['rank']}: {frame['output_path']}")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch

from . import utils
from .config import PipelineConfig, PipelineResult
from .models import IntentAnalyzer, FrameReranker, VideoEmbedder
from .models.embedding import compute_video_features
from .retrieval import SpatiotemporalGraphBuilder, VideoRetriever
from .preprocessing import VideoSegmenter
from .time_utils import seconds_to_timestamp, timestamp_label
from .checkpoint import CheckpointManager

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


_NEAR_ZERO_DURATION = 1e-3


def _setup_logging(output_dir: str) -> logging.Logger:
    """Set up file-based logging for the pipeline.

    Args:
        output_dir: Base output directory for the pipeline.

    Returns:
        Configured logger instance.
    """
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "pipeline.log")

    logger = logging.getLogger("videostir_pipeline")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Also add console handler for visibility
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_format)
    logger.addHandler(console_handler)

    return logger


class Timer:
    """Simple timing context manager for pipeline stages."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[timer] Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"[timer] {self.name} completed in {self.duration:.2f}s")


def _combine_segment_scores(info: Dict[str, Any]) -> float:
    """Compute combined score from individual similarity scores."""
    vision_score = float(info.get("similarity") or 0.0)
    subtitle_score = float(info.get("subtitle_similarity") or 0.0)
    time_score = float(info.get("time_similarity") or 0.0)
    combined = vision_score + subtitle_score + time_score
    info["combined_score"] = combined
    return combined


def _time_attribute_text(attrs: Dict[str, Any]) -> str:
    """Generate text description for time-based retrieval."""
    start = attrs.get("start_sec")
    end = attrs.get("end_sec")
    if start is None or end is None:
        return ""
    timestamp_start = seconds_to_timestamp(start)
    timestamp_end = seconds_to_timestamp(end)
    duration = max(float(end) - float(start), 0.0)
    return (
        f"Clip spanning {timestamp_start} to {timestamp_end} (duration {duration:.2f} seconds)."
    )


def _subtitle_attribute_text(attrs: Dict[str, Any]) -> str:
    """Extract subtitle text for subtitle-based retrieval."""
    text = attrs.get("subtitle_text")
    if not text:
        return ""
    return str(text)


def _sparse_sample_time_range(
    video_path: str,
    start_sec: float,
    end_sec: float,
    frame_interval: int,
    output_dir: str,
    prefix: str,
) -> List[Dict[str, Any]]:
    """Sample frames from a time range in the video."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)  # type: ignore[name-defined]
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    if fps <= 0.0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(int(start_sec * fps), 0)
    end_frame = min(int(end_sec * fps), total_frames - 1 if total_frames > 0 else start_frame)
    if end_frame < start_frame:
        end_frame = start_frame

    frame_interval = max(int(frame_interval), 1)

    sampled: List[Dict[str, Any]] = []
    current = start_frame
    while current <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        success, frame = cap.read()
        if not success:
            break

        timestamp = current / fps if fps else 0.0
        filename = f"t{timestamp_label(timestamp)}_{prefix}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame)

        sampled.append({
            "frame_index": current,
            "timestamp": timestamp,
            "output_path": output_path,
        })

        current += frame_interval

    cap.release()
    return sampled


def _update_segment_paths_to_video(
    segment_infos: List[Dict[str, Any]], video_path: str
) -> List[Dict[str, Any]]:
    """Update segment paths to use the original video file.

    After temp directory cleanup, segments won't have valid paths.
    This function updates them to use the original video with frame ranges.

    Args:
        segment_infos: List of segment info dictionaries.
        video_path: Path to the original video file.

    Returns:
        Updated segment info dictionaries with video_path as the segment path.
    """
    new_segment_infos = []
    for seg in segment_infos:
        new_seg = dict(seg)
        new_seg["path"] = video_path
        new_segment_infos.append(new_seg)
    return new_segment_infos


def _copy_segments_to_output(
    segment_infos: List[Dict[str, Any]], output_segments_dir: str
) -> List[Dict[str, Any]]:
    """Copy segment files to output directory and update paths.

    Args:
        segment_infos: List of segment info dictionaries with 'path' keys.
        output_segments_dir: Directory to copy segments to.

    Returns:
        New list of segment info dictionaries with updated paths.
    """
    os.makedirs(output_segments_dir, exist_ok=True)
    new_segment_infos = []

    for seg in segment_infos:
        seg_path = seg.get("path", "")
        if not seg_path or not os.path.exists(seg_path):
            new_segment_infos.append(seg)
            continue

        # Extract segment index from path if possible
        seg_filename = os.path.basename(seg_path)
        output_path = os.path.join(output_segments_dir, seg_filename)

        # Copy file if it doesn't already exist
        if output_path != seg_path:
            if not os.path.exists(output_path):
                shutil.copy2(seg_path, output_path)
            new_seg = dict(seg)
            new_seg["path"] = output_path
            new_segment_infos.append(new_seg)
        else:
            new_segment_infos.append(seg)

    return new_segment_infos


def _collect_temporal_neighbors(graph, node_id: int, hops: int) -> List[int]:
    """Collect nodes within N temporal hops of a given node."""
    if hops <= 0:
        return []

    visited = {node_id}
    frontier = {node_id}
    collected = set()

    for _ in range(hops):
        next_frontier = set()
        for current in frontier:
            for neighbor in graph.neighbors(current):
                edge = graph.edges[current, neighbor]
                if edge.get("type") != "temporal":
                    continue
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                collected.add(neighbor)
                next_frontier.add(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier

    return list(collected)


def _merge_attribute_results(
    graph,
    base_segments: Iterable[Dict[str, Any]],
    intent: Dict[str, Any],
    query: str,
    attribute_top_k: int,
    subtitle_query: Optional[str] = None,
    subtitle_neighbor_hops: int = 3,
) -> List[Dict[str, Any]]:
    """Merge subtitle and semantic search results."""
    graph_builder = SpatiotemporalGraphBuilder()

    aggregated = {info["node_id"]: dict(info) for info in base_segments if "node_id" in info}

    for info in aggregated.values():
        info.setdefault("subtitle_similarity", 0.0)
        info.setdefault("time_similarity", 0.0)

    subtitle_hits: List[Tuple[int, float, str]] = []
    if intent.get("subtitle_search"):
        subtitle_query_text = subtitle_query if subtitle_query else query
        subtitle_hits = graph_builder.retrieve_by_attribute(
            graph, subtitle_query_text, _subtitle_attribute_text, attribute_top_k
        )
        print("------------subtitle_hits:")
        print(subtitle_hits)
        for node_id, sim, _ in subtitle_hits:
            info = aggregated.get(node_id)
            if info is None:
                attrs = dict(graph.nodes[node_id])
                attrs["node_id"] = node_id
                attrs["similarity"] = 0.0
                attrs["subtitle_similarity"] = 0.0
                attrs["time_similarity"] = 0.0
                aggregated[node_id] = attrs
                info = attrs
            info["subtitle_similarity"] = float(sim)

    if intent.get("subtitle_search") and subtitle_neighbor_hops > 0:
        subtitle_hit_scores = {node_id: float(sim) for node_id, sim, _ in subtitle_hits}
        for node_id, sim in subtitle_hit_scores.items():
            temporal_neighbors = _collect_temporal_neighbors(
                graph, node_id, subtitle_neighbor_hops
            )
            for neighbor_id in temporal_neighbors:
                info = aggregated.get(neighbor_id)
                if info is None:
                    attrs = dict(graph.nodes[neighbor_id])
                    attrs["node_id"] = neighbor_id
                    attrs["similarity"] = 0.0
                    attrs["subtitle_similarity"] = 0.0
                    attrs["time_similarity"] = 0.0
                    aggregated[neighbor_id] = attrs
                    info = attrs
                current = float(info.get("subtitle_similarity") or 0.0)
                if sim > current:
                    info["subtitle_similarity"] = sim

    merged = list(aggregated.values())
    for info in merged:
        _combine_segment_scores(info)

    merged.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return merged


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the complete VideoStir retrieval pipeline.

    Args:
        config: PipelineConfig with all parameters for the pipeline.

    Returns:
        PipelineResult containing reranked frames, time focus frames,
        and detailed retrieval information.
    """
    start_time = time.time()
    temp_root = tempfile.mkdtemp(prefix="pipeline_tmp_")
    # Note: segments_dir is in temp_root but segments are also copied to output_dir
    segments_dir = os.path.join(temp_root, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    output_dir = os.path.abspath(config.output_dir)
    if os.path.commonpath([output_dir, temp_root]) == temp_root:
        raise ValueError(
            "Output directory must be outside the pipeline's temporary workspace."
        )

    # Create video-specific subfolder in output directory
    video_name = os.path.splitext(os.path.basename(config.video_path))[0]
    output_dir = os.path.join(output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # Also create segments in output directory for reranker access
    # Use output_dir (which is already video-specific) for segment storage
    output_segments_dir = os.path.join(output_dir, "segments")
    os.makedirs(output_segments_dir, exist_ok=True)

    # Initialize logger
    logger = _setup_logging(output_dir)
    logger.info(f"Pipeline started for video: {config.video_path}")
    logger.info(f"Query: {config.query}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize checkpoint manager
    checkpoint_manager: Optional[CheckpointManager] = None
    if config.enable_checkpoint:
        checkpoint_manager = CheckpointManager(
            output_dir,
            checkpoint_dir=config.checkpoint_dir,
            cache_dir=config.cache_dir,
        )

    try:
        # Probe video metadata
        video_metadata = utils._probe_video_metadata(config.video_path)
        video_fps = float(video_metadata.get("fps", 0.0) or 0.0)
        video_total_frames = float(video_metadata.get("total_frames", 0.0) or 0.0)
        approx_duration = float(video_metadata.get("duration", 0.0) or 0.0)

        short_video_mode = (
            config.short_video_threshold > 0.0
            and approx_duration > 0.0
            and approx_duration <= config.short_video_threshold
        )

        if short_video_mode:
            logger.info(
                f"Short video detected (duration {approx_duration:.2f}s). "
                "Skipping spatiotemporal retrieval and sampling at 3 FPS."
            )
            print(
                f"[pipeline] Short video detected (duration {approx_duration:.2f}s). "
                "Skipping spatiotemporal retrieval and sampling at 3 FPS."
            )

        # Load subtitle entries if provided
        subtitle_entries = utils._load_subtitle_entries(config.subtitle_json)

        if short_video_mode:
            end_frame_index = int(video_total_frames) - 1 if video_total_frames > 0.0 else 0
            segment_infos = [
                {
                    "segment_index": 0,
                    "path": config.video_path,
                    "start_sec": 0.0,
                    "end_sec": approx_duration if approx_duration > 0.0 else 0.0,
                    "start_frame": 0,
                    "end_frame": max(end_frame_index, 0),
                    "fps": video_fps if video_fps > 0.0 else None,
                }
            ]
            if subtitle_entries:
                segment_infos = utils._attach_subtitles(segment_infos, subtitle_entries)
            else:
                segment_infos = list(segment_infos)
            graph = None
            # For short video mode, segments already point to the original video
            # No need to copy - just update paths if needed
            segment_infos = _update_segment_paths_to_video(segment_infos, config.video_path)
        else:
            # Check if segments already exist in checkpoint
            if checkpoint_manager is not None and checkpoint_manager.has_checkpoint("segments"):
                print("[pipeline] Loading segments from checkpoint...")
                segment_infos = checkpoint_manager.load_segments()
            else:
                # Segment the video
                print("[pipeline] Segmenting video...")
                segmenter = VideoSegmenter(
                    frame_interval=config.frame_interval,
                    n_clusters=config.n_clusters,
                    min_segment_sec=config.min_segment_sec,
                    output_dir=segments_dir,
                    compile_model=config.compile_model,
                    batch_size=config.batch_size,
                )
                segment_infos = segmenter.segment_video(config.video_path)

                if not segment_infos:
                    raise RuntimeError("No segments were generated from the input video.")

                if checkpoint_manager is not None:
                    checkpoint_manager.save_segments(segment_infos)
                    print(f"[pipeline] Saved segments to checkpoint")

            if subtitle_entries:
                segment_infos = utils._attach_subtitles(segment_infos, subtitle_entries)
            else:
                segment_infos = list(segment_infos)

            # Copy segments to output directory for reranker access
            print(f"[pipeline] Copying segments to output directory...")
            segment_infos = _copy_segments_to_output(segment_infos, output_segments_dir)
            print(f"[pipeline] Segments copied to {output_segments_dir}")

            # Update segment paths to use original video with frame ranges
            # This ensures reranker can access frames even after temp cleanup
            segment_infos = _update_segment_paths_to_video(
                segment_infos, config.video_path
            )

            # Check if segment features (with embeddings) already exist in checkpoint
            if checkpoint_manager is not None and checkpoint_manager.has_checkpoint("segment_features"):
                print("[pipeline] Loading segment features from checkpoint...")
                segment_features = checkpoint_manager.load_segment_features()
            else:
                # Compute video features (embeddings)
                print(f"[pipeline] Computing video features for {len(segment_infos)} segments...")
                with Timer("compute_embeddings"):
                    # Use parallel processing across all available GPUs
                    import torch
                    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
                    print(f"[pipeline] Using {num_gpus} GPUs for parallel processing")

                    # For segments that point to the original video, we need to update paths
                    # to use the copied segment files if they exist
                    segment_paths = [seg.get("path") for seg in segment_infos]
                    if segment_paths and os.path.exists(segment_paths[0]):
                        # Segments are already copied to output_dir/segments
                        pass
                    else:
                        # Update to use copied segments
                        copied_segments_dir = os.path.join(output_segments_dir, "segments")
                        if os.path.exists(copied_segments_dir):
                            segment_infos = _update_segment_paths_to_video(
                                segment_infos, config.video_path
                            )

                    segment_features = compute_video_features(
                        segment_infos,
                        frame_interval=config.embedding_frame_interval,
                        compile_model=config.compile_model,
                        num_workers=config.num_gpus
                    )

                if checkpoint_manager is not None and config.checkpoint_embeddings:
                    # Save segment features with embeddings
                    checkpoint_manager.save_segment_features(segment_features)
                    print(f"[pipeline] Saved segment features to checkpoint")

        print("--------------------segment_infos:")
        print(segment_infos)

        # Analyze query intent
        original_query = config.query.strip()
        intent = None
        time_focus_analysis: Optional[Dict[str, Any]] = None
        time_focus_results: List[Dict[str, Any]] = []
        subtitle_query_text: Optional[str] = None
        cleaned_query: str = original_query

        # Check if intent analysis is already in checkpoint (and matches current query)
        if checkpoint_manager is not None and checkpoint_manager.has_checkpoint("intent_analysis"):
            intent, saved_query = checkpoint_manager.load_intent_analysis()
            if saved_query != original_query:
                print(f"[pipeline] Query changed, re-analyzing intent...")
                intent = None  # Trigger re-analysis
            else:
                print("[pipeline] Loading intent analysis from checkpoint...")
                logger.info(f"Loaded intent analysis for query: {saved_query}")
        else:
            intent = None

        # Run intent analysis if not loaded from checkpoint
        if intent is None:
            with Timer("intent_analysis"):
                with IntentAnalyzer(model_id=config.intent_model_id) as intent_analyzer:
                    intent = intent_analyzer.analyze(original_query, keep_model_loaded=False)
                    intent_str = json.dumps({
                        "subtitle_search": intent.get("subtitle_search"),
                        "time_search": intent.get("time_search"),
                        "reason": intent.get("reason"),
                    }, ensure_ascii=False)
                    print(f"Query intent: {intent_str}")
                    logger.info(f"Query intent analysis: {intent_str}")

                # Save intent analysis to checkpoint
                if checkpoint_manager is not None and config.checkpoint_intent:
                    checkpoint_manager.save_intent_analysis(intent, original_query)
                    logger.info("Intent analysis saved to checkpoint")
                    print(f"[pipeline] Saved intent analysis to checkpoint")

        # Handle subtitle extraction if intent indicates subtitle search
        subtitle_analysis: Optional[Dict[str, Any]] = None
        if intent.get("subtitle_search"):
            with Timer("subtitle_extraction"):
                with IntentAnalyzer(model_id=config.intent_model_id) as subtitle_analyzer:
                    subtitle_analysis = subtitle_analyzer.extract_subtitles(
                        original_query, keep_model_loaded=False
                    )
                extracted_subtitle = subtitle_analysis.get("subtitle_text", "") if subtitle_analysis else ""
                print("----------extracted_subtitles:")
                print(extracted_subtitle)
                if extracted_subtitle:
                    subtitle_query_text = extracted_subtitle

                    cleaned_candidate = (
                        subtitle_analysis.get("cleaned_query") if subtitle_analysis else ""
                    )
                    if cleaned_candidate:
                        cleaned_query = cleaned_candidate
                else:
                    # Subtitle extraction failed - check if we have subtitle JSON
                    # If yes, use it for subtitle search; otherwise, disable subtitle search
                    if subtitle_entries:
                        # Use subtitle JSON content for subtitle search
                        # Combine all subtitle texts as the search query
                        subtitle_query_text = " ".join(
                            entry.get("text", "") for entry in subtitle_entries
                        ).strip()
                    else:
                        # No subtitle text extracted and no JSON available
                        # Disable subtitle search and use original query
                        intent["subtitle_search"] = False
                        subtitle_query_text = None
                        cleaned_query = original_query

                print(
                    "Subtitle rewrite:",
                    json.dumps(
                        {
                            "subtitle_text": subtitle_query_text or "",
                            "cleaned_query": cleaned_query,
                            "reason": subtitle_analysis.get("reason") if subtitle_analysis else "",
                        },
                        ensure_ascii=False,
                    ),
                )

        vision_query = cleaned_query.strip() or original_query or config.query

        print("----------query with no subtitles:")
        print(vision_query)

        total_duration = max((info.get("end_sec") or 0.0) for info in segment_infos) if segment_infos else 0.0
        if total_duration > 0.0:
            approx_duration = total_duration

        # Build graph from segment features
        if checkpoint_manager is not None and checkpoint_manager.has_checkpoint("graph"):
            logger.info("Loading graph from checkpoint...")
            graph_info = checkpoint_manager.load_graph()
            graph_builder = SpatiotemporalGraphBuilder(temporal_weight=config.temporal_weight)
            graph = graph_builder.build(segment_features)
            logger.info(f"Loaded graph info: {graph_info}")
            print("[pipeline] Loading graph from checkpoint...")
            print(f"[pipeline] Loaded graph info: {graph_info}")
        else:
            with Timer("build_graph"):
                graph_builder = SpatiotemporalGraphBuilder(temporal_weight=config.temporal_weight)
                graph = graph_builder.build(segment_features)
                logger.info(f"Built graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                print(f"[pipeline] Built graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

                # Save graph to checkpoint
                if checkpoint_manager is not None and config.checkpoint_graph:
                    checkpoint_manager.save_graph({
                        "num_nodes": graph.number_of_nodes(),
                        "num_edges": graph.number_of_edges(),
                    })
                    logger.info("Graph saved to checkpoint")
                    print(f"[pipeline] Saved graph to checkpoint")

        # Analyze time focus
        if intent.get("time_search"):
            with IntentAnalyzer(model_id=config.intent_model_id) as time_analyzer:
                time_focus_analysis = time_analyzer.analyze_time_focus(
                    original_query, keep_model_loaded=False
                )
            time_focus_str = json.dumps(
                {
                    "mode": time_focus_analysis.get("mode"),
                    "start_time_sec": time_focus_analysis.get("start_time_sec"),
                    "end_time_sec": time_focus_analysis.get("end_time_sec"),
                    "reason": time_focus_analysis.get("reason"),
                },
                ensure_ascii=False,
            )
            print(f"Time focus analysis: {time_focus_str}")
            logger.info(f"Time focus analysis: {time_focus_str}")

            mode = (time_focus_analysis.get("mode") or "none").lower()
            start_sec = float(time_focus_analysis.get("start_time_sec") or 0.0)
            end_sec = float(time_focus_analysis.get("end_time_sec") or 0.0)

            clamped_ratio = max(min(float(config.time_focus_ratio), 1.0), 0.0)
            # Use a minimum duration to ensure time focus range is valid
            effective_duration = max(total_duration, _NEAR_ZERO_DURATION)
            if mode == "start":
                end_sec = effective_duration * clamped_ratio
                start_sec = 0.0
            elif mode == "end":
                start_sec = max(effective_duration * (1.0 - clamped_ratio), 0.0)
                end_sec = effective_duration
            elif mode == "range":
                start_sec = max(min(start_sec, effective_duration), 0.0)
                end_sec = max(min(end_sec, effective_duration), 0.0)
            else:
                start_sec = 0.0
                end_sec = 0.0

            if end_sec < start_sec:
                start_sec, end_sec = end_sec, start_sec

            if end_sec - start_sec < max(float(config.time_min_window), 0.0):
                mid = (start_sec + end_sec) / 2.0
                padding = max(float(config.time_range_padding), 0.0)
                half_window = max(float(config.time_min_window) / 2.0, padding)
                start_sec = max(mid - half_window, 0.0)
                end_sec = min(mid + half_window, effective_duration)

            if end_sec > start_sec:
                time_output_dir = os.path.join(output_dir, "time_focus_frames")
                mode_label = mode if mode in {"start", "end", "range"} else "custom"
                prefix = f"time_{mode_label}"
                time_focus_results = _sparse_sample_time_range(
                    config.video_path,
                    start_sec,
                    end_sec,
                    frame_interval=config.time_sampling_interval,
                    output_dir=time_output_dir,
                    prefix=prefix,
                )

                for item in time_focus_results:
                    item["mode"] = mode_label
                    item["start_sec"] = start_sec
                    item["end_sec"] = end_sec
                    item["timestamp_label"] = timestamp_label(item.get("timestamp", 0.0))

        # Perform retrieval
        if short_video_mode:
            selected_segments = list(segment_infos)
            rerank_inputs = list(segment_infos)
        else:
            effective_top_k = config.top_k
            effective_spatial_k = config.spatial_k
            if intent.get("subtitle_search"):
                effective_top_k = 1
                effective_spatial_k = 1

            with Timer("retrieval"):
                print("[pipeline] Retrieving top-k segments...")
                retriever = VideoRetriever()
                selected_segments = retriever.retrieve(
                    graph,
                    vision_query,
                    top_k=effective_top_k,
                    spatial_k=effective_spatial_k,
                    verbose=True,
                )
                retriever.unload()

            if not selected_segments:
                raise RuntimeError("No segments were selected from the retrieval stage.")

            merged_segments = _merge_attribute_results(
                graph,
                selected_segments,
                intent=intent,
                query=vision_query,
                attribute_top_k=config.attribute_top_k,
                subtitle_query=subtitle_query_text,
                subtitle_neighbor_hops=config.subtitle_neighbor_hops,
            )

            max_segments = max(config.top_k + config.attribute_top_k, len(selected_segments))
            rerank_inputs = merged_segments[:max_segments]

            # Update subtitle_query_text to use the matched subtitle text from the top hit
            # This ensures the answer generator gets the relevant subtitle text, not the full transcript
            if intent.get("subtitle_search") and merged_segments:
                # Find the segment with highest subtitle_similarity
                best_subtitle_segment = max(
                    merged_segments,
                    key=lambda x: x.get("subtitle_similarity", 0.0),
                    default=None
                )
                if best_subtitle_segment and best_subtitle_segment.get("subtitle_similarity", 0.0) > 0:
                    matched_subtitle = best_subtitle_segment.get("subtitle_text", "")
                    if matched_subtitle:
                        subtitle_query_text = matched_subtitle

            # If time_search is enabled, add time focus frames to rerank_inputs
            # so they get scored along with retrieved segments
            if intent.get("time_search") and time_focus_results:
                print(f"[pipeline] Adding {len(time_focus_results)} time focus frames to reranking...")
                for item in time_focus_results:
                    # Convert time focus frame to segment-like format for scoring
                    # Use frame_index as segment boundaries (single frame segment)
                    frame_idx = item.get("frame_index", 0)
                    # Get FPS from video metadata
                    fps = float(video_metadata.get("fps", 0.0) or 0.0)
                    if fps <= 0.0:
                        fps = 30.0
                    segment = {
                        "path": config.video_path,
                        "start_frame": frame_idx,
                        "end_frame": frame_idx,
                        "fps": fps,
                        "segment_index": -1,  # Mark as time focus frame
                        "start_sec": item.get("timestamp", 0.0),
                        "end_sec": item.get("timestamp", 0.0),
                        "similarity": 0.0,  # No retrieval score - will be scored by reranker
                        "subtitle_similarity": 0.0,
                        "time_similarity": 1.0,  # Give high time similarity since this is time-focused
                    }
                    rerank_inputs.append(segment)
                print(f"[pipeline] Total rerank inputs now: {len(rerank_inputs)}")

            # Save retrieval results to checkpoint
            if checkpoint_manager is not None:
                checkpoint_manager.save_retrieval_results(
                    merged_segments,
                    vision_query,
                )
                checkpoint_manager.save_graph({
                    "num_nodes": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                    "selected_nodes": [s.get("node_id") for s in merged_segments],
                })
                print(f"[pipeline] Saved graph and retrieval results to checkpoint")

        if not rerank_inputs:
            raise RuntimeError("No segments are available for reranking.")

        # Check if rerank results already exist in checkpoint (and query matches)
        rerank_results_loaded = False
        if checkpoint_manager is not None and checkpoint_manager.has_checkpoint("rerank_results"):
            saved_frames, saved_time_focus, saved_query = checkpoint_manager.load_rerank_results()
            # Only load if the query matches the current vision_query
            if saved_query == vision_query:
                print("[pipeline] Loading rerank results from checkpoint...")
                final_frames = saved_frames
                time_focus_results = saved_time_focus
                rerank_results_loaded = True
            else:
                print(f"[pipeline] Query changed ({saved_query} -> {vision_query}), re-running reranking...")

        if not rerank_results_loaded:
            # Perform reranking
            print(f"[pipeline] Reranking frames ({len(rerank_inputs)} segments)...")
            with Timer("reranking"):
                with FrameReranker(
                    model_id=config.reranker_model_id,
                    adapter_dir=config.reranker_adapter_dir,
                    compile_model=config.reranker_compile_model,
                ) as reranker:
                    # Clear time_focus_results - they'll be scored as part of rerank_inputs
                    if intent.get("time_search"):
                        print("[pipeline] Clearing original time focus sampling - will use scored frames from reranking")
                        time_focus_results = []
                    final_frames = reranker.rerank(
                        rerank_inputs,
                        query=vision_query,
                        frame_interval=config.rerank_frame_interval,
                        top_frames=config.top_frames,
                        output_dir=output_dir,
                        min_frames_per_clip=config.min_frames_per_clip,
                        target_sample_fps=3.0 if short_video_mode else None,
                        subtitle_query=subtitle_query_text,
                        verbose=True,
                    )

            # Save rerank results to checkpoint
            if checkpoint_manager is not None and config.checkpoint_rerank:
                # When time_search is enabled, time focus frames are merged into rerank_inputs
                # and scored together. We save the scored time focus frames from final_frames.
                scored_time_focus = [f for f in final_frames if f.get("segment_index") == -1]
                checkpoint_manager.save_rerank_results(final_frames, scored_time_focus, vision_query)
                print(f"[pipeline] Saved rerank results to checkpoint")

        # Also save to final results for backward compatibility
        if checkpoint_manager is not None:
            checkpoint_manager.save_final_results(
                final_frames,
                time_focus_results,
                intent,
            )
            print(f"[pipeline] Saved final results to checkpoint")

        # Save results
        results_path = os.path.join(output_dir, "rerank_results.json")
        utils._save_json(final_frames, results_path)

        # Save scored time focus frames (from reranking)
        if intent.get("time_search"):
            scored_time_focus = [f for f in final_frames if f.get("segment_index") == -1]
            if scored_time_focus:
                time_results_path = os.path.join(output_dir, "time_focus_results.json")
                utils._save_json(scored_time_focus, time_results_path)

        plan_path = os.path.join(output_dir, "retrieval_plan.json")
        utils._save_json(
            {
                "intent": intent,
                "selected_segments": utils._to_serializable(selected_segments),
                "query_variants": {
                    "original": config.query,
                    "vision": vision_query,
                    "subtitle": subtitle_query_text or "",
                },
                "subtitle_analysis": subtitle_analysis,
                "time_focus_analysis": time_focus_analysis,
                "time_focus_range": {
                    "start_sec": time_focus_results[0]["start_sec"] if time_focus_results else 0.0,
                    "end_sec": time_focus_results[0]["end_sec"] if time_focus_results else 0.0,
                },
                "short_video_mode": short_video_mode,
                "video_metadata": {
                    "fps": video_fps,
                    "total_frames": video_total_frames,
                    "duration_sec": approx_duration,
                },
                "correct_choice": config.correct_choice,
            },
            plan_path,
        )

        # Print timing summary
        print("\n" + "=" * 50)
        print("  Pipeline Timing Summary")
        print("=" * 50)
        print(f"  Total pipeline time: {time.time() - start_time:.2f}s")
        print("=" * 50)

        # Extract subtitle frames (frames with is_subtitle_frame=True)
        subtitle_frames = [f for f in final_frames if f.get("is_subtitle_frame", False)]

        return PipelineResult(
            reranked_frames=final_frames,
            time_focus_frames=time_focus_results,
            subtitle_frames=subtitle_frames,
            intent=intent,
            retrieval_plan={
                "selected_segments": selected_segments,
                "query_variants": {
                    "original": config.query,
                    "vision": vision_query,
                    "subtitle": subtitle_query_text or "",
                },
                "subtitle_analysis": subtitle_analysis,
                "time_focus_analysis": time_focus_analysis,
            },
            video_metadata={
                "fps": video_fps,
                "total_frames": video_total_frames,
                "duration_sec": approx_duration,
            },
        )

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def run_batch_pipeline(config: PipelineConfig, **kwargs) -> Dict[str, Dict[str, List[Dict]]]:
    """Run the pipeline for multiple videos from a batch configuration.

    Args:
        config: Base PipelineConfig (will be overridden by batch entries).
        **kwargs: Additional config options.

    Returns:
        Dictionary mapping entry IDs to their results.
    """
    # This is a placeholder - batch processing should use run_batch_from_config
    raise NotImplementedError("Batch processing uses run_batch_from_config with JSON input")


def run_batch_from_config(
    config_path: str,
    output_root: str,
    *,
    video_root: Optional[str] = None,
    subtitle_root: Optional[str] = None,
    default_query_field: str = "query_retrieval",
    fallback_query_fields: Optional[Iterable[str]] = None,
    skip_existing: bool = True,
    **pipeline_kwargs,
) -> Dict[str, Dict[str, List[Dict]]]:
    """Run the pipeline for every entry defined in a batch configuration file.

    Args:
        config_path: Path to the JSON file describing the batch inputs.
        output_root: Directory where per-video results will be written.
        video_root: Optional base directory for resolving video paths.
        subtitle_root: Optional base directory for resolving subtitle paths.
        default_query_field: Preferred key in each JSON entry for the query text.
        fallback_query_fields: Additional keys to look up if the default is missing.
        skip_existing: When True (default), skip entries whose output directory
            already exists under output_root.
        **pipeline_kwargs: Additional keyword arguments for PipelineConfig.

    Returns:
        A mapping from entry identifiers to the resulting data.
    """
    import json as _json

    with open(config_path, "r", encoding="utf-8") as f:
        entries = _json.load(f)

    if not isinstance(entries, list):
        raise ValueError("Batch configuration must be a JSON list of entries.")

    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)

    batch_results: Dict[str, Dict[str, List[Dict]]] = {}
    fallback_fields = list(fallback_query_fields or ["query", "query_vlm"])

    def _with_metadata(results: Dict[str, Any], entry_data: Dict[str, Any]) -> Dict[str, Any]:
        if "correct_choice" not in entry_data:
            return results
        merged = dict(results)
        merged["correct_choice"] = entry_data.get("correct_choice")
        return merged

    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            print(f"[pipeline] Skipping non-dict entry at index {idx - 1}.")
            continue

        video_rel_path = entry.get("video_path")
        video_path = utils._resolve_path(video_root, video_rel_path)
        if not video_path or not os.path.exists(video_path):
            print(
                f"[pipeline] Skipping entry {entry.get('id', idx)}: video not found"
                f" ({video_rel_path})."
            )
            continue

        subtitle_rel_path = entry.get("subtitle_path")
        subtitle_path = utils._resolve_path(subtitle_root, subtitle_rel_path)
        if subtitle_path and not os.path.exists(subtitle_path):
            subtitle_path = None

        query = entry.get(default_query_field)
        if not query:
            for field in fallback_fields:
                query = entry.get(field)
                if query:
                    break
        if not query:
            print(f"[pipeline] Skipping entry {entry.get('id', idx)}: missing query text.")
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        item_id = entry.get("id") or video_name
        video_output_dir = os.path.join(output_root, video_name)

        if skip_existing and os.path.isdir(video_output_dir):
            print(
                f"[pipeline] Skipping entry {idx}/{len(entries)} (id={item_id}): "
                f"existing output found at {video_output_dir}."
            )

            existing_results: Dict[str, List[Dict]] = {}
            rerank_path = os.path.join(video_output_dir, "rerank_results.json")
            time_path = os.path.join(video_output_dir, "time_focus_results.json")

            if os.path.exists(rerank_path):
                try:
                    with open(rerank_path, "r", encoding="utf-8") as f:
                        existing_results["reranked_frames"] = _json.load(f)
                except (OSError, _json.JSONDecodeError) as exc:
                    print(
                        f"[pipeline] Warning: Failed to load existing rerank results "
                        f"for {item_id}: {exc}"
                    )

            if os.path.exists(time_path):
                try:
                    with open(time_path, "r", encoding="utf-8") as f:
                        existing_results["time_focus_frames"] = _json.load(f)
                except (OSError, _json.JSONDecodeError) as exc:
                    print(
                        f"[pipeline] Warning: Failed to load existing time focus results "
                        f"for {item_id}: {exc}"
                    )

            if existing_results:
                batch_results[item_id] = _with_metadata(existing_results, entry)
            else:
                batch_results[item_id] = _with_metadata({"skipped": True}, entry)
            continue

        print("=" * 80)
        progress_msg = f"[pipeline] Processing entry {idx}/{len(entries)}: id={item_id}, video={video_rel_path}"
        print(progress_msg)

        try:
            config = PipelineConfig(
                video_path=video_path,
                query=query,
                output_dir=video_output_dir,
                **pipeline_kwargs,
            )
            results = run_pipeline(config)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[pipeline] Skipping entry {idx}/{len(entries)} (id={item_id}) due to error: {exc}"
            )
            if os.path.isdir(video_output_dir):
                try:
                    shutil.rmtree(video_output_dir)
                except OSError as cleanup_exc:
                    print(
                        f"[pipeline] Warning: Failed to clean output directory for {item_id}: {cleanup_exc}"
                    )

            batch_results[item_id] = _with_metadata({"skipped": True, "error": str(exc)}, entry)
            continue

        batch_results[item_id] = _with_metadata(results.to_dict(), entry)

    summary_path = os.path.join(output_root, "batch_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        _json.dump(utils._to_serializable(batch_results), f, ensure_ascii=False, indent=2)

    print(f"[pipeline] Batch processing complete. Summary saved to {summary_path}")

    return batch_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full long-video retrieval pipeline")
    parser.add_argument("--video", type=str, default=None, help="Path to the input video")
    parser.add_argument("--query", type=str, default=None, help="Text query for retrieval")
    parser.add_argument("--output", type=str, default="./output/", help="Directory to save the final ranked frames")
    parser.add_argument("--frame-interval", type=int, default=45, dest="frame_interval")
    parser.add_argument("--clusters", type=int, default=30, dest="n_clusters")
    parser.add_argument("--min-segment-sec", type=float, default=1, dest="min_segment_sec")
    parser.add_argument("--embed-frame-interval", type=int, default=10, dest="embedding_frame_interval")
    parser.add_argument("--top-k", type=int, default=1, dest="top_k")
    parser.add_argument("--spatial-k", type=int, default=1, dest="spatial_k")
    parser.add_argument("--rerank-frame-interval", type=int, default=5, dest="rerank_frame_interval")
    parser.add_argument("--top-frames", type=int, default=256, dest="top_frames")
    parser.add_argument("--temporal-weight", type=float, default=1.0, dest="temporal_weight")
    parser.add_argument("--subtitle-json", type=str, default=None, dest="subtitle_json")
    parser.add_argument("--attribute-top-k", type=int, default=1, dest="attribute_top_k")
    parser.add_argument("--min-frames-per-clip", type=int, default=4, dest="min_frames_per_clip")
    parser.add_argument(
        "--subtitle-neighbor-hops", type=int, default=3, dest="subtitle_neighbor_hops"
    )
    parser.add_argument(
        "--time-focus-ratio", type=float, default=0.04, dest="time_focus_ratio"
    )
    parser.add_argument(
        "--time-sampling-interval", type=int, default=20, dest="time_sampling_interval"
    )
    parser.add_argument(
        "--time-range-padding", type=float, default=1.0, dest="time_range_padding"
    )
    parser.add_argument(
        "--time-min-window", type=float, default=2.0, dest="time_min_window"
    )
    parser.add_argument(
        "--short-video-threshold",
        type=float,
        default=240.0,
        dest="short_video_threshold",
        help="Duration (in seconds) below which the pipeline skips graph retrieval and samples at 3 FPS.",
    )
    parser.add_argument("--disable-checkpoint", action="store_true", dest="disable_checkpoint")
    parser.add_argument("--disable-reranker-compile", action="store_true", dest="disable_reranker_compile", help="Disable torch.compile for reranker (useful for debugging or unsupported hardware)")
    parser.add_argument("--enable-segment-compile", action="store_true", dest="enable_segment_compile", help="Enable torch.compile for vision model during segmentation")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", dest="checkpoint_dir")
    parser.add_argument("--cache-dir", type=str, default="cache", dest="cache_dir")
    parser.add_argument("--batch-config", type=str, default=None, dest="batch_config", help="JSON file describing batch inputs")
    parser.add_argument("--video-root", type=str, default="./videos", dest="video_root", help="Base directory for resolving relative video paths in batch mode")
    parser.add_argument(
        "--subtitle-root",
        type=str,
        default="./subtitles",
        dest="subtitle_root",
        help="Base directory for resolving relative subtitle paths in batch mode",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reprocess entries even if their output directory already exists. "
            "By default, the batch runner skips completed videos so that runs can resume."
        ),
    )

    args = parser.parse_args()

    pipeline_kwargs = dict(
        frame_interval=args.frame_interval,
        n_clusters=args.n_clusters,
        min_segment_sec=args.min_segment_sec,
        embedding_frame_interval=args.embedding_frame_interval,
        top_k=args.top_k,
        spatial_k=args.spatial_k,
        rerank_frame_interval=args.rerank_frame_interval,
        top_frames=args.top_frames,
        temporal_weight=args.temporal_weight,
        attribute_top_k=args.attribute_top_k,
        min_frames_per_clip=args.min_frames_per_clip,
        subtitle_neighbor_hops=args.subtitle_neighbor_hops,
        time_focus_ratio=args.time_focus_ratio,
        time_sampling_interval=args.time_sampling_interval,
        time_range_padding=args.time_range_padding,
        time_min_window=args.time_min_window,
        short_video_threshold=args.short_video_threshold,
        enable_checkpoint=not args.disable_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        cache_dir=args.cache_dir,
        reranker_compile_model=not args.disable_reranker_compile,
        compile_model=args.enable_segment_compile,
    )

    batch_config = args.batch_config.strip() if args.batch_config else None
    if batch_config:
        run_batch_from_config(
            config_path=batch_config,
            output_root=args.output,
            video_root=args.video_root,
            subtitle_root=args.subtitle_root,
            skip_existing=not args.force,
            **pipeline_kwargs,
        )
    else:
        video_path = utils._resolve_path(args.video_root, args.video)
        subtitle_path = utils._resolve_path(args.subtitle_root, args.subtitle_json)

        config = PipelineConfig(
            video_path=video_path or "",
            query=args.query or "",
            output_dir=args.output,
            subtitle_json=subtitle_path,
            **pipeline_kwargs,
        )
        results = run_pipeline(config)

        print(json.dumps(utils._to_serializable(results.to_dict()), ensure_ascii=False, indent=2))