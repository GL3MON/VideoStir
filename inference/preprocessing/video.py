"""Video segmentation utilities for VideoStir."""

from __future__ import annotations

import cv2
import numpy as np
import os
import sys
import torch
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .. import utils
from sklearn.cluster import KMeans, DBSCAN


class _StderrSuppressor:
    """Context manager to suppress stderr output (for FFmpeg warnings)."""
    def __init__(self):
        self.devnull = None
        self.old_stderr = None

    def __enter__(self):
        # Flush stderr first
        sys.stderr.flush()
        # Save original stderr
        self.old_stderr = os.dup(2)
        # Open /dev/null
        self.devnull = open(os.devnull, 'wb')
        # Redirect stderr to /dev/null
        os.dup2(self.devnull.fileno(), 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.devnull:
            self.devnull.close()
        if self.old_stderr is not None:
            os.dup2(self.old_stderr, 2)
            os.close(self.old_stderr)

def extract_visual_embeddings(
    video_path: str, frame_interval: int = 30, compile_model: bool = False,
    batch_size: int = 32, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract visual embeddings from a video using PE-Core-G14-448 model.

    Uses batching for significantly faster inference.

    Args:
        video_path: Path to the video file.
        frame_interval: Frame sampling interval.
        compile_model: If True, use torch.compile to accelerate the model.
        batch_size: Number of frames to process in each batch.
        verbose: Print progress information.

    Returns:
        Tuple of (embeddings, frames) arrays.
    """
    import sys
    sys.path.insert(0, "/hfcache/harissh/VideoStir/inference")
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms
    from PIL import Image

    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False

    frames_list, embeddings_list = [], []
    idx = 0

    # Suppress FFmpeg/H.264 decoder warnings from the very beginning
    with _StderrSuppressor():
        # Open video file
        cap = cv2.VideoCapture(video_path)

        # Get total frames for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_count = (total_frames + frame_interval - 1) // frame_interval

        # Load model once
        model_name = "PE-Core-G14-448"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = pe.CLIP.from_config(model_name, pretrained=True).to(device)
        model.eval()

        # Compile the model for faster inference (requires PyTorch 2.0+)
        if compile_model and torch.__version__ >= "2.0.0":
            try:
                model = torch.compile(model)
                print(f"[extract_visual_embeddings] Model compiled with torch.compile")
            except Exception as e:
                print(f"[extract_visual_embeddings] Warning: torch.compile failed: {e}")

        preprocess = transforms.get_image_transform(model.image_size)

        batch_images = []
        batch_frame_indices = []

        # Progress bar setup
        if HAS_TQDM and verbose:
            pbar = tqdm(total=sampled_count, desc="Embedding frames", unit="frame")
        else:
            pbar = None

    # Suppress FFmpeg/H.264 decoder warnings during frame reading
    with _StderrSuppressor():
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % frame_interval == 0:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    preprocessed = preprocess(image)
                    batch_images.append(preprocessed)
                    batch_frame_indices.append(idx)

                    # Process batch when full
                    if len(batch_images) >= batch_size:
                        batch_tensor = torch.stack(batch_images).to(device)
                        batch_emb = model.encode_image(batch_tensor)
                        batch_emb = batch_emb.detach().cpu().numpy()

                        for i, frame_idx in enumerate(batch_frame_indices):
                            embeddings_list.append(batch_emb[i])
                            frames_list.append(frame_idx)

                        batch_images = []
                        batch_frame_indices = []

                        if pbar:
                            pbar.update(len(batch_frame_indices))

                idx += 1

        # Process remaining frames in incomplete batch
        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            batch_emb = model.encode_image(batch_tensor)
            batch_emb = batch_emb.detach().cpu().numpy()

            for i, frame_idx in enumerate(batch_frame_indices):
                embeddings_list.append(batch_emb[i])
                frames_list.append(frame_idx)

            if pbar:
                pbar.update(len(batch_frame_indices))

        if pbar:
            pbar.close()

    cap.release()

    return np.array(embeddings_list), np.array(frames_list)


def cluster_and_segment(
    video_path: str,
    embeddings: np.ndarray,
    frames: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 5,
) -> Tuple[List[int], np.ndarray]:
    """Cluster video frames and determine segment boundaries.

    Args:
        video_path: Path to the video file.
        embeddings: Embedding vectors for each sampled frame.
        frames: Frame indices for each embedding.
        method: Clustering method ('kmeans', 'dbscan', 'pelt').
        n_clusters: Number of clusters for kmeans.

    Returns:
        Tuple of (change_points, labels).
    """
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(embeddings)
    elif method == "dbscan":
        clusterer = DBSCAN(eps=1.5, min_samples=3)
        labels = clusterer.fit_predict(embeddings)
    elif method == "pelt":
        from sklearn.metrics.pairwise import cosine_similarity
        import ruptures as rpt

        if len(embeddings) != len(frames):
            raise ValueError("The lengths of the embeddings and the frames do not match.")
        if len(embeddings) < 2:
            return [int(frames[0]), int(frames[-1])]

        pairwise_sim = cosine_similarity(embeddings)
        semantic_shift = 1.0 - np.clip(np.diag(pairwise_sim, k=1), 0.0, 1.0)
        series = np.concatenate([[0.0], semantic_shift])[:, None]

        algo = rpt.Pelt(model="rbf", jump=1, min_size=5).fit(series)
        idx_change_points = algo.predict(pen=5.0)

        change_points = [int(frames[0])]
        for idx in idx_change_points:
            if idx >= len(frames):
                continue
            change_points.append(int(frames[idx]))

        if change_points[-1] != int(frames[-1]):
            change_points.append(int(frames[-1]))

        change_points = sorted(dict.fromkeys(change_points))
        return change_points, np.zeros(len(frames), dtype=int)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    if method != "pelt":
        change_points = [int(frames[0])]
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                change_points.append(int(frames[i]))
        change_points.append(int(frames[-1]))

    return change_points, labels


def export_segments(
    video_path: str,
    change_points: List[int],
    output_prefix: str = "segment",
    min_segment_sec: float = 10,
    output_dir: str = "segments",
) -> List[Dict[str, Any]]:
    """Export video segments to files.

    Args:
        video_path: Path to the source video.
        change_points: List of frame indices for segment boundaries.
        output_prefix: Prefix for output filenames.
        min_segment_sec: Minimum segment duration in seconds.
        output_dir: Directory to save segments.

    Returns:
        List of segment metadata dictionaries.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    min_frames = int(fps * min_segment_sec)

    merged_points = [change_points[0]]
    i = 1
    while i < len(change_points):
        start, end = change_points[i - 1], change_points[i]
        seg_len = end - start

        if seg_len < min_frames and i < len(change_points) - 1:
            i += 1
            continue
        else:
            merged_points.append(change_points[i])
            i += 1

    segment_infos = []
    seg_id = 0
    for i in range(len(merged_points) - 1):
        start, end = merged_points[i], merged_points[i + 1]
        out_path = os.path.join(output_dir, f"{output_prefix}_{seg_id}.mp4")

        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for j in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        start_sec = float(start) / fps if fps else 0.0
        end_sec = float(end) / fps if fps else start_sec
        segment_infos.append(
            {
                "path": out_path,
                "start_frame": start,
                "end_frame": end,
                "fps": fps,
                "segment_index": seg_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": max(end_sec - start_sec, 0.0),
            }
        )
        seg_id += 1

    cap.release()
    return segment_infos


class VideoSegmenter:
    """High-level video segmentation interface."""

    def __init__(
        self,
        frame_interval: int = 30,
        n_clusters: int = 10,
        min_segment_sec: float = 10,
        output_dir: str = "segments",
        compile_model: bool = False,
        verbose: bool = True,
        batch_size: int = 32,
    ):
        """Initialize the video segmenter.

        Args:
            frame_interval: Frame sampling interval for embeddings.
            n_clusters: Number of clusters for kmeans segmentation.
            min_segment_sec: Minimum segment duration.
            output_dir: Directory to save segment files.
            compile_model: If True, use torch.compile to accelerate the vision model.
            verbose: Print progress information.
            batch_size: Number of frames to process in each batch (larger = faster but more VRAM).
        """
        self.frame_interval = frame_interval
        self.n_clusters = n_clusters
        self.min_segment_sec = min_segment_sec
        self.output_dir = output_dir
        self.compile_model = compile_model
        self.verbose = verbose
        self.batch_size = batch_size

    def segment_video(self, video_path: str) -> List[Dict[str, Any]]:
        """Segment a video into meaningful clips.

        Args:
            video_path: Path to the input video.

        Returns:
            List of segment metadata dictionaries.
        """
        if self.verbose:
            print(f"[VideoSegmenter] Extracting embeddings from {os.path.basename(video_path)}...")
        embeddings, frames = extract_visual_embeddings(
            video_path, frame_interval=self.frame_interval,
            compile_model=self.compile_model, verbose=self.verbose,
            batch_size=self.batch_size
        )
        if self.verbose:
            print(f"[VideoSegmenter] Clustering {len(embeddings)} embeddings into {self.n_clusters} segments...")
        change_points, _ = cluster_and_segment(
            video_path, embeddings, frames,
            method="pelt"
        )
        if self.verbose:
            print(f"[VideoSegmenter] Exporting {len(change_points) - 1} segments...")
        return export_segments(
            video_path, change_points,
            min_segment_sec=self.min_segment_sec,
            output_dir=self.output_dir,
        )