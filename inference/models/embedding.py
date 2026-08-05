"""Video embedding model for VideoStir.

Computes visual embeddings for video segments using CLIP-based models.
These embeddings are used for semantic similarity search in the retrieval phase.
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image

from .. import core


class VideoEmbedder:
    """Computes visual embeddings for video segments.

    Uses a CLIP-based model to encode video segments into embeddings
    that can be used for semantic similarity search.

    Example:
        >>> embedder = VideoEmbedder()
        >>> segments = [{"path": "segment1.mp4"}, {"path": "segment2.mp4"}]
        >>> results = embedder.embed_segments(segments, frame_interval=10)
    """

    def __init__(
        self,
        model_name: str = "PE-Core-G14-448",
        device: str | None = None,
        compile_model: bool = False,
    ):
        """Initialize the video embedder.

        Args:
            model_name: Name of the CLIP model to use.
            device: Device to run on ('cuda' or 'cpu'). Defaults to 'cuda' if available.
            compile_model: If True, use torch.compile to accelerate the model.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compile_model = compile_model

        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Load the CLIP model and preprocessing transform."""
        if self._model is None:
            import core.vision_encoder.pe as pe
            import core.vision_encoder.transforms as transforms

            model = pe.CLIP.from_config(self.model_name, pretrained=True).to(self.device)

            # Compile the model for faster inference (requires PyTorch 2.0+)
            if self.compile_model and torch.__version__ >= "2.0.0":
                try:
                    model = torch.compile(model)
                    print(f"[VideoEmbedder] Model compiled with torch.compile")
                except Exception as e:
                    print(f"[VideoEmbedder] Warning: torch.compile failed: {e}")

            self._model = model
            self._preprocess = transforms.get_image_transform(self._model.image_size)
            self._tokenizer = transforms.get_text_tokenizer(self._model.context_length)

    def _preprocess_video(
        self, video_path: str, frame_interval: int = 5
    ) -> torch.Tensor:
        """Load and preprocess video frames.

        Args:
            video_path: Path to the video file.
            frame_interval: Sampling interval (every Nth frame).

        Returns:
            Tensor of shape (num_frames, channels, height, width).
        """
        import decord
        from PIL import Image

        self._load_model()

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        frame_indices = list(range(0, total_frames, frame_interval))
        frames = vr.get_batch(frame_indices).asnumpy()

        # Preprocess frames in batches for efficiency
        batch_size = 32
        preprocessed_list = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_preprocessed = [
                self._preprocess(Image.fromarray(frame)) for frame in batch_frames
            ]
            preprocessed_list.extend(batch_preprocessed)

        return torch.stack(preprocessed_list, dim=0)

    @torch.no_grad()
    def compute_video_features(
        self,
        segment_infos: Iterable[Dict[str, Any]],
        frame_interval: int = 10,
        chunk_size: int = 16,
        compile_model: bool | None = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Compute visual features for video segments.

        Args:
            segment_infos: List of segment metadata dictionaries (must contain 'path').
            frame_interval: Frame sampling interval.
            chunk_size: Number of frames processed per chunk (for VRAM efficiency).
            compile_model: Override compile_model setting if provided.
            verbose: Print progress information.

        Returns:
            List of segment dictionaries with added 'image_features' field.
        """
        # Allow compile_model override
        if compile_model is not None:
            self.compile_model = compile_model
        self._load_model()

        results: List[Dict[str, Any]] = []
        device = torch.device(self.device)

        # Try to import tqdm for progress bar
        try:
            from tqdm import tqdm
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False

        segment_infos_list = list(segment_infos)
        total_segments = len(segment_infos_list)

        # Pre-compute total frames for overall progress estimation
        total_frames_to_process = 0
        for info in segment_infos_list:
            video_path = info.get("path")
            if video_path and os.path.exists(video_path):
                try:
                    import decord
                    vr = decord.VideoReader(video_path)
                    total_frames_to_process += len(vr)
                except:
                    pass

        if HAS_TQDM and verbose:
            overall_pbar = tqdm(total=total_frames_to_process, desc="Processing video features", unit="frame")
            segment_pbar = tqdm(total=total_segments, desc=f"Segments ({total_segments})", unit="seg", leave=False)
        else:
            overall_pbar = None
            segment_pbar = None

        with torch.no_grad():
            for info in segment_infos_list:
                video_path = info.get("path")
                if not video_path or not os.path.exists(video_path):
                    print(f"[x] Missing file: {video_path}")
                    continue

                try:
                    video = self._preprocess_video(video_path, frame_interval=frame_interval)
                    total_frames = video.shape[0]
                    if total_frames == 0:
                        print(f"[x] No valid frames in {video_path}")
                        continue

                    feats: List[torch.Tensor] = []
                    for i in range(0, total_frames, chunk_size):
                        chunk = video[i:i + chunk_size].unsqueeze(0).to(device, non_blocking=True)
                        f = self._model.encode_video(chunk)
                        f = f / f.norm(dim=-1, keepdim=True)
                        feats.append(f.cpu())

                        if overall_pbar:
                            overall_pbar.update(min(chunk_size, total_frames - i))
                        del chunk, f
                        torch.cuda.empty_cache()

                    image_features = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)

                    enriched = dict(info)
                    enriched["image_features"] = image_features
                    results.append(enriched)

                    if segment_pbar:
                        segment_pbar.update(1)

                    if verbose:
                        print(f"[✓] Processed {os.path.basename(video_path)} ({total_frames} frames)")

                except Exception as e:
                    print(f"[x] Failed on {video_path}: {e}")
                    torch.cuda.empty_cache()
                    continue

        if overall_pbar:
            overall_pbar.close()
        if segment_pbar:
            segment_pbar.close()

        return results

    def unload(self) -> None:
        """Unload the model from memory to free GPU resources."""
        if self._model is not None:
            del self._model
            self._model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def __enter__(self) -> "VideoEmbedder":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures model is unloaded."""
        self.unload()

    def __del__(self) -> None:
        """Destructor - ensures model is unloaded."""
        try:
            self.unload()
        except Exception:
            pass


def _process_segment_worker(args):
    """Worker function for parallel segment processing.

    Args:
        args: Tuple of (segment_info, model_name, device, frame_interval, compile_model)

    Returns:
        Updated segment info with image_features, or None if failed.
    """
    segment_info, model_name, device, frame_interval, compile_model = args

    try:
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms
        import decord
        from PIL import Image

        # Load model
        model = pe.CLIP.from_config(model_name, pretrained=True).to(device)
        model.eval()

        if compile_model and torch.__version__ >= "2.0.0":
            try:
                model = torch.compile(model)
            except Exception:
                pass

        preprocess = transforms.get_image_transform(model.image_size)
        vr = decord.VideoReader(segment_info.get("path"))

        # Get frames for this segment
        start_frame = segment_info.get("start_frame", 0)
        end_frame = segment_info.get("end_frame", len(vr) - 1)
        frame_indices = list(range(start_frame, min(end_frame + 1, len(vr)), frame_interval))

        frames = vr.get_batch(frame_indices).asnumpy()
        preprocessed = [preprocess(Image.fromarray(frame)) for frame in frames]
        video = torch.stack(preprocessed).to(device, non_blocking=True)

        # Encode
        with torch.no_grad():
            feats = []
            chunk_size = 16
            for i in range(0, len(video), chunk_size):
                chunk = video[i:i + chunk_size].unsqueeze(0)
                f = model.encode_video(chunk)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu())
                del chunk, f
                torch.cuda.empty_cache()

            image_features = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)

        result = dict(segment_info)
        result["image_features"] = image_features
        return result

    except Exception as e:
        print(f"  [x] Failed on segment: {e}")
        return None


def compute_video_features(
    segment_infos: Iterable[Dict[str, Any]],
    frame_interval: int = 10,
    chunk_size: int = 16,
    compile_model: bool = False,
    num_workers: int = 8,  # Number of parallel workers (GPUs)
) -> List[Dict[str, Any]]:
    """Convenience function to compute video features.

    Args:
        segment_infos: List of segment metadata dictionaries.
        frame_interval: Frame sampling interval.
        chunk_size: Chunk size for VRAM efficiency.
        compile_model: If True, use torch.compile to accelerate the model.
        num_workers: Number of parallel workers (recommend 1 per GPU).

    Returns:
        List of segment dictionaries with image features.
    """
    # If only 1 worker, use sequential processing
    if num_workers <= 1:
        with VideoEmbedder(compile_model=compile_model) as embedder:
            return embedder.compute_video_features(
                segment_infos=segment_infos,
                frame_interval=frame_interval,
                chunk_size=chunk_size,
            )

    # Parallel processing
    model_name = "PE-Core-G14-448"

    # Distribute segments across workers (round-robin by GPU index)
    worker_segments = [[] for _ in range(num_workers)]
    for i, seg in enumerate(segment_infos):
        worker_segments[i % num_workers].append(seg)

    # Prepare arguments for each worker
    all_args = []
    for gpu_idx in range(num_workers):
        device = f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu"
        for seg in worker_segments[gpu_idx]:
            all_args.append((seg, model_name, device, frame_interval, compile_model))

    print(f"[compute_video_features] Processing {len(all_args)} segments across {num_workers} GPUs...")

    # Use 'spawn' start method for CUDA compatibility
    import multiprocessing as mp
    ctx = mp.get_context('spawn')

    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(_process_segment_worker, all_args)

    # Filter out None results
    return [r for r in results if r is not None]