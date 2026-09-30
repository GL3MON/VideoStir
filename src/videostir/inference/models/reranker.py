"""Frame reranking model for VideoStir.

Scores frames for relevance to a query and produces a ranked list
of the most relevant frames for downstream reasoning.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ..time_utils import timestamp_label

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# Default model for reranking
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_ADAPTER_DIR = "./result"

# Batch size for parallel frame scoring
DEFAULT_BATCH_SIZE = 64


class FrameReranker:
    """Scores video frames for relevance to a query and reranks them.

    This model uses a Qwen2.5-VL model with a LoRA adapter to score
    individual frames based on their relevance to the user's query.
    Frames are scored from 1-5, where:
    - 1 = completely irrelevant
    - 5 = highly relevant (decisive evidence)

    Example:
        >>> reranker = FrameReranker()
        >>> frames = reranker.rerank(segments, query="Show me the opening")
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        adapter_dir: str = DEFAULT_ADAPTER_DIR,
        device: Optional[str] = None,
        torch_dtype: str = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        compile_model: bool = False,
        verbose: bool = True,
    ):
        """Initialize the frame reranker.

        Args:
            model_id: Hugging Face model ID for the base Qwen2.5-VL model.
            adapter_dir: Path to the LoRA adapter weights.
            device: Device to run the model on (e.g., 'cuda', 'cpu').
            torch_dtype: PyTorch dtype for model weights.
            batch_size: Number of frames to process in parallel (default: 64).
            compile_model: Whether to torch.compile the model for faster inference (default: True).
            verbose: Print timing information (default: True).
        """
        self.model_id = model_id
        self.adapter_dir = adapter_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size
        self.compile_model = compile_model
        self.verbose = verbose

        self._processor: Optional[AutoProcessor] = None
        self._model: Optional[Qwen2_5_VLForConditionalGeneration] = None
        self._score_ids: Optional[List[int]] = None
        self._compiled_model: Any = None

    def _load_model(self) -> None:
        """Load the model and processor if not already loaded."""
        if self._processor is None or self._model is None:
            print(f"[reranker] Loading model: {self.model_id}")
            print(f"[reranker] Device: {self.device}, dtype: {self.torch_dtype}")
            print(f"[reranker] Loading LoRA adapter: {self.adapter_dir}")
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self._resolve_dtype(),
                device_map="auto" if self.device == "cuda" else self.device,
            )
            self._model = PeftModel.from_pretrained(
                self._model, self.adapter_dir
            )
            self._model = self._model.merge_and_unload()
            self._model.eval()
            print(f"[reranker] Model loaded successfully")

            # Pre-compute token IDs for scores 1-5
            _tokenizer = self._processor.tokenizer
            self._score_ids = [
                _tokenizer.convert_tokens_to_ids(str(i))
                for i in range(1, 6)
            ]

            # Compile model for faster inference if enabled (before warmup)
            if self.compile_model and self.device == "cuda":
                try:
                    print(f"[reranker] Compiling model with torch.compile (mode=reduce-overhead, dynamic=True)...")
                    self._compiled_model = torch.compile(
                        self._model,
                        mode="reduce-overhead",
                        dynamic=True,
                    )
                    print(f"[reranker] Model compiled successfully - using compiled mode")
                except Exception as e:
                    print(f"[reranker] torch.compile failed: {e}, falling back to eager mode")
                    self._compiled_model = self._model
            else:
                if self.compile_model and self.device != "cuda":
                    print(f"[reranker] torch.compile disabled - device is {self.device} (requires CUDA)")
                else:
                    print(f"[reranker] Using eager mode (compile_model={self.compile_model})")
                self._compiled_model = self._model

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve the torch dtype."""
        if self.torch_dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        if self.torch_dtype == "float16":
            return torch.float16
        if self.torch_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _create_frame_messages(self, frame_path: str, query: str) -> tuple:
        """Create messages and processor inputs for a frame.

        Returns:
            Tuple of (text, image_inputs).
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": frame_path},
                {
                    "type": "text",
                    "text": (
                        "Given the image, which is a frame from a video, rate how relevant this frame is for "
                        f"answering the question: '{query}'.\n"
                        "Output only one number from 1 to 5, where:\n"
                        "1 = completely irrelevant — the frame provides no visual or contextual information related to the "
                        "question or its answer.\n"
                        "2 = slightly relevant — the frame shows general background or context, but it is unlikely to "
                        "contribute to answering.\n"
                        "3 = moderately relevant — the frame includes partial clues or indirect context that might help "
                        "infer the answer, but the key evidence is missing.\n"
                        "4 = mostly relevant — the frame provides substantial visual or contextual information that can be "
                        "used to answer the question, though not fully decisive.\n"
                        "5 = highly relevant — the frame clearly contains the decisive evidence or strong contextual cues "
                        "that directly or indirectly support the correct answer."
                    ),
                },
            ],
        }]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = [frame_path]
        return text, image_inputs

    @torch.no_grad()
    def _score_frames_batch(
        self,
        frame_paths: List[str],
        query: str,
        progress_desc: str = "Scoring frames",
        verbose: bool = True,
    ) -> List[float]:
        """Score multiple frames in a single batch for efficiency.

        Args:
            frame_paths: List of frame image paths.
            query: The user query.
            progress_desc: Description for progress bar.
            verbose: Show progress bar.

        Returns:
            List of relevance scores (1.0 to 5.0).
        """
        batch_start = time.time()
        self._load_model()

        # Progress bar for scoring
        if HAS_TQDM and self.verbose and verbose:
            score_pbar = tqdm(total=len(frame_paths), desc=progress_desc, unit="fr")
        else:
            score_pbar = None

        # Create messages for all frames
        texts = []
        image_inputs_list = []
        for i, frame_path in enumerate(frame_paths):
            text, image_inputs = self._create_frame_messages(frame_path, query)
            texts.append(text)
            image_inputs_list.append(image_inputs[0])

            # Update progress bar
            if score_pbar:
                score_pbar.update(1)

        # Process with processor
        inputs = self._processor(
            text=texts,
            images=image_inputs_list,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Compute logits for the batch
        batch_scores = self._compiled_model(**inputs).logits[:, -1, :]
        score_vectors = [
            batch_scores[:, id_] for id_ in self._score_ids  # type: ignore[index]
        ]
        batch_scores = torch.stack(score_vectors, dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores.exp()

        # Compute weighted sum for each frame in batch
        result_scores = []
        for frame_idx in range(scores.shape[0]):
            frame_probs = scores[frame_idx].tolist()
            weighted_sum = sum((i + 1) * p for i, p in enumerate(frame_probs))
            result_scores.append(weighted_sum)

        # Print batch timing
        batch_time = time.time() - batch_start
        if self.verbose:
            avg_time = batch_time / len(frame_paths) if frame_paths else 0
            print(f"[reranker] Batch of {len(frame_paths)} frames: {batch_time:.2f}s (avg: {avg_time:.2f}s/frame)")

        if score_pbar:
            score_pbar.close()

        return result_scores

    def _score_segment_frames(
        self,
        segment_info: Dict[str, Any],
        query: str,
        frame_interval: int,
        temp_dir: str,
        target_sample_fps: Optional[float] = None,
        subtitle_query: Optional[str] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Score all frames in a video segment.

        Args:
            segment_info: Dictionary with segment metadata (path, fps, etc.).
            query: The user query.
            frame_interval: Frame sampling interval.
            temp_dir: Temporary directory for extracted frames.
            target_sample_fps: Target FPS for sampling (overrides interval).
            subtitle_query: Optional subtitle text to mark matching frames.
            verbose: Print progress information.

        Returns:
            List of frame scoring results.
        """
        os.makedirs(temp_dir, exist_ok=True)
        video_path = segment_info["path"]
        cap = cv2.VideoCapture(video_path)

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        if fps <= 0.0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Seek to start_frame if available (to read only this segment's frames)
        start_frame = segment_info.get("start_frame", 0) or 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Get end_frame if available (to stop reading at segment boundary)
        segment_end_frame = segment_info.get("end_frame")
        if segment_end_frame is None:
            # Read until end of video
            segment_end_frame = float('inf')
        else:
            segment_end_frame = int(segment_end_frame)

        # Calculate effective interval (may be overridden by target_sample_fps)
        effective_interval = frame_interval
        if target_sample_fps and target_sample_fps > 0.0 and fps > 0.0:
            computed = int(round(fps / target_sample_fps))
            effective_interval = max(computed, 1)

        # Timing
        total_time = 0.0
        frame_count = 0

        # Count total frames to sample for progress bar
        total_segment_frames = int((segment_end_frame - start_frame) / effective_interval) if effective_interval > 0 else 0

        # Collect frames and their paths for batch processing
        frame_results: List[Dict[str, Any]] = []
        frame_paths: List[str] = []
        frame_data: List[Dict[str, Any]] = []
        relative_idx = 0  # Frame index relative to start_frame

        # Progress bar for frame sampling
        use_verbose = verbose and self.verbose
        if HAS_TQDM and use_verbose and total_segment_frames > 0:
            sample_pbar = tqdm(total=total_segment_frames, desc="Sampling frames", unit="fr", leave=False)
        else:
            sample_pbar = None

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Check if we've reached the end of this segment
            current_frame_pos = start_frame + relative_idx
            if current_frame_pos > segment_end_frame:
                break

            if relative_idx % effective_interval == 0:
                segment_index = segment_info.get("segment_index", 0)
                if segment_index is not None:
                    segment_index = int(segment_index)

                frame_filename = f"seg{segment_index:04d}_frame_{relative_idx:05d}.jpg"
                frame_path = os.path.join(temp_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

                # Store frame info for later
                global_frame_index = current_frame_pos
                timestamp = global_frame_index / fps if fps else 0.0

                frame_data.append({
                    "temp_path": frame_path,
                    "segment_index": segment_index,
                    "segment_path": video_path,
                    "frame_in_segment": relative_idx,
                    "global_frame_index": global_frame_index,
                    "timestamp": timestamp,
                })
                frame_paths.append(frame_path)

                # Progress update for sampling
                if sample_pbar:
                    sample_pbar.update(1)

                # Process in batches
                if len(frame_paths) >= self.batch_size:
                    scores = self._score_frames_batch(frame_paths, query)
                    for i, (frame_info, score) in enumerate(zip(frame_data, scores)):
                        frame_results.append({**frame_info, "score": score})
                    frame_paths = []
                    frame_data = []

            relative_idx += 1

        cap.release()

        if sample_pbar:
            sample_pbar.close()

        # Process remaining frames
        if frame_paths:
            scores = self._score_frames_batch(frame_paths, query)
            for frame_info, score in zip(frame_data, scores):
                frame_results.append({**frame_info, "score": score})

        # Mark frames as subtitle frames if segment matches subtitle query
        # Method 1: Check subtitle_similarity score (set by _merge_attribute_results)
        subtitle_similarity = segment_info.get("subtitle_similarity", 0.0)
        # Method 2: Check subtitle_text field for direct match
        if subtitle_query:
            subtitle_query_lower = subtitle_query.lower().strip()
            segment_text = (segment_info.get("subtitle_text") or "").lower()
            if subtitle_query_lower in segment_text or segment_text in subtitle_query_lower:
                subtitle_similarity = 1.0  # Direct match

        # Mark frames if subtitle_similarity is significant (lower threshold for better recall)
        if subtitle_similarity >= 0.25:  # Threshold for subtitle match
            for frame_result in frame_results:
                frame_result["is_subtitle_frame"] = True

        return frame_results

    def rerank(
        self,
        segment_infos: Iterable[Dict[str, Any]],
        query: str,
        frame_interval: int = 10,
        top_frames: int = 128,
        output_dir: str = "reranker_output",
        min_frames_per_clip: int = 6,
        target_sample_fps: Optional[float] = None,
        subtitle_query: Optional[str] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Rerank frames across multiple segments and return the most relevant ones.

        Args:
            segment_infos: List of segment metadata dictionaries.
            query: The user query for relevance scoring.
            frame_interval: Frame sampling interval (default: 10).
            top_frames: Maximum number of frames to return (default: 128).
            output_dir: Directory to save selected frames.
            min_frames_per_clip: Minimum frames to keep from each segment.
            target_sample_fps: Target FPS for sampling (optional).
            subtitle_query: Optional subtitle text to highlight matching frames.
            verbose: Print progress information.

        Returns:
            List of frame dictionaries sorted by relevance score, each containing:
                - rank: Position in the final ranking
                - output_path: Path to saved frame image
                - score: Relevance score (1-5)
                - is_subtitle_frame: True if this frame comes from a subtitle-matched segment
                - timestamp: Time in video
                - segment_index: Which segment this frame came from
        """
        segment_infos_list = list(segment_infos)
        num_segments = len(segment_infos_list)

        temp_dir = tempfile.mkdtemp(prefix="frames_tmp_")
        os.makedirs(output_dir, exist_ok=True)

        # Progress bar for segment processing
        if HAS_TQDM and verbose and num_segments > 1:
            segment_pbar = tqdm(total=num_segments, desc="Processing segments", unit="seg")
        else:
            segment_pbar = None

        try:
            all_frames: List[Dict[str, Any]] = []
            for info in segment_infos_list:
                all_frames.extend(
                    self._score_segment_frames(
                        info,
                        query=query,
                        frame_interval=max(int(frame_interval), 1),
                        temp_dir=temp_dir,
                        target_sample_fps=target_sample_fps,
                        subtitle_query=subtitle_query,
                        verbose=verbose,
                    )
                )

                if segment_pbar:
                    segment_pbar.update(1)

            if segment_pbar:
                segment_pbar.close()

            if not all_frames:
                return []

            if top_frames <= 0:
                return []

            # Group frames by segment
            grouped_frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for frame_info in all_frames:
                grouped_frames[int(frame_info["segment_index"])].append(frame_info)

            # Sort each segment's frames by score
            for frame_list in grouped_frames.values():
                frame_list.sort(key=lambda x: x["score"], reverse=True)

            # Initial selection: ensure at least min_frames_per_clip from each segment
            initial_selection: List[Dict[str, Any]] = []
            if min_frames_per_clip > 0:
                for frame_list in grouped_frames.values():
                    initial_selection.extend(frame_list[:min_frames_per_clip])

            initial_selection.sort(key=lambda x: x["score"], reverse=True)

            # Select top frames, preserving initial selection
            if len(initial_selection) >= top_frames:
                selected_frames = initial_selection[:top_frames]
            else:
                selected_frames = list(initial_selection)
                remaining_frames: List[Dict[str, Any]] = []
                for frame_list in grouped_frames.values():
                    start_idx = min_frames_per_clip if min_frames_per_clip > 0 else 0
                    remaining_frames.extend(frame_list[start_idx:])

                remaining_frames.sort(key=lambda x: x["score"], reverse=True)
                for frame_info in remaining_frames:
                    if len(selected_frames) >= top_frames:
                        break
                    selected_frames.append(frame_info)

            selected_frames = selected_frames[:top_frames]
            selected_frames.sort(
                key=lambda x: (x["timestamp"], x["segment_index"], x["frame_in_segment"])
            )

            # Copy selected frames to output directory with proper naming
            results: List[Dict[str, Any]] = []
            for rank, frame_info in enumerate(selected_frames, start=1):
                timestamp = float(frame_info.get("timestamp") or 0.0)
                timestamp_tag = timestamp_label(timestamp)
                filename = (
                    f"t{timestamp_tag}_"
                    f"seg{frame_info['segment_index']:04d}_"
                    f"frame{frame_info['frame_in_segment']:05d}.jpg"
                )
                dest_path = os.path.join(output_dir, filename)
                shutil.copy2(frame_info["temp_path"], dest_path)

                enriched = dict(frame_info)
                enriched["rank"] = rank
                enriched["output_path"] = dest_path
                enriched["timestamp_label"] = timestamp_tag
                results.append(enriched)

            return results

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def unload(self) -> None:
        """Unload the model from memory to free GPU resources."""
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._model is not None:
            del self._model
            self._model = None
        if self._compiled_model is not None:
            del self._compiled_model
            self._compiled_model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def __enter__(self) -> "FrameReranker":
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


def rerank_segments(
    segment_infos: Iterable[Dict[str, Any]],
    query: str,
    frame_interval: int = 10,
    top_frames: int = 128,
    output_dir: str = "reranker_output",
    min_frames_per_clip: int = 6,
    target_sample_fps: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Convenience function to rerank segments.

    This is a drop-in replacement for the original function, maintaining
    backward compatibility while using the new class-based approach.

    Args:
        segment_infos: List of segment metadata dictionaries.
        query: The user query.
        frame_interval: Frame sampling interval.
        top_frames: Maximum number of frames to return.
        output_dir: Directory to save selected frames.
        min_frames_per_clip: Minimum frames to keep from each segment.
        target_sample_fps: Target FPS for sampling.

    Returns:
        List of ranked frame dictionaries.
    """
    with FrameReranker() as reranker:
        return reranker.rerank(
            segment_infos=segment_infos,
            query=query,
            frame_interval=frame_interval,
            top_frames=top_frames,
            output_dir=output_dir,
            min_frames_per_clip=min_frames_per_clip,
            target_sample_fps=target_sample_fps,
        )