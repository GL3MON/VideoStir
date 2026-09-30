"""Answer generation using MLLM for VideoStir.

Generates natural language answers to queries based on retrieved and reranked frames.
Uses Qwen2.5-VL or LLaVA-Video models for reasoning over visual content.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class AnswerGenerator:
    """Generates answers to queries using retrieved frames and MLLM.

    This component takes the reranked frames from VideoStir's pipeline
    and uses an MLLM (Qwen2.5-VL or similar) to generate a natural language
    answer to the user's query.

    Example:
        >>> generator = AnswerGenerator()
        >>> answer = generator.generate(
        ...     frames=[frame1, frame2, frame3],
        ...     query="What instrument is he using?"
        ... )
        >>> print(answer)
        "He is using a stethoscope to listen to the patient's chest."
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = None,
        torch_dtype: str = "auto",
    ):
        """Initialize the answer generator.

        Args:
            model_id: Hugging Face model ID for the MLLM.
            device: Device to run the model on (e.g., 'cuda', 'cpu').
            torch_dtype: PyTorch dtype for model weights.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

        self._processor: Optional[AutoProcessor] = None
        self._model: Optional[Qwen2_5_VLForConditionalGeneration] = None

    def _load_model(self) -> None:
        """Load the model and processor if not already loaded."""
        if self._processor is None or self._model is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self._resolve_dtype(),
                device_map="auto" if self.device == "cuda" else self.device,
            )
            self._model.eval()

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve the torch dtype."""
        if self.torch_dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        if self.torch_dtype == "float16":
            return torch.float16
        if self.torch_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def generate(
        self,
        frames: List[Dict[str, Any]],
        query: str,
        time_focus_frames: Optional[List[Dict[str, Any]]] = None,
        subtitle_frames: Optional[List[Dict[str, Any]]] = None,
        subtitle_text: Optional[str] = None,
        keep_model_loaded: bool = False,
    ) -> str:
        """Generate an answer to a query based on retrieved frames.

        Args:
            frames: List of frame dictionaries with 'output_path' and 'timestamp'.
                    Frames are sorted by score (descending) before use.
            query: The user's natural language query.
            time_focus_frames: Optional list of time-focused frames for context.
            subtitle_frames: Optional list of frames from subtitle-matched segments.
            subtitle_text: Optional subtitle text to verify against frames.
            keep_model_loaded: If True, keep the model loaded for subsequent calls.

        Returns:
            Natural language answer to the query.
        """
        self._load_model()

        # Build messages with frames and query
        messages = self._build_messages(
            frames, query, time_focus_frames, subtitle_frames, subtitle_text
        )

        try:
            # Generate response
            input_text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Extract image paths from messages
            image_inputs = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "image":
                            image_inputs.append(item["image"])

            if os.environ.get("VIDEOSTIR_DEBUG") == "1":
                self._log_llm_input(
                    input_text=input_text,
                    image_inputs=image_inputs,
                    frames=frames,
                    time_focus_frames=time_focus_frames,
                    subtitle_frames=subtitle_frames,
                    subtitle_text=subtitle_text,
                )

            inputs = self._processor(
                text=[input_text],
                images=image_inputs if image_inputs else None,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )

            # Decode the response
            response = self._processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )[0].strip()

            return response

        finally:
            if not keep_model_loaded:
                self.unload()

    def _log_llm_input(
        self,
        input_text: str,
        image_inputs: List[str],
        frames: List[Dict[str, Any]],
        time_focus_frames: Optional[List[Dict[str, Any]]],
        subtitle_frames: Optional[List[Dict[str, Any]]],
        subtitle_text: Optional[str],
    ) -> None:
        """Log the LLM input for debugging purposes.

        Args:
            input_text: The full text input to the model.
            image_inputs: List of image paths included in the input.
            frames: Main frames passed to the model.
            time_focus_frames: Time-focused frames (if any).
            subtitle_frames: Subtitle-matched frames (if any).
            subtitle_text: Subtitle text (if any).
        """
        print("\n" + "=" * 60)
        print("  LLM INPUT LOG (AnswerGenerator)")
        print("=" * 60)

        # Count frame types in the final input
        print(f"\n📊 Frame Summary:")
        print(f"  Total images: {len(image_inputs)}")
        print(f"  Main frames input: {len(frames)}")
        if time_focus_frames:
            print(f"  Time-focus frames input: {len(time_focus_frames)}")
        if subtitle_frames:
            print(f"  Subtitle frames input: {len(subtitle_frames)}")
        if subtitle_text:
            print(f"  Subtitle text: \"{subtitle_text[:100]}...\"")

        # Show input structure summary
        print(f"\n📝 Input Structure:")
        print(f"  <retained_frames>: {len(frames)} images")
        if time_focus_frames:
            print(f"  <time-focus-frames>: {len(time_focus_frames)} images")
        if subtitle_frames and subtitle_text:
            print(f"  <subtitle-frames>: {len(subtitle_frames)} images")
            print(f"  <subtitle-text>: \"{subtitle_text[:80]}...\"")

        # Show a preview of the text input - clean it up for readability
        print(f"\n📝 Text Input Preview:")
        # Clean up the preview by removing special tokens and long image markers
        import re

        clean_text = input_text
        # Replace image paths with simple markers
        for i, img_path in enumerate(image_inputs[:5]):
            if img_path:
                clean_text = clean_text.replace(img_path, f"<image_{i}>")
        # Truncate and show
        clean_preview = re.sub(r'<[^>]+>', '', clean_text[:800])
        print(f"  {clean_preview}")

        # Log full image paths
        print(f"\n🖼️  All Image Paths:")
        for i, path in enumerate(image_inputs):
            print(f"  [{i}] {path}")

        print("=" * 60 + "\n")

    def _build_messages(
        self,
        frames: List[Dict[str, Any]],
        query: str,
        time_focus_frames: Optional[List[Dict[str, Any]]] = None,
        subtitle_frames: Optional[List[Dict[str, Any]]] = None,
        subtitle_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build the message structure for the MLLM.

        Args:
            frames: List of frame dictionaries (main/retrieved frames).
            query: The user query.
            time_focus_frames: Optional list of time-focused frames for context.
            subtitle_frames: Optional list of frames from subtitle-matched segments.
            subtitle_text: Optional subtitle text to verify against frames.

        Returns:
            Message list for the model.
        """
        # Sort main frames by timestamp
        sorted_frames = sorted(frames, key=lambda x: x.get("timestamp", 0))
        # Sort by score (descending) and take top frames
        sorted_frames = sorted(sorted_frames, key=lambda x: x.get("score", 0), reverse=True)

        # Build frame context for main frames
        frame_context = []
        for frame in sorted_frames:
            timestamp = frame.get("timestamp", 0.0)
            output_path = frame.get("output_path", "")
            frame_context.append({
                "type": "image",
                "image": output_path,
            })

        # Build the full message
        text_parts = [
            f"You are analyzing frames from a video to answer a query.\n\n"
            f"Query: {query}\n\n"
            f"Here are {len(sorted_frames)} key frames from the video, "
            f"sorted by relevance to the query. "
            f"Use these frames as the primary context to answer the question accurately."
        ]

        # Add time focus frames context if provided
        if time_focus_frames and len(time_focus_frames) > 0:
            sorted_time_focus = sorted(time_focus_frames, key=lambda x: x.get("timestamp", 0))
            text_parts.append(
                f"\n\n"
                f"{'='*60}\n"
                f"TIME-FOCUSED CONTEXT:\n"
                f"{'='*60}\n"
                f"The following {len(sorted_time_focus)} frames were identified as being from the "
                f"time range relevant to your query (e.g., specific timestamps, start/end of video).\n"
                f"These frames provide temporal context that may help with answering questions "
                f"about WHEN events occur in the video.\n"
                f"{'='*60}\n"
            )
            for frame in sorted_time_focus:
                timestamp = frame.get("timestamp", 0.0)
                output_path = frame.get("output_path", "")
                frame_context.append({
                    "type": "image",
                    "image": output_path,
                })
            text_parts.append(
                f"Total time-focused frames: {len(sorted_time_focus)}\n"
                f"(These appear as additional images interleaved above)"
            )

        # Add subtitle frames context if provided
        # if subtitle_frames and len(subtitle_frames) > 0 and subtitle_text:
        #     sorted_subtitle_frames = sorted(subtitle_frames, key=lambda x: x.get("timestamp", 0))
        #     text_parts.append(
        #         f"\n\n"
        #         f"{'='*60}\n"
        #         f"SUBTITLE-VERIFICATION CONTEXT:\n"
        #         f"{'='*60}\n"
        #         f"The extracted subtitle text from the query was:\n"
        #         f'"{subtitle_text}"\n\n'
        #         f"The following {len(sorted_subtitle_frames)} frames come from video segments "
        #         f"that were identified as potentially containing this subtitle text.\n"
        #         f"Verify whether the subtitle text matches what you see in these frames.\n"
        #         f"{'='*60}\n"
        #     )
        #     for frame in sorted_subtitle_frames:
        #         timestamp = frame.get("timestamp", 0.0)
        #         output_path = frame.get("output_path", "")
        #         frame_context.append({
        #             "type": "image",
        #             "image": output_path,
        #         })
        #     text_parts.append(
        #         f"Total subtitle frames: {len(sorted_subtitle_frames)}\n"
        #         f"(These appear as additional images interleaved above)"
        #     )

        text_parts.append(
            "\n\nProvide a concise, direct answer based on what you see in the frames."
        )

        messages = [
            {
                "role": "user",
                "content": frame_context + [
                    {
                        "type": "text",
                        "text": "\n".join(text_parts),
                    },
                ],
            }
        ]

        return messages

    def unload(self) -> None:
        """Unload the model from memory to free GPU resources."""
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._model is not None:
            del self._model
            self._model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    def __enter__(self) -> "AnswerGenerator":
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


def generate_answer(
    frames: List[Dict[str, Any]],
    query: str,
    **kwargs,
) -> str:
    """Convenience function to generate an answer.

    Args:
        frames: List of frame dictionaries.
        query: The user query.
        **kwargs: Additional arguments for AnswerGenerator.

    Returns:
        Natural language answer.
    """
    with AnswerGenerator(**kwargs) as generator:
        return generator.generate(frames, query, keep_model_loaded=False)