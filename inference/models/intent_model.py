"""Intent analysis model for VideoStir.

Analyzes queries to determine the appropriate retrieval strategy
(subtitle search, time-focused search, or both).
"""

from __future__ import annotations

import gc
import json
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# Default model for intent analysis
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Prompts for intent analysis
INTENT_PROMPT = (
    "You are assisting with long-video retrieval. Analyze the user's query to determine "
    "whether the system should activate additional subtitle-based retrieval and/or time-range retrieval.\n"
    "Return a strict JSON object with the following keys: 'subtitle_search' (true or false), "
    "'time_search' (true or false), and 'reason' (a brief sentence explaining your decision).\n"
    "Focus only on explicit or strongly implied cues in the query.\n"
    "For example, if the query mentions a specific time segment of the video (e.g., 'opening', 'beginning', 'end'), "
    "enable time_search; otherwise, set it to false.\n"
    "If the query refers to a particular subtitle or quoted line, enable subtitle_search; otherwise, set it to false.\n"
    "Query: {query}\n"
    "JSON:"
)

SUBTITLE_REWRITE_PROMPT = (
    "You assist with long-video question answering. The user query may contain quoted "
    "or paraphrased subtitles mixed with other instructions. Extract only the subtitle "
    "text that should be matched against a subtitle index, and rewrite the query so "
    "that it no longer contains any literal subtitle-related text (for example: \"After the subtitle '......'\"; \"When the phrase '.....'\") while keeping all other "
    "statements and the final question.\n"
    "Return a strict JSON object with keys: 'subtitle_text' (a single string with the "
    "subtitle text separated by spaces, or an empty string if none), 'cleaned_query' "
    "(the query rewritten without subtitle-related text but preserving the rest), and 'reason' "
    "(briefly explain your extraction).\n"
    "Query: {query}\n"
    "JSON:"
)

TIME_FOCUS_PROMPT = (
    "You help a video-retrieval pipeline understand time-oriented requests.\n"
    "Given the query, decide whether it refers to the beginning, the ending, or a specific time span of the video.\n"
    "Return a JSON object with keys: 'mode', 'start_time_sec', 'end_time_sec', and 'reason'.\n"
    "'mode' must be one of: 'start', 'end', 'range', or 'none'.\n"
    "For queries about the start/opening/first moments, return mode 'start'.\n"
    "For queries about the end/closing/final moments, return mode 'end'.\n"
    "If the query specifies concrete timestamps (e.g., 'at 1:23', 'between 00:30 and 01:10'), return mode 'range' and fill "
    "'start_time_sec' and 'end_time_sec' with the interpreted seconds (use the same value for both if only one point is given).\n"
    "If no clear temporal focus exists, return mode 'none'.\n"
    "Always include 'reason' explaining the interpretation.\n"
    "Query: {query}\n"
    "JSON:"
)


class IntentAnalyzer:
    """Analyzes video queries to determine retrieval strategy.

    This model determines:
    - Whether subtitle-based retrieval should be activated
    - Whether time-focused retrieval should be activated
    - The temporal focus range if applicable

    Example:
        >>> analyzer = IntentAnalyzer()
        >>> result = analyzer.analyze("What happens in the opening scene?")
        >>> print(result.subtitle_search)  # False
        >>> print(result.time_search)      # True
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        torch_dtype: str = "float16",
    ):
        """Initialize the intent analyzer.

        Args:
            model_id: Hugging Face model ID for the Qwen VL model.
            device: Device to run the model on (e.g., 'cuda', 'cpu').
                    Defaults to 'cuda' if available.
            torch_dtype: PyTorch dtype for model weights ('float16', 'bfloat16', 'auto').
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

        # Lazy loading
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

    def _generate_response(
        self, messages: List[Dict[str, Any]], max_new_tokens: int = 256
    ) -> str:
        """Generate a response from the model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The generated response string.
        """
        self._load_model()

        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(text=[chat_text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
        return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    @staticmethod
    def _extract_json_object(response: str) -> Dict[str, Any]:
        """Extract a JSON object from a response string."""
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            return {}
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _to_bool(value: Any) -> bool:
        """Convert a value to boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
        return False

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """Convert a value to float."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    def analyze(
        self, query: str, keep_model_loaded: bool = False
    ) -> Dict[str, Any]:
        """Analyze a query to determine retrieval strategy.

        Args:
            query: The natural language query to analyze.
            keep_model_loaded: If True, keep the model loaded for subsequent calls.
                               If False, unload after processing to save GPU memory.

        Returns:
            Dictionary with keys:
                - subtitle_search (bool): Whether subtitle search should be enabled
                - time_search (bool): Whether time-focused search should be enabled
                - reason (str): Explanation of the analysis
        """
        formatted_prompt = INTENT_PROMPT.format(query=query.strip())
        messages = [
            {"role": "system", "content": "You decide retrieval strategies for video search."},
            {"role": "user", "content": formatted_prompt},
        ]

        try:
            response = self._generate_response(messages)
        finally:
            if not keep_model_loaded:
                self.unload()

        payload = self._extract_json_object(response)

        return {
            "subtitle_search": self._to_bool(payload.get("subtitle_search")),
            "time_search": self._to_bool(payload.get("time_search")),
            "reason": payload.get("reason", ""),
            "raw_response": response.strip(),
        }

    def analyze_time_focus(
        self, query: str, keep_model_loaded: bool = False
    ) -> Dict[str, Any]:
        """Analyze the temporal focus of a query.

        Args:
            query: The natural language query to analyze.
            keep_model_loaded: If True, keep the model loaded for subsequent calls.

        Returns:
            Dictionary with keys:
                - mode (str): One of 'start', 'end', 'range', or 'none'
                - start_time_sec (float): Start time in seconds (if applicable)
                - end_time_sec (float): End time in seconds (if applicable)
                - reason (str): Explanation of the analysis
        """
        formatted_prompt = TIME_FOCUS_PROMPT.format(query=query.strip())
        messages = [
            {"role": "system", "content": "You interpret temporal hints in video-retrieval queries."},
            {"role": "user", "content": formatted_prompt},
        ]

        try:
            response = self._generate_response(messages, max_new_tokens=256)
        finally:
            if not keep_model_loaded:
                self.unload()

        payload = self._extract_json_object(response)

        mode = payload.get("mode", "none")
        if isinstance(mode, str):
            mode = mode.strip().lower()
        else:
            mode = "none"

        return {
            "mode": mode,
            "start_time_sec": self._to_float(payload.get("start_time_sec")),
            "end_time_sec": self._to_float(payload.get("end_time_sec")),
            "reason": payload.get("reason", ""),
            "raw_response": response.strip(),
        }

    def extract_subtitles(
        self, query: str, keep_model_loaded: bool = False
    ) -> Dict[str, Any]:
        """Extract subtitle text from a query and rewrite it without subtitle references.

        Args:
            query: The natural language query (may contain subtitle references).
            keep_model_loaded: If True, keep the model loaded for subsequent calls.

        Returns:
            Dictionary with keys:
                - subtitle_text (str): The extracted subtitle text
                - cleaned_query (str): The query without subtitle references
                - reason (str): Explanation of the extraction
        """
        formatted_prompt = SUBTITLE_REWRITE_PROMPT.format(query=query.strip())
        messages = [
            {
                "role": "system",
                "content": "You extract subtitle text and rewrite queries for video retrieval.",
            },
            {"role": "user", "content": formatted_prompt},
        ]

        try:
            response = self._generate_response(messages, max_new_tokens=384)
        finally:
            if not keep_model_loaded:
                self.unload()

        payload = self._extract_json_object(response)

        return {
            "subtitle_text": payload.get("subtitle_text", "").strip(),
            "cleaned_query": payload.get("cleaned_query", "").strip(),
            "reason": payload.get("reason", ""),
            "raw_response": response.strip(),
        }

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

    def __enter__(self) -> "IntentAnalyzer":
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


def analyze_query_intent(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to analyze a query intent.

    This is a drop-in replacement for the original function, maintaining
    backward compatibility while using the new class-based approach.

    Args:
        query: The query to analyze.
        **kwargs: Additional arguments passed to IntentAnalyzer.

    Returns:
        The analysis result dictionary.
    """
    with IntentAnalyzer(**kwargs) as analyzer:
        return analyzer.analyze(query)


def analyze_time_focus(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to analyze time focus.

    Args:
        query: The query to analyze.
        **kwargs: Additional arguments passed to IntentAnalyzer.

    Returns:
        The time focus analysis result dictionary.
    """
    with IntentAnalyzer(**kwargs) as analyzer:
        return analyzer.analyze_time_focus(query)


def rewrite_query_and_extract_subtitles(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to extract subtitles and rewrite query.

    Args:
        query: The query to process.
        **kwargs: Additional arguments passed to IntentAnalyzer.

    Returns:
        The subtitle extraction result dictionary.
    """
    with IntentAnalyzer(**kwargs) as analyzer:
        return analyzer.extract_subtitles(query)


def unload_intent_model() -> None:
    """Unload any cached intent model (for backward compatibility)."""
    # This is now a no-op since the class manages its own state
    pass