"""Data handling for VideoStir training.

This module provides dataset and collation utilities for training
with multimodal (text + image) data.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import Qwen2_5_VLProcessor

from qwen_vl_utils import process_vision_info


class MMDataset:
    """Dataset for multimodal training data.

    This dataset handles training data with chat-style conversations
    containing text and images/videos.

    Args:
        data: List of conversation records.
        processor: Qwen2_5_VLProcessor for tokenization.
        max_length: Maximum sequence length.

    Example:
        >>> with open("train.json") as f:
        ...     data = json.load(f)
        >>> processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        >>> dataset = MMDataset(data, processor)
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        processor: Optional[Qwen2_5_VLProcessor] = None,
        max_length: int = 512,
    ):
        """Initialize the multimodal dataset."""
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        return self.data[int(idx)]

    def process_sample(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single sample through the processor.

        Args:
            sample: A single conversation record.

        Returns:
            Processed sample with inputs and labels.
        """
        if self.processor is None:
            raise ValueError("Processor must be set to process samples")

        # Apply chat template
        text = self.processor.apply_chat_template(sample, tokenize=False)

        # Process vision info
        images, _ = process_vision_info(sample)

        # Tokenize
        batch = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        )

        # Create labels
        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Remove image token IDs from loss
        if isinstance(self.processor, Qwen2_5_VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [
                self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
            ]

        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return {k: v.squeeze(0) for k, v in batch.items()}


class MultimodalCollator:
    """Collator for multimodal training data.

    This collator processes a batch of multimodal samples and prepares
    them for model training.

    Args:
        processor: Qwen2_5_VLProcessor for tokenization.
        max_length: Maximum sequence length.

    Example:
        >>> processor = Qwen2_5_VLProcessor.from_pretrained("...")
        >>> collator = MultimodalCollator(processor, max_length=512)
        >>> dataloader = DataLoader(dataset, collate_fn=collator, batch_size=4)
    """

    def __init__(
        self,
        processor: Qwen2_5_VLProcessor,
        max_length: int = 512,
    ):
        """Initialize the collator."""
        self.processor = processor
        self.max_length = max_length

    def __call__(
        self,
        examples: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples.

        Args:
            examples: List of sample dictionaries.

        Returns:
            Batch dictionary with tensors.
        """
        texts = []
        images = []

        for example in examples:
            # Apply chat template
            text = self.processor.apply_chat_template(example, tokenize=False)
            texts.append(text)

            # Process vision info
            image, _ = process_vision_info(example)
            if image:
                images.append(image)
            else:
                images.append(None)

        # Tokenize
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Create labels
        labels = batch["input_ids"].clone()

        # Replace pad tokens with -100 for loss computation
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Remove image token IDs from loss
        if isinstance(self.processor, Qwen2_5_VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [
                self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
            ]

        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels

        return {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}


def load_training_data(
    labeled_path: str,
    instruction_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load training data from JSON files.

    Args:
        labeled_path: Path to labeled data JSON file.
        instruction_path: Optional path to instruction data JSON file.

    Returns:
        Combined training data list.
    """
    # Load labeled data
    with open(labeled_path, "r", encoding="utf-8") as f:
        labeled_data = json.load(f)

    # Optionally load instruction data
    if instruction_path and os.path.exists(instruction_path):
        with open(instruction_path, "r", encoding="utf-8") as f:
            instruction_data = json.load(f)
        # Merge data (instruction data can provide additional context)
        return labeled_data + instruction_data

    return labeled_data