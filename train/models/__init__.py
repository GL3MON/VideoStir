"""Model classes for VideoStir training."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .distiller import DistillSFTTrainer
    from .data import MMDataset, MultimodalCollator

__all__ = [
    "DistillSFTTrainer",
    "MMDataset",
    "MultimodalCollator",
]