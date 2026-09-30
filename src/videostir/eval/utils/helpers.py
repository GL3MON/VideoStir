"""Utility functions and helpers."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def load_split_data(medvidqa_root: str, split: str) -> List[Dict]:
    """Load one MedVidQA split (train/val/test) from its JSON file.

    Args:
        medvidqa_root: Path to the MedVidQA data directory.
        split: Split name ("train", "val", or "test").

    Returns:
        List of sample dictionaries (empty list if the file is missing).
    """
    filepath = Path(medvidqa_root) / f"{split}.json"
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return []

    with open(filepath, "r") as f:
        return json.load(f)


def setup_logging(level=logging.INFO) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def ensure_directory(path: str) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_medvidqa_stats(medvidqa_root: str = "./MedVidQA") -> dict:
    """Get MedVidQA dataset statistics."""
    return {split: len(load_split_data(medvidqa_root, split)) for split in ["train", "val", "test"]}


def print_metrics_summary(aggregate: dict) -> None:
    """Pretty print metrics summary."""
    if not aggregate:
        print("No metrics available")
        return
    
    print("\n" + "=" * 60)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 60)
    
    print(f"\nSamples: {aggregate.get('num_samples', 'N/A')}")
    print(f"\nMRR: {aggregate.get('mrr_avg', 0):.4f} ± {aggregate.get('mrr_std', 0):.4f}")
    
    for k in [10, 25, 50, 128]:
        recall_key = f"recall@{k}_avg"
        if recall_key in aggregate:
            print(f"Recall@{k}: {aggregate[recall_key]:.4f}")
    
    print("=" * 60)
