"""Evaluation metrics computation for MedVidQA."""

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils import load_split_data

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class MedVidQAEvaluator:
    """Evaluate retrieval performance on MedVidQA dataset."""

    def __init__(
        self,
        medvidqa_root: str = "./MedVidQA",
        output_dir: str = "./eval_results",
        retrieval_dir: str = "./medvidqa_retrieval",
    ):
        """Initialize evaluator.

        Args:
            medvidqa_root: Path to MedVidQA data directory.
            output_dir: Directory to save evaluation results.
            retrieval_dir: Directory with cached retrieval frames
                (written by the ``retrieve`` command).
        """
        self.medvidqa_root = Path(medvidqa_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval_dir = Path(retrieval_dir)

        self.train_data = load_split_data(str(self.medvidqa_root), "train")
        self.val_data = load_split_data(str(self.medvidqa_root), "val")
        self.test_data = load_split_data(str(self.medvidqa_root), "test")

        logger.info(f"Loaded MedVidQA: train={len(self.train_data)}, "
                    f"val={len(self.val_data)}, test={len(self.test_data)}")

        self.results = []

    def _get_dataset(self, dataset: str) -> List[Dict]:
        """Get dataset by name."""
        if dataset == "train":
            return self.train_data
        elif dataset == "val":
            return self.val_data
        elif dataset == "test":
            return self.test_data
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def _is_timestamp_in_range(
        self,
        timestamp: float,
        start_sec: float,
        end_sec: float,
        tolerance: float = 2.0,
    ) -> bool:
        """Check if timestamp is within ground truth range."""
        return (timestamp >= start_sec - tolerance and 
                timestamp <= end_sec + tolerance)

    def _compute_recall_at_k(
        self,
        retrieved_frames: List[Dict],
        start_sec: float,
        end_sec: float,
        k: int = 10,
    ) -> float:
        """Compute Recall@K."""
        if not retrieved_frames or k <= 0:
            return 0.0
        
        top_k_frames = retrieved_frames[:k]
        relevant_count = sum(
            1 for frame in top_k_frames
            if self._is_timestamp_in_range(frame["timestamp"], start_sec, end_sec)
        )
        
        gt_duration = end_sec - start_sec
        gt_frames_approx = max(1, int(gt_duration * 25))
        
        return min(1.0, relevant_count / max(1, min(k, gt_frames_approx)))

    def _compute_mrr(
        self,
        retrieved_frames: List[Dict],
        start_sec: float,
        end_sec: float,
    ) -> float:
        """Compute Mean Reciprocal Rank."""
        for i, frame in enumerate(retrieved_frames, 1):
            if self._is_timestamp_in_range(frame["timestamp"], start_sec, end_sec):
                return 1.0 / i
        return 0.0

    def _compute_map_at_k(
        self,
        retrieved_frames: List[Dict],
        start_sec: float,
        end_sec: float,
        k: int = 10,
    ) -> float:
        """Compute Mean Average Precision@K."""
        if not retrieved_frames or k <= 0:
            return 0.0
        
        top_k_frames = retrieved_frames[:k]
        ap_sum = 0.0
        num_relevant = 0
        
        for i, frame in enumerate(top_k_frames, 1):
            if self._is_timestamp_in_range(frame["timestamp"], start_sec, end_sec):
                num_relevant += 1
                precision_at_i = num_relevant / i
                ap_sum += precision_at_i
        
        if num_relevant == 0:
            return 0.0
        
        return ap_sum / min(k, num_relevant)

    def _compute_coverage(
        self,
        retrieved_frames: List[Dict],
        start_sec: float,
        end_sec: float,
        k: int = 128,
    ) -> float:
        """Compute % of ground truth time range covered."""
        top_k_frames = retrieved_frames[:k]
        
        num_buckets = 10
        bucket_size = (end_sec - start_sec) / max(1, num_buckets)
        covered_buckets = set()
        
        for frame in top_k_frames:
            if self._is_timestamp_in_range(frame["timestamp"], start_sec, end_sec):
                bucket_idx = int((frame["timestamp"] - start_sec) / max(0.01, bucket_size))
                bucket_idx = min(bucket_idx, num_buckets - 1)
                covered_buckets.add(bucket_idx)
        
        return len(covered_buckets) / num_buckets if num_buckets > 0 else 0.0

    def evaluate_sample(
        self,
        sample: Dict,
        retrieved_frames: List[Dict],
    ) -> Dict:
        """Evaluate a single sample."""
        start_sec = sample["answer_start_second"]
        end_sec = sample["answer_end_second"]
        
        metrics = {
            "sample_id": sample["sample_id"],
            "question": sample["question"],
            "answer_start": start_sec,
            "answer_end": end_sec,
            "retrieved_count": len(retrieved_frames),
        }
        
        for k in [10, 25, 50, 128]:
            metrics[f"recall@{k}"] = self._compute_recall_at_k(
                retrieved_frames, start_sec, end_sec, k
            )
            metrics[f"map@{k}"] = self._compute_map_at_k(
                retrieved_frames, start_sec, end_sec, k
            )
        
        metrics["mrr"] = self._compute_mrr(retrieved_frames, start_sec, end_sec)
        metrics["coverage@128"] = self._compute_coverage(
            retrieved_frames, start_sec, end_sec, k=128
        )
        
        return metrics

    def evaluate_dataset(
        self,
        dataset: str = "val",
        num_samples: Optional[int] = None,
        skip_missing_videos: bool = True,
    ) -> Tuple[List[Dict], Dict]:
        """Evaluate on a dataset."""
        data = self._get_dataset(dataset)
        
        if num_samples:
            data = data[:num_samples]
        
        results = []
        
        logger.info(f"Evaluating {len(data)} samples from {dataset} set...")
        
        for sample_idx, sample in enumerate(data):
            video_id = sample.get("video_id")
            question = sample.get("question", "")
            
            logger.info(f"[{sample_idx + 1}/{len(data)}] {question[:60]}")
            
            result_file = self.retrieval_dir / f"{video_id}_{sample['sample_id']}_frames.json"
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    retrieved_frames = json.load(f)
            else:
                logger.warning(f"No cached results for {video_id}, skipping...")
                continue
            
            metrics = self.evaluate_sample(sample, retrieved_frames)
            results.append(metrics)
        
        aggregate = self._aggregate_metrics(results)
        
        return results, aggregate

    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate statistics."""
        import numpy as np

        if not results:
            return {}

        aggregate = {
            "num_samples": len(results),
            "mrr_avg": float(np.mean([r["mrr"] for r in results])),
            "mrr_std": float(np.std([r["mrr"] for r in results])),
        }

        for k in [10, 25, 50, 128]:
            recall_key = f"recall@{k}"
            map_key = f"map@{k}"

            if recall_key in results[0]:
                aggregate[f"recall@{k}_avg"] = float(np.mean([r[recall_key] for r in results]))
                aggregate[f"recall@{k}_std"] = float(np.std([r[recall_key] for r in results]))

            if map_key in results[0]:
                aggregate[f"map@{k}_avg"] = float(np.mean([r[map_key] for r in results]))
                aggregate[f"map@{k}_std"] = float(np.std([r[map_key] for r in results]))

        if "coverage@128" in results[0]:
            aggregate["coverage_avg"] = float(np.mean([r["coverage@128"] for r in results]))
            aggregate["coverage_std"] = float(np.std([r["coverage@128"] for r in results]))

        return aggregate

    def save_results(
        self,
        results: List[Dict],
        aggregate: Dict,
        output_file: str = "eval_results.json",
    ) -> None:
        """Save evaluation results."""
        output_path = self.output_dir / output_file
        
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "aggregate_metrics": aggregate,
            "sample_results": results,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

    def print_summary(self, aggregate: Dict) -> None:
        """Print summary of evaluation results."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        if not aggregate:
            print("No results to display")
            return
        
        print(f"\nNumber of samples: {aggregate.get('num_samples', 'N/A')}")
        print(f"\nMRR (Mean Reciprocal Rank):")
        print(f"  Mean: {aggregate.get('mrr_avg', 0):.4f} ± {aggregate.get('mrr_std', 0):.4f}")
        
        for k in [10, 25, 50, 128]:
            recall_key = f"recall@{k}_avg"
            map_key = f"map@{k}_avg"
            
            if recall_key in aggregate and map_key in aggregate:
                print(f"\n@{k}:")
                print(f"  Recall: {aggregate[recall_key]:.4f} ± {aggregate.get(f'recall@{k}_std', 0):.4f}")
                print(f"  MAP:    {aggregate[map_key]:.4f} ± {aggregate.get(f'map@{k}_std', 0):.4f}")
        
        if "coverage_avg" in aggregate:
            print(f"\nTemporal Coverage @128:")
            print(f"  Mean: {aggregate['coverage_avg']:.4f} ± {aggregate.get('coverage_std', 0):.4f}")
        
        print("\n" + "=" * 60)
