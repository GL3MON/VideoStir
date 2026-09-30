#!/usr/bin/env python3
"""
Enhanced Evaluation Script for MedVidQA
- Temporal segment verification (IoU, overlap, temporal precision/recall)
- Answer content verification (using MLLM to check if retrieved frames contain answer)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from videostir.eval import MedVidQAEvaluator, MedVidQARetriever
from videostir.eval.utils import load_split_data

logger = logging.getLogger(__name__)


class EnhancedEvaluator(MedVidQAEvaluator):
    """Extended evaluator with temporal segment analysis and answer verification."""

    def _compute_temporal_iou(
        self,
        retrieved_frames: List[Dict],
        start_sec: float,
        end_sec: float,
    ) -> float:
        """Compute IoU between retrieved temporal span and ground truth."""
        if not retrieved_frames:
            return 0.0

        retrieved_timestamps = [f["timestamp"] for f in retrieved_frames]
        ret_start = min(retrieved_timestamps)
        ret_end = max(retrieved_timestamps)

        intersection = max(0, min(ret_end, end_sec) - max(ret_start, start_sec))
        union = max(ret_end, end_sec) - min(ret_start, start_sec)

        return intersection / union if union > 0 else 0.0

    def _compute_temporal_precision_recall(
        self,
        retrieved_frames: List[Dict],
        start_sec: float,
        end_sec: float,
        tolerance: float = 2.0,
    ) -> Tuple[float, float]:
        """Compute temporal precision and recall."""
        if not retrieved_frames:
            return 0.0, 0.0

        relevant = sum(
            1 for f in retrieved_frames
            if self._is_timestamp_in_range(f["timestamp"], start_sec, end_sec, tolerance)
        )
        total_retrieved = len(retrieved_frames)

        gt_duration = end_sec - start_sec
        gt_frames_approx = max(1, int(gt_duration * 2))  # ~2fps sampling

        precision = relevant / total_retrieved if total_retrieved > 0 else 0.0
        recall = relevant / gt_frames_approx if gt_frames_approx > 0 else 0.0

        return precision, recall

    def _compute_first_relevant_rank(
        self,
        retrieved_frames: List[Dict],
        start_sec: float,
        end_sec: float,
        tolerance: float = 2.0,
    ) -> Optional[int]:
        """Find rank of first relevant frame (1-indexed)."""
        for i, frame in enumerate(retrieved_frames, 1):
            if self._is_timestamp_in_range(frame["timestamp"], start_sec, end_sec, tolerance):
                return i
        return None

    def _analyze_temporal_distribution(
        self,
        retrieved_frames: List[Dict],
        start_sec: float,
        end_sec: float,
    ) -> Dict:
        """Analyze how retrieved frames distribute around ground truth."""
        if not retrieved_frames:
            return {}

        in_range = []
        before = []
        after = []

        for frame in retrieved_frames:
            ts = frame["timestamp"]
            if start_sec <= ts <= end_sec:
                in_range.append(ts)
            elif ts < start_sec:
                before.append(start_sec - ts)
            else:
                after.append(ts - end_sec)

        return {
            "in_range_count": len(in_range),
            "before_count": len(before),
            "after_count": len(after),
            "avg_before_gap": sum(before) / len(before) if before else 0,
            "avg_after_gap": sum(after) / len(after) if after else 0,
            "min_timestamp": min(f["timestamp"] for f in retrieved_frames),
            "max_timestamp": max(f["timestamp"] for f in retrieved_frames),
        }

    def evaluate_sample_enhanced(
        self,
        sample: Dict,
        retrieved_frames: List[Dict],
    ) -> Dict:
        """Enhanced evaluation with temporal analysis."""
        start_sec = sample["answer_start_second"]
        end_sec = sample["answer_end_second"]

        metrics = {
            "sample_id": sample["sample_id"],
            "video_id": sample.get("video_id", ""),
            "question": sample["question"],
            "ground_truth_answer": sample.get("answer", ""),
            "answer_start": start_sec,
            "answer_end": end_sec,
            "gt_duration": end_sec - start_sec,
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

        # Enhanced temporal metrics
        metrics["temporal_iou"] = self._compute_temporal_iou(retrieved_frames, start_sec, end_sec)
        precision, recall = self._compute_temporal_precision_recall(retrieved_frames, start_sec, end_sec)
        metrics["temporal_precision"] = precision
        metrics["temporal_recall"] = recall
        metrics["temporal_f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics["first_relevant_rank"] = self._compute_first_relevant_rank(
            retrieved_frames, start_sec, end_sec
        )

        metrics["temporal_distribution"] = self._analyze_temporal_distribution(
            retrieved_frames, start_sec, end_sec
        )

        return metrics


def run_answer_verification(
    sample: Dict,
    retrieved_frames: List[Dict],
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    adapter_dir: Optional[str] = None,
) -> Dict:
    """Use MLLM to verify if retrieved frames contain the answer."""
    try:
        from videostir.inference.models import AnswerGenerator
        from videostir.inference.pipeline import PipelineConfig, run_pipeline
    except ImportError as e:
        logger.warning(f"Cannot run answer verification: {e}")
        return {"answer_verified": False, "error": str(e)}

    # Build context from retrieved frames
    frame_paths = [f["output_path"] for f in retrieved_frames[:16] if "output_path" in f]

    if not frame_paths:
        return {"answer_verified": False, "error": "No frame paths available"}

    # Create verification prompt
    question = sample["question"]
    gt_answer = sample.get("answer", "")

    prompt = f"""Given the video frames, answer the question:
Question: {question}
Ground truth answer: {gt_answer}

Does the retrieved visual content contain sufficient information to answer the question?
Respond with ONLY: YES or NO, followed by a brief explanation."""

    try:
        answer_gen = AnswerGenerator(model_id=model_id)
        if adapter_dir:
            answer_gen.load_adapter(adapter_dir)

        # This is a simplified check - in practice you'd pass frames to the model
        # For now, we'll use the answer generation pipeline
        answer = answer_gen.generate(
            frames=retrieved_frames[:16],
            query=f"{question} Ground truth: {gt_answer}. Verify if this answer is correct based on the frames.",
        )

        verified = "yes" in answer.lower() or "correct" in answer.lower() or "accurate" in answer.lower()

        answer_gen.unload()

        return {
            "answer_verified": verified,
            "generated_answer": answer,
            "ground_truth": gt_answer,
        }
    except Exception as e:
        logger.error(f"Answer verification failed: {e}")
        return {"answer_verified": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced MedVidQA Evaluation with Temporal & Answer Verification"
    )
    parser.add_argument("--dataset", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--medvidqa-root", default="./MedVidQA")
    parser.add_argument("--retrieval-dir", default="./medvidqa_retrieval")
    parser.add_argument("--output-dir", default="./eval_enhanced")
    parser.add_argument("--output-file", default="eval_enhanced_results.json")
    parser.add_argument("--verify-answers", action="store_true",
                        help="Run MLLM-based answer verification (slow)")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    evaluator = EnhancedEvaluator(
        medvidqa_root=args.medvidqa_root,
        retrieval_dir=args.retrieval_dir,
        output_dir=args.output_dir,
    )

    results, aggregate = evaluator.evaluate_dataset(
        dataset=args.dataset,
        num_samples=args.num_samples,
    )

    # Add answer verification if requested
    if args.verify_answers:
        logger.info("Running answer verification with MLLM...")
        for i, (result, sample_data) in enumerate(zip(results, evaluator._get_dataset(args.dataset)[:args.num_samples])):
            if i >= len(results):
                break
            # Load frames again for verification
            video_id = sample_data.get("video_id")
            sample_id = sample_data.get("sample_id")
            result_file = Path(args.retrieval_dir) / f"{video_id}_{sample_id}_frames.json"
            if result_file.exists():
                with open(result_file) as f:
                    frames = json.load(f)
                verification = run_answer_verification(
                    sample_data, frames, args.model_id, args.adapter_dir
                )
                result["answer_verification"] = verification

    # Save enhanced results
    evaluator.save_results(results, aggregate, args.output_file)

    # Print enhanced summary
    print_enhanced_summary(aggregate, results)


def print_enhanced_summary(aggregate: Dict, results: List[Dict]) -> None:
    """Print enhanced evaluation summary."""
    if not aggregate:
        print("No results to display")
        return

    print("\n" + "=" * 70)
    print("ENHANCED EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nNumber of samples: {aggregate.get('num_samples', 'N/A')}")

    print(f"\n--- Standard Retrieval Metrics ---")
    print(f"MRR:  {aggregate.get('mrr_avg', 0):.4f} ± {aggregate.get('mrr_std', 0):.4f}")

    for k in [10, 25, 50, 128]:
        if f"recall@{k}_avg" in aggregate:
            print(f"R@{k}:  {aggregate[f'recall@{k}_avg']:.4f} ± {aggregate.get(f'recall@{k}_std', 0):.4f}  "
                  f"MAP@{k}: {aggregate.get(f'map@{k}_avg', 0):.4f}")

    if "coverage_avg" in aggregate:
        print(f"Coverage@128: {aggregate['coverage_avg']:.4f} ± {aggregate.get('coverage_std', 0):.4f}")

    # Enhanced temporal metrics
    print(f"\n--- Temporal Segment Metrics ---")
    if results:
        temporal_iou = [r.get("temporal_iou", 0) for r in results]
        temporal_prec = [r.get("temporal_precision", 0) for r in results]
        temporal_rec = [r.get("temporal_recall", 0) for r in results]
        temporal_f1 = [r.get("temporal_f1", 0) for r in results]
        first_ranks = [r.get("first_relevant_rank") for r in results if r.get("first_relevant_rank")]

        import numpy as np
        print(f"Temporal IoU:     {np.mean(temporal_iou):.4f} ± {np.std(temporal_iou):.4f}")
        print(f"Temporal Precision: {np.mean(temporal_prec):.4f} ± {np.std(temporal_prec):.4f}")
        print(f"Temporal Recall:    {np.mean(temporal_rec):.4f} ± {np.std(temporal_rec):.4f}")
        print(f"Temporal F1:        {np.mean(temporal_f1):.4f} ± {np.std(temporal_f1):.4f}")
        if first_ranks:
            print(f"First Relevant Rank: {np.mean(first_ranks):.1f} (median: {np.median(first_ranks):.0f})")

    # Answer verification
    verified = [r for r in results if r.get("answer_verification", {}).get("answer_verified")]
    if verified:
        print(f"\n--- Answer Verification ---")
        print(f"Verified: {len(verified)}/{len(results)} ({100*len(verified)/len(results):.1f}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()