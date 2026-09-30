#!/usr/bin/env python3
"""Command-line interface for MedVidQA Evaluation."""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from . import (
    MedVidQAEvaluator,
    EvaluationReportGenerator,
    MedVidQARetriever,
)
from .utils import setup_logging, get_medvidqa_stats


def cmd_evaluate(args):
    """Evaluate retrieval performance."""
    setup_logging(level=logging.INFO)
    
    evaluator = MedVidQAEvaluator(
        medvidqa_root=args.medvidqa_root,
        output_dir=args.output_dir,
        retrieval_dir=args.retrieval_dir,
    )
    
    results, aggregate = evaluator.evaluate_dataset(
        dataset=args.dataset,
        num_samples=args.num_samples,
    )
    
    evaluator.save_results(results, aggregate, args.output_file)
    evaluator.print_summary(aggregate)


def cmd_report(args):
    """Generate evaluation reports."""
    setup_logging(level=logging.INFO)
    
    if not Path(args.results).exists():
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)
    
    generator = EvaluationReportGenerator(output_dir=args.output_dir)
    aggregate, samples = generator.load_results(args.results)
    
    print(f"Generating reports from {len(samples)} samples...")
    generator.generate_text_report(aggregate, samples)
    generator.generate_markdown_report(aggregate, samples)
    generator.generate_csv_export(samples)
    
    print(f"Reports saved to {generator.output_dir}")


def cmd_retrieve(args):
    """Run VideoStir retrieval on MedVidQA samples."""
    setup_logging(level=logging.INFO)

    retriever = MedVidQARetriever(
        medvidqa_root=args.medvidqa_root,
        video_root=args.video_root,
        output_dir=args.output_dir,
    )

    stats = retriever.run_on_dataset(
        dataset=args.dataset,
        num_samples=args.num_samples,
        skip_existing=args.skip_existing,
        auto_download_missing=args.download_missing,
    )

    print(f"\nProcessing complete!")
    print(f"  Total: {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")

    if args.download_missing and stats['failed']:
        print("\nSome samples still failed because the video could not be downloaded or processed.")
        print("Tip: verify that yt-dlp is installed and the video URL is accessible.")


def cmd_info(args):
    """Show dataset information."""
    setup_logging(level=logging.INFO)
    
    stats = get_medvidqa_stats(args.medvidqa_root)
    
    print("\nMedVidQA Dataset Statistics")
    print("=" * 40)
    for split, count in stats.items():
        print(f"{split:10s}: {count:5d} samples")
    print(f"{'Total':10s}: {sum(stats.values()):5d} samples")


def cmd_demo(args):
    """Run evaluation on a random training sample."""
    setup_logging(level=logging.INFO)

    # Load training data
    medvidqa_root = Path(args.medvidqa_root)
    train_file = medvidqa_root / "train.json"

    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        sys.exit(1)

    with open(train_file, 'r') as f:
        train_data = json.load(f)

    # Pick a random sample
    random_sample = random.choice(train_data)
    sample_id = random_sample.get("sample_id", "unknown")
    video_id = random_sample.get("video_id", "unknown")
    question = random_sample.get("question", "")
    answer = random_sample.get("answer", "")
    answer_start = random_sample.get("answer_start_second", 0)
    answer_end = random_sample.get("answer_end_second", 0)

    print("\n" + "=" * 70)
    print("DEMO: Random Sample from Training Set")
    print("=" * 70)
    print(f"Sample ID:     {sample_id}")
    print(f"Video ID:      {video_id}")
    print(f"Question:      {question}")
    print(f"Answer:        {answer}")
    print(f"Answer Time:   {answer_start:.1f}s - {answer_end:.1f}s")
    print("=" * 70 + "\n")

    # Wrap the sample for evaluation
    wrapped_sample = {
        "sample_id": sample_id,
        "video_id": video_id,
        "question": question,
        "answer": answer,
        "answer_start_second": answer_start,
        "answer_end_second": answer_end,
        "video_length": random_sample.get("video_length", 0),
    }

    # Check if retrieval results exist
    retrieval_dir = Path(args.retrieval_dir)
    result_file = retrieval_dir / f"{video_id}_{sample_id}_frames.json"

    if not result_file.exists():
        print(f"No retrieval results found for this sample.")
        print(f"Expected: {result_file}")
        print(f"\nTo generate results, run:")
        print(f"  python -m videostir.eval retrieve --dataset train --num-samples 10")
        sys.exit(1)

    # Load retrieval results (a plain list of frame dicts)
    with open(result_file, 'r') as f:
        retrieval_data = json.load(f)

    frames = retrieval_data if isinstance(retrieval_data, list) else retrieval_data.get("frames", [])
    if not frames:
        print(f"No frames found in {result_file}")
        sys.exit(1)

    print(f"Retrieved {len(frames)} frames\n")

    # Evaluate this sample
    evaluator = MedVidQAEvaluator(medvidqa_root=args.medvidqa_root)
    sample_result = evaluator.evaluate_sample(wrapped_sample, frames)

    # Display results
    print("\nEvaluation Results for Random Sample")
    print("-" * 70)
    print(f"MRR (Mean Reciprocal Rank)    : {sample_result.get('mrr', 0):.4f}")
    print(f"Recall@10                     : {sample_result.get('recall@10', 0):.4f}")
    print(f"Recall@25                     : {sample_result.get('recall@25', 0):.4f}")
    print(f"Recall@50                     : {sample_result.get('recall@50', 0):.4f}")
    print(f"Recall@128 (All frames)       : {sample_result.get('recall@128', 0):.4f}")
    print(f"MAP@10                        : {sample_result.get('map@10', 0):.4f}")
    print(f"Coverage@128                  : {sample_result.get('coverage@128', 0):.4f}")
    print("-" * 70)

    # Show first relevant frame
    tolerance = 2.0
    first_relevant_idx = next(
        (
            i
            for i, frame in enumerate(frames)
            if answer_start - tolerance <= frame.get("timestamp", -1.0) <= answer_end + tolerance
        ),
        None,
    )
    if first_relevant_idx is not None:
        print(f"\n✓ First relevant frame at rank #{first_relevant_idx + 1}")
        if first_relevant_idx < len(frames):
            ts = frames[first_relevant_idx].get("timestamp", 0.0)
            print(f"  Timestamp: {ts:.2f}s (Answer range: {answer_start:.1f}s - {answer_end:.1f}s)")
    else:
        print(f"\n✗ No relevant frames found in top-128")

    print("\n" + "=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MedVidQA Retrieval Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View dataset info
  medvidqa-eval info
  
  # Run demo on a random training sample
  medvidqa-eval demo
  
  # Quick evaluation
  medvidqa-eval evaluate --dataset val --num-samples 5
  
  # Full evaluation with reports
  medvidqa-eval evaluate --dataset val
  medvidqa-eval report --results eval_results/eval_results.json
  
  # Run retrieval (requires videos)
  medvidqa-eval retrieve --dataset train --num-samples 10
  
  # Auto-download missing videos before retrieval
  medvidqa-eval retrieve --dataset train --num-samples 1 --download-missing
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate retrieval")
    eval_parser.add_argument("--dataset", default="val", 
                            choices=["train", "val", "test"],
                            help="Dataset to evaluate")
    eval_parser.add_argument("--num-samples", type=int, default=None,
                            help="Number of samples to evaluate")
    eval_parser.add_argument("--medvidqa-root", default="./MedVidQA",
                            help="MedVidQA root directory")
    eval_parser.add_argument("--output-dir", default="./eval_results",
                             help="Output directory for results")
    eval_parser.add_argument("--retrieval-dir", default="./medvidqa_retrieval",
                             help="Directory with cached retrieval frames")
    eval_parser.add_argument("--output-file", default="eval_results.json",
                             help="Output JSON filename")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument("--results", required=True,
                              help="Path to eval_results.json")
    report_parser.add_argument("--output-dir", default="./eval_reports",
                              help="Output directory for reports")
    report_parser.set_defaults(func=cmd_report)
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Run retrieval")
    retrieve_parser.add_argument("--dataset", default="val",
                                choices=["train", "val", "test"],
                                help="Dataset to process")
    retrieve_parser.add_argument("--num-samples", type=int, default=None,
                                help="Number of samples to process")
    retrieve_parser.add_argument("--medvidqa-root", default="./MedVidQA",
                                help="MedVidQA root directory")
    retrieve_parser.add_argument("--video-root", default="./videos",
                                help="Video root directory")
    retrieve_parser.add_argument("--output-dir", default="./medvidqa_retrieval",
                                help="Cache directory for results")
    retrieve_parser.add_argument("--skip-existing", action="store_true",
                                default=True, help="Skip cached results")
    retrieve_parser.add_argument("--download-missing", action="store_true",
                                help="Download missing videos from YouTube before running retrieval")
    retrieve_parser.set_defaults(func=cmd_retrieve)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset info")
    info_parser.add_argument("--medvidqa-root", default="./MedVidQA",
                            help="MedVidQA root directory")
    info_parser.set_defaults(func=cmd_info)
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo on random train sample")
    demo_parser.add_argument("--medvidqa-root", default="./MedVidQA",
                            help="MedVidQA root directory")
    demo_parser.add_argument("--retrieval-dir", default="./medvidqa_retrieval",
                            help="Directory with cached retrieval results")
    demo_parser.set_defaults(func=cmd_demo)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
