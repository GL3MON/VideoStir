"""Command-line interface for VideoStir.

Usage:
    python -m videostir.inference run --video video.mp4 --query "query"
    python -m videostir.inference batch --input config.json --output results/
    python -m videostir.inference answer --video video.mp4
"""

import argparse
import json
import os
import sys

from .config import PipelineConfig
from .pipeline import run_pipeline, run_batch_from_config
from .utils import _resolve_path
from .models import AnswerGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Run the VideoStir long-video retrieval pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video retrieval
  python -m videostir.inference run --video video.mp4 --query "Show me the opening"

  # With custom options
  python -m videostir.inference run --video video.mp4 --query "query" --top-frames 64

  # Batch processing
  python -m videostir.inference batch --input samples.json --output results/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single video parser
    run_parser = subparsers.add_parser("run", help="Run single video retrieval")
    run_parser.add_argument("--video", type=str, default=None, help="Path to input video")
    run_parser.add_argument("--query", type=str, default=None, help="Text query for retrieval")
    run_parser.add_argument("--output", type=str, default="./output/", help="Output directory")
    run_parser.add_argument("--frame-interval", type=int, default=30, dest="frame_interval")
    run_parser.add_argument("--clusters", type=int, default=10, dest="n_clusters")
    run_parser.add_argument("--min-segment-sec", type=float, default=1, dest="min_segment_sec")
    run_parser.add_argument("--embed-frame-interval", type=int, default=10, dest="embedding_frame_interval")
    run_parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    run_parser.add_argument("--top-k", type=int, default=3, dest="top_k")
    run_parser.add_argument("--spatial-k", type=int, default=3, dest="spatial_k")
    run_parser.add_argument("--rerank-frame-interval", type=int, default=5, dest="rerank_frame_interval")
    run_parser.add_argument("--top-frames", type=int, default=128, dest="top_frames")
    run_parser.add_argument("--temporal-weight", type=float, default=1.0, dest="temporal_weight")
    run_parser.add_argument("--subtitle-json", type=str, default=None, dest="subtitle_json")
    run_parser.add_argument("--attribute-top-k", type=int, default=3, dest="attribute_top_k")
    run_parser.add_argument("--min-frames-per-clip", type=int, default=6, dest="min_frames_per_clip")
    run_parser.add_argument("--subtitle-neighbor-hops", type=int, default=2, dest="subtitle_neighbor_hops")
    run_parser.add_argument("--time-focus-ratio", type=float, default=0.05, dest="time_focus_ratio")
    run_parser.add_argument("--time-sampling-interval", type=int, default=10, dest="time_sampling_interval")
    run_parser.add_argument("--time-range-padding", type=float, default=1.0, dest="time_range_padding")
    run_parser.add_argument("--time-min-window", type=float, default=2.0, dest="time_min_window")
    run_parser.add_argument("--short-video-threshold", type=float, default=240.0, dest="short_video_threshold")
    run_parser.add_argument("--disable-checkpoint", action="store_true", dest="disable_checkpoint")
    run_parser.add_argument("--enable-segment-compile", action="store_true", dest="enable_segment_compile", help="Enable torch.compile for vision model during segmentation")
    run_parser.add_argument("--enable-embed-compile", action="store_true", dest="enable_embed_compile", help="Enable torch.compile for vision model during video feature computation")
    run_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", dest="checkpoint_dir")
    run_parser.add_argument("--cache-dir", type=str, default="cache", dest="cache_dir")
    run_parser.add_argument("--intent-model-id", default=PipelineConfig.__dataclass_fields__["intent_model_id"].default)
    run_parser.add_argument("--reranker-model-id", default=PipelineConfig.__dataclass_fields__["reranker_model_id"].default)
    run_parser.add_argument("--embedding-model-name", default=PipelineConfig.__dataclass_fields__["embedding_model_name"].default)
    run_parser.add_argument("--reranker-adapter-dir", default=PipelineConfig.__dataclass_fields__["reranker_adapter_dir"].default)

    # Batch parser
    batch_parser = subparsers.add_parser("batch", help="Run batch processing")
    batch_parser.add_argument("--input", type=str, required=True, help="Path to batch config JSON")
    batch_parser.add_argument("--output", type=str, required=True, help="Output directory")
    batch_parser.add_argument("--video-root", type=str, default=None, dest="video_root")
    batch_parser.add_argument("--subtitle-root", type=str, default=None, dest="subtitle_root")
    batch_parser.add_argument("--skip-existing", action="store_true", default=True, dest="skip_existing")
    batch_parser.add_argument("--force", action="store_false", dest="skip_existing")
    batch_parser.add_argument("--intent-model-id", default=PipelineConfig.__dataclass_fields__["intent_model_id"].default)
    batch_parser.add_argument("--reranker-model-id", default=PipelineConfig.__dataclass_fields__["reranker_model_id"].default)
    batch_parser.add_argument("--embedding-model-name", default=PipelineConfig.__dataclass_fields__["embedding_model_name"].default)
    batch_parser.add_argument("--reranker-adapter-dir", default=PipelineConfig.__dataclass_fields__["reranker_adapter_dir"].default)

    # Answer parser
    answer_parser = subparsers.add_parser("answer", help="Run pipeline and generate answer with Qwen2.5-VL-3B")
    answer_parser.add_argument("--video", required=True, help="Path to input video")
    answer_parser.add_argument("--output", default="./output/", help="Output directory")
    answer_parser.add_argument("--top-frames", type=int, default=5, help="Number of frames for answer generation")
    answer_parser.add_argument(
        "--model-id", "--answer-model-id", dest="model_id",
        default="Qwen/Qwen2.5-VL-3B-Instruct", help="MLLM model ID",
    )
    answer_parser.add_argument("--device", default="cuda", help="Device to run on (cuda/cpu)")
    answer_parser.add_argument("--frame-interval", type=int, default=30, help="Frame sampling interval")
    answer_parser.add_argument("--subtitle-json", type=str, default=None, dest="subtitle_json", help="Path to subtitle JSON file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        # Resolve paths
        video_path = _resolve_path(None, args.video) if args.video else None
        subtitle_path = _resolve_path(None, args.subtitle_json) if args.subtitle_json else None

        # Create config
        config = PipelineConfig(
            video_path=video_path or "",
            query=args.query or "",
            output_dir=args.output,
            frame_interval=args.frame_interval,
            n_clusters=args.n_clusters,
            min_segment_sec=args.min_segment_sec,
            embedding_frame_interval=args.embedding_frame_interval,
            top_k=args.top_k,
            spatial_k=args.spatial_k,
            rerank_frame_interval=args.rerank_frame_interval,
            top_frames=args.top_frames,
            temporal_weight=args.temporal_weight,
            subtitle_json=subtitle_path,
            attribute_top_k=args.attribute_top_k,
            min_frames_per_clip=args.min_frames_per_clip,
            subtitle_neighbor_hops=args.subtitle_neighbor_hops,
            time_focus_ratio=args.time_focus_ratio,
            time_sampling_interval=args.time_sampling_interval,
            time_range_padding=args.time_range_padding,
            time_min_window=args.time_min_window,
            short_video_threshold=args.short_video_threshold,
            enable_checkpoint=not args.disable_checkpoint,
            checkpoint_dir=args.checkpoint_dir,
            cache_dir=args.cache_dir,
            compile_model=args.enable_segment_compile or args.enable_embed_compile,
            batch_size=args.batch_size,
            intent_model_id=args.intent_model_id,
            reranker_model_id=args.reranker_model_id,
            embedding_model_name=args.embedding_model_name,
            reranker_adapter_dir=args.reranker_adapter_dir,
        )

        # Run pipeline
        result = run_pipeline(config)

        # Print summary
        print(f"\nProcessed {len(result.reranked_frames)} frames")
        print(f"Time-focused frames: {len(result.time_focus_frames)}")
        print(f"Intent: {result.intent}")

        # Save results
        print(f"\nResults saved to {args.output}")

    elif args.command == "batch":
        results = run_batch_from_config(
            config_path=args.input,
            output_root=args.output,
            video_root=args.video_root,
            subtitle_root=args.subtitle_root,
            skip_existing=args.skip_existing,
            intent_model_id=args.intent_model_id,
            reranker_model_id=args.reranker_model_id,
            embedding_model_name=args.embedding_model_name,
            reranker_adapter_dir=args.reranker_adapter_dir,
        )

        total = len(results)
        completed = sum(1 for r in results.values() if not r.get("skipped", False))
        print(f"\nProcessed {completed}/{total} videos")

    elif args.command == "answer":
        # Resolve subtitle path if provided
        subtitle_path = _resolve_path(None, args.subtitle_json) if args.subtitle_json else None

        # Run the pipeline
        config = PipelineConfig(
            video_path=args.video,
            query="dummy",  # Will be replaced in interactive mode
            output_dir=args.output,
            top_frames=args.top_frames,
            frame_interval=args.frame_interval,
            subtitle_json=subtitle_path,
            enable_checkpoint=True,
        )

        # Initialize answer generator
        answer_gen = AnswerGenerator(model_id=args.model_id, device=args.device)

        # Print header
        print("=" * 60)
        print("  VideoStir Answer Generator (Qwen2.5-VL-3B)")
        print("=" * 60)
        print(f"\n🎬 Video: {args.video}")
        print(f"📊 Top frames: {args.top_frames}")
        print("🚀 Loading models...")

        # Run pipeline once to load models
        print("  • Loading video segmentation model...")
        print("  • Loading CLIP embeddings...")
        print("  • Loading Qwen2.5-VL-3B...")
        print("\n✅ Models loaded!\n")

        # Interactive loop
        try:
            while True:
                try:
                    query = input("Your query (or 'quit' to exit): ").strip()
                except EOFError:
                    break

                if not query:
                    continue

                if query.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                print(f"\n🔍 Query: {query}")

                # Update query and run pipeline
                config.query = query
                result = run_pipeline(config)

                print(f"✅ Retrieved {len(result.reranked_frames)} frames")
                print(f"🔍 Intent: subtitle_search={result.intent.get('subtitle_search', False)}, time_search={result.intent.get('time_search', False)}")
                for i, frame in enumerate(result.reranked_frames[:5], 1):
                    ts = frame.get("timestamp", 0)
                    mins, secs = int(ts // 60), int(ts % 60)
                    print(f"  Frame #{i}: {mins:02d}:{secs:02d} (score: {frame.get('score', 0):.2f})")

                # Generate answer
                print("\n🤖 Generating answer...")
                # Extract subtitle frames (frames with is_subtitle_frame=True)
                subtitle_frames = [f for f in result.reranked_frames if f.get("is_subtitle_frame", False)]
                subtitle_text = result.retrieval_plan.get("query_variants", {}).get("subtitle", "")

                # Use time focus frames as separate context when time_search is enabled
                if result.intent.get("time_search"):
                    print(f"  Reranked frames: {len(result.reranked_frames)}")
                    print(f"  Time focus frames: {len(result.time_focus_frames)}")
                    print(f"  Subtitle frames: {len(subtitle_frames)}")
                    answer = answer_gen.generate(
                        frames=result.reranked_frames,
                        query=query,
                        time_focus_frames=result.time_focus_frames,
                        subtitle_frames=subtitle_frames,
                        subtitle_text=subtitle_text,
                    )
                else:
                    answer = answer_gen.generate(
                        frames=result.reranked_frames,
                        query=query,
                        subtitle_frames=subtitle_frames,
                        subtitle_text=subtitle_text,
                    )
                print(f"\n💡 Answer: {answer}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        finally:
            answer_gen.unload()


if __name__ == "__main__":
    main()
