"""Test script: Build graph and retrieve from medical video.

This script demonstrates the VideoStir pipeline with checkpointing support.
Checkpointing allows you to restart from the last completed stage if the
pipeline is interrupted.

Usage:
    python test_medical_retrieval.py [resume]

If 'resume' is passed, it will skip completed stages from the checkpoint.

Checkpoint Stages:
    - segments: Video segmentation results
    - segment_features: Visual embeddings for each segment
    - intent_analysis: Query intent analysis
    - graph: Spatiotemporal graph structure
    - retrieval: Top-k segment retrieval results
    - rerank_results: Final reranked frames
    - final_results: Complete pipeline results
"""

import os
import sys

# Ensure the repository root is importable so `inference` resolves
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Set environment to avoid GUI issues
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from videostir.inference.preprocessing import VideoSegmenter
from videostir.inference.models import VideoEmbedder, IntentAnalyzer, FrameReranker
from videostir.inference.retrieval import SpatiotemporalGraphBuilder, VideoRetriever
from videostir.inference.checkpoint import CheckpointManager

video_path = os.path.join(repo_root, "artifacts", "test_medical_video.mp4")
query = "Show me the medical procedure steps"
output_dir = os.path.join(repo_root, "artifacts", "output")

# Enable checkpointing
enable_checkpoint = True
checkpoint_dir = os.path.join(output_dir, "checkpoints")
cache_dir = os.path.join(output_dir, "cache")

print(f"Processing: {video_path}")
print(f"Query: {query}")
print(f"Output: {output_dir}")
print(f"Checkpointing: {'enabled' if enable_checkpoint else 'disabled'}")

if enable_checkpoint:
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

# Create checkpoint manager
checkpoint_manager = CheckpointManager(
    output_dir,
    checkpoint_dir="checkpoints",
    cache_dir="cache"
)

# Check if we can resume
resume_mode = len(sys.argv) > 1 and sys.argv[1] == "resume"
if resume_mode:
    print("\n[RESUME MODE] Checking for existing checkpoints...")
    stages = [
        ("segments", "Video segments"),
        ("segment_features", "Embeddings"),
        ("intent_analysis", "Intent analysis"),
        ("graph", "Graph structure"),
        ("retrieval", "Retrieval results"),
        ("rerank_results", "Rerank results"),
    ]
    for checkpoint_name, description in stages:
        status = "available" if checkpoint_manager.has_checkpoint(checkpoint_name) else "not found"
        print(f"  {description}: {status}")


def run_pipeline():
    """Run the complete VideoStir pipeline with checkpointing."""

    # Step 1: Segment the video
    print("\n[1] Segmenting video...")
    if enable_checkpoint and checkpoint_manager.has_checkpoint("segments"):
        print("   Loading segments from checkpoint...")
        segments = checkpoint_manager.load_segments()
    else:
        segmenter = VideoSegmenter(
            frame_interval=30,   # Extract features every 30 frames
            n_clusters=10,       # Create 10 clusters
            min_segment_sec=1.0,
            output_dir=os.path.join(output_dir, "segments")
        )
        segments = segmenter.segment_video(video_path)
        if enable_checkpoint:
            checkpoint_manager.save_segments(segments)
            print(f"   Saved segments to checkpoint")
    print(f"   Created {len(segments)} segments")

    # Step 2: Compute embeddings
    print("\n[2] Computing video embeddings...")
    if enable_checkpoint and checkpoint_manager.has_checkpoint("segment_features"):
        print("   Loading segment features from checkpoint...")
        segment_features = checkpoint_manager.load_segment_features()
    else:
        embedder = VideoEmbedder()
        segment_features = embedder.compute_video_features(
            segments,
            frame_interval=10
        )
        embedder.unload()
        if enable_checkpoint:
            checkpoint_manager.save_segment_features(segment_features)
            print(f"   Saved segment features to checkpoint")
    print(f"   Computed embeddings for {len(segment_features)} segments")

    # Step 3: Build graph
    print("\n[3] Building spatiotemporal graph...")
    if enable_checkpoint and checkpoint_manager.has_checkpoint("graph"):
        print("   Loading graph from checkpoint...")
        graph_info = checkpoint_manager.load_graph()
        graph_builder = SpatiotemporalGraphBuilder(temporal_weight=1.0)
        graph = graph_builder.build(segment_features)
        print(f"   Loaded graph info: {graph_info}")
    else:
        builder = SpatiotemporalGraphBuilder(temporal_weight=1.0)
        graph = builder.build(segment_features)
        if enable_checkpoint:
            checkpoint_manager.save_graph({
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
            })
            print(f"   Saved graph to checkpoint")
    print(f"   Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Step 4: Analyze query intent
    print("\n[4] Analyzing query intent...")
    if enable_checkpoint and checkpoint_manager.has_checkpoint("intent_analysis"):
        print("   Loading intent analysis from checkpoint...")
        intent, _ = checkpoint_manager.load_intent_analysis()
    else:
        with IntentAnalyzer() as analyzer:
            intent = analyzer.analyze(query, keep_model_loaded=False)
        if enable_checkpoint:
            checkpoint_manager.save_intent_analysis(intent, query)
            print(f"   Saved intent analysis to checkpoint")
    print(f"   Intent: subtitle_search={intent.get('subtitle_search')}, time_search={intent.get('time_search')}")

    # Step 5: Retrieve segments
    print("\n[5] Retrieving top-k segments...")
    if enable_checkpoint and checkpoint_manager.has_checkpoint("retrieval"):
        print("   Loading retrieval results from checkpoint...")
        results, saved_query = checkpoint_manager.load_retrieval_results()
        print(f"   Loaded {len(results)} segments (query: '{saved_query}')")
    else:
        retriever = VideoRetriever()
        results = retriever.retrieve(graph, query, top_k=3, spatial_k=3)
        retriever.unload()
        if enable_checkpoint:
            checkpoint_manager.save_retrieval_results(results, query)
            print(f"   Saved retrieval results to checkpoint")
    print(f"   Retrieved {len(results)} segments")

    # Step 6: Rerank frames
    print("\n[6] Reranking frames...")
    if enable_checkpoint and checkpoint_manager.has_checkpoint("rerank_results"):
        print("   Loading rerank results from checkpoint...")
        final_frames, time_focus_frames, _ = checkpoint_manager.load_rerank_results()
    else:
        with FrameReranker() as reranker:
            final_frames = reranker.rerank(
                results,
                query=query,
                frame_interval=5,
                top_frames=128,
                output_dir=output_dir,
            )
        if enable_checkpoint:
            checkpoint_manager.save_rerank_results(final_frames, [])
            print(f"   Saved rerank results to checkpoint")
    print(f"   Reranked {len(final_frames)} frames")

    return final_frames


# Run the pipeline
if resume_mode:
    # In resume mode, load from checkpoint or run missing stages
    print("\n[RESUME] Running pipeline with checkpoint support...")

    # Load or compute each stage
    print("\n[1] Loading segments from checkpoint...")
    segments = checkpoint_manager.load_segments()
    print(f"   Loaded {len(segments)} segments")

    print("\n[2] Loading embeddings from checkpoint...")
    segment_features = checkpoint_manager.load_segment_features()
    print(f"   Loaded {len(segment_features)} segment features")

    print("\n[3] Loading graph from checkpoint...")
    graph_builder = SpatiotemporalGraphBuilder(temporal_weight=1.0)
    graph = graph_builder.build(segment_features)
    graph_info = checkpoint_manager.load_graph()
    print(f"   Loaded graph: {graph_info}")

    print("\n[4] Loading intent analysis from checkpoint...")
    intent, _ = checkpoint_manager.load_intent_analysis()
    print(f"   Loaded intent: {intent}")

    print("\n[5] Loading retrieval results from checkpoint...")
    results, saved_query = checkpoint_manager.load_retrieval_results()
    print(f"   Loaded {len(results)} results")

    print("\n[6] Loading rerank results from checkpoint...")
    final_frames, time_focus_frames, _ = checkpoint_manager.load_rerank_results()
    print(f"   Loaded {len(final_frames)} final frames")
else:
    # Run full pipeline
    final_frames = run_pipeline()

# Step 7: Display results
print("\n[7] Retrieval Results:")
if enable_checkpoint and checkpoint_manager.has_checkpoint("rerank_results"):
    final_frames, _, _ = checkpoint_manager.load_rerank_results()

for i, result in enumerate(final_frames[:5], 1):
    node_id = result.get("node_id", result.get("rank", "N/A"))
    score = result.get("score", result.get("similarity", 0.0))
    output_path = result.get("output_path", "N/A")
    timestamp = result.get("timestamp", result.get("timestamp_label", "N/A"))

    print(f"\n   #{i}:")
    print(f"      Path: {output_path}")
    print(f"      Timestamp: {timestamp}")
    print(f"      Score: {score:.4f}")

if enable_checkpoint:
    print(f"\n[Checkpoint directory: {output_dir}/checkpoints]")
    print(f"[Cache directory: {output_dir}/cache]")

    # List available checkpoints
    print("\n[Available checkpoints]")
    checkpoints = checkpoint_manager.list_checkpoints()
    for cp in checkpoints:
        print(f"  - {cp['name']}: {cp['path']}")

print("\nDone!")