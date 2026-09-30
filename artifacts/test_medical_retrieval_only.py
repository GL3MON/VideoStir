"""Test script: Run only retrieval from medical video checkpoints.

This script demonstrates running only the retrieval stage after
segments and features have been computed and checkpointed.

Usage:
    python test_medical_retrieval_only.py

Requirements:
    - Must have existing checkpoints for 'segments' and 'segment_features'
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

from videostir.inference.models import FrameReranker
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

def run_retrieval_only():
    """Run only the retrieval stage using checkpointed data."""

    # Step 1: Load segments from checkpoint
    print("\n[1] Loading segments from checkpoint...")
    if enable_checkpoint and checkpoint_manager.has_checkpoint("segments"):
        segments = checkpoint_manager.load_segments()
        print(f"   Loaded {len(segments)} segments from checkpoint")
    else:
        print("   ERROR: No segments checkpoint found!")
        print("   Run the full pipeline first to generate segments.")
        return []

    # Step 2: Load segment features from checkpoint
    print("\n[2] Loading segment features from checkpoint...")
    if enable_checkpoint and checkpoint_manager.has_checkpoint("segment_features"):
        segment_features = checkpoint_manager.load_segment_features()
        print(f"   Loaded {len(segment_features)} segment features from checkpoint")
    else:
        print("   ERROR: No segment_features checkpoint found!")
        print("   Run the full pipeline first to generate features.")
        return []

    # Step 3: Build graph (rebuilds from features, doesn't load graph structure)
    print("\n[3] Building spatiotemporal graph...")
    if enable_checkpoint and checkpoint_manager.has_checkpoint("graph"):
        graph_info = checkpoint_manager.load_graph()
        print(f"   Loaded graph info: {graph_info}")

    builder = SpatiotemporalGraphBuilder(temporal_weight=1.0)
    graph = builder.build(segment_features)
    print(f"   Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Step 4: Retrieve segments
    print("\n[4] Retrieving top-k segments...")
    if enable_checkpoint and checkpoint_manager.has_checkpoint("retrieval"):
        results, saved_query = checkpoint_manager.load_retrieval_results()
        print(f"   Loaded {len(results)} segments from checkpoint (query: '{saved_query}')")
    else:
        retriever = VideoRetriever()
        results = retriever.retrieve(graph, query, top_k=3, spatial_k=3)
        retriever.unload()
        if enable_checkpoint:
            checkpoint_manager.save_retrieval_results(results, query)
            print(f"   Saved retrieval results to checkpoint")
    print(f"   Retrieved {len(results)} segments")

    # Step 5: Rerank frames (optional - if you want final frames)
    print("\n[5] Reranking frames...")
    if enable_checkpoint and checkpoint_manager.has_checkpoint("rerank_results"):
        final_frames, time_focus_frames, _ = checkpoint_manager.load_rerank_results()
        print(f"   Loaded {len(final_frames)} frames from checkpoint")
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


# Run retrieval only
final_frames = run_retrieval_only()

# Display results
print("\n[6] Retrieval Results:")
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
    print("\n[Available checkpoints]")
    checkpoints = checkpoint_manager.list_checkpoints()
    for cp in checkpoints:
        print(f"  - {cp['name']}: {cp['path']}")

print("\nDone!")