# VideoStir Pipeline Analysis Report

## Overview
VideoStir is a long-video retrieval pipeline that performs **intent-aware multimodal search** across video content using visual features and temporal context.

---

## Pipeline Architecture

### Stage 1: Video Preprocessing & Segmentation
```
Input: Video file (e.g., test_medical_video.mp4)
Output: Segment metadata with frame ranges
```
- **VideoSegmenter** splits the video into semantically coherent segments
- Uses CLIP-based video embeddings + temporal clustering
- Parameters: `n_clusters=10`, `min_segment_sec=1.0`, `frame_interval=30`

**From test output:**
- Created 30 segments from the medical video
- Each segment has: `path`, `start_frame`, `end_frame`, `start_sec`, `end_sec`
- Example segment (index 9): 72.0s - 79.2s duration (7.2s)

---

### Stage 2: Embedding Computation
```
Input: Segments
Output: Visual embeddings (CLIP features) for each segment
```
- Uses **PE-Core-G14-448** CLIP model for video encoding
- Parallel processing across multiple GPUs
- Frame sampling interval: 10 frames

**From test output:**
- Computed embeddings for all 30 segments
- Each embedding: 768-dimensional vector (normalized)
- Stored in checkpoint for reuse

---

### Stage 3: Spatiotemporal Graph Construction
```
Input: Segments + embeddings
Output: Graph with semantic and temporal edges
```
- **SpatiotemporalGraphBuilder** creates a graph where:
  - **Nodes**: Video segments (with visual embeddings)
  - **Semantic Edges**: Based on CLIP similarity between segments
  - **Temporal Edges**: Connect consecutive segments (bidirectional)

**From test output:**
- Graph: 30 nodes, multiple edges (exact count varies)
- Each segment gets `node_id` for graph navigation

---

### Stage 4: Query Intent Analysis
```
Input: Natural language query
Output: Intent metadata
```
Uses **Qwen2.5-VL-7B-Instruct** to analyze query and determine:
- `time_search`: True if query references specific time range
- `reason`: Natural language explanation

**From test output (query: "Show me the medical procedure steps"):**
```json
{
  "time_search": false,
  "reason": "The query focuses on visual content rather than specific time range."
}
```

---

### Stage 5: Multi-Hop Retrieval
```
Input: Graph + query + intent
Output: Top-k relevant segments
```

**Retrieval Process:**
1. **Semantic Retrieval**: Vector search over segment embeddings using query
2. **Spatiotemporal Expansion**: For each top-k segment, include spatial neighbors

**Merging Strategy (`_merge_attribute_results`):**
- Combines semantic similarity + time similarity
- Final score = sum of all similarity components

**From test output:**
- `selected_segments`: Base retrieved segments
- `merged_segments`: Expanded with temporal neighbors
- Each segment has: `similarity`, `time_similarity`, `combined_score`

---

### Stage 6: Frame Reranking
```
Input: Selected segments + query
Output: Top frames with relevance scores
```
Uses **Qwen2.5-VL-3B-Instruct** + **LoRA adapter** to score individual frames:

**Input to reranker:**
- Each segment → sampled frames at `rerank_frame_interval`
- Prompt includes: query, segment metadata

**Output structure:**
```json
{
  "output_path": "/path/to/frame.jpg",
  "timestamp": 79.6,
  "score": 4.51,           // Relevance score (reranker output)
  "segment_index": 10,
  "global_frame_index": 1990,
  "rank": 1
}
```

**Reranking behavior:**
- Frames are sorted by score (descending)
- Top `top_frames` selected (default 128)

---

## Test Case Walkthrough

**Video**: `test_medical_video.mp4` (5+ minutes, medical consultation)
**Query**: "Show me the medical procedure steps"

### Pipeline Execution:

1. **Segmentation**: Created 30 segments (10 clusters)
2. **Embeddings**: Computed visual embeddings for all segments
3. **Graph**: Built spatiotemporal graph (30 nodes, multiple edges)
4. **Intent Analysis**: 
   - `time_search=false` (no time reference in query)
5. **Retrieval**:
   - Semantic: Retrieved segments with highest visual similarity
   - Merged: Combined results with combined scores
6. **Reranking**:
   - Sampled frames from retrieved segments
   - Qwen2.5-VL-3B scored each frame
   - Top frames output with scores

### Key Output Analysis:

**Top reranked frames (from `rerank_results.json`):**
| Rank | Timestamp | Score |
|------|-----------|-------|
| 1 | 79.6s | 4.51 |
| 2 | 80.6s | 4.47 |
| 3 | 80.8s | 4.51 |
| 4 | 81.4s | 4.49 |
| 5 | 82.2s | 4.48 |

All top frames are from **segment 10**, which had highest semantic similarity to the query.

**Time-focused frames** (from `time_focus_results.json`):
- 10 frames from ~11-16s range
- Lower scores (~3.8-4.0) since query didn't specify temporal focus
- Used as additional context for answer generation

---

## Data Flow Summary

```
Video → [Segmenter] → Segments
Segments + [Embedder] → Features
Features + [GraphBuilder] → Graph
Query + [IntentAnalyzer] → Intent
Graph + Query + Intent → [Retriever] → Selected Segments
Merged + [Reranker] → Final Frames
```

---

## Checkpoint System

VideoStir supports **resumable execution** via checkpoints:
- `segments`: Segment metadata
- `segment_features`: Visual embeddings
- `intent_analysis`: Intent metadata
- `graph`: Graph structure
- `retrieval`: Retrieval results
- `rerank_results`: Final scores

Checkpoints stored in: `{output_dir}/checkpoints/` and `{output_dir}/cache/`

---

## Performance Characteristics

| Stage | GPU Requirement | Time Complexity |
|-------|-----------------|-----------------|
| Segmentation | Low | O(n_frames / interval) |
| Embeddings | High | O(n_segments × frames_per_seg) |
| Graph | Low | O(n²) for full similarity |
| Intent Analysis | Medium (7B model) | Constant |
| Retrieval | Low | O(n) vector search |
| Reranking | High (3B model) | O(n_frames × model_time) |

---

## Key Design Decisions

1. **Two-stage retrieval**: coarse (semantic) → fine (reranking) balances accuracy vs speed
2. **Spatiotemporal graph**: Enables multi-hop retrieval with temporal constraints
3. **Checkpointing**: Allows resuming long-running pipelines
4. **Frame sampling**: Adjustable intervals balance detail vs computation

---

## Output Structure

```
output/<video_name>/
├── retrieval_plan.json      # Full retrieval metadata
├── rerank_results.json      # Scored frames
├── time_focus_results.json  # Time-focused frames (if applicable)
├── checkpoints/             # Checkpointed intermediate results
│   ├── segments.meta.json
│   ├── graph.meta.json
│   ├── intent_analysis.meta.json
│   └── ...
└── segments/                # Extracted segment videos
```

---

## Query Pattern Recognition

### Time-Focused Queries
These queries benefit from time-focused retrieval:

---

## Query Pattern Recognition

### Time-Focused Queries
These queries benefit from time-focused retrieval:
- "What happens in the opening?"
- "What is at the end of the video?"
- "Show me the scene at 1:30"

### Visual Content Queries
These queries benefit from semantic retrieval:
- "Show me the medical procedure steps"
- "What happens during the consultation?"

---

## Graph Metrics Analysis

### Graph Construction Summary

| Metric | Value |
|--------|-------|
| Nodes (|V|) | 38 |
| Edges (|E|) | 703 |
| Temporal Edges | 37 |
| Spatial Edges | 666 |

### 1. Graph Density (D)

**Formula**:
```
D = 2|E| / (|V| × (|V| - 1))
```

**Calculation**:
```
D = 2 × 703 / (38 × 37)
D = 1406 / 1406
D = 1.0000
```

**Interpretation**: The graph is **maximally dense** (D = 1.0). This occurs because the construction adds spatial edges between ALL non-consecutive node pairs. With 38 nodes:
- Maximum possible edges = 38 × 37 = 1406
- Actual edges = 703 (temporal + all spatial pairs)
- This represents a **complete graph minus temporal edges** (each temporal edge pair appears once)

### 2. Average Node Degree (d̄)

**Formula**:
```
d̄ = (2 × |E|) / |V|
```

**Calculation**:
```
d̄ = (2 × 703) / 38
d̄ = 1406 / 38
d̄ = 37.00
```

**Interpretation**: Each node connects to **37 other nodes on average**. This is consistent with a near-complete graph where each node connects to almost all other nodes.

### 3. Clustering Coefficient (C)

**Formula**:
```
Ci = 2Ti / (di × (di - 1))
C = (1/|V|) × ΣCi
```

**Calculation**:
```
C = 1.0000
```

**Interpretation**: **High clustering** (C = 1.0) indicates strong local connectivity - neighbors of any node are highly likely to be connected to each other. This creates dense local neighborhoods.

### 4. Graph Transitivity

**Calculation**:
```
T = 1.0000
```

**Interpretation**: Transitivity measures the global clustering tendency. Value of 1.0 confirms the graph is effectively a clique with temporal edge modifications.

### 5. Connectivity Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Connected Components | 1 | Fully connected |
| Largest Component Size | 38 | All nodes in one component |
| Largest Component Ratio (R) | 1.0000 | No disconnected nodes |

### 6. Average Shortest Path Length

**Calculation**:
```
L = 1.0000
```

**Interpretation**: In a near-complete graph, any two nodes are typically connected by a direct edge, resulting in an average path length of ~1.

### 7. Spatial Edge Weights

| Stat | Value |
|------|-------|
| Mean | 0.7783 |
| Std | 0.1174 |
| Min | 0.4380 |
| Max | 0.9919 |

**Interpretation**: Spatial edges have high average similarity (0.78), indicating that non-consecutive segments still share strong visual features.

---

## Key Findings

1. **Dense Graph Structure**: The graph is nearly complete (D = 1.0), enabling rich connectivity between segments
2. **High Clustering**: Local neighborhoods are well-connected (C = 1.0), supporting local retrieval
3. **Fully Connected**: All segments are reachable (R = 1.0), ensuring comprehensive retrieval
4. **Strong Spatial Similarity**: Non-consecutive segments maintain high visual similarity (mean = 0.78)
5. **Dual Edge Types**: Temporal edges preserve chronology; spatial edges enable semantic search

---

*Report generated from analysis of test_medical_video.mp4 execution*