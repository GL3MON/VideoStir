"""Spatiotemporal graph construction for VideoStir.

Builds and manages the graph structure that represents the temporal
and spatial relationships between video segments.
"""

from __future__ import annotations

import networkx as nx
import torch
from typing import Any, Dict, Iterable, List, Optional


class SpatiotemporalGraphBuilder:
    """Builds spatiotemporal graphs from video segment features.

    The graph represents:
    - Nodes: Video segments with their visual features
    - Temporal edges: Consecutive segments (i to i+1)
    - Spatial edges: Non-consecutive segments based on visual similarity

    Example:
        >>> builder = SpatiotemporalGraphBuilder(temporal_weight=1.0)
        >>> graph = builder.build(segments_with_features)
    """

    def __init__(self, temporal_weight: float = 1.0):
        """Initialize the graph builder.

        Args:
            temporal_weight: Weight assigned to temporal edges.
        """
        self.temporal_weight = temporal_weight

    def build(
        self, segment_features: Iterable[Dict[str, Any]]
    ) -> nx.Graph:
        """Build a spatiotemporal graph from segment features.

        Args:
            segment_features: List of segment dictionaries containing
                'image_features' key with visual embeddings.

        Returns:
            NetworkX graph with nodes for segments and edges for relationships.
        """
        segment_features = list(segment_features)
        if not segment_features:
            raise ValueError("segment_features is empty")

        features = torch.cat([d["image_features"] for d in segment_features], dim=0)
        num_segments = len(segment_features)

        G = nx.Graph()

        # Add nodes with their features
        for idx, info in enumerate(segment_features):
            node_attr = dict(info)
            node_attr["feature"] = info["image_features"].squeeze(0)
            node_attr.pop("image_features", None)
            G.add_node(idx, **node_attr)

        # Add temporal edges (consecutive segments)
        for i in range(num_segments - 1):
            G.add_edge(i, i + 1, type="temporal", weight=self.temporal_weight)

        # Add spatial edges (non-consecutive segments based on similarity)
        sim_matrix = torch.nn.functional.cosine_similarity(
            features.unsqueeze(1),
            features.unsqueeze(0),
            dim=-1,
        )

        for i in range(num_segments):
            for j in range(num_segments):
                if i == j:
                    continue
                if abs(i - j) == 1:  # Skip consecutive (already have temporal edges)
                    continue
                sim = float(sim_matrix[i, j].item())
                G.add_edge(i, j, type="spatial", weight=sim)

        return G

    def retrieve_by_attribute(
        self,
        graph: nx.Graph,
        query: str,
        text_getter,
        top_k: int = 3,
    ) -> List[Any]:
        """Retrieve nodes by attribute text similarity.

        Args:
            graph: The graph to search.
            query: The text query.
            text_getter: Function that extracts text from node attributes.
            top_k: Number of top results to return.

        Returns:
            List of (node_id, similarity, text) tuples.
        """
        documents: List[str] = []
        node_ids: List[int] = []

        for node_id, attrs in graph.nodes(data=True):
            text = text_getter(attrs)
            if not text:
                continue
            node_ids.append(node_id)
            documents.append(text)

        if not documents:
            return []

        stripped_query = (query or "").strip()
        results: List[Any] = []

        # Prefer exact substring matches for subtitle text
        if stripped_query:
            normalized_query = stripped_query.lower()
            for node_id, doc in zip(node_ids, documents):
                if normalized_query in doc.lower():
                    results.append((node_id, 1.0, doc))

            if results:
                return results[: max(top_k, 1)]

        # Fall back to semantic similarity
        from ..text_embedding import compute_similarities

        sims = compute_similarities(query, documents)
        scored = list(zip(node_ids, sims, documents))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(top_k, 1)]

    def collect_temporal_neighbors(
        self, graph: nx.Graph, node_id: int, hops: int = 3
    ) -> List[int]:
        """Collect nodes within N temporal hops of a given node.

        Args:
            graph: The graph to traverse.
            node_id: Starting node ID.
            hops: Number of temporal hops to traverse.

        Returns:
            List of neighbor node IDs.
        """
        if hops <= 0:
            return []

        visited = {node_id}
        frontier = {node_id}
        collected: set[int] = set()

        for _ in range(hops):
            next_frontier: set[int] = set()
            for current in frontier:
                for neighbor in graph.neighbors(current):
                    edge = graph.edges[current, neighbor]
                    if edge.get("type") != "temporal":
                        continue
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    collected.add(neighbor)
                    next_frontier.add(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier

        return list(collected)

    def merge_attribute_results(
        self,
        graph: nx.Graph,
        base_segments: Iterable[Dict[str, Any]],
        query: str,
        text_getter,
        attribute_top_k: int = 3,
        subtitle_neighbor_hops: int = 3,
    ) -> List[Dict[str, Any]]:
        """Merge subtitle and semantic search results.

        Args:
            graph: The graph to search.
            base_segments: Base segment results from semantic search.
            query: The original query.
            text_getter: Function to extract text from node attributes.
            attribute_top_k: Number of subtitle matches to find.
            subtitle_neighbor_hops: How many temporal neighbors to include.

        Returns:
            Merged and scored segment list.
        """
        aggregated: Dict[int, Dict[str, Any]] = {
            info["node_id"]: dict(info) for info in base_segments if "node_id" in info
        }

        for info in aggregated.values():
            info.setdefault("subtitle_similarity", 0.0)
            info.setdefault("time_similarity", 0.0)

        subtitle_hits = self.retrieve_by_attribute(
            graph, query, text_getter, attribute_top_k
        )

        for node_id, sim, _ in subtitle_hits:
            info = aggregated.get(node_id)
            if info is None:
                attrs = dict(graph.nodes[node_id])
                attrs["node_id"] = node_id
                attrs["similarity"] = 0.0
                attrs["subtitle_similarity"] = 0.0
                attrs["time_similarity"] = 0.0
                aggregated[node_id] = attrs
                info = attrs
            info["subtitle_similarity"] = float(sim)

        # Expand subtitle hits to temporal neighbors
        if subtitle_neighbor_hops > 0:
            subtitle_hit_scores = {node_id: float(sim) for node_id, sim, _ in subtitle_hits}
            for node_id, sim in subtitle_hit_scores.items():
                temporal_neighbors = self.collect_temporal_neighbors(
                    graph, node_id, subtitle_neighbor_hops
                )
                for neighbor_id in temporal_neighbors:
                    info = aggregated.get(neighbor_id)
                    if info is None:
                        attrs = dict(graph.nodes[neighbor_id])
                        attrs["node_id"] = neighbor_id
                        attrs["similarity"] = 0.0
                        attrs["subtitle_similarity"] = 0.0
                        attrs["time_similarity"] = 0.0
                        aggregated[neighbor_id] = attrs
                        info = attrs
                    current = float(info.get("subtitle_similarity") or 0.0)
                    if sim > current:
                        info["subtitle_similarity"] = sim

        merged = list(aggregated.values())
        for info in merged:
            info["combined_score"] = (
                float(info.get("similarity") or 0.0)
                + float(info.get("subtitle_similarity") or 0.0)
                + float(info.get("time_similarity") or 0.0)
            )

        merged.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        return merged


def build_spatiotemporal_graph(
    segment_features: Iterable[Dict[str, Any]],
    temporal_weight: float = 1.0,
) -> nx.Graph:
    """Convenience function to build a spatiotemporal graph.

    Args:
        segment_features: List of segment dictionaries with 'image_features'.
        temporal_weight: Weight for temporal edges.

    Returns:
        NetworkX graph.
    """
    builder = SpatiotemporalGraphBuilder(temporal_weight=temporal_weight)
    return builder.build(segment_features)


def visualize_graph(
    graph: nx.Graph,
    output_path: str,
    node_size: int = 500,
    node_color: str = "lightblue",
    edge_color: str = "gray",
    font_size: int = 8,
    show: bool = False,
    figsize: tuple = (12, 8),
    highlight_nodes: Optional[List[int]] = None,
    highlight_color: str = "red",
) -> None:
    """Visualize a spatiotemporal graph.

    Args:
        graph: The NetworkX graph to visualize.
        output_path: Path to save the visualization image.
        node_size: Size of node circles.
        node_color: Default color for nodes.
        edge_color: Color for edges.
        font_size: Font size for node labels.
        show: Whether to display the plot interactively.
        figsize: Figure size as (width, height).
        highlight_nodes: List of node IDs to highlight.
        highlight_color: Color for highlighted nodes.

    Example:
        >>> from videostir.inference.retrieval.graph import build_spatiotemporal_graph, visualize_graph
        >>> builder = SpatiotemporalGraphBuilder()
        >>> graph = builder.build(segment_features)
        >>> visualize_graph(graph, "graph.png", highlight_nodes=[0, 5, 10])
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
    except ImportError:
        raise ImportError(
            "matplotlib is required for graph visualization. "
            "Install with: pip install matplotlib"
        )

    pos = _layout_graph(graph)

    plt.figure(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_size,
        node_color=node_color,
        alpha=0.8,
    )

    # Draw highlighted nodes if specified
    if highlight_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=highlight_nodes,
            node_size=node_size,
            node_color=highlight_color,
            alpha=0.9,
        )

    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color=edge_color,
        alpha=0.5,
        width=1.0,
    )

    # Draw labels
    labels = {node: f"{node}" for node in graph.nodes()}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=font_size,
        font_color="black",
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_graph_interactive(
    graph: nx.Graph,
    highlight_nodes: Optional[List[int]] = None,
) -> None:
    """Display graph interactively in Jupyter notebook.

    Args:
        graph: The NetworkX graph to visualize.
        highlight_nodes: List of node IDs to highlight.

    Example:
        >>> visualize_graph_interactive(graph, highlight_nodes=[0, 1, 2])
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for graph visualization. "
            "Install with: pip install matplotlib"
        )

    pos = _layout_graph(graph)

    plt.figure(figsize=(12, 8))

    # Get node colors
    node_colors = []
    for node in graph.nodes():
        if highlight_nodes and node in highlight_nodes:
            node_colors.append("red")
        else:
            node_colors.append("lightblue")

    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=500,
        node_color=node_colors,
        alpha=0.8,
    )

    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="gray",
        alpha=0.5,
        width=1.0,
    )

    # Draw labels
    labels = {node: f"{node}" for node in graph.nodes()}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=8,
        font_color="black",
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def _layout_graph(graph: nx.Graph) -> dict:
    """Compute graph layout using spring layout with temporal ordering.

    Args:
        graph: The NetworkX graph.

    Returns:
        Dictionary mapping node IDs to (x, y) positions.
    """
    import matplotlib.pyplot as plt

    # Get node positions based on graph structure
    # Use spring layout for general graphs
    pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)

    # For chain-like graphs (mostly temporal edges), use circular layout
    # Check if graph is mostly a chain
    temporal_edges = sum(
        1 for u, v, d in graph.edges(data=True) if d.get("type") == "temporal"
    )
    total_edges = graph.number_of_edges()

    if total_edges > 0 and temporal_edges / total_edges > 0.7:
        # Use circular layout for chain-like graphs
        pos = nx.circular_layout(graph)

    return pos