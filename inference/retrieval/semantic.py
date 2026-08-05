"""Semantic retrieval for VideoStir.

Uses visual feature embeddings to retrieve the most relevant
video segments for a given query.
"""

from __future__ import annotations

import networkx as nx
import os
import sys
from typing import Any, Dict, List

import torch

from .graph import SpatiotemporalGraphBuilder

# Add inference directory to path for core module import
_inference_dir = os.path.dirname(os.path.dirname(__file__))
if _inference_dir not in sys.path:
    sys.path.insert(0, _inference_dir)


class VideoRetriever:
    """Retrieves relevant video segments using visual feature similarity.

    Performs multi-hop retrieval:
    1. Find top-k visually similar segments to the query
    2. Expand to spatial neighbors of top segments
    3. Include temporal neighbors for context

    Example:
        >>> retriever = VideoRetriever()
        >>> segments = retriever.retrieve(graph, "Show me the car chase")
    """

    def __init__(
        self,
        device: str | None = None,
    ):
        """Initialize the video retriever.

        Args:
            device: Device to run computations on ('cuda' or 'cpu').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._graph_builder = SpatiotemporalGraphBuilder()

    def _encode_query(self, query: str) -> torch.Tensor:
        """Encode a text query into a feature vector.

        Args:
            query: The text query.

        Returns:
            Normalized query embedding tensor.
        """
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        model_name = "PE-Core-G14-448"
        model = pe.CLIP.from_config(model_name, pretrained=True).to(self.device)
        tokenizer = transforms.get_text_tokenizer(model.context_length)

        text = tokenizer([query]).to(self.device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.squeeze(0)

    def retrieve(
        self,
        graph: nx.Graph,
        query: str,
        top_k: int = 3,
        spatial_k: int = 3,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant segments from a graph.

        Args:
            graph: The spatiotemporal graph to search.
            query: The text query.
            top_k: Number of most similar nodes to select initially.
            spatial_k: Number of spatial neighbors to expand for each top node.
            verbose: Print progress information.

        Returns:
            List of segment dictionaries with node_id and similarity scores.
        """
        if graph.number_of_nodes() == 0:
            return []

        # Try to import tqdm for progress bar
        try:
            from tqdm import tqdm
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False

        text_feat = self._encode_query(query)

        # Compute similarities for all nodes
        similarities: List[Any] = []
        num_nodes = graph.number_of_nodes()

        if HAS_TQDM and verbose:
            pbar = tqdm(total=num_nodes, desc="Computing similarities", unit="node")
        else:
            pbar = None

        for node_id, attrs in graph.nodes(data=True):
            img_feat = attrs["feature"].to(self.device)
            img_feat = img_feat / img_feat.norm()
            sim = torch.dot(img_feat, text_feat).item()
            similarities.append((node_id, sim))

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_nodes = similarities[: max(top_k, 1)]

        # Collect selected nodes (top + spatial + temporal)
        selected_nodes: set[int] = set(nid for nid, _ in top_nodes)

        # Add spatial neighbors
        for nid, _ in top_nodes:
            spatial_neighbors = [
                (nbr, graph.edges[nid, nbr]["weight"])
                for nbr in graph.neighbors(nid)
                if graph.edges[nid, nbr]["type"] == "spatial"
            ]
            spatial_neighbors.sort(key=lambda x: x[1], reverse=True)
            selected_nodes.update(nbr for nbr, _ in spatial_neighbors[:spatial_k])

        # Add temporal neighbors
        for nid in list(selected_nodes):
            for nbr in graph.neighbors(nid):
                if graph.edges[nid, nbr]["type"] == "temporal":
                    selected_nodes.add(nbr)

        # Build results
        ordered_nodes = sorted(selected_nodes)
        node_lookup = dict(similarities)

        results: List[Dict[str, Any]] = []
        for nid in ordered_nodes:
            info = dict(graph.nodes[nid])
            info["node_id"] = nid
            info["similarity"] = node_lookup.get(nid)
            results.append(info)

        return results

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
            top_k: Number of results to return.

        Returns:
            List of (node_id, similarity, text) tuples.
        """
        return self._graph_builder.retrieve_by_attribute(
            graph, query, text_getter, top_k
        )

    def collect_temporal_neighbors(
        self, graph: nx.Graph, node_id: int, hops: int = 3
    ) -> List[int]:
        """Collect nodes within N temporal hops.

        Args:
            graph: The graph to traverse.
            node_id: Starting node ID.
            hops: Number of temporal hops.

        Returns:
            List of neighbor node IDs.
        """
        return self._graph_builder.collect_temporal_neighbors(graph, node_id, hops)

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
            base_segments: Base segment results.
            query: The original query.
            text_getter: Function to extract text from node attributes.
            attribute_top_k: Number of subtitle matches.
            subtitle_neighbor_hops: Temporal expansion for subtitle hits.

        Returns:
            Merged and scored segment list.
        """
        return self._graph_builder.merge_attribute_results(
            graph, base_segments, query, text_getter,
            attribute_top_k, subtitle_neighbor_hops
        )

    def unload(self) -> None:
        """Unload the model from memory to free GPU resources."""
        # VideoRetriever uses the same CLIP model as VideoEmbedder
        # Clear any cached model state
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def retrieve_topk_segments(
    graph: nx.Graph,
    query: str,
    top_k: int = 3,
    spatial_k: int = 3,
) -> List[Dict[str, Any]]:
    """Convenience function to retrieve top-k segments.

    Args:
        graph: The spatiotemporal graph.
        query: The text query.
        top_k: Number of segments to retrieve.
        spatial_k: Number of spatial neighbors to expand.

    Returns:
        List of retrieved segment dictionaries.
    """
    retriever = VideoRetriever()
    return retriever.retrieve(graph, query, top_k=top_k, spatial_k=spatial_k)