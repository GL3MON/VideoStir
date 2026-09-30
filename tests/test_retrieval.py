"""Tests for VideoStir retrieval module - structure tests only."""

import sys
import unittest
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestRetrievalStructure(unittest.TestCase):
    """Test retrieval module structure."""

    def test_graph_builder_structure(self):
        """Test SpatiotemporalGraphBuilder has expected methods."""
        from videostir.inference.retrieval import SpatiotemporalGraphBuilder

        builder = SpatiotemporalGraphBuilder()
        self.assertTrue(hasattr(builder, 'build'))
        self.assertTrue(hasattr(builder, 'retrieve_by_attribute'))
        self.assertTrue(hasattr(builder, 'collect_temporal_neighbors'))

    def test_video_retriever_structure(self):
        """Test VideoRetriever has expected methods."""
        from videostir.inference.retrieval import VideoRetriever

        retriever = VideoRetriever()
        self.assertTrue(hasattr(retriever, 'retrieve'))
        self.assertTrue(hasattr(retriever, 'retrieve_by_attribute'))
        self.assertTrue(hasattr(retriever, 'collect_temporal_neighbors'))

    def test_build_graph_function(self):
        """Test build_spatiotemporal_graph is callable."""
        from videostir.inference.retrieval import build_spatiotemporal_graph
        import networkx as nx

        self.assertTrue(callable(build_spatiotemporal_graph))

    def test_retrieve_function(self):
        """Test retrieve_topk_segments is callable."""
        from videostir.inference.retrieval import retrieve_topk_segments

        self.assertTrue(callable(retrieve_topk_segments))


if __name__ == "__main__":
    unittest.main()