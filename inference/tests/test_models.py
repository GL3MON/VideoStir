"""Tests for VideoStir models module - structure tests only."""

import sys
import unittest
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestConfigStructure(unittest.TestCase):
    """Test configuration class structure without dependencies."""

    def test_pipeline_config_attributes(self):
        """Test PipelineConfig has all required attributes."""
        from inference.config import PipelineConfig

        # Check all expected attributes exist
        config = PipelineConfig.__dataclass_fields__
        expected_attrs = [
            'video_path', 'query', 'output_dir', 'frame_interval',
            'n_clusters', 'top_k', 'top_frames', 'subtitle_json'
        ]
        for attr in expected_attrs:
            self.assertIn(attr, config, f"Missing attribute: {attr}")

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        from inference.config import PipelineConfig

        config = PipelineConfig(
            video_path="video.mp4",
            query="test",
            output_dir="out"
        )
        self.assertEqual(config.frame_interval, 30)
        self.assertEqual(config.n_clusters, 10)
        self.assertEqual(config.top_k, 3)
        self.assertEqual(config.top_frames, 128)
        self.assertIsNone(config.subtitle_json)


class TestRetrievalStructure(unittest.TestCase):
    """Test retrieval module structure."""

    def test_graph_builder_structure(self):
        """Test SpatiotemporalGraphBuilder has expected methods."""
        from inference.retrieval import SpatiotemporalGraphBuilder

        builder = SpatiotemporalGraphBuilder()
        self.assertTrue(hasattr(builder, 'build'))
        self.assertTrue(hasattr(builder, 'retrieve_by_attribute'))
        self.assertTrue(hasattr(builder, 'collect_temporal_neighbors'))

    def test_video_retriever_structure(self):
        """Test VideoRetriever has expected methods."""
        from inference.retrieval import VideoRetriever

        retriever = VideoRetriever()
        self.assertTrue(hasattr(retriever, 'retrieve'))
        self.assertTrue(hasattr(retriever, 'retrieve_by_attribute'))
        self.assertTrue(hasattr(retriever, 'collect_temporal_neighbors'))


class TestUtilityStructure(unittest.TestCase):
    """Test utility module structure."""

    def test_utility_functions_exist(self):
        """Test utility functions are present."""
        from inference.utils import (
            _to_serializable,
            _normalize_subtitle_time,
            _resolve_path,
            _probe_video_metadata,
            _load_subtitle_entries,
            _attach_subtitles,
            _combine_segment_scores,
            _save_json,
            _load_json,
        )

    def test_time_functions(self):
        """Test time utility functions."""
        from inference.time_utils import seconds_to_timestamp, timestamp_label

        self.assertEqual(seconds_to_timestamp(0), "00:00:00")
        self.assertEqual(seconds_to_timestamp(60), "00:01:00")
        self.assertEqual(timestamp_label(60), "00h01m00s")


class TestPipelineStructure(unittest.TestCase):
    """Test pipeline module structure."""

    def test_run_pipeline_signature(self):
        """Test run_pipeline accepts PipelineConfig."""
        import inspect
        from inference import run_pipeline

        sig = inspect.signature(run_pipeline)
        params = list(sig.parameters.keys())
        self.assertIn('config', params)

    def test_pipeline_result_structure(self):
        """Test PipelineResult has expected attributes."""
        from inference.config import PipelineResult
        import dataclasses

        fields = dataclasses.fields(PipelineResult)
        field_names = [f.name for f in fields]
        expected = ['reranked_frames', 'time_focus_frames', 'intent', 'retrieval_plan', 'video_metadata']
        for exp in expected:
            self.assertIn(exp, field_names, f"Missing field: {exp}")


class TestPreprocessingStructure(unittest.TestCase):
    """Test preprocessing module structure."""

    def test_video_segmenter_structure(self):
        """Test VideoSegmenter has expected methods."""
        from inference.preprocessing import VideoSegmenter

        segmenter = VideoSegmenter()
        self.assertTrue(hasattr(segmenter, 'segment_video'))

    def test_frame_extractor_structure(self):
        """Test FrameExtractor has expected methods."""
        from inference.preprocessing import FrameExtractor

        extractor = FrameExtractor()
        self.assertTrue(hasattr(extractor, 'extract_frames'))
        self.assertTrue(hasattr(extractor, 'extract_time_range'))


if __name__ == "__main__":
    unittest.main()