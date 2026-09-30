"""Tests for VideoStir pipeline - structure tests only."""

import sys
import unittest
import os
import tempfile
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestPipelineConfig(unittest.TestCase):
    """Test cases for pipeline config."""

    def test_config_import(self):
        """Test config import."""
        from videostir.inference.config import PipelineConfig, PipelineResult
        self.assertIsNotNone(PipelineConfig)
        self.assertIsNotNone(PipelineResult)

    def test_config_serialization(self):
        """Test config serialization."""
        from videostir.inference.config import PipelineConfig

        config = PipelineConfig(
            video_path="video.mp4",
            query="test query",
            output_dir="output",
            top_frames=64
        )
        d = config.to_dict()
        self.assertEqual(d["top_frames"], 64)
        self.assertEqual(d["video_path"], "video.mp4")


class TestPipelineStructure(unittest.TestCase):
    """Test pipeline module structure."""

    def test_run_pipeline_signature(self):
        """Test run_pipeline accepts PipelineConfig."""
        import inspect
        from videostir import run_pipeline

        sig = inspect.signature(run_pipeline)
        params = list(sig.parameters.keys())
        self.assertIn('config', params)

    def test_run_batch_from_config_signature(self):
        """Test run_batch_from_config has expected parameters."""
        import inspect
        from videostir.inference.pipeline import run_batch_from_config

        sig = inspect.signature(run_batch_from_config)
        params = list(sig.parameters.keys())
        self.assertIn('config_path', params)
        self.assertIn('output_root', params)

    def test_pipeline_result_structure(self):
        """Test PipelineResult has expected attributes."""
        from videostir.inference.config import PipelineResult
        import dataclasses

        fields = dataclasses.fields(PipelineResult)
        field_names = [f.name for f in fields]
        expected = ['reranked_frames', 'time_focus_frames', 'intent', 'retrieval_plan', 'video_metadata']
        for exp in expected:
            self.assertIn(exp, field_names, f"Missing field: {exp}")


class TestUtilityStructure(unittest.TestCase):
    """Test utility module structure."""

    def test_utility_functions_exist(self):
        """Test utility functions are present."""
        from videostir.inference.utils import (
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
        from videostir.inference.time_utils import seconds_to_timestamp, timestamp_label

        self.assertEqual(seconds_to_timestamp(0), "00:00:00")
        self.assertEqual(seconds_to_timestamp(60), "00:01:00")
        self.assertEqual(timestamp_label(60), "00h01m00s")


if __name__ == "__main__":
    unittest.main()