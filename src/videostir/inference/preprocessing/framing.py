"""Frame extraction utilities for VideoStir."""

from __future__ import annotations

import cv2
import os
from typing import Any, Dict, List


class FrameExtractor:
    """Extracts frames from videos at regular intervals."""

    def __init__(
        self,
        frame_interval: int = 10,
        output_dir: str | None = None,
    ):
        """Initialize the frame extractor.

        Args:
            frame_interval: Sample every Nth frame.
            output_dir: Directory to save extracted frames (optional).
        """
        self.frame_interval = frame_interval
        self.output_dir = output_dir or "frames"

    def extract_frames(
        self, video_path: str, output_prefix: str = "frame"
    ) -> List[Dict[str, Any]]:
        """Extract frames from a video.

        Args:
            video_path: Path to the video file.
            output_prefix: Prefix for output filenames.

        Returns:
            List of frame metadata dictionaries.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_results: List[Dict[str, Any]] = []
        idx = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            if idx % self.frame_interval == 0:
                timestamp = idx / fps
                filename = f"{output_prefix}_t{timestamp:.3f}_idx{idx:06d}.jpg"
                output_path = os.path.join(self.output_dir, filename)

                cv2.imwrite(output_path, frame)

                frame_results.append({
                    "path": output_path,
                    "frame_index": idx,
                    "timestamp": timestamp,
                })

            idx += 1

        cap.release()
        return frame_results

    def extract_time_range(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        frame_interval: int = 10,
        output_prefix: str = "time_range",
    ) -> List[Dict[str, Any]]:
        """Extract frames from a specific time range.

        Args:
            video_path: Path to the video file.
            start_sec: Start time in seconds.
            end_sec: End time in seconds.
            frame_interval: Sample every Nth frame.
            output_prefix: Prefix for output filenames.

        Returns:
            List of frame metadata dictionaries.
        """
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        os.makedirs(self.output_dir, exist_ok=True)

        frame_results: List[Dict[str, Any]] = []
        current = start_frame

        while current <= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current)
            success, frame = cap.read()

            if success:
                timestamp = current / fps
                filename = f"{output_prefix}_t{timestamp:.3f}_idx{current:06d}.jpg"
                output_path = os.path.join(self.output_dir, filename)

                cv2.imwrite(output_path, frame)

                frame_results.append({
                    "path": output_path,
                    "frame_index": current,
                    "timestamp": timestamp,
                })

            current += frame_interval

        cap.release()
        return frame_results