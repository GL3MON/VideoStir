"""VideoStir pipeline runner for MedVidQA."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..utils import load_split_data

logger = logging.getLogger(__name__)


class MedVidQARetriever:
    """Run VideoStir retrieval on MedVidQA samples."""

    def __init__(
        self,
        medvidqa_root: str = "./MedVidQA",
        video_root: str = "./videos",
        output_dir: str = "./medvidqa_retrieval",
        pipeline_output_dir: str = "./pipeline_output",
    ):
        """Initialize retriever."""
        self.medvidqa_root = Path(medvidqa_root)
        self.video_root = Path(video_root)
        self.output_dir = Path(output_dir)
        self.pipeline_output_dir = Path(pipeline_output_dir)

        self.video_root.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_output_dir.mkdir(parents=True, exist_ok=True)

        self.train_data = load_split_data(str(self.medvidqa_root), "train")
        self.val_data = load_split_data(str(self.medvidqa_root), "val")
        self.test_data = load_split_data(str(self.medvidqa_root), "test")

        logger.info(f"Loaded MedVidQA: train={len(self.train_data)}, "
                    f"val={len(self.val_data)}, test={len(self.test_data)}")

    def _get_dataset(self, dataset: str) -> List[Dict]:
        """Get dataset by name."""
        if dataset == "train":
            return self.train_data
        elif dataset == "val":
            return self.val_data
        elif dataset == "test":
            return self.test_data
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def _find_video(self, video_id: str) -> Optional[Path]:
        """Find a local video file for the given video ID (any extension)."""
        mp4_path = self.video_root / f"{video_id}.mp4"
        if mp4_path.exists():
            return mp4_path
        candidates = sorted(self.video_root.glob(f"{video_id}.*"))
        return candidates[0] if candidates else None

    def _download_video_if_missing(self, sample: Dict) -> bool:
        """Download the YouTube video for a sample when it is not already on disk."""
        video_id = sample.get("video_id")
        if not video_id:
            return False

        video_path = self.video_root / f"{video_id}.mp4"
        if video_path.exists():
            return True

        video_url = sample.get("video_url") or f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"Video not found at {video_path}; trying to download from {video_url}")

        try:
            import yt_dlp
        except Exception:
            logger.warning("yt_dlp is not installed; auto-download is unavailable.")
            return False

        output_template = str(self.video_root / f"{video_id}.%(ext)s")
        ydl_opts = {
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "restrictfilenames": False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as exc:
            logger.warning(f"Download failed for {video_id}: {exc}")
            return False

        if video_path.exists():
            logger.info(f"Downloaded video to {video_path}")
            return True

        candidates = list(self.video_root.glob(f"{video_id}.*"))
        if candidates:
            logger.info(f"Video available at {candidates[0]}")
            return True

        return False

    def _run_videostir_pipeline(
        self,
        video_path: str,
        query: str,
        sample_id: int,
    ) -> Optional[List[Dict]]:
        """Run VideoStir pipeline on a single video."""
        try:
            from videostir.inference.pipeline import PipelineConfig, run_pipeline

            output_path = self.pipeline_output_dir / f"sample_{sample_id}"
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Running VideoStir pipeline: {query}")
            logger.info(f"Video: {video_path}")

            config = PipelineConfig(
                video_path=str(video_path),
                query=query,
                output_dir=str(output_path),
                top_frames=128,
                frame_interval=30,
                top_k=3,
                spatial_k=3,
                batch_size=64,
            )

            result = run_pipeline(config)

            frames = result.reranked_frames
            if frames:
                logger.info(f"Retrieved {len(frames)} frames")
                return frames

            logger.warning(f"No reranked frames returned by the pipeline")
            return None

        except Exception as e:
            logger.error(f"Error running pipeline: {e}", exc_info=True)
            return None

    def run_on_dataset(
        self,
        dataset: str = "val",
        num_samples: Optional[int] = None,
        skip_existing: bool = True,
        auto_download_missing: bool = False,
    ) -> Dict:
        """Run retrieval on all samples in a dataset."""
        data = self._get_dataset(dataset)

        if num_samples:
            data = data[:num_samples]

        stats = {
            "total": len(data),
            "processed": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "failed_samples": [],
        }

        logger.info(f"Processing {len(data)} samples from {dataset} set...")

        for idx, sample in enumerate(data, 1):
            sample_id = sample.get("sample_id")
            video_id = sample.get("video_id")
            question = sample.get("question", "")

            logger.info(f"\n[{idx}/{len(data)}] Sample {sample_id}: {question[:80]}")

            cache_file = self.output_dir / f"{video_id}_{sample_id}_frames.json"

            if cache_file.exists() and skip_existing:
                logger.info(f"Results already cached, skipping...")
                stats["skipped"] += 1
                continue

            video_path = self._find_video(video_id)

            if video_path is None:
                if auto_download_missing:
                    logger.info(f"Video missing, attempting automatic download for {video_id}")
                    if not self._download_video_if_missing(sample):
                        logger.warning(f"Could not download video for sample {sample_id}")
                        stats["failed"] += 1
                        stats["failed_samples"].append({
                            "sample_id": sample_id,
                            "reason": "video_download_failed",
                        })
                        continue
                    video_path = self._find_video(video_id)
                    if video_path is None:
                        logger.warning(f"Video still missing after attempted download: {video_id}")
                        stats["failed"] += 1
                        stats["failed_samples"].append({
                            "sample_id": sample_id,
                            "reason": "video_not_found_after_download",
                        })
                        continue
                else:
                    logger.warning(f"Video not found for {video_id} under {self.video_root}")
                    logger.info(f"Hint: Download from {sample.get('video_url')}")
                    stats["failed"] += 1
                    stats["failed_samples"].append({
                        "sample_id": sample_id,
                        "reason": "video_not_found",
                    })
                    continue

            frames = self._run_videostir_pipeline(
                str(video_path),
                question,
                sample_id,
            )

            stats["processed"] += 1

            if frames is not None:
                with open(cache_file, 'w') as f:
                    json.dump(frames, f, indent=2)
                logger.info(f"Saved {len(frames)} frames to {cache_file}")
                stats["success"] += 1
            else:
                logger.warning(f"Pipeline failed for sample {sample_id}")
                stats["failed"] += 1
                stats["failed_samples"].append({
                    "sample_id": sample_id,
                    "reason": "pipeline_error",
                })

        logger.info(f"\n{'=' * 60}")
        logger.info(f"PROCESSING SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total samples: {stats['total']}")
        logger.info(f"Processed: {stats['processed']}")
        logger.info(f"Success: {stats['success']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Skipped: {stats['skipped']}")

        return stats
