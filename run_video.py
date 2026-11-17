from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

from app.config import PipelineConfig
from app.pipeline import VideoDamagePipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a car video with the existing segmentation models.")
    parser.add_argument("video", help="Path to an input video (mp4/mov/avi/mkv/webm)")
    parser.add_argument("--video-id", default=None, help="Optional report/video ID")
    parser.add_argument("--output", default=None, help="Artifact directory (default: data/artifacts/<video-id>)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        parser.error(f"Video file does not exist: {video_path}")
    config = PipelineConfig()
    pipeline = VideoDamagePipeline(config)
    pipeline.load_models()
    video_id = args.video_id or video_path.stem
    output_dir = Path(args.output).expanduser() if args.output else config.output_dir / video_id
    report = pipeline.analyze_video(video_path, output_dir=output_dir, video_id=video_id)
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {output_dir / 'damage_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
