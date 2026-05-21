#!/usr/bin/env python3
"""
P-Flow Quick Run Script.

Simple entry point for running P-Flow experiments.

Usage:
    # Basic usage
    python run.py --video reference.mp4 --prompt "fire effect" --output /data/outputs/test

    # With full paper config
    python run.py --video reference.mp4 --prompt "fire effect" \
        --config configs/paper_default.yaml --seed 42

    # Mock VLM (for testing without API key)
    python run.py --video reference.mp4 --prompt "fire effect" --mock_vlm
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import PFlowPipeline


def main():
    parser = argparse.ArgumentParser(
        description="P-Flow: Training-Free Visual Effects Customization"
    )
    parser.add_argument("--video", type=str, required=True, help="Reference video path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, default="/root/autodl-tmp/outputs/pflow/quick_run",
                       help="Output directory")
    parser.add_argument("--config", type=str, default=None,
                       help="Config YAML (default: configs/paper_default.yaml)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="t2v", choices=["t2v", "i2v"])
    parser.add_argument("--mock_vlm", action="store_true", help="Use mock VLM (testing)")
    parser.add_argument("--effect", type=str, default=None, help="Desired effect description")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Default config
    config_path = args.config
    if config_path is None:
        default_config = Path(__file__).parent / "configs" / "paper_default.yaml"
        if default_config.exists():
            config_path = str(default_config)

    # Run pipeline
    pipeline = PFlowPipeline(
        config_path=config_path,
        use_mock_vlm=args.mock_vlm,
    )

    result = pipeline.run(
        reference_video_path=args.video,
        prompt=args.prompt,
        output_dir=args.output,
        seed=args.seed,
        desired_visual_effect=args.effect or args.prompt,
        mode=args.mode,
    )

    print(f"\nDone! Output: {result['output_dir']}")
    print(f"Videos: {len(result['video_paths'])} iterations saved")
    print(f"Total time: {result['total_time']:.1f}s")


if __name__ == "__main__":
    main()
