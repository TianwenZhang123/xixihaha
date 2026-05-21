#!/usr/bin/env python3
"""
P-Flow Experiment Runner.

Runs the complete P-Flow pipeline on individual samples or batches from Open-VFX.

Usage:
    # Single sample
    python scripts/run_experiment.py \
        --video /data/datasets/Open-VFX/videos/fire_effects/fire_0001.mp4 \
        --prompt "A campfire with dancing flames" \
        --output_dir /data/outputs/pflow/fire_0001 \
        --seed 42

    # Batch from dataset
    python scripts/run_experiment.py \
        --dataset /data/datasets/Open-VFX \
        --split test \
        --output_dir /data/outputs/pflow/batch_test \
        --start_index 0 --end_index 100

    # With custom config
    python scripts/run_experiment.py \
        --config configs/paper_default.yaml \
        --video /path/to/video.mp4 \
        --prompt "description" \
        --seed 42
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PFlowPipeline


def setup_logging(output_dir: str, verbose: bool = False):
    """Setup logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "experiment.log")

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def run_single(args):
    """Run P-Flow on a single video."""
    logger = logging.getLogger(__name__)

    config_path = args.config if args.config else str(
        Path(__file__).parent.parent / "configs" / "paper_default.yaml"
    )

    pipeline = PFlowPipeline(
        config_path=config_path,
        use_mock_vlm=args.mock_vlm,
    )

    result = pipeline.run(
        reference_video_path=args.video,
        prompt=args.prompt,
        output_dir=args.output_dir,
        seed=args.seed,
        desired_visual_effect=args.effect or args.prompt,
        subject=args.subject or "",
        environment=args.environment or "",
        mode=args.mode,
        reference_image_path=args.reference_image,
    )

    logger.info(f"Experiment complete. Output: {result['output_dir']}")
    return result


def run_batch(args):
    """Run P-Flow on a batch of samples from Open-VFX dataset."""
    logger = logging.getLogger(__name__)

    # Load dataset split
    dataset_dir = Path(args.dataset)
    split_file = dataset_dir / "splits" / f"{args.split}.json"

    if not split_file.exists():
        logger.error(f"Split file not found: {split_file}")
        sys.exit(1)

    with open(split_file) as f:
        split_data = json.load(f)

    samples = split_data["samples"]
    start = args.start_index or 0
    end = args.end_index or len(samples)
    samples = samples[start:end]

    logger.info(f"Running batch: {len(samples)} samples (index {start}-{end})")

    config_path = args.config or str(
        Path(__file__).parent.parent / "configs" / "paper_default.yaml"
    )

    pipeline = PFlowPipeline(
        config_path=config_path,
        use_mock_vlm=args.mock_vlm,
    )

    results = []
    failed = []

    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        category = sample["category"]
        prompt = sample["prompt"]
        video_file = sample["video_file"]
        video_path = str(dataset_dir / "videos" / category / video_file)

        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}, skipping")
            failed.append({"id": sample_id, "reason": "video_not_found"})
            continue

        sample_output = os.path.join(args.output_dir, f"{sample_id}")
        logger.info(f"\n{'='*50}")
        logger.info(f"Sample {i+1}/{len(samples)}: {sample_id} ({category})")
        logger.info(f"{'='*50}")

        try:
            result = pipeline.run(
                reference_video_path=video_path,
                prompt=prompt,
                output_dir=sample_output,
                seed=args.seed,
                desired_visual_effect=prompt,
                mode=args.mode,
            )
            results.append({"id": sample_id, "category": category, "output_dir": sample_output})
        except Exception as e:
            logger.error(f"Failed: {sample_id} - {e}")
            failed.append({"id": sample_id, "reason": str(e)})

    # Save batch summary
    summary = {
        "total_samples": len(samples),
        "successful": len(results),
        "failed": len(failed),
        "results": results,
        "failures": failed,
    }
    summary_path = os.path.join(args.output_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nBatch complete: {len(results)}/{len(samples)} successful")
    logger.info(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="P-Flow Experiment Runner")

    # Common args
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="t2v", choices=["t2v", "i2v"])
    parser.add_argument("--mock_vlm", action="store_true", help="Use mock VLM")
    parser.add_argument("--verbose", action="store_true")

    # Single sample args
    parser.add_argument("--video", type=str, help="Reference video path")
    parser.add_argument("--prompt", type=str, help="Text prompt")
    parser.add_argument("--effect", type=str, default=None, help="Desired effect")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--environment", type=str, default=None)
    parser.add_argument("--reference_image", type=str, default=None, help="For I2V mode")

    # Batch args
    parser.add_argument("--dataset", type=str, help="Dataset root directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)

    args = parser.parse_args()

    setup_logging(args.output_dir, args.verbose)

    if args.dataset:
        run_batch(args)
    elif args.video and args.prompt:
        run_single(args)
    else:
        parser.error("Provide either --dataset for batch or --video + --prompt for single")


if __name__ == "__main__":
    main()
