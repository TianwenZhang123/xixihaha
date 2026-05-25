#!/usr/bin/env python3
"""
Unified Batch Runner for Video Reproduction Experiments.

Supports two methods:
    --method baseline : Direct Caption → T2V (one-shot, no optimization)
    --method pflow   : Iterative Prompt Optimization + Noise Prior (ours)

Both methods use the same:
    - Data: 200 videos from MovieGenVideoBench (motion_level=high)
    - Model: Wan2.1-T2V-1.3B for generation, Qwen2.5-VL-7B for captioning
    - Evaluation: CLIP / X-CLIP / STREAM (run separately after generation)

Usage:
    # Run baseline on all 200 samples
    python run_batch.py --method baseline --data_dir /path/to/video-200

    # Run P-Flow (ours) on all 200 samples
    python run_batch.py --method pflow --data_dir /path/to/video-200

    # Run specific samples
    python run_batch.py --method baseline --sample_ids 1 2 3

    # Run with pre-computed captions (skip VLM captioning)
    python run_batch.py --method baseline --caption_dir /path/to/captions_qwen

    # Resume from where you left off (skip existing outputs)
    python run_batch.py --method pflow --resume
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch runner for video reproduction experiments"
    )

    # Method selection
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["baseline", "pflow"],
        help="Method to run: 'baseline' (direct caption) or 'pflow' (iterative optimization)"
    )

    # Data paths
    parser.add_argument(
        "--data_dir", type=str,
        default="/root/autodl-tmp/data/video-200/water_mark_out",
        help="Directory containing reference videos ({id}.mp4)"
    )
    parser.add_argument(
        "--caption_dir", type=str, default=None,
        help="Directory with pre-computed captions ({id}.txt). "
             "If provided, skip VLM captioning step."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory. Default: /root/autodl-tmp/outputs/{method}_batch"
    )

    # Sample selection
    parser.add_argument(
        "--sample_ids", type=int, nargs="+", default=None,
        help="Specific sample IDs to process. Default: all found in data_dir."
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start index (for slicing the sample list)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of samples to process"
    )

    # Generation parameters
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps (aligned with collaborator)")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=15, help="FPS (aligned with collaborator: 15)")

    # P-Flow specific
    parser.add_argument(
        "--alpha", type=float, default=0.001,
        help="Noise prior blending weight (P-Flow only)"
    )
    parser.add_argument(
        "--i_max", type=int, default=10,
        help="Number of optimization iterations (P-Flow only)"
    )

    # Execution control
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip samples that already have output files"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config YAML file (overrides CLI args)"
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def build_config(args) -> dict:
    """Build configuration dict from CLI args."""
    import yaml

    # Start with defaults
    config = {
        "model": {
            "t2v_path": "/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers",
            "dtype": "bfloat16",
        },
        "video": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.steps,
        },
        "noise_prior": {
            "alpha": args.alpha,
            "rho_s": 0.1,
            "rho_m": 0.9,
            "inversion_steps": 50,
            "guidance_scale": 1.0,
        },
        "optimization": {
            "i_max": args.i_max,
        },
        "vlm": {
            "provider": "local",
            "model_path": "/root/models/Qwen2.5-VL-7B-Instruct",
            "temperature": 0.7,
            "max_tokens": 2048,
            "max_retries": 3,
            "use_video_mode": True,
            "quantization": None,
            "lazy_load": True,
        },
    }

    # Override with YAML config if provided
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        _deep_merge(config, yaml_config)

    return config


def _deep_merge(base: dict, override: dict):
    """Deep merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def discover_samples(data_dir: str, sample_ids=None, start=0, limit=None):
    """
    Discover video samples in data directory.

    Returns list of (sample_id, video_path) tuples.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all .mp4 files
    video_files = sorted(data_path.glob("*.mp4"), key=lambda p: int(p.stem))

    if sample_ids:
        # Filter to specific IDs
        id_set = set(sample_ids)
        samples = [(int(p.stem), str(p)) for p in video_files if int(p.stem) in id_set]
    else:
        samples = [(int(p.stem), str(p)) for p in video_files]

    # Apply slicing
    samples = samples[start:]
    if limit:
        samples = samples[:limit]

    return samples


def load_caption(caption_dir: str, sample_id: int) -> str:
    """Load pre-computed caption for a sample."""
    caption_path = Path(caption_dir) / f"{sample_id}.txt"
    if caption_path.exists():
        return caption_path.read_text(encoding="utf-8").strip()
    return None


def run_baseline(args, config, samples):
    """Run baseline method on all samples."""
    from src.baseline import BaselinePipeline

    output_dir = args.output_dir or "/root/autodl-tmp/outputs/baseline_batch"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pipeline = BaselinePipeline(config)
    results = []
    total = len(samples)

    # If we have pre-computed captions, we can skip VLM loading for captioning
    # But still need VLM for initial caption if no caption_dir
    has_captions = args.caption_dir is not None

    for idx, (sample_id, video_path) in enumerate(samples, 1):
        # Check if already done
        final_output = Path(output_dir) / f"{sample_id}.mp4"
        if args.resume and final_output.exists() and final_output.stat().st_size > 0:
            print(f"[{idx}/{total}] Skipping (exists): {sample_id}")
            continue

        print(f"[{idx}/{total}] Processing sample {sample_id}...")

        # Get caption
        caption = None
        if has_captions:
            caption = load_caption(args.caption_dir, sample_id)
            if caption:
                print(f"  Using pre-computed caption: {caption[:60]}...")

        # Run
        result = pipeline.run_single(
            video_path=video_path,
            output_dir=output_dir,
            sample_id=sample_id,
            seed=args.seed,
            caption=caption,
        )
        results.append(result)

    # Save batch results
    summary = {
        "method": "baseline",
        "total_samples": total,
        "completed": len(results),
        "config": config,
        "results": results,
    }
    with open(Path(output_dir) / "batch_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"Baseline batch complete: {len(results)}/{total} samples")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


def run_pflow(args, config, samples):
    """Run P-Flow method on all samples."""
    from src.baseline import BaselinePipeline
    from src.enhancement import PFlowEnhancer

    output_dir = args.output_dir or "/root/autodl-tmp/outputs/pflow_batch"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # First, get captions (reuse baseline's captioning)
    baseline = BaselinePipeline(config)
    has_captions = args.caption_dir is not None

    # Create enhancer (shares Wan pipeline with baseline)
    enhancer = PFlowEnhancer(config)

    results = []
    total = len(samples)

    for idx, (sample_id, video_path) in enumerate(samples, 1):
        # Check if already done
        sample_output_dir = Path(output_dir) / f"sample_{sample_id}"
        final_output = sample_output_dir / f"{sample_id}.mp4"
        if args.resume and final_output.exists() and final_output.stat().st_size > 0:
            print(f"[{idx}/{total}] Skipping (exists): {sample_id}")
            continue

        print(f"[{idx}/{total}] Processing sample {sample_id} with P-Flow...")

        # Step 1: Get initial caption
        caption = None
        if has_captions:
            caption = load_caption(args.caption_dir, sample_id)

        if caption is None:
            print(f"  Generating caption...")
            caption = baseline.caption_video(video_path)
            # Save caption for reuse
            caption_save_dir = Path(output_dir) / "captions"
            caption_save_dir.mkdir(parents=True, exist_ok=True)
            (caption_save_dir / f"{sample_id}.txt").write_text(
                caption + "\n", encoding="utf-8"
            )

        print(f"  Caption: {caption[:60]}...")

        # Step 2: Run P-Flow enhancement
        # Share the Wan pipeline between baseline and enhancer
        enhancer.pipe = baseline.pipe

        result = enhancer.run_single(
            video_path=video_path,
            output_dir=str(sample_output_dir),
            sample_id=sample_id,
            seed=args.seed,
            initial_caption=caption,
        )
        results.append(result)

        # Also copy final video to flat output dir for easy evaluation
        import shutil
        flat_output = Path(output_dir) / f"{sample_id}.mp4"
        if final_output.exists():
            shutil.copy2(str(final_output), str(flat_output))

    # Save batch results
    summary = {
        "method": "pflow",
        "total_samples": total,
        "completed": len(results),
        "alpha": args.alpha,
        "i_max": args.i_max,
        "config": config,
        "results": results,
    }
    with open(Path(output_dir) / "batch_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"P-Flow batch complete: {len(results)}/{total} samples")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


def main():
    args = parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build config
    config = build_config(args)

    # Discover samples
    samples = discover_samples(
        data_dir=args.data_dir,
        sample_ids=args.sample_ids,
        start=args.start,
        limit=args.limit,
    )
    print(f"Found {len(samples)} samples to process")
    print(f"Method: {args.method}")
    print(f"Seed: {args.seed}, Steps: {args.steps}, Guidance: {args.guidance_scale}")
    if args.method == "pflow":
        print(f"Alpha: {args.alpha}, Iterations: {args.i_max}")
    print()

    # Run
    if args.method == "baseline":
        run_baseline(args, config, samples)
    else:
        run_pflow(args, config, samples)


if __name__ == "__main__":
    main()
