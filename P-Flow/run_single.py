#!/usr/bin/env python3
"""
Single Video Runner — demonstrates baseline vs P-Flow on one video.

Usage:
    # Baseline: direct caption → generate (fast, ~1 min)
    python run_single.py --method baseline --video /path/to/ref.mp4

    # P-Flow: iterative optimization (slow, ~10 min)
    python run_single.py --method pflow --video /path/to/ref.mp4

    # P-Flow with custom alpha and iterations
    python run_single.py --method pflow --video ref.mp4 --alpha 0.01 --i_max 5

    # Provide initial prompt instead of auto-captioning
    python run_single.py --method baseline --video ref.mp4 --prompt "a cat jumping"
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline or P-Flow on a single video"
    )
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "pflow"],
                        help="'baseline' or 'pflow'")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to reference video")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Initial prompt (skip VLM captioning if provided)")
    parser.add_argument("--output", type=str,
                        default="/root/autodl-tmp/outputs/single_test",
                        help="Output directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="Noise prior alpha (P-Flow only)")
    parser.add_argument("--i_max", type=int, default=10,
                        help="Iterations (P-Flow only)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build config
    import yaml
    config = {
        "model": {
            "t2v_path": "/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers",
            "dtype": "bfloat16",
        },
        "video": {
            "height": 480, "width": 832, "num_frames": 81,
            "fps": 16, "guidance_scale": 5.0, "num_inference_steps": 50,
        },
        "noise_prior": {
            "alpha": args.alpha, "rho_s": 0.1, "rho_m": 0.9,
            "inversion_steps": 50, "guidance_scale": 1.0,
        },
        "optimization": {"i_max": args.i_max},
        "vlm": {
            "provider": "local",
            "model_path": "/root/models/Qwen2.5-VL-7B-Instruct",
            "temperature": 0.7, "max_tokens": 2048, "max_retries": 3,
            "use_video_mode": True, "quantization": None, "lazy_load": True,
        },
    }

    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        for key, value in yaml_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value

    output_dir = Path(args.output) / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "baseline":
        from src.baseline import BaselinePipeline

        pipeline = BaselinePipeline(config)
        result = pipeline.run_single(
            video_path=args.video,
            output_dir=str(output_dir),
            sample_id=0,
            seed=args.seed,
            caption=args.prompt,
        )

        print(f"\n{'='*60}")
        print(f"[Baseline] Done!")
        print(f"  Caption: {result['caption'][:80]}...")
        print(f"  Output: {result['generated_video']}")
        print(f"  Time: {result['time_seconds']:.1f}s")
        print(f"{'='*60}")

    else:  # pflow
        from src.baseline import BaselinePipeline
        from src.enhancement import PFlowEnhancer

        # Step 1: Get caption (baseline step)
        if args.prompt:
            caption = args.prompt
            print(f"Using provided prompt: {caption[:60]}...")
        else:
            print("Generating caption from video...")
            baseline = BaselinePipeline(config)
            caption = baseline.caption_video(args.video)
            baseline.unload_vlm()  # Free VLM memory for T2V
            print(f"Caption: {caption[:80]}...")

        # Step 2: Run P-Flow enhancement
        enhancer = PFlowEnhancer(config)
        result = enhancer.run_single(
            video_path=args.video,
            output_dir=str(output_dir),
            sample_id=0,
            seed=args.seed,
            initial_caption=caption,
        )

        print(f"\n{'='*60}")
        print(f"[P-Flow] Done!")
        print(f"  Initial caption: {caption[:60]}...")
        print(f"  Final prompt: {result['final_prompt'][:60]}...")
        print(f"  Best iteration: {result['best_iteration']}/{result['total_iterations']}")
        print(f"  Alpha: {result['alpha']}")
        print(f"  Time: {result['total_time_seconds']:.1f}s")
        print(f"  Output: {output_dir}/{result['sample_id']}.mp4")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
