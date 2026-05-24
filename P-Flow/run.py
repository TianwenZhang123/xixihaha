#!/usr/bin/env python3
"""
Video Reproduction via Iterative Prompt Optimization.

Given a reference video, this script iteratively optimizes a T2V prompt
so that the generated video faithfully reproduces the reference.

Usage:
    # Basic: provide a reference video + initial prompt
    python run.py --video reference.mp4 --prompt "a cat jumping on a table"

    # Auto-generate initial prompt from video (recommended)
    python run.py --video reference.mp4 --auto_prompt

    # With hint for auto-prompt
    python run.py --video reference.mp4 --auto_prompt --hint "a cat playing"

    # Adjust motion guidance strength (alpha)
    python run.py --video reference.mp4 --prompt "..." --alpha 0.2

    # Quick test with mock VLM (no API key needed)
    python run.py --video reference.mp4 --prompt "..." --mock_vlm

    # Full config
    python run.py --video reference.mp4 --auto_prompt \
        --config configs/paper_default.yaml --seed 42 --alpha 0.1
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
        description="Video Reproduction via Iterative Prompt Optimization + Noise Prior"
    )
    parser.add_argument("--video", type=str, required=True,
                       help="Reference video path (the video to reproduce)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Initial text prompt describing the video content")
    parser.add_argument("--auto_prompt", action="store_true",
                       help="Auto-generate initial prompt from reference video using VLM")
    parser.add_argument("--hint", type=str, default="",
                       help="Optional hint for auto-prompt generation")
    parser.add_argument("--output", type=str, default="/root/autodl-tmp/outputs/video_reproduction",
                       help="Output directory (default: /root/autodl-tmp/outputs/video_reproduction)")
    parser.add_argument("--config", type=str, default=None,
                       help="Config YAML (default: configs/paper_default.yaml)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=None,
                       help="Noise prior alpha (motion guidance strength). "
                            "0.001=minimal, 0.1=moderate, 0.3=strong. Default from config.")
    parser.add_argument("--i_max", type=int, default=None,
                       help="Number of optimization iterations (default: 10)")
    parser.add_argument("--mode", type=str, default="t2v", choices=["t2v", "i2v"])
    parser.add_argument("--mock_vlm", action="store_true", help="Use mock VLM (testing)")
    parser.add_argument("--noise_prior_only", action="store_true",
                       help="Run only noise prior (no prompt optimization, for testing)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Validate args
    if not args.auto_prompt and args.prompt is None:
        parser.error("Either --prompt or --auto_prompt is required")

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

    # Override i_max if specified
    config_override = None
    if args.i_max is not None:
        config_override = {"optimization": {"i_max": args.i_max}}

    # Create pipeline
    pipeline = PFlowPipeline(
        config_path=config_path,
        config=config_override,
        use_mock_vlm=args.mock_vlm,
    )

    # Determine initial prompt
    if args.auto_prompt:
        print("Generating initial prompt from reference video...")
        prompt = pipeline.generate_initial_prompt(
            reference_video_path=args.video,
            user_hint=args.hint,
        )
        print(f"Generated prompt: {prompt}\n")
    else:
        prompt = args.prompt

    # Run
    if args.noise_prior_only:
        # Quick test: noise prior only
        output_path = str(Path(args.output) / "noise_prior_only.mp4")
        result_path = pipeline.run_noise_prior_only(
            reference_video_path=args.video,
            prompt=prompt,
            output_path=output_path,
            seed=args.seed,
            alpha_override=args.alpha,
        )
        print(f"\nDone! Output: {result_path}")
        print("This video uses noise prior only (no prompt optimization).")
        print("Compare with full pipeline to see the effect of iterative refinement.")
    else:
        # Full pipeline
        result = pipeline.run(
            reference_video_path=args.video,
            prompt=prompt,
            output_dir=args.output,
            seed=args.seed,
            video_description=args.hint,
            mode=args.mode,
            alpha_override=args.alpha,
        )

        print(f"\n{'='*60}")
        print(f"Done! Output directory: {result['output_dir']}")
        print(f"{'='*60}")
        print(f"Reference video: {result['reference_video']}")
        print(f"Initial prompt: {result['initial_prompt'][:80]}...")
        print(f"Final prompt: {result['final_prompt'][:80]}...")
        print(f"Noise prior alpha: {result['noise_prior_alpha']}")
        print(f"Total iterations: {result['num_iterations']}")
        print(f"Total time: {result['total_time']:.1f}s ({result['total_time']/60:.1f} min)")
        print(f"\nGenerated videos:")
        for vp in result['video_paths']:
            print(f"  {vp}")
        print(f"\nReview the generated videos and compare with reference.mp4")
        print(f"The later iterations should be closer to the reference.")


if __name__ == "__main__":
    main()
